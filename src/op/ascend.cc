// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tl/op/ascend.cc
 *
 * Define ascend-related operators.
 */

#include "ascend.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "builtin.h"

namespace tvm {
namespace tl {

using namespace tir;

AscendCopy::AscendCopy(Array<PrimExpr> args, BufferMap vmap) : args_(args) {
  Array<Range> rgs[2];
  Buffer bf[2];
  for (int i = 0; i < 2; i++) {
    auto expr = args[i];
    auto call = expr.as<CallNode>();
    ICHECK(call);
    auto region = RegionOp(call->args, vmap);
    rgs[i] = region.GetRanges();
    bf[i] = region.GetBuffer();
  }
  if (args.size() >= 3) {
    srcN = static_cast<int>(args[2].as<IntImmNode>()->value);
    enRelu = args[3].as<Bool>().value();
  }
  std::tie(this->src, this->dst) = std::tie(bf[0], bf[1]);
  std::tie(this->src_range, this->dst_range) = std::tie(rgs[0], rgs[1]);
}

Stmt AscendCopy::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  auto get_dtype = [](const Buffer &buf) -> std::string {
    auto dtype = buf->dtype;
    if (dtype.is_float16()) {
      return "half";
    } else if (dtype.is_float() && dtype.bits() == 32) {
      return "float";
    } else if (dtype.is_int()) {
      return "int";
    } else if (dtype.is_uint() && dtype.bits() == 8) {
      return "uint8_t";
    } else if (dtype.is_uint() && dtype.bits() == 32) {
      return "uint32_t";
    }
    LOG(FATAL) << "Unsupported data type: " << dtype;
    return "";
  };

  std::stringstream ss;
  ss << "tl::ascend::";
  bool flag = false;
  bool gm2ub_flag = false;
  bool print_dst_layout = false;
  bool print_src_layout = false;
  bool print_gm_layout = false;
  bool print_ub = false;
  if (src.scope() == "global" && dst.scope() == "shared.dyn") {
    ss << "copy_gm_to_l1";
  } else if (src.scope() == "shared.dyn" && dst.scope() == "wmma.matrix_a") {
    ss << "copy_l1_to_l0a";
    print_src_layout = true;
  } else if (src.scope() == "shared.dyn" && dst.scope() == "wmma.matrix_b") {
    ss << "copy_l1_to_l0b";
    print_src_layout = true;
  } else if (src.scope() == "wmma.accumulator" && dst.scope() == "global") {
    ss << "copy_l0c_to_gm";
    flag = true;
    print_gm_layout = true;
  } else if (src.scope() == "shared" || dst.scope() == "shared") {
    if (src.scope() == "global") {
      gm2ub_flag = true;
      ss << "copy_gm_to_ub<";
      ss << get_dtype(src) << ", ";
      ss << src->shape[src->shape.size() - 1] << ", ";
      ss << dst->shape[dst->shape.size() - 1];
      if (dst->shape.size() > 1) {
        ss << ", " << dst->shape[dst->shape.size() - 2];
      }
      ss << ">";
    } else if (dst.scope() == "global") {
      ss << "copy_ub_to_gm<";
      ss << get_dtype(dst) << ", ";
      ss << dst->shape[dst->shape.size() - 1] << ", ";
      ss << src->shape[src->shape.size() - 1];
      if (src->shape.size() > 1) {
        ss << ", " << src->shape[src->shape.size() - 2];
      }
      ss << ">";
    } else {
      ss << "copy_ub_to_ub<";
      ss << get_dtype(dst) << ", ";
      ss << get_dtype(src) << ", ";
      PrimExpr len = 1;
      for (auto &shape : src->shape) {
        len *= shape;
      }
      ss << len;
      ss << ">";
    }

    print_ub = true;
  } else {
    LOG(FATAL) << "Unsupported scope: src = " << src.scope()
               << ", dst = " << dst.scope();
  }
  if (!print_ub) {
    ss << "<" << get_dtype(src) << ", ";

    if (flag) {
      ss << get_dtype(dst) << ", ";
    }
    if (print_gm_layout) {
      if (T.layout_map.count(src))
        ss << T.layout_map[src]->AscendLayoutStr() << ", ";
      else
        ss << "layout::RowMajor, ";
    }

    if (print_src_layout) {
      ICHECK(T.layout_map.count(src))
          << "Layout map does not contain source buffer: " << src->name;
      ss << T.layout_map[src]->AscendLayoutStr() << ", ";
    } else if (print_dst_layout) {
      ICHECK(T.layout_map.count(dst))
          << "Layout map does not contain destination buffer: " << dst->name;
      ss << T.layout_map[dst]->AscendLayoutStr() << ", ";
    }
    int src_ndim = src->shape.size(), dst_ndim = dst->shape.size();
    ss << src->shape[src_ndim - 2] << ", " << src->shape[src_ndim - 1] << ", "
       << dst->shape[dst_ndim - 2] << ", " << dst->shape[dst_ndim - 1] << ">";
  }
  Array<PrimExpr> src_indices, dst_indices;

  for (size_t i = 0; i < src_range.size(); i++) {
    src_indices.push_back(src_range[i]->min);
  }

  for (size_t i = 0; i < dst_range.size(); i++) {
    dst_indices.push_back(dst_range[i]->min);
  }

  auto src_new_indices = T.layout_map.count(src)
                             ? T.layout_map[src]->Forward(src_indices)
                             : src_indices;
  auto dst_new_indices = T.layout_map.count(dst)
                             ? T.layout_map[dst]->Forward(dst_indices)
                             : dst_indices;

  auto src_new_buffer = T.buffer_remap.count(src) ? T.buffer_remap[src] : src;
  auto dst_new_buffer = T.buffer_remap.count(dst) ? T.buffer_remap[dst] : dst;
  auto src_ptr = src_new_buffer.access_ptr(
      1, DataType::Handle(), 1,
      src_new_buffer.OffsetOf(src_new_indices).back());
  auto dst_ptr = dst_new_buffer.access_ptr(
      2, DataType::Handle(), 1,
      dst_new_buffer.OffsetOf(dst_new_indices).back());
  Array<PrimExpr> new_args;
  new_args.push_back(StringImm(ss.str()));

  new_args.push_back(src_ptr);
  new_args.push_back(dst_ptr);

  if (gm2ub_flag) {
    new_args.push_back(srcN);
  }
  if (flag) {
    new_args.push_back(enRelu);
  }

  auto new_call = Call(DataType::Handle(), builtin::call_extern(), new_args);
  return Evaluate(new_call);
}

LayoutMap AscendCopy::InferLayout(const LayoutInferArgs &T, InferLevel level) {
  LayoutMap results;
  // TODO: add logic to infer layout for AscendCopy
  return results;
}

TIR_REGISTER_TL_OP(AscendCopy, ascend_copy)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
} // namespace tl
} // namespace tvm