// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file target/codegen.cc
 */

#include "codegen_ascend.h"
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/op.h>

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "../op/builtin.h"

#include "arith/pattern_match.h"

namespace tvm {
namespace codegen {

std::string getType(const DataType &dtype) {
  if (dtype.is_float16()) {
    return "half";
  } else if (dtype.is_float() && dtype.bits() == 32) {
    return "float";
  }
  LOG(FATAL) << "Unsupported data type: " << dtype;
  return "";
}

static std::string GetFP8Type(DataType type) {
  std::stringstream stream;
  int32_t lanes = type.lanes();
  std::string vec;
  if (type.is_scalar()) {
    vec = "";
  } else if (lanes == 2) {
    vec = "_2";
  } else if (lanes == 4) {
    vec = "_4";
  } else if (lanes == 8) {
    vec = "_8";
  } else if (lanes == 16) {
    vec = "_16";
  } else {
    LOG(FATAL) << "Only support scalar and vector types of width (2, 4, 8, 16) "
                  "for FP8";
  }
  if (type.code() == DataType::kFloat8_e4m3fn) {
    stream << "fp8_e4" << vec << "_t";
  } else if (type.code() == DataType::kFloat8_e5m2) {
    stream << "fp8_e5" << vec << "_t";
  } else {
    LOG(FATAL) << "Unsupported FP8 type in CUDA codegen";
  }
  return stream.str();
}

CodeGenTileLangAscend::CodeGenTileLangAscend() {
  restrict_keyword_ = "GM_ADDR";
}

void CodeGenTileLangAscend::PrintFuncPrefix(std::ostream &os) {
  os << "CATLASS_GLOBAL\n";
}

std::string CodeGenTileLangAscend::Finish() {
  decl_stream << "#include \"tl_templates/ascend/common.h\"\n";
  decl_stream << "#include \"acl/acl.h\"\n";
  decl_stream << "using namespace Catlass;\n";
  decl_stream << "\n";
  std::ostringstream code;
  code << decl_stream.str();
  code << stream.str();
  return code.str();
}

void CodeGenTileLangAscend::VisitStmt_(const tir::ForNode *op) {
  auto flush = false;
  if (flush_out_) {
    flush = true;
    flush_out_ = false;
  }
  if (op->kind == tir::ForKind::kUnrolled) {
    PrintIndent();
    stream << "#pragma unroll\n";
  }
  std::string extent =
      PrintExpr(arith::Analyzer().Simplify(op->extent + op->min));
  PrintIndent();
  std::string vid = AllocVarID(op->loop_var.get());
  std::string start = PrintExpr(op->min);
  stream << "for (";
  PrintType(op->loop_var.dtype(), stream);
  stream << ' ' << vid << " = " << start << "; " << vid << " < " << extent
         << "; ++" << vid << ") {\n";
  int for_scope = BeginScope();
  PrintStmt(op->body);
  this->EndScope(for_scope);
  PrintIndent();
  stream << "}\n";
  if (flush) {
    while (!inst_.empty()) {
      PrintIndent();
      stream << inst_.back();
      inst_.pop_back();
    }
  }
}

void CodeGenTileLangAscend::PrintType(DataType t,
                                      std::ostream &os) { // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    ICHECK(t.is_scalar()) << "do not yet support vector types";
    os << "void*";
    return;
  }

  if (t.is_void()) {
    os << "void";
    return;
  }

  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
    case 16:
      enable_fp16_ = true;
      if (t.is_scalar()) {
        os << "half_t";
      } else if (lanes <= 8) {
        // Emit CUDA code to access fp16 vector elements.
        //
        // half4 is stored as uint2
        //
        // h4.x is emitted as *(half2*)(&(u2.x)).x
        // h4.y is emitted as *(half2*)(&(u2.x)).y
        // h4.z is emitted as *(half2*)(&(u2.y)).x
        // h4.w is emitted as *(half2*)(&(u2.y)).y
        //
        ICHECK_EQ(lanes % 2, 0) << "only support even lane for half type";
        os << "uint" << lanes / 2;
      } else {
        fail = true;
      }
      break;
    case 32:
      if (lanes <= 4) {
        os << "float";
      } else if (lanes <= 8) {
        // Emit CUDA code to access fp32 vector elements for 4 < lanes <= 8.
        //
        // float8 is stored as ulonglong4
        //
        // f8.v1 is emitted as *(float2*)(&(ul4.x)).x
        // f8.v2 is emitted as *(float2*)(&(ul4.x)).y
        //
        ICHECK_EQ(lanes % 2, 0)
            << "only support even lane for float type with lanes > 4";
        os << "ulonglong" << lanes / 2;
      } else {
        fail = true;
      }
      break;
    case 64:
      os << "double";
      break;
    default:
      fail = true;
      break;
    }
    if (!fail && (t.is_scalar() || t.bits() == 16))
      return;
    if (!fail && (lanes > 4 && lanes <= 8 && t.bits() == 32))
      return;
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  } else if (t.is_bfloat16()) {
    enable_bf16_ = true;
    if (t.is_scalar()) {
      os << "bfloat16_t";
    } else if (lanes <= 8) {
      ICHECK_EQ(lanes % 2, 0) << "only support even lane for half type";
      os << "uint" << lanes / 2;
    } else {
      fail = true;
    }
    if (!fail)
      return;
  } else if (t.is_float8()) {
    enable_fp8_ = true;
    os << GetFP8Type(t);
    return;
  } else if (t == DataType::Bool()) {
    os << "bool";
    return;
  } else if (t.is_vector_bool()) {
    // CUDA does not support bool vectors.
    // Use ushort vectors to represent instead.
    int n = t.lanes();
    if (n <= 4) {
      os << "ushort" << n;
      return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << "u";
    }
    switch (t.bits()) {
    case 1: {
      if (t.is_scalar()) {
        os << "int";
        return;
      } else if (t.lanes() == 8) {
        os << "int8_t";
        return;
      } else if (t.lanes() == 16) {
        os << "int16_t";
        return;
      } else if (t.lanes() == 32) {
        os << "int";
        return;
      } else {
        LOG(FATAL) << "Cannot convert type " << t << " to CUDA type!";
      }
    }
    case 4: {
      if (t.is_scalar()) {
        os << "int";
        return;
      } else if (t.lanes() == 4) {
        os << "int16_t";
        return;
      } else if (t.lanes() == 8) {
        // directly 8 4-bit int in integer.
        os << "int";
        return;
      } else if (t.lanes() == 16) {
        os << "int2";
        return;
      } else if (t.lanes() == 32) {
        os << "int4";
        return;
      } else if (t.lanes() == 64) {
        os << "int8";
        return;
      } else {
        LOG(FATAL) << "Cannot convert type " << t << " to CUDA type!";
      }
    }
    case 8: {
      if (t.lanes() == 4) {
        // directly 4 8 bit int in integer.
        enable_int8_ = true;

        // We use int for int8x4 instead of char4 because using char4 is
        // likely to produce extra instructions to pack four int8 elements
        // into 32-bit data.
        os << "int";
        return;
      } else if (t.lanes() == 8) {
        enable_int8_ = true;
        os << "int2";
        return;
      } else if (t.lanes() == 16) {
        enable_int8_ = true;
        os << "int4";
        return;
      } else if (!t.is_uint() && t.is_scalar()) {
        os << "signed char";
        break;
      } else {
        os << "char";
        break;
      }
    }
    case 16: {
      if (t.is_scalar()) {
        os << "short";
      } else if (t.lanes() <= 4) {
        os << "short" << lanes;
      } else if (t.lanes() <= 8) {
        // Emit CUDA code to access int16 vector elements.
        //
        // short4 is stored as int2
        //
        // s4.x is emitted as *(short2*)(&(i2.x)).x
        // s4.y is emitted as *(short2*)(&(i2.x)).y
        // s4.z is emitted as *(short2*)(&(i2.y)).x
        // s4.w is emitted as *(short2*)(&(i2.y)).y
        //
        ICHECK_EQ(t.lanes() % 2, 0)
            << "only support even lane for shorT type with lanes > 4";
        os << "int" << t.lanes() / 2;
      } else {
        fail = true;
      }
      if (!fail) {
        return;
      }
      break;
    }
    case 32: {
      if (t.is_scalar()) {
        os << "int";
      } else if (t.lanes() <= 4) {
        os << "int" << t.lanes();
      } else if (t.lanes() <= 8) {
        // Emit CUDA code to access int32 vector elements for 4 < lanes <= 8.
        //
        // int8 is stored as longlong4
        //
        // i8.v1 is emitted as *(int2*)(&(l4.x)).x
        // i8.v2 is emitted as *(int2*)(&(l4.x)).y
        //
        ICHECK_EQ(lanes % 2, 0)
            << "only support even lane for int32 type with lanes > 4";
        os << "longlong" << lanes / 2;
      } else {
        fail = true;
      }
      if (!fail) {
        return;
      }
      break;
    }
    case 64: {
      if (t.is_scalar()) {
        os << "int64_t";
      } else if (t.lanes() == 2) {
        os << "longlong2";
      } else if (t.lanes() == 3) {
        os << "longlong3";
      } else if (t.lanes() == 4) {
        os << "longlong4";
      }
      return;
    }
    default:
      fail = true;
      break;
    }
    if (!fail && lanes == 1) {
      return;
    }
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to CUDA type";
}

void CodeGenTileLangAscend::PrintStorageScope(const std::string &scope,
                                              std::ostream &os) { // NOLINT(*)
}

void CodeGenTileLangAscend::VisitExpr_(const FloorDivNode *op,
                                       std::ostream &os) {
  os << "(";
  PrintExpr(op->a, os);
  os << " / ";
  PrintExpr(op->b, os);
  os << ")";
}

void CodeGenTileLangAscend::VisitExpr_(const FloorModNode *op,
                                       std::ostream &os) {
  os << "(";
  PrintExpr(op->a, os);
  os << " % ";
  PrintExpr(op->b, os);
  os << ")";
}

void CodeGenTileLangAscend::VisitExpr_(const CallNode *op, std::ostream &os) {
  if (op->op.same_as(builtin::call_extern())) {
    std::string op_name = Downcast<StringImm>(op->args[0])->value;
    if (op_name.find("copy") != std::string::npos) {
      this->PrintIndent();
      this->stream << "{\n";
      int func_scope = this->BeginScope();

      auto src_var = op->args[1].as<CallNode>()->args[1].as<VarNode>();
      auto dst_var = op->args[2].as<CallNode>()->args[1].as<VarNode>();

      auto src_var_id = var_idmap_[src_var];
      auto dst_var_id = var_idmap_[dst_var];
      if (src_var_id == "") {
        src_var_id = src_var->name_hint;
      }
      if (dst_var_id == "") {
        dst_var_id = dst_var->name_hint;
      }

      auto src_offset = PrintExpr(op->args[1].as<CallNode>()->args[2]);
      auto dst_offset = PrintExpr(op->args[2].as<CallNode>()->args[2]);

      auto src_type = op->args[1].as<CallNode>()->args[0].as<CallNode>()->dtype;
      auto dst_type = op->args[2].as<CallNode>()->args[0].as<CallNode>()->dtype;

      if (op_name.find("copy_l0c_to_gm") != std::string::npos) {
        this->PrintIndent();
        this->stream << "auto " << src_var_id << "_ = " << src_var_id << ".Get<"
                     << getType(src_type) << ">();\n";

        this->PrintIndent();
        this->stream << op_name << "(" << dst_var_id << "[" << dst_offset
                     << "], " << src_var_id << "_[" << src_offset << "]);\n";
      } else if (op_name.find("copy_gm_to_l1") != std::string::npos) {
        this->PrintIndent();
        this->stream << "auto " << dst_var_id << "_ = " << dst_var_id << ".Get<"
                     << getType(dst_type) << ">();\n";
        this->PrintIndent();
        this->stream << op_name << "(" << dst_var_id << "_[" << dst_offset
                     << "], " << src_var_id << "[" << src_offset << "]);\n";
      } else if (op_name.find("copy_l1_to_l0a") != std::string::npos) {

        auto type = getType(src_type);
        this->PrintIndent();
        this->stream << "auto " << src_var_id << "_ = " << src_var_id << ".Get<"
                     << type << ">();\n";
        this->PrintIndent();
        this->stream << "auto " << dst_var_id << "_ = " << dst_var_id << ".Get<"
                     << type << ">();\n";
        this->PrintIndent();
        this->stream << op_name << "(" << dst_var_id << "_[" << dst_offset
                     << "], " << src_var_id << "_[" << src_offset << "]);\n";
      } else if (op_name.find("copy_l1_to_l0b") != std::string::npos) {

        auto type = getType(src_type);
        this->PrintIndent();
        this->stream << "auto " << src_var_id << "_ = " << src_var_id << ".Get<"
                     << type << ">();\n";
        this->PrintIndent();
        this->stream << "auto " << dst_var_id << "_ = " << dst_var_id << ".Get<"
                     << type << ">();\n";
        this->PrintIndent();

        this->stream << op_name << "(" << dst_var_id << "_[" << dst_offset
                     << "], " << src_var_id << "_[" << src_offset << "]);\n";
      } else if (op_name.find("copy_gm_to_ub") != std::string::npos) {
        this->PrintIndent();
        this->stream << "auto " << dst_var_id << "_ = " << dst_var_id << ".Get<"
                     << getType(dst_type) << ">();\n";
        this->PrintIndent();
        this->stream << op_name << "(" << dst_var_id << "_[" << dst_offset
                     << "], " << src_var_id << "[" << src_offset << "]);\n";
      } else if (op_name.find("copy_ub_to_gm") != std::string::npos) {
        this->PrintIndent();
        this->stream << "auto " << src_var_id << "_ = " << src_var_id << ".Get<"
                     << getType(src_type) << ">();\n";
        this->PrintIndent();
        this->stream << op_name << "(" << dst_var_id << "[" << dst_offset
                     << "], " << src_var_id << "_[" << src_offset << "]);\n";
      } else {
        this->PrintIndent();
        this->stream << "not implemented yet\n";
      }
      this->EndScope(func_scope);
      this->PrintIndent();
      this->stream << "}\n";
    } else if (op_name == "npu.fill") {
    } else if (op_name.find("mma") != std::string::npos) {

      this->PrintIndent();
      auto a_var = op->args[1].as<CallNode>()->args[1].as<VarNode>();
      auto b_var = op->args[2].as<CallNode>()->args[1].as<VarNode>();
      auto c_var = op->args[3].as<CallNode>()->args[1].as<VarNode>();

      auto a_offset = PrintExpr(op->args[1].as<CallNode>()->args[2]);
      auto b_offset = PrintExpr(op->args[2].as<CallNode>()->args[2]);
      auto c_offset = PrintExpr(op->args[3].as<CallNode>()->args[2]);

      auto a_name = var_idmap_[a_var];
      auto b_name = var_idmap_[b_var];
      auto c_name = var_idmap_[c_var];

      auto src_type = op->args[1].as<CallNode>()->args[0].as<CallNode>()->dtype;
      auto dst_type = op->args[3].as<CallNode>()->args[0].as<CallNode>()->dtype;

      stream << "{\n";
      int func_scope = this->BeginScope();
      this->PrintIndent();
      this->stream << "auto " << a_name << "_ = " << a_name << ".Get<"
                   << getType(src_type) << ">();\n";
      this->PrintIndent();
      this->stream << "auto " << b_name << "_ = " << b_name << ".Get<"
                   << getType(src_type) << ">();\n";
      auto init = Downcast<Bool>(op->args[9])->value;
      if (!init) {
        this->PrintIndent();
        this->stream << "auto " << c_name << "_ = " << c_name << ".Get<"
                     << getType(dst_type) << ">();\n";
      } else {
        this->PrintIndent();
        this->stream << "auto " << c_name << "_ = " << c_name << ".Get<"
                     << getType(dst_type) << ">();\n";
      }
      this->PrintIndent();
      this->stream << op_name << "(" << a_name << "_[" << a_offset << "],"
                   << b_name << "_[" << b_offset << "]," << c_name << "_["
                   << c_offset << "]);\n";
      this->EndScope(func_scope);
      this->PrintIndent();
      this->stream << "}\n";
    } else if (op_name.find("AscendC") != std::string::npos) {
      this->PrintIndent();
      this->stream << op_name << "(";
      for (size_t i = 1; i < op->args.size(); ++i) {
        if (i > 1) {
          this->stream << ", ";
        }
        PrintExpr(op->args[i], this->stream);
      }
      this->stream << ");\n";
    } else if (op_name.find("tile_add") != std::string::npos) {
      this->PrintIndent();
      auto a_var = op->args[1].as<CallNode>()->args[1].as<VarNode>();
      auto b_var = op->args[2].as<CallNode>()->args[1].as<VarNode>();
      auto c_var = op->args[3].as<CallNode>()->args[1].as<VarNode>();

      auto a_offset = PrintExpr(op->args[1].as<CallNode>()->args[2]);
      auto b_offset = PrintExpr(op->args[2].as<CallNode>()->args[2]);
      auto c_offset = PrintExpr(op->args[3].as<CallNode>()->args[2]);

      auto a_name = var_idmap_[a_var];
      auto b_name = var_idmap_[b_var];
      auto c_name = var_idmap_[c_var];

      auto src_type = op->args[1].as<CallNode>()->args[0].as<CallNode>()->dtype;

      stream << "{\n";
      int func_scope = this->BeginScope();
      this->PrintIndent();
      this->stream << "auto " << a_name << "_ = " << a_name << ".Get<"
                   << getType(src_type) << ">();\n";
      this->PrintIndent();
      this->stream << "auto " << b_name << "_ = " << b_name << ".Get<"
                   << getType(src_type) << ">();\n";
      if (c_name != a_name && c_name != b_name) {
        this->PrintIndent();
        this->stream << "auto " << c_name << "_ = " << c_name << ".Get<"
                     << getType(src_type) << ">();\n";
      }

      this->PrintIndent();
      this->stream << op_name << "(" << a_name << "_[" << a_offset << "],"
                   << b_name << "_[" << b_offset << "]," << c_name << "_["
                   << c_offset << "]);\n";
      this->EndScope(func_scope);
      this->PrintIndent();
      this->stream << "}\n";
    } else if (op_name.find("reduce") != std::string::npos) {
      this->PrintIndent();
      auto in_var = op->args[1].as<CallNode>()->args[1].as<VarNode>();
      auto out_var = op->args[2].as<CallNode>()->args[1].as<VarNode>();

      auto in_offset = PrintExpr(op->args[1].as<CallNode>()->args[2]);
      auto out_offset = PrintExpr(op->args[2].as<CallNode>()->args[2]);

      auto in_name = var_idmap_[in_var];
      auto out_name = var_idmap_[out_var];

      auto src_type = op->args[1].as<CallNode>()->args[0].as<CallNode>()->dtype;

      stream << "{\n";
      int func_scope = this->BeginScope();
      this->PrintIndent();
      this->stream << "auto " << in_name << "_ = " << in_name << ".Get<"
                   << getType(src_type) << ">();\n";
      this->PrintIndent();
      this->stream << "auto " << out_name << "_ = " << out_name << ".Get<"
                   << getType(src_type) << ">();\n";

      this->PrintIndent();
      this->stream << op_name << "(" << in_name << "_[" << in_offset << "],"
                   << out_name << "_[" << out_offset << "]);\n";
      this->EndScope(func_scope);
      this->PrintIndent();
      this->stream << "}\n";      
    }
  } else if (op->op.same_as(tl::add())) {

  } else if (op->op.same_as(tl::sub())) {

  } else if (op->op.same_as(tl::mul())) {

  } else if (op->op.same_as(tl::div())) {

  } else if (op->op.same_as(tl::adds())) {

  } else if (op->op.same_as(tl::muls())) {

  } else if (op->op.same_as(tl::exp())) {

  }
  else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenTileLangAscend::VisitStmt_(const AttrStmtNode *op) {
  if (op->attr_key == "threadblock_swizzle_pattern") {
    this->PrintIndent();
    const StringImmNode *pattern = op->value.as<StringImmNode>();
    ICHECK(pattern);
    this->stream << this->block_id_ << " = " << pattern->value << "("
                 << this->block_id_ << ");\n";
    this->VisitStmt(op->body);
    return;
  } else if (op->attr_key == "thread_extent") {
    IterVar iv = Downcast<IterVar>(op->node);
    if (iv->thread_tag == "blockIdx.x" && iv->var->name_hint != "_") {
      this->block_id_ = AllocVarID(iv->var.get());
      this->PrintIndent();
      this->stream << "auto " << this->block_id_
                   << " = AscendC::GetBlockIdx();\n";
      this->core_num_ = op->value.as<IntImmNode>()->value;
    } else if (iv->thread_tag == "blockIdx.y" && iv->var->name_hint != "_") {
      auto vec_id_ = AllocVarID(iv->var.get());
      this->PrintIndent();
      this->stream << "auto " << vec_id_ << " = AscendC::GetSubBlockIdx();\n";
    }
    this->VisitStmt(op->body);
    return;
  } else if (op->attr_key == "init_flag" || op->attr_key == "clear_flag") {
    const StringImmNode *instn = op->value.as<StringImmNode>();

    std::string inst = std::string(instn->value);
    size_t st = 0;
    for (size_t i = 0; i < inst.size(); ++i) {
      if (inst[i] == '\n') {
        this->PrintIndent();
        stream << inst.substr(st, i - st) << "\n";
        st = i + 1;
      }
    }
    this->VisitStmt(op->body);
    return;
  } else if (op->attr_key == "resource_scope") {
    auto resource_id = Downcast<IntImm>(op->value)->value;
    auto resource_name = resource_id == 0 ? "AIC" : "AIV";

    this->PrintIndent();
    stream << "if (g_coreType == AscendC::" << resource_name << ") {\n";
    int func_scope = this->BeginScope();
    this->VisitStmt(op->body);
    this->EndScope(func_scope);
    this->PrintIndent();
    stream << "}\n";
    return;
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangAscend::VisitStmt_(const AllocateNode *op) {
  ICHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());
  std::string scope = GetPtrStorageScope(op->buffer_var);
  const VarNode *buffer = op->buffer_var.as<VarNode>();

  auto print_buffer = [&](const std::string &pos) {
    this->PrintIndent();
    stream << "AscendC::TBuf<AscendC::TPosition::" << pos << "> " << vid
           << ";\n";
    this->PrintIndent();
    stream << "pipe.InitBuffer(" << vid << ", " << op->ConstantAllocationSize()
           << " * " << op->dtype.bytes() << ");\n";
  };

  if (scope == "wmma.matrix_a") {
    print_buffer("A2");
  } else if (scope == "wmma.matrix_b") {
    print_buffer("B2");
  } else if (scope == "wmma.accumulator") {
    print_buffer("CO1");
  } else if (scope == "shared.dyn") {
    print_buffer("A1");
  } else if (scope == "shared") {
    print_buffer("VECIN");
  }
  // add pipe.Destroy()
  if (!op->body.as<AllocateNode>()) {
    this->PrintIndent();
    this->stream << "pipe.Destroy();\n\n";
  }
  this->PrintStmt(op->body);
}

inline void PrintConst(const FloatImmNode *op, std::ostream &os,
                       CodeGenTileLangAscend *p) { // NOLINT(*)
  // Type code is kBFloat
  if (op->dtype.is_bfloat16()) {
    os << "bfloat16_t";
    os << '(' << std::scientific << op->value << 'f' << ')';
    return;
  }
  // Type code is kFloat8_e5m2 or kE4M4Float
  if (op->dtype.is_float8() || op->dtype.is_float4()) {
    p->PrintType(op->dtype, os);
    os << '(' << std::scientific << op->value << 'f' << ')';
    return;
  }
  // Type code is kFloat
  switch (op->dtype.bits()) {
  case 64:
  case 32: {
    std::ostringstream temp;
    if (std::isinf(op->value)) {
      if (op->value < 0) {
        temp << "-";
      }
      temp << ((op->dtype.bits() == 32) ? "CUDART_INF_F" : "CUDART_INF");
      p->need_math_constants_h_ = true;
    } else if (std::isnan(op->value)) {
      temp << ((op->dtype.bits() == 32) ? "CUDART_NAN_F" : "CUDART_NAN");
      p->need_math_constants_h_ = true;
    } else {
      temp << std::scientific << op->value;
      if (op->dtype.bits() == 32)
        temp << 'f';
    }
    p->MarkConst(temp.str());
    os << temp.str();
    break;
  }
  case 16: {
    os << "half_t" << '(';
    FloatImm const_f32 = FloatImm(DataType::Float(32), op->value);
    PrintConst(const_f32.get(), os, p);
    os << ')';
    break;
  }
  default:
    LOG(FATAL) << "Bad bit-width for float: " << op->dtype << "\n";
  }
}

void CodeGenTileLangAscend::VisitExpr_(const FloatImmNode *op,
                                       std::ostream &os) { // NOLINT(*)
  PrintConst(op, os, this);
}

void CodeGenTileLangAscend::PreFunctionBody(const PrimFunc &f) {
  int func_scope = this->BeginScope();
  this->PrintIndent();
  stream << "AscendC::TPipe pipe;\n\n";
  ICHECK(this->para_.size() % 3 == 0)
      << "CodeGenTileLangAscend: parameters should be in pairs of (var, "
         "handle, dtype)";
  for (size_t i = 0; i < this->para_.size(); i += 3) {
    this->PrintIndent();
    stream << "AscendC::GlobalTensor<" << this->para_[i + 2] << "> "
           << this->para_[i + 1] << ";\n";
    this->PrintIndent();
    stream << this->para_[i + 1] << ".SetGlobalBuffer((__gm__ "
           << this->para_[i + 2] << "*)" << this->para_[i] << ");\n";
  }
  stream << "\n";
  this->EndScope(func_scope);
}

void CodeGenTileLangAscend::VisitExpr_(const SelectNode *op, std::ostream &os) {
  auto condition = PrintExpr(op->condition);
  auto true_value = PrintExpr(op->true_value);
  auto false_value = PrintExpr(op->false_value);

  os << "(" << condition << " ? "
     << "" << true_value << " : " << false_value << ")";
}

void PrintHostFunc(const PrimFunc &f, const std::string &name, std::ostream &os,
                   int core) {
  os << "extern \"C\" void call(";
  std::vector<std::string> arg_names;
  for (size_t i = 0; i < f->params.size(); ++i) {
    auto v = f->params[i];
    if (i != 0) {
      os << ", ";
    }
    arg_names.push_back(v->name_hint);
    os << "uint8_t* " << v->name_hint;
  }
  os << ", aclrtStream stream) {\n  ";

  os << name << "<<<" << core << ", nullptr, stream>>>(";
  for (auto &arg_name : arg_names) {
    os << arg_name;
    if (arg_name != arg_names.back()) {
      os << ", ";
    }
  }
  os << ");\n";
  os << "}\n";
}

void CodeGenTileLangAscend::AddFunction(const GlobalVar &gvar,
                                        const PrimFunc &f) {
  // If the function has already been forward-declared, this is a
  // no-op.
  CodeGenC::DeclareFunction(gvar, f);
  // clear previous generated state.
  this->InitFuncState(f);

  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.defined())
      << "CodeGenC: Expect PrimFunc to have the global_symbol attribute";
  bool no_alias = f->HasNonzeroAttr(tir::attr::kNoAlias);

  this->PrintFuncPrefix(stream);
  CodeGenC::PrintType(f->ret_type, stream);
  auto func_name = static_cast<std::string>(global_symbol.value()) + "_kernel";
  this->stream << " " << func_name << "(";

  for (size_t i = 0; i < f->params.size(); ++i) {
    tir::Var v = f->params[i];
    std::string vid = AllocVarID(v.get());

    if (i != 0)
      stream << ", ";
    if (v.dtype().is_handle()) {
      auto real_v = f->buffer_map[v]->data;
      this->para_.push_back(vid);
      // vid = AllocVarID(real_v.get());
      this->para_.push_back(AllocVarID(real_v.get()));
      this->para_.push_back(getType(f->buffer_map[v]->dtype));
      PrintRestrict(v, stream);

      auto it = alloc_storage_scope_.find(v.get());
      if (it != alloc_storage_scope_.end()) {
        PrintStorageScope(it->second, stream);
      }

      if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
          RegisterHandleType(v.get(), prim->dtype);
        }
      }

    } else {
      CodeGenC::PrintType(GetType(v), stream);
    }
    stream << ' ' << vid;
  }
  stream << ") {\n";
  this->PreFunctionBody(f);
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";

  PrintHostFunc(f, func_name, stream, this->core_num_);
}

} // namespace codegen
} // namespace tvm
