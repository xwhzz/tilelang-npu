// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file lower_tile_op.cc
 * \brief Lower the tile op for further codegen.
 */

#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>

#include "../layout/layout.h"
#include "../layout/utils.h"
#include "../op/builtin.h"
#include "../op/op.h"

#include "arith/ir_mutator_with_analyzer.h"
#include "loop_partition.h"

namespace tvm {
namespace tl {

using namespace tir;

static Buffer makeBufferWithLayout(const Buffer &buffer, const Layout &layout,
                                   Map<Var, Var> &var_remap) {
  const auto *ptr_type =
      TVM_TYPE_AS(buffer->data->type_annotation, PointerTypeNode);
  Type new_type;
  // convert fragments to normal local buffer
  if (ptr_type->storage_scope == "local.fragment") {
    new_type = PointerType(ptr_type->element_type, "local");
  } else {
    new_type = buffer->data->type_annotation;
  }
  Var new_var;
  if (ptr_type->storage_scope == "global") {
    new_var = buffer->data;
  } else {
    if (var_remap.count(buffer->data)) {
      new_var = var_remap[buffer->data];
    } else {
      new_var = Var(buffer->data->name_hint, new_type);
      var_remap.Set(buffer->data, new_var);
    }
  }
  Array<PrimExpr> layout_shape = layout->OutputShape();
  Array<PrimExpr> output_shape = layout_shape;

  if (ptr_type->storage_scope == "shared" ||
      ptr_type->storage_scope == "shared.dyn") {
    int replicate_extent = 1;
    Array<PrimExpr> buffer_shape = buffer->shape;
    int buffer_extent = 1;
    int layout_extent = 1;
    for (size_t i = 0; i < buffer_shape.size(); i++) {
      auto shape = buffer_shape[i].as<IntImmNode>();
      buffer_extent *= shape->value;
    }
    for (size_t i = 0; i < layout_shape.size(); i++) {
      auto shape = layout_shape[i].as<IntImmNode>();
      layout_extent *= shape->value;
    }
    replicate_extent = buffer_extent / layout_extent;
    if (replicate_extent > 1) {
      output_shape.insert(output_shape.begin(), replicate_extent);
    }
  }
  return Buffer(new_var, buffer->dtype, output_shape, {}, buffer->elem_offset,
                buffer->name, buffer->data_alignment, buffer->offset_factor,
                buffer->buffer_type);
}

/*!
 * \brief A class that rewrites buffer references in a statement based on a
 * given buffer remapping.
 *
 * This class is used to update buffer references in a statement after buffer
 * transformations have been applied. It specifically handles the remapping of
 * padding annotations.
 */
class RemapBufferRewriter : public arith::IRMutatorWithAnalyzer {
public:
  /*!
   * \brief Substitute buffer references in a statement based on a given buffer
   * remapping. \param stmt The statement to rewrite. \param buffer_remap A map
   * from old buffers to new buffers. \return The rewritten statement.
   */
  static Stmt Substitute(Stmt stmt, Map<Buffer, Buffer> buffer_remap) {
    arith::Analyzer analyzer;
    RemapBufferRewriter substituter(&analyzer);
    substituter.buffer_remap_ = std::move(buffer_remap);
    return substituter.VisitStmt(stmt);
  }

private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  Stmt VisitStmt_(const BlockNode *op) final {
    if (op->annotations.count(attr::kPaddingMap)) {
      return RewritePaddingMap(op);
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  /*!
   * \brief Rewrite the padding map annotation of a block.
   * \param op The block node to rewrite.
   * \return The rewritten block.
   */
  Stmt RewritePaddingMap(const BlockNode *op) {
    auto padding_map =
        op->annotations.Get(attr::kPaddingMap).as<Map<Var, PrimExpr>>().value();

    Map<Var, Var> var_remap = CreateVarRemap();
    Map<Var, PrimExpr> new_padding_map =
        RemapPaddingMap(padding_map, var_remap);

    auto block = Downcast<Block>(IRMutatorWithAnalyzer::VisitStmt_(op));
    auto block_ptr = block.CopyOnWrite();
    block_ptr->annotations.Set(attr::kPaddingMap, new_padding_map);
    return block;
  }

  /*!
   * \brief Create a mapping from old variables to new variables based on buffer
   * remapping. \return A map from old variables to new variables.
   */
  Map<Var, Var> CreateVarRemap() const {
    Map<Var, Var> var_remap;
    for (const auto &[buffer, buffer_remap] : buffer_remap_) {
      var_remap.Set(buffer->data, buffer_remap->data);
    }
    return var_remap;
  }

  /*!
   * \brief Remap the padding map using the variable remapping.
   * \param padding_map The original padding map.
   * \param var_remap The variable remapping.
   * \return The remapped padding map.
   */
  Map<Var, PrimExpr> RemapPaddingMap(const Map<Var, PrimExpr> &padding_map,
                                     const Map<Var, Var> &var_remap) const {
    Map<Var, PrimExpr> new_padding_map;
    for (const auto &[var, padding] : padding_map) {
      if (var_remap.count(var)) {
        new_padding_map.Set(var_remap.at(var), padding);
      } else {
        new_padding_map.Set(var, padding);
      }
    }
    return new_padding_map;
  }

  Map<Buffer, Buffer> buffer_remap_;
};

class LowerTileOpPass : arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f) {
    arith::Analyzer analyzer;
    LowerTileOpPass substituter(&analyzer);
    // Trace the buffer map for tvm_access_ptr
    substituter.buffer_map_.insert(f->buffer_map.begin(), f->buffer_map.end());
    for (const auto &[_, buffer] : f->buffer_map) {
      substituter.buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target.defined()) << "LowerTileOpPass: Require the target attribute";
    substituter.target_ = target.value();
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = substituter.VisitStmt(f->body);
    fptr->body =
        RemapBufferRewriter::Substitute(fptr->body, substituter.buffer_remap_);
    auto fn_attr = fptr->attrs.CopyOnWrite();
    // for (auto op : alloc_ops) {
    //   int32_t address = addrMapWithBC[op];
    fn_attr->dict.Set("address_map", substituter.address_map_);
    return f;
  }

private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  Stmt VisitStmt_(const BlockNode *op) final {
    // Record the mapping from buffer data var to buffer for later lookup
    for (auto buffer : op->alloc_buffers) {
      buffer_map_.insert({buffer->data, buffer});
    }
    for (auto match_buffer : op->match_buffers) {
      buffer_map_.insert({match_buffer->buffer->data, match_buffer->buffer});
    }
    for (auto buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    Map<Var, Layout> vmap;
    if (op->annotations.count(attr::kLayoutMap)) {
      auto layout_map = op->annotations.at(attr::kLayoutMap)
                            .as<Map<Buffer, Layout>>()
                            .value();
      for (auto [buffer, layout] : layout_map) {
        buffer_remap_.Set(buffer,
                          makeBufferWithLayout(buffer, layout, var_remap_));
        layout_map_.Set(buffer, layout);
      }
    } else if (op->annotations.count("address_map")) {
      address_map_ = op->annotations.at("address_map").as<Map<Var, PrimExpr>>().value();
    }
    auto block = Downcast<Block>(arith::IRMutatorWithAnalyzer::VisitStmt_(op));
    auto block_ptr = block.CopyOnWrite();
    for (size_t i = 0; i < block->alloc_buffers.size(); i++) {
      auto buffer = block->alloc_buffers[i];
      if (buffer_remap_.count(buffer)) {
        block_ptr->alloc_buffers.Set(i, buffer_remap_[buffer]);
      }
    }
    for (const auto &buffer : workspaces_)
      block_ptr->alloc_buffers.push_back(buffer);
    workspaces_.clear();
    block_ptr->annotations.erase(attr::kLayoutMap);
    return block;
  }

  int CheckAndGetBufferRowSize(Buffer buffer) {
    CHECK(buffer->shape.size() >= 2)
        << "The dimension of Buffer \"" << buffer->name << "\" with shape "
        << buffer->shape << " should be at least 2";

    auto dim = buffer->shape.size();
    auto buffer_row_size = buffer->shape[dim - 1].as<IntImmNode>()->value;
    return buffer_row_size;
  }

  PrimExpr HandleAccessPtrAndOffset(PrimExpr access_ptr,
                                    Optional<PrimExpr> offset = NullOpt,
                                    DataType dtype = DataType::Int(32)) {
    // The 2th arg of T.tvm_access_ptr call is offset, we set it to 0 and
    // accumulate it to smem_offset
    CHECK(access_ptr->IsInstance<CallNode>())
        << "Invalid access ptr for permuted layout: " << access_ptr;
    auto access_ptr_call = Downcast<Call>(access_ptr);
    if (access_ptr_call->op.same_as(builtin::tvm_access_ptr())) {
      LOG(FATAL) << "Transformation for tvm_access_ptr is not implemented yet";
    } else if (access_ptr_call->op.same_as(builtin::address_of())) {
      BufferLoad load = Downcast<BufferLoad>(access_ptr_call->args[0]);
      Array<PrimExpr> indices = load->indices;
      Array<PrimExpr> shape = load->buffer->shape;

      CHECK_EQ(indices.size(), shape.size())
          << "Indices size and shape size must match for general N-dimensional "
             "buffer "
          << "but got indices size: " << indices.size()
          << " and shape size: " << shape.size();

      PrimExpr elem_offset = 0;
      PrimExpr stride = 1;

      for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        elem_offset += indices[i] * stride;
        stride *= shape[i];
      }

      PrimExpr smem_offset =
          elem_offset + (offset.defined() ? offset.value() : 0);

      auto new_buffer = buffer_remap_[load->buffer];

      auto buffer_map_iter =
          buffer_map_.find(Downcast<Var>(load->buffer->data));
      CHECK(buffer_map_iter != buffer_map_.end())
          << "The buffer corresponding to data Var " << access_ptr_call->args[0]
          << " is not found";

      int buffer_row_size = CheckAndGetBufferRowSize(buffer_map_iter->second);
      (void)buffer_row_size;

      // Convert offset to target-dimension, reindex it and convert it back
      Array<PrimExpr> multi_dim_indices;
      PrimExpr remaining_offset = smem_offset;

      for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        multi_dim_indices.insert(multi_dim_indices.begin(),
                                 floormod(remaining_offset, shape[i]));
        remaining_offset = floordiv(remaining_offset, shape[i]);
      }

      auto forward_indices =
          layout_map_[load->buffer]->Forward(multi_dim_indices);
      PrimExpr new_offset = 0;
      PrimExpr stride_offset = 1;
      for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        new_offset += forward_indices[i] * stride_offset;
        stride_offset *= shape[i];
      }
      new_offset = analyzer_->Simplify(new_offset);

      Array<PrimExpr> new_indices;
      for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        new_indices.insert(new_indices.begin(), floormod(new_offset, shape[i]));
        new_offset = floordiv(new_offset, shape[i]);
      }

      auto new_access_ptr = access_ptr_call.CopyOnWrite();
      new_access_ptr->args.Set(0, BufferLoad(new_buffer, new_indices));
    } else {
      LOG(FATAL) << "Invalid access op for permuted layout: " << access_ptr;
    }

    return access_ptr_call;
  }

  PrimExpr VisitExpr_(const tir::CallNode *op) final {
    Array<RelayExpr> ptx_instructions = {builtin::ptx_ldmatrix(),
                                         builtin::mma_store()};

    if (std::find(ptx_instructions.begin(), ptx_instructions.end(), op->op) ==
        ptx_instructions.end()) {
      auto call = Downcast<Call>(IRMutatorWithAnalyzer::VisitExpr_(op));
      return call;
    } else {
      is_ptx_ = true;
    }
    // Rewrite from/to shared or shared.dyn to/from local
    auto call = Downcast<Call>(IRMutatorWithAnalyzer::VisitExpr_(op));
    if (call->op.same_as(builtin::ptx_ldmatrix())) {
      // form: T.ptx_ldmatrix(..., smem_ptr, smem_offset)
      // smem_ptr: T.tvm_access_ptr(ptype, data, offset, extent, rw_mask)
      // or T.address_of(buffer, offset)
      auto access_ptr = call->args[5];
      PrimExpr smem_offset = call->args[6];
      Call address_of_call = Downcast<Call>(access_ptr);
      if (!address_of_call->op.same_as(builtin::address_of())) {
        LOG(FATAL) << "Invalid access ptr for permuted layout: " << access_ptr;
      }
      BufferLoad load = Downcast<BufferLoad>(address_of_call->args[0]);

      if (buffer_remap_.count(load->buffer)) {
        auto new_access_ptr =
            HandleAccessPtrAndOffset(access_ptr, smem_offset, call->dtype);
        auto new_call = call.CopyOnWrite();
        new_call->args.Set(5, new_access_ptr);
        new_call->args.Set(6, IntImm(smem_offset->dtype, 0));
      }
    } else if (call->op.same_as(builtin::mma_store())) {
      // because we will directly store result to Buffer instead of calling
      // mma_store now
      auto access_ptr = call->args[2];
      auto new_access_ptr =
          HandleAccessPtrAndOffset(access_ptr, NullOpt, call->dtype);
      auto new_call = call.CopyOnWrite();
      new_call->args.Set(2, new_access_ptr);
    } else {
      LOG(FATAL) << "Invalid call node: " << call;
    }
    is_ptx_ = false;
    return call;
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto load = Downcast<BufferLoad>(IRMutatorWithAnalyzer::VisitExpr_(op));
    if (is_ptx_) {
      return load;
    }
    auto buffer = load->buffer;
    if (buffer_remap_.count(buffer)) {
      auto new_indices = layout_map_[buffer]->Forward(load->indices);
      auto new_buffer = buffer_remap_[load->buffer];
      return BufferLoad(new_buffer, new_indices);
    } else if (var_remap_.count(buffer->data)) {
      auto new_buffer = Buffer(
          var_remap_[buffer->data], buffer->dtype, buffer->shape,
          buffer->strides, buffer->elem_offset, buffer->name,
          buffer->data_alignment, buffer->offset_factor, buffer->buffer_type);
      return BufferLoad(new_buffer, load->indices);
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto store = Downcast<BufferStore>(IRMutatorWithAnalyzer::VisitStmt_(op));
    auto buffer = store->buffer;
    if (buffer_remap_.count(buffer)) {
      auto new_indices = layout_map_[buffer]->Forward(store->indices);
      auto new_buffer = buffer_remap_[store->buffer];
      return BufferStore(new_buffer, store->value, new_indices);
    } else if (var_remap_.count(buffer->data)) {
      auto new_buffer = Buffer(
          var_remap_[buffer->data], buffer->dtype, buffer->shape,
          buffer->strides, buffer->elem_offset, buffer->name,
          buffer->data_alignment, buffer->offset_factor, buffer->buffer_type);
      return BufferStore(new_buffer, store->value, store->indices);
    }
    return store;
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    auto var = Downcast<Var>(IRMutatorWithAnalyzer::VisitExpr_(op));
    if (buffer_data_to_buffer_.count(var)) {
      auto buffer = buffer_data_to_buffer_[var];
      if (buffer_remap_.count(buffer))
        return buffer_remap_[buffer]->data;
    }
    return var;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const CallNode *call = op->value.as<CallNode>();
    // Do not analysis the call node to the global function.
    if (call && call->op.as<GlobalVarNode>())
      return Downcast<Evaluate>(IRMutatorWithAnalyzer::VisitStmt_(op));

    auto tile_op = ParseOperator(GetRef<Stmt>(op), buffer_data_to_buffer_);
    if (tile_op == nullptr)
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    AddWorkspaceCallback callback = [this](int num_elem, DataType dtype) {
      auto workspace =
          decl_buffer({PrimExpr(num_elem)}, dtype, "workspace", "shared.dyn");
      workspaces_.push_back(workspace);
      return workspace.access_ptr(2); // write
    };

    // Get pass config `tl.disable_tma_lower`
    tvm::transform::PassContext ctxt = tvm::transform::PassContext::Current();
    Optional<Bool> opt_disable_tma_lower =
        ctxt->GetConfig(kDisableTMALower, Optional<Bool>());
    bool disable_tma_lower = opt_disable_tma_lower.value_or(Bool(false));
    Range thread_bounds;

    if (analyzer_->const_int_bound.IsBound(thread_var_->var)) {
      auto const_int_bound = analyzer_->const_int_bound(thread_var_);
      auto min_value = const_int_bound->min_value;
      auto max_value = const_int_bound->max_value;
      auto extent = max_value + 1 - min_value;
      thread_bounds =
          Range::FromMinExtent(IntImm(thread_var_->var.dtype(), min_value),
                               IntImm(thread_var_->var.dtype(), extent));
    } else {
      thread_bounds = Range::FromMinExtent(0, 1);
    }

    auto lowered = tile_op->Lower(
        LowerArgs{target_, thread_bounds, thread_var_->var, callback,
                  layout_map_, buffer_remap_, disable_tma_lower, resource_scope_},
        analyzer_);
    return IRMutatorWithAnalyzer::VisitStmt(lowered);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      ICHECK_NE(iv->thread_tag.length(), 0U);
      if (iv->thread_tag == "threadIdx.x") {
        thread_var_ = iv;
        ICHECK(iv->dom->extent.as<IntImmNode>());
        thread_block_size_ = iv->dom->extent.as<IntImmNode>()->value;
      }
    } else if (op->attr_key == "resource_scope") {
      resource_scope_ = Downcast<IntImm>(op->value)->value;
    }
    return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Target target_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  Map<Buffer, Layout> layout_map_;
  Map<Buffer, Buffer> buffer_remap_;
  // This is a workaround for cpu backend,
  // we need to define a thread_var for the serial loop.
  IterVar thread_var_ = IterVar(Range::FromMinExtent(0, 1), Var("v_thread"),
                                IterVarType::kDataPar);
  size_t thread_block_size_ = 0;
  Array<Buffer> workspaces_;
  // For ptx Node, we need to remap the buffer and indices
  // By access CallNode instead of BufferLoad Node.
  bool is_ptx_{false};
  // Mapping from data Var of a Buffer to Buffer, for lookup
  std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_map_;
  Map<Var, Var> var_remap_;
  int resource_scope_ = 0;
  Map<Var, PrimExpr> address_map_;
};

namespace transform {

using namespace tir::transform;

tvm::transform::Pass LowerTileOp() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return LowerTileOpPass::Substitute(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerTileOp", {});
}

TVM_REGISTER_GLOBAL("tl.transform.LowerTileOp").set_body_typed(LowerTileOp);
} // namespace transform

} // namespace tl
} // namespace tvm