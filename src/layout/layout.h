// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file Layout.h
 *
 */

#ifndef TVM_TL_LAYOUT_LAYOUT_H_
#define TVM_TL_LAYOUT_LAYOUT_H_

#include <tvm/arith/analyzer.h>

namespace tvm {
namespace tl {

using namespace tir;

class Layout;
class Fragment;

enum class AscendLayout {
  kRowMajor = 0,
  kColMajor = 1,
  kzN = 2,
  kzZ = 3,
  knZ = 4,
};

static const std::unordered_map<AscendLayout, std::string> ascendLayoutMap = {
    {AscendLayout::kRowMajor, "layout::RowMajor"},
    {AscendLayout::kColMajor, "layout::ColumnMajor"},
    {AscendLayout::kzN, "layout::zN"},
    {AscendLayout::kzZ, "layout::zZ"},
    {AscendLayout::knZ, "layout::nZ"}};

class LayoutNode : public Object {
public:
  LayoutNode() = default;
  LayoutNode(Array<PrimExpr> input_size, Array<PrimExpr> forward_index,
             PrimExpr ascend_layout = 0);

  size_t InputDim() const { return input_size_.size(); }

  size_t OutputDim() const { return forward_index_.size(); }

  Array<PrimExpr> InputShape() const { return input_size_; }

  Array<PrimExpr> OutputShape() const;

  Array<PrimExpr> GetForwardIndex() const { return forward_index_; }

  std::string AscendLayoutStr() const { return ascend_layout_str_; }

  virtual Array<PrimExpr> GetForwardVars() const;

  virtual Array<PrimExpr> Forward(const Array<PrimExpr> &vars) const;

  virtual Layout Inverse() const;

  virtual std::string DebugOutput() const;

  virtual bool IsEqual(const LayoutNode *other, bool skip_index = false) const;

  static constexpr bool _type_has_method_sequal_reduce = true;
  static constexpr const char *_type_key = "tl.Layout";
  bool SEqualReduce(const LayoutNode *other, SEqualReducer equal) const;
  void VisitAttrs(tvm::AttrVisitor *v);
  TVM_DECLARE_BASE_OBJECT_INFO(LayoutNode, Object);

protected:
  virtual Map<Var, Range> getVarMap() const;
  void UpdateAnalyzer(arith::Analyzer *analyzer) const;
  Array<PrimExpr> forward_index_;
  Array<PrimExpr> input_size_;
  AscendLayout ascend_layout_;
  std::string ascend_layout_str_;
};

/*!
 * \brief Layout reference class.
 */
class Layout : public ObjectRef {
public:
  TVM_DLL Layout(Array<IterVar> forward_var, Array<PrimExpr> forward_index,
                 PrimExpr ascend_layout = 0);
  TVM_DLL Layout(Array<PrimExpr> input_size, Array<PrimExpr> forward_index,
                 PrimExpr ascend_layout = 0);

  TVM_DEFINE_OBJECT_REF_METHODS(Layout, ObjectRef, LayoutNode);
};

class FragmentNode : public LayoutNode {
public:
  FragmentNode() = default;
  FragmentNode(Array<PrimExpr> input_size, Array<PrimExpr> forward_index,
               PrimExpr forward_thread, PrimExpr replicate_size);

  PrimExpr GetForwardThread() const { return forward_thread_; }

  Array<PrimExpr> GetForwardVars() const final;

  Layout Inverse() const final;

  PrimExpr ThreadExtent() const;

  PrimExpr ReplicateExtent() const { return replicate_size_; };

  PrimExpr ForwardThread(const Array<PrimExpr> &vars,
                         const Optional<PrimExpr> &rep_var) const;

  Fragment Repeat(const Array<PrimExpr> &repeats, bool repeat_on_thread,
                  bool lower_dim_first = true) const;

  Fragment Replicate(int repeats) const;

  Fragment DeReplicate() const;

  Fragment CondenseReplicateVar() const;

  std::string DebugOutput() const final;

  Fragment BindThreadRange(Range thread_range) const;

  Range ThreadRange() const { return thread_range_; }

  bool IsEqual(const FragmentNode *other, bool skip_index = false) const;

  void VisitAttrs(tvm::AttrVisitor *v);

  bool SEqualReduce(const FragmentNode *other, SEqualReducer equal) const;
  static constexpr const char *_type_key = "tl.Fragment";
  TVM_DECLARE_FINAL_OBJECT_INFO(FragmentNode, LayoutNode);

protected:
  Map<Var, Range> getVarMap() const final;
  Range thread_range_;
  PrimExpr forward_thread_;
  PrimExpr replicate_size_;
};

/*!
 * \brief Fragment reference class.
 */
class Fragment : public Layout {
public:
  TVM_DLL Fragment(Array<IterVar> forward_var, Array<PrimExpr> forward_index,
                   PrimExpr forward_thread, IterVar thread_replicate);

  TVM_DLL Fragment(Array<PrimExpr> input_size, Array<PrimExpr> forward_index,
                   PrimExpr forward_thread, PrimExpr replicate_size,
                   Optional<Var> replicate_var);

  TVM_DEFINE_OBJECT_REF_METHODS(Fragment, Layout, FragmentNode);
};

Var InputPlaceholder(size_t idx);
Var ReplicationPlaceholder();

Fragment makeGemmFragment8x8();
Fragment makeGemmFragment8x8Transposed();
Fragment makeGemmFragmentC(const int block_m, const int block_n,
                           const int warp_m, const int warp_n,
                           const int element_size);
Fragment makeGemmFragmentCCDNA(const int block_m, const int block_n,
                               const int warp_m, const int warp_n,
                               const int element_size);
Fragment makeGemmFragmentCHopper(const int block_m, const int block_n,
                                 const int warp_m, const int warp_n,
                                 const int element_size);
Fragment makeGemmFragmentA(const int block_m, const int block_n,
                           const int block_k, const int warp_m,
                           const int warp_n, const int element_size,
                           bool transposed = false);
Fragment makeGemmFragmentB(const int block_m, const int block_n,
                           const int block_k, const int warp_m,
                           const int warp_n, bool transposed = false);

Fragment makeGemmFragmentACDNA(const int block_m, const int block_n,
                               const int block_k, const int warp_m,
                               const int warp_n, const int element_size,
                               bool transposed = false);

// Default Memory Layout
Layout makeGemmLayoutLinear(int stride, int continuous);
Layout makeGemmABLayoutPadded(int stride, int continuous, int element_size);
Layout makeGemmABLayout(int mat_stride, int mat_continuous, int continuity,
                        int element_size, int kfactor);
Layout makeGemmABLayoutCDNA(int stride, int continuous, int element_size,
                            int kfactor);

Fragment makeGemmVoltaFragmentC(const int block_m, const int block_n,
                                const int warp_m, const int warp_n,
                                const int element_size);
Fragment makeGemmVoltaFragmentA(const int block_m, const int block_n,
                                const int block_k, const int warp_m,
                                const int warp_n);
Layout makeGemmVoltaABLayout(int stride, int continuous, bool is_a,
                             int kfactor);

Layout makeFullBankSwizzleLayout(int stride, int continuous, int element_size);
Layout makeHalfBankSwizzleLayout(int stride, int continuous, int element_size);

namespace attr {
// BlockAttr, Containing the layout for all the buffers in the block
constexpr const char *kLayoutMap = "layout_map";
} // namespace attr

} // namespace tl
} // namespace tvm

#endif // TVM_TL_LAYOUT_LAYOUT_H_
