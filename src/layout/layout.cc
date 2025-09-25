// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file layout/layout.cc
 *
 */

#include "layout.h"

#include <tvm/arith/pattern.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "arith/pattern_match.h"
#include "utils.h"

namespace tvm {
namespace tl {

using namespace tir;

static Var getPlaceholder(const std::string &s) {
  static std::unordered_map<std::string, Var> map;
  if (map.find(s) == map.end()) {
    map[s] = Var(s);
  }
  return map[s];
}

Var ReplicationPlaceholder() { return getPlaceholder("_rep"); }
Var InputPlaceholder(size_t idx) {
  return getPlaceholder(std::string{'_', char('i' + idx)});
}

Map<Var, Range> LayoutNode::getVarMap() const {
  Map<Var, Range> map;
  for (size_t i = 0; i < InputDim(); i++) {
    map.Set(InputPlaceholder(i), {0, input_size_[i]});
  }
  return map;
}

Map<Var, Range> FragmentNode::getVarMap() const {
  auto map = LayoutNode::getVarMap();
  map.Set(ReplicationPlaceholder(), {0, ReplicateExtent()});
  return map;
}

LayoutNode::LayoutNode(Array<PrimExpr> input_size,
                       Array<PrimExpr> forward_index, PrimExpr ascend_layout) {
  input_size_ = input_size;
  arith::Analyzer analyzer;
  UpdateAnalyzer(&analyzer);
  forward_index_ = forward_index.Map(
      [&](const PrimExpr &e) { return analyzer.Simplify(e); });
  ascend_layout_ =
      static_cast<AscendLayout>(Downcast<IntImm>(ascend_layout)->value);
  ascend_layout_str_ = ascendLayoutMap.at(ascend_layout_);
}

Layout::Layout(Array<IterVar> forward_var, Array<PrimExpr> forward_index,
               PrimExpr ascend_layout) {
  Map<Var, PrimExpr> vmap;
  Array<PrimExpr> input_size;
  for (size_t i = 0; i < forward_var.size(); i++) {
    vmap.Set(forward_var[i]->var, InputPlaceholder(i));
    CHECK(is_zero(forward_var[i]->dom->min));
    input_size.push_back(forward_var[i]->dom->extent);
  }
  forward_index =
      forward_index.Map([&](const PrimExpr &e) { return Substitute(e, vmap); });

  auto n = make_object<LayoutNode>(input_size, forward_index, ascend_layout);
  data_ = std::move(n);
}

Layout::Layout(Array<PrimExpr> input_size, Array<PrimExpr> forward_index,
               PrimExpr ascend_layout) {
  auto n = make_object<LayoutNode>(input_size, forward_index, ascend_layout);
  data_ = std::move(n);
}

void LayoutNode::VisitAttrs(AttrVisitor *v) {
  v->Visit("input_size", &input_size_);
  v->Visit("forward_index", &forward_index_);
}

void LayoutNode::UpdateAnalyzer(arith::Analyzer *analyzer) const {
  for (const auto &[var, dom] : getVarMap()) {
    analyzer->Bind(var, dom);
  }
}

Array<PrimExpr> LayoutNode::GetForwardVars() const {
  Array<PrimExpr> vars;
  for (size_t i = 0; i < InputDim(); i++) {
    vars.push_back(InputPlaceholder(i));
  }
  return vars;
}

Array<PrimExpr> LayoutNode::OutputShape() const {
  Array<PrimExpr> ret(OutputDim(), 1);
  arith::Analyzer analyzer;
  UpdateAnalyzer(&analyzer);
  for (size_t i = 0; i < ret.size(); i++) {
    auto ist = analyzer.int_set(forward_index_[i] + 1);
    if (arith::is_neg_inf(ist.min()) && arith::is_pos_inf(ist.max())) {
      // X-OR Expression
      ret.Set(i, input_size_[i]);
    } else {
      // CHECK(is_one(ist.min())) << ist.min();
      ret.Set(i, ist.max());
    }
  }
  return ret;
}

Array<PrimExpr> LayoutNode::Forward(const Array<PrimExpr> &vars) const {
  if (vars.empty())
    return forward_index_;
  ICHECK_EQ(vars.size(), InputDim());
  Map<Var, PrimExpr> vmap;
  for (size_t i = 0; i < InputDim(); i++) {
    vmap.Set(InputPlaceholder(i), vars[i]);
  }
  return forward_index_.Map(
      [&](const PrimExpr &e) { return Substitute(e, vmap); });
}

Fragment FragmentNode::Repeat(const Array<PrimExpr> &repeats,
                              bool repeat_on_thread,
                              bool lower_dim_first) const {
  ICHECK_EQ(repeats.size(), InputDim());
  Array<PrimExpr> new_input_size;
  Map<Var, PrimExpr> vmap;
  for (size_t i = 0; i < InputDim(); i++) {
    new_input_size.push_back(input_size_[i] * repeats[i]);
    vmap.Set(InputPlaceholder(i),
             FloorMod(InputPlaceholder(i), InputShape()[i]));
  }

  PrimExpr repeats_index = 0, repeat_stride = 1;
  if (lower_dim_first) {
    for (int i = InputDim() - 1; i >= 0; i--) {
      repeats_index +=
          repeat_stride * FloorDiv(InputPlaceholder(i), InputShape()[i]);
      repeat_stride *= repeats[i];
    }
  } else {
    for (size_t i = 0; i < InputDim(); i++) {
      repeats_index +=
          repeat_stride * FloorDiv(InputPlaceholder(i), InputShape()[i]);
      repeat_stride *= repeats[i];
    }
  }

  if (repeat_on_thread) {
    PrimExpr thread_size = ThreadExtent();
    auto new_forward_index = forward_index_.Map(
        [&](const PrimExpr &e) { return Substitute(e, vmap); });
    auto new_forward_thread =
        Substitute(forward_thread_, vmap) + thread_size * repeats_index;
    return Fragment(new_input_size, new_forward_index, new_forward_thread,
                    replicate_size_, NullOpt);
  } else {
    ICHECK(OutputDim() == 1);
    PrimExpr frag_len = OutputShape()[0];
    Array<PrimExpr> new_forward_index = {Substitute(forward_index_[0], vmap) +
                                         frag_len * repeats_index};
    PrimExpr new_forward_thread = Substitute(forward_thread_, vmap);
    return Fragment(new_input_size, new_forward_index, new_forward_thread,
                    replicate_size_, NullOpt);
  }
}

Fragment FragmentNode::Replicate(int repeats) const {
  ICHECK(repeats >= 1);
  Map<Var, PrimExpr> vmap;
  vmap.Set(ReplicationPlaceholder(),
           FloorMod(ReplicationPlaceholder(), ReplicateExtent()));
  PrimExpr new_forward_thread =
      Substitute(forward_thread_, vmap) +
      ThreadExtent() * FloorDiv(ReplicationPlaceholder(), ReplicateExtent());
  return Fragment(input_size_, forward_index_, new_forward_thread,
                  ReplicateExtent() * repeats, NullOpt);
}

Fragment FragmentNode::DeReplicate() const {
  ICHECK(OutputDim() == 1);
  arith::Analyzer analyzer;
  UpdateAnalyzer(&analyzer);
  int factor = 1;
  auto rep_size = as_const_int(ReplicateExtent());
  auto idx_size = as_const_int(OutputShape()[0]);
  if (rep_size && idx_size) {
    factor = arith::ZeroAwareGCD(*rep_size, *idx_size);
  }
  if (factor == 1)
    return GetRef<Fragment>(this);

  Map<Var, PrimExpr> vmap;
  vmap.Set(ReplicationPlaceholder(), ReplicationPlaceholder() * factor +
                                         FloorMod(forward_index_[0], factor));
  PrimExpr new_forward_thread = Substitute(forward_thread_, vmap);
  Array<PrimExpr> new_forward_index = {FloorDiv(forward_index_[0], factor)};
  return Fragment(input_size_, new_forward_index, new_forward_thread,
                  int(*rep_size) / factor, NullOpt);
}

Fragment FragmentNode::BindThreadRange(Range thread_range) const {
  auto n = make_object<FragmentNode>(*this);
  n->thread_range_ = thread_range;
  return Fragment(n);
}

Layout LayoutNode::Inverse() const {
  arith::Analyzer analyzer;
  arith::IterMapResult res =
      arith::DetectIterMap(forward_index_, getVarMap(), 1,
                           arith::IterMapLevel::Bijective, &analyzer);
  ICHECK(res->errors.empty())
      << "Layout " << DebugOutput() << " has errors: " << res->errors;

  auto outputs_shape = OutputShape();
  Array<PrimExpr> outputs;
  for (size_t i = 0; i < OutputDim(); i++) {
    outputs.push_back(InputPlaceholder(i));
  }

  auto inv = arith::InverseAffineIterMap(res->indices, outputs);

  Array<PrimExpr> backward_index;
  for (size_t i = 0; i < InputDim(); i++) {
    if (inv.find(InputPlaceholder(i)) != inv.end()) {
      backward_index.push_back(inv[InputPlaceholder(i)]);
    } else {
      backward_index.push_back(0);
    }
  }

  return Layout(outputs_shape, backward_index);
}

PrimExpr infer_fragment_index(const Map<Var, Range> &input_iters,
                              const PrimExpr &forward_thread,
                              arith::Analyzer *analyzer) {
  Array<arith::IterSplitExpr> splits = DivideUnusedIterators(
      {forward_thread}, ToIterVars(input_iters), analyzer);

  Array<arith::IterSplitExpr> split_without_rep;
  for (const auto &split : splits) {
    CHECK(split->source->source.as<Var>());
    if (split->source->source.as<Var>().value().same_as(
            ReplicationPlaceholder()))
      continue;
    split_without_rep.push_back(split);
  }
  return MakeFlattenedExpression(split_without_rep);
}

FragmentNode::FragmentNode(Array<PrimExpr> input_size,
                           Array<PrimExpr> forward_index,
                           PrimExpr forward_thread, PrimExpr replicate_size) {
  input_size_ = input_size;
  replicate_size_ = replicate_size;
  arith::Analyzer analyzer;
  UpdateAnalyzer(&analyzer);
  forward_thread_ = analyzer.Simplify(forward_thread);
  if (forward_index.empty()) {
    forward_index = {
        infer_fragment_index(getVarMap(), forward_thread_, &analyzer)};
  }
  forward_index_ = forward_index.Map(
      [&](const PrimExpr &e) { return analyzer.Simplify(e); });
}

Fragment::Fragment(Array<IterVar> forward_var, Array<PrimExpr> forward_index,
                   PrimExpr forward_thread, IterVar thread_replicate) {
  Map<Var, PrimExpr> vmap;
  Array<PrimExpr> input_size;
  PrimExpr replicate_size = 1;
  for (size_t i = 0; i < forward_var.size(); i++) {
    vmap.Set(forward_var[i]->var, InputPlaceholder(i));
    CHECK(is_zero(forward_var[i]->dom->min));
    input_size.push_back(forward_var[i]->dom->extent);
  }
  if (thread_replicate.defined()) {
    ICHECK(is_zero(thread_replicate->dom->min));
    replicate_size = thread_replicate->dom->extent;
    vmap.Set(thread_replicate->var, ReplicationPlaceholder());
  }
  forward_index =
      forward_index.Map([&](const PrimExpr &e) { return Substitute(e, vmap); });
  forward_thread = Substitute(forward_thread, vmap);

  auto n = make_object<FragmentNode>(input_size, forward_index, forward_thread,
                                     replicate_size);
  data_ = std::move(n);
}

Fragment::Fragment(Array<PrimExpr> input_size, Array<PrimExpr> forward_index,
                   PrimExpr forward_thread, PrimExpr replicate_size,
                   Optional<Var> replicate_var) {
  if (replicate_var.defined()) {
    forward_thread = Substitute(
        forward_thread, {{replicate_var.value(), ReplicationPlaceholder()}});
  }
  auto n = make_object<FragmentNode>(input_size, forward_index, forward_thread,
                                     replicate_size);
  data_ = std::move(n);
}

void FragmentNode::VisitAttrs(tvm::AttrVisitor *v) {
  LayoutNode::VisitAttrs(v);
  v->Visit("forward_thread", &forward_thread_);
  v->Visit("replicate_size", &replicate_size_);
}

PrimExpr FragmentNode::ThreadExtent() const {
  Array<PrimExpr> ret(OutputDim(), 1);
  arith::Analyzer analyzer;
  UpdateAnalyzer(&analyzer);
  auto ist = analyzer.int_set(forward_thread_ + 1);
  // CHECK(is_one(ist.min()));
  return ist.max();
}

Array<PrimExpr> FragmentNode::GetForwardVars() const {
  Array<PrimExpr> vars;
  if (*as_const_int(ReplicateExtent()) > 1) {
    vars.push_back(ReplicationPlaceholder());
  }
  for (size_t i = 0; i < InputDim(); i++) {
    vars.push_back(InputPlaceholder(i));
  }
  return vars;
}

PrimExpr FragmentNode::ForwardThread(const Array<PrimExpr> &vars,
                                     const Optional<PrimExpr> &rep_var) const {
  Map<Var, PrimExpr> vmap;
  ICHECK_EQ(vars.size(), InputDim());
  for (size_t i = 0; i < InputDim(); i++) {
    vmap.Set(InputPlaceholder(i), vars[i]);
  }
  if (rep_var.defined())
    vmap.Set(ReplicationPlaceholder(), rep_var.value());

  return Substitute(forward_thread_, vmap);
}

Layout FragmentNode::Inverse() const {
  auto input_size_copy = input_size_;
  input_size_copy.push_back(ReplicateExtent());
  auto forward_index_copy = forward_index_;
  forward_index_copy.push_back(
      Substitute(forward_thread_,
                 {{ReplicationPlaceholder(), InputPlaceholder(InputDim())}}));
  auto fwd = Layout(input_size_copy, forward_index_copy);
  auto bwd = fwd->Inverse();
  return bwd;
}

Fragment FragmentNode::CondenseReplicateVar() const {
  arith::Analyzer analyzer;
  auto input_iters = getVarMap();
  input_iters.Set(ReplicationPlaceholder(), {0, ReplicateExtent()});
  PrimExpr new_forward_thread;
  IterVar new_thread_replicate;
  std::tie(new_forward_thread, new_thread_replicate) =
      CompressIterator(forward_thread_, ToIterVars(input_iters),
                       ReplicationPlaceholder(), &analyzer);
  return Fragment(input_size_, forward_index_, new_forward_thread,
                  new_thread_replicate->dom->extent, new_thread_replicate->var);
}

std::string LayoutNode::DebugOutput() const {
  std::stringstream ss;
  ss << "Layout Shape: " << InputShape() << " -> " << OutputShape() << " -> "
     << GetForwardIndex();
  return ss.str();
}

std::string FragmentNode::DebugOutput() const {
  std::stringstream ss;
  ss << "Fragment Shape: " << InputShape() << " -> " << OutputShape();
  ss << " -> replicate: " << ReplicateExtent();
  ss << " -> thread: " << ThreadExtent();
  ss << " -> forward_thread: " << forward_thread_;
  ss << " -> forward_index: " << GetForwardIndex();
  ss << " -> thread_range: " << thread_range_;
  return ss.str();
}

bool LayoutNode::SEqualReduce(const LayoutNode *other,
                              SEqualReducer equal) const {
  return equal(this->InputShape(), other->InputShape()) &&
         equal(this->forward_index_, other->forward_index_);
}

bool FragmentNode::SEqualReduce(const FragmentNode *other,
                                SEqualReducer equal) const {
  return equal(this->ReplicateExtent(), other->ReplicateExtent()) &&
         equal(this->InputShape(), other->InputShape()) &&
         equal(this->ThreadExtent(), other->ThreadExtent()) &&
         equal(this->forward_index_, other->forward_index_) &&
         equal(this->forward_thread_, other->forward_thread_);
}

bool LayoutNode::IsEqual(const LayoutNode *other, bool skip_index) const {
  bool ret = StructuralEqual()(this->InputShape(), other->InputShape());
  ret &= StructuralEqual()(this->OutputShape(), other->OutputShape());
  if (!skip_index) {
    ret &= StructuralEqual()(this->forward_index_, other->forward_index_);
  }
  return ret;
}

bool FragmentNode::IsEqual(const FragmentNode *other, bool skip_index) const {
  // Fragment Layout Comparison can skip the index comparison
  // when the output shape is the same, as we can do
  // a[i, j] = b[j, i] in register level.

  bool ret = StructuralEqual()(this->InputShape(), other->InputShape());
  if (!ret) {
    // may be broadcast case
    return true;
  }
  if (this->thread_range_.defined() && other->thread_range_.defined()) {
    ret &= StructuralEqual()(this->thread_range_, other->thread_range_);
  }
  ret &= StructuralEqual()(this->OutputShape(), other->OutputShape());
  ret &= StructuralEqual()(this->ReplicateExtent(), other->ReplicateExtent());
  ret &= StructuralEqual()(this->ThreadExtent(), other->ThreadExtent());
  if (!skip_index) {
    ret &= StructuralEqual()(this->forward_index_, other->forward_index_);
  }
  return ret;
}

TVM_REGISTER_NODE_TYPE(LayoutNode);
TVM_REGISTER_NODE_TYPE(FragmentNode);

TVM_REGISTER_GLOBAL("tl.Layout").set_body([](TVMArgs args, TVMRetValue *ret) {
  *ret = Layout(Array<IterVar>(args[0]), Array<PrimExpr>(args[1]),
                PrimExpr(args[2]));
});

TVM_REGISTER_GLOBAL("tl.Layout_input_shape").set_body_typed([](Layout layout) {
  return layout->InputShape();
});

TVM_REGISTER_GLOBAL("tl.Layout_output_shape").set_body_typed([](Layout layout) {
  return layout->OutputShape();
});

TVM_REGISTER_GLOBAL("tl.Layout_inverse").set_body_typed([](Layout layout) {
  return layout->Inverse();
});

TVM_REGISTER_GLOBAL("tl.Layout_index").set_body_typed([](Layout layout) {
  return layout->GetForwardIndex();
});

TVM_REGISTER_GLOBAL("tl.Layout_forward_vars").set_body_typed([](Layout layout) {
  return layout->GetForwardVars();
});

TVM_REGISTER_GLOBAL("tl.Fragment").set_body([](TVMArgs args, TVMRetValue *ret) {
  *ret = Fragment(args[0], args[1], args[2], args[3]);
});

TVM_REGISTER_GLOBAL("tl.Fragment_thread_size")
    .set_body_typed([](Fragment fragment) { return fragment->ThreadExtent(); });

TVM_REGISTER_GLOBAL("tl.Fragment_thread").set_body_typed([](Fragment fragment) {
  return fragment->GetForwardThread();
});

TVM_REGISTER_GLOBAL("tl.Fragment_repeat")
    .set_body_typed([](Fragment fragment, Array<PrimExpr> repeats,
                       bool repeat_on_thread, bool lower_dim_first) {
      return fragment->Repeat(repeats, repeat_on_thread, lower_dim_first);
    });

TVM_REGISTER_GLOBAL("tl.Fragment_replicate")
    .set_body_typed([](Fragment fragment, int repeats) {
      return fragment->Replicate(repeats);
    });

TVM_REGISTER_GLOBAL("tl.Fragment_condense_rep_var")
    .set_body_typed([](Fragment fragment) {
      return fragment->CondenseReplicateVar();
    });

TVM_REGISTER_GLOBAL("tl.make_swizzled_layout")
    .set_body_typed([](int stride, int continuous, int element_size) {
      return makeGemmABLayout(stride, continuous, continuous, element_size, 0);
    });

} // namespace tl
} // namespace tvm
