// The scheduled-TIR walker, natively.
//
// A port of npu_compiler/tir_codegen_v09.py's ``Walker``.  The Python version
// remains the definition of what a kernel lowers to; this one exists because
// the machine has no branch or loop instruction, so emission is O(unrolled
// iterations), and in Python every iteration reads TIR nodes whose attribute
// access and hashing are FFI calls.  Here the same walk touches plain C++
// objects.
//
// Anything this walker does not recognise raises ``Unsupported``, which the
// caller turns into a fall back to the Python walker for that kernel.  That
// keeps coverage a performance question rather than a correctness one: a
// kernel is either emitted identically or emitted by Python.
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "v09_asm.h"

namespace npu {
namespace {

using namespace tvm;
using namespace tvm::tir;
using tvm::runtime::NDArray;

constexpr int64_t kTile = 64;

class Unsupported : public std::runtime_error {
 public:
  explicit Unsupported(const std::string& what) : std::runtime_error(what) {}
};

// SRAM nibbles and global bytes per element, by dtype -- tir_codegen_v09.py's
// _SRAM_WIDTH / _ITEMSIZE.  A buffer with no recorded dtype is fp16, which is
// what every staged activation is.
struct Unit {
  int64_t width = 4;     // SRAM nibbles
  int64_t itemsize = 2;  // global bytes
};

Unit UnitOf(DataType dtype) {
  if (dtype.is_float() && dtype.bits() == 16) return {4, 2};
  if (dtype.is_float() && dtype.bits() == 32) return {8, 4};
  if (dtype.is_int() && dtype.bits() == 8) return {2, 1};
  if (dtype.is_int() && dtype.bits() == 4) return {1, 1};
  throw Unsupported("no SRAM width for dtype " + DLDataType2String(dtype));
}

// ---------------------------------------------------------------- walker

class Walker {
 public:
  Asm a;
  SramEmitter stage{&a};

  std::unordered_map<const VarNode*, int64_t> env;
  std::unordered_map<const VarNode*, int64_t> bases;   // global element offset
  std::unordered_map<const VarNode*, int64_t> sram;    // SRAM nibble
  std::unordered_map<const VarNode*, Unit> units;
  std::map<double, int64_t> constants;                 // value -> SRAM nibble
  std::vector<int64_t> scratch_slots;
  int64_t one_fp32 = -1;

  int depth = 0;
  int64_t inner_base = 0;
  bool has_pending = false;
  int64_t pending_c = 0, pending_sc = 0;
  std::unordered_set<int64_t> zeroed;
  std::unordered_map<int64_t, int64_t> stored;

  // ---- units -----------------------------------------------------------

  Unit UnitFor(const VarNode* data) const {
    auto it = units.find(data);
    return it == units.end() ? Unit{} : it->second;
  }
  int64_t Width(const VarNode* data) const { return UnitFor(data).width; }
  int64_t ItemSize(const VarNode* data) const { return UnitFor(data).itemsize; }
  bool InSram(const VarNode* data) const { return sram.count(data) != 0; }

  // ---- expression evaluation -------------------------------------------
  // Unlike the Python evaluator there is no slow path: every node kind the
  // schedules produce is handled here, and anything else is Unsupported.

  int64_t Ev(const PrimExpr& expr) const {
    if (const auto* imm = expr.as<IntImmNode>()) return imm->value;
    if (const auto* var = expr.as<VarNode>()) {
      auto it = env.find(var);
      if (it == env.end()) {
        throw Unsupported("free variable " + var->name_hint);
      }
      return it->second;
    }
    if (const auto* node = expr.as<AddNode>()) return Ev(node->a) + Ev(node->b);
    if (const auto* node = expr.as<SubNode>()) return Ev(node->a) - Ev(node->b);
    if (const auto* node = expr.as<MulNode>()) return Ev(node->a) * Ev(node->b);
    if (const auto* node = expr.as<FloorDivNode>()) {
      return FloorDiv(Ev(node->a), Ev(node->b));
    }
    if (const auto* node = expr.as<FloorModNode>()) {
      return FloorMod(Ev(node->a), Ev(node->b));
    }
    if (const auto* node = expr.as<MinNode>()) {
      return std::min(Ev(node->a), Ev(node->b));
    }
    if (const auto* node = expr.as<MaxNode>()) {
      return std::max(Ev(node->a), Ev(node->b));
    }
    if (const auto* node = expr.as<LTNode>()) return Ev(node->a) < Ev(node->b);
    if (const auto* node = expr.as<LENode>()) return Ev(node->a) <= Ev(node->b);
    if (const auto* node = expr.as<GTNode>()) return Ev(node->a) > Ev(node->b);
    if (const auto* node = expr.as<GENode>()) return Ev(node->a) >= Ev(node->b);
    if (const auto* node = expr.as<EQNode>()) return Ev(node->a) == Ev(node->b);
    if (const auto* node = expr.as<NENode>()) return Ev(node->a) != Ev(node->b);
    if (const auto* node = expr.as<AndNode>()) {
      return Ev(node->a) && Ev(node->b);
    }
    if (const auto* node = expr.as<OrNode>()) return Ev(node->a) || Ev(node->b);
    if (const auto* node = expr.as<NotNode>()) return !Ev(node->a);
    if (const auto* node = expr.as<CastNode>()) return Ev(node->value);
    throw Unsupported("cannot evaluate " + std::string(expr->GetTypeKey()));
  }

  // TVM's floordiv/floormod round toward negative infinity for every sign;
  // C++ division does not, so the two would disagree on padded tiles.
  static int64_t FloorDiv(int64_t a, int64_t b) {
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
  }
  static int64_t FloorMod(int64_t a, int64_t b) { return a - FloorDiv(a, b) * b; }

  // ---- addresses -------------------------------------------------------

  // Absolute address of a buffer access, in the unit its space counts in:
  // nibbles for SRAM, bytes for global memory (the DMA is global memory's
  // only consumer, and bytes are the one unit every dtype shares).
  int64_t Flat(const Buffer& buffer, const Array<PrimExpr>& indices) const {
    int64_t offset = 0, scale = 1;
    int rank = static_cast<int>(buffer->shape.size());
    int count = static_cast<int>(indices.size());
    for (int i = rank - 1, j = count - 1; i >= 0 && j >= 0; --i, --j) {
      offset += Ev(indices[j]) * scale;
      scale *= Ev(buffer->shape[i]);
    }
    const VarNode* data = buffer->data.get();
    auto in_sram = sram.find(data);
    if (in_sram != sram.end()) return in_sram->second + offset * Width(data);
    auto in_global = bases.find(data);
    if (in_global == bases.end()) {
      throw Unsupported("unplaced buffer " + buffer->name);
    }
    return in_global->second * 2 + offset * ItemSize(data);
  }

  // (address, stride along one axis), taken from inner_base so a row can be
  // emitted in pieces.
  std::pair<int64_t, int64_t> AffineFlat(const Buffer& buffer,
                                         const Array<PrimExpr>& indices,
                                         const VarNode* axis) {
    env[axis] = inner_base;
    int64_t base = Flat(buffer, indices);
    env[axis] = inner_base + 1;
    int64_t stride = Flat(buffer, indices) - base;
    env[axis] = inner_base;
    return {base, stride};
  }

  int64_t Ptr(const PrimExpr& expr) const {
    const auto* call = expr.as<CallNode>();
    if (!call || !call->op.same_as(builtin::tvm_access_ptr())) {
      throw Unsupported("expected access_ptr");
    }
    const VarNode* data = call->args[1].as<VarNode>();
    if (data == nullptr) throw Unsupported("access_ptr without a data var");
    auto in_sram = sram.find(data);
    if (in_sram != sram.end()) return in_sram->second + Ev(call->args[2]);
    auto in_global = bases.find(data);
    if (in_global == bases.end()) throw Unsupported("unknown buffer in ptr");
    return in_global->second * 2 + Ev(call->args[2]) * ItemSize(data);
  }

  // ---- statement walk --------------------------------------------------

  void Visit(const Stmt& stmt) {
    if (const auto* loop = stmt.as<ForNode>()) {
      if (MatchNest(stmt)) return;
      int64_t begin = Ev(loop->min), extent = Ev(loop->extent);
      const VarNode* var = loop->loop_var.get();
      for (int64_t value = begin; value < begin + extent; ++value) {
        env[var] = value;
        Visit(loop->body);
      }
      env.erase(var);
      return;
    }
    if (const auto* seq = stmt.as<SeqStmtNode>()) {
      for (const Stmt& sub : seq->seq) Visit(sub);
      return;
    }
    if (const auto* realize = stmt.as<BlockRealizeNode>()) {
      const BlockNode* block = realize->block.get();
      for (size_t i = 0; i < block->iter_vars.size(); ++i) {
        env[block->iter_vars[i]->var.get()] = Ev(realize->iter_values[i]);
      }
      for (const MatchBufferRegion& match : block->match_buffers) {
        BindMatch(match);
      }
      if (block->init.defined()) Visit(block->init.value());
      Visit(block->body);
      return;
    }
    if (const auto* evaluate = stmt.as<EvaluateNode>()) {
      Call(evaluate->value);
      return;
    }
    if (stmt.as<BufferStoreNode>() || stmt.as<LetStmtNode>()) {
      throw Unsupported("un-tensorized statement reached codegen");
    }
    if (const auto* block = stmt.as<BlockNode>()) {
      Visit(block->body);
      return;
    }
    if (const auto* branch = stmt.as<IfThenElseNode>()) {
      // the program is fully unrolled, so every guard is decidable here
      if (Ev(branch->condition)) {
        Visit(branch->then_case);
      } else if (branch->else_case.defined()) {
        Visit(branch->else_case.value());
      }
      return;
    }
    if (const auto* attr = stmt.as<AttrStmtNode>()) {
      Visit(attr->body);
      return;
    }
    if (const auto* alloc = stmt.as<AllocateConstNode>()) {
      Visit(alloc->body);
      return;
    }
    throw Unsupported(std::string("unhandled TIR node ") + stmt->GetTypeKey());
  }

  // ---- whole-loop-nest patterns ---------------------------------------

  bool MatchNest(const Stmt& stmt);
  std::unordered_map<const VarNode*, int64_t> Narrow(
      const PrimExpr& predicate, const std::vector<const VarNode*>& spatial,
      std::unordered_map<const VarNode*, int64_t> extents);

  void EmitReduction(const BufferStore& store, const BufferStore& init,
                     const std::vector<const VarNode*>& spatial,
                     const VarNode* axis,
                     const std::unordered_map<const VarNode*, int64_t>& extents);
  void EmitPointwise(const BufferStore& store,
                     const std::vector<const VarNode*>& spatial,
                     const std::unordered_map<const VarNode*, int64_t>& extents);
  void EmitMovement(const BufferStore& store,
                    const std::vector<const VarNode*>& spatial,
                    const std::unordered_map<const VarNode*, int64_t>& extents);
  void EmitDma(const BufferStore& store, const BufferLoad& load,
               const std::vector<const VarNode*>& spatial,
               const std::unordered_map<const VarNode*, int64_t>& extents,
               bool to_sram);

  // ---- expression materialization -------------------------------------

  int64_t Slot() {
    if (depth >= static_cast<int>(scratch_slots.size())) {
      throw Unsupported("expression nesting exceeds the scratch slots");
    }
    return scratch_slots[depth++];
  }

  int64_t Materialize(const PrimExpr& expr, const VarNode* inner,
                      int64_t length, int64_t into = -1);
  int64_t MaterializeCast(const CastNode* cast, const VarNode* inner,
                          int64_t length, int64_t into);
  int64_t MaterializeSigmoid(const CallNode* call, const VarNode* inner,
                             int64_t length, int64_t into);
  int64_t MaterializeTanh(const CallNode* call, const VarNode* inner,
                          int64_t length, int64_t into);
  int64_t MaterializeRsqrt(const CallNode* call, const VarNode* inner,
                           int64_t length, int64_t into);

  // ---- tensorized markers ---------------------------------------------

  void Call(const PrimExpr& expr);
  void EmitGemm(int64_t c, int64_t sc, int64_t a_addr, int64_t sa,
                int64_t b_addr, int64_t sb);
  void BindMatch(const MatchBufferRegion& match);

  void Flush() {
    if (!has_pending) return;
    stage.region(kDst, pending_c, pending_sc, kTile, kTile);
    a.save(1);
    stored[pending_c] = pending_sc;
    has_pending = false;
  }

  void Run(const PrimFunc& func) {
    Visit(func->body);
    Flush();
  }
};

// ------------------------------------------------------------- helpers

// Concrete values for every axis, last one varying fastest -- the order
// _outer_positions yields, which the DMA segment merging depends on.
class Positions {
 public:
  Positions(const std::vector<const VarNode*>& axes,
            const std::unordered_map<const VarNode*, int64_t>& extents)
      : axes_(axes) {
    total_ = 1;
    for (const VarNode* axis : axes) {
      int64_t count = extents.at(axis);
      counts_.push_back(count);
      total_ *= count;
    }
    current_.assign(axes.size(), 0);
  }

  int64_t total() const { return total_; }

  const std::vector<int64_t>& at(int64_t flat) {
    int64_t rest = flat;
    for (int i = static_cast<int>(counts_.size()) - 1; i >= 0; --i) {
      current_[i] = rest % counts_[i];
      rest /= counts_[i];
    }
    return current_;
  }

  const std::vector<const VarNode*>& axes() const { return axes_; }

 private:
  std::vector<const VarNode*> axes_;
  std::vector<int64_t> counts_;
  std::vector<int64_t> current_;
  int64_t total_ = 1;
};

// ---------------------------------------------------------- nest matching

bool Walker::MatchNest(const Stmt& stmt) {
  std::vector<const ForNode*> loops;
  Stmt node = stmt;
  while (const auto* loop = node.as<ForNode>()) {
    loops.push_back(loop);
    node = loop->body;
  }
  const auto* realize = node.as<BlockRealizeNode>();
  if (realize == nullptr) return false;
  const BlockNode* block = realize->block.get();
  if (!block->match_buffers.empty()) return false;   // tensorized

  Stmt body = block->body;
  bool has_init = block->init.defined();
  PrimExpr guard;
  bool has_guard = false;
  // LowerInitBlock rewrites a reduction into a guarded initial store followed
  // by the accumulating store, and ConvertBlocksToOpaque then strips the iter
  // vars, so the reduce axis has to come from the guard.
  if (const auto* seq = body.as<SeqStmtNode>()) {
    if (seq->seq.size() == 2) {
      const auto* branch = seq->seq[0].as<IfThenElseNode>();
      if (branch != nullptr && !branch->else_case.defined() &&
          branch->then_case.as<BufferStoreNode>() != nullptr &&
          seq->seq[1].as<BufferStoreNode>() != nullptr) {
        has_init = true;
        guard = branch->condition;
        has_guard = true;
        body = seq->seq[1];
      }
    }
  }
  const auto* store_node = body.as<BufferStoreNode>();
  if (store_node == nullptr) return false;

  // after compute_at the block's iter values are affine in the enclosing
  // loops, so substitute them away and work in terms of the local loops
  std::unordered_map<const VarNode*, PrimExpr> subst;
  for (size_t i = 0; i < block->iter_vars.size(); ++i) {
    subst[block->iter_vars[i]->var.get()] = realize->iter_values[i];
  }
  auto lookup = [&subst](const Var& var) -> Optional<PrimExpr> {
    auto it = subst.find(var.get());
    if (it == subst.end()) return NullOpt;
    return it->second;
  };
  BufferStore store = Downcast<BufferStore>(Substitute(body, lookup));

  std::unordered_set<const VarNode*> reduce_vars;
  if (has_guard) {
    guard = Substitute(guard, lookup);
    for (const Var& var : UndefinedVars(guard)) reduce_vars.insert(var.get());
  }
  for (size_t i = 0; i < block->iter_vars.size(); ++i) {
    if (block->iter_vars[i]->iter_type == kCommReduce) {
      for (const Var& var : UndefinedVars(realize->iter_values[i])) {
        reduce_vars.insert(var.get());
      }
    }
  }

  std::vector<const VarNode*> local, spatial, reduce_axes;
  std::unordered_map<const VarNode*, int64_t> extents;
  for (const ForNode* loop : loops) {
    const VarNode* var = loop->loop_var.get();
    local.push_back(var);
    extents[var] = Ev(loop->extent);
    if (reduce_vars.count(var)) {
      reduce_axes.push_back(var);
    } else {
      spatial.push_back(var);
    }
  }

  const auto* always = realize->predicate.as<IntImmNode>();
  if (always == nullptr || always->value != 1) {
    extents = Narrow(realize->predicate, spatial, extents);
  }

  std::vector<std::pair<const VarNode*, bool>> saved;
  std::vector<int64_t> saved_values;
  for (const VarNode* var : local) {
    auto it = env.find(var);
    saved.emplace_back(var, it != env.end());
    saved_values.push_back(it != env.end() ? it->second : 0);
    env[var] = 0;
  }
  bool matched = true;
  try {
    if (has_init && reduce_axes.size() == 1) {
      EmitReduction(store, store, spatial, reduce_axes[0], extents);
    } else if (reduce_axes.empty()) {
      EmitPointwise(store, spatial, extents);
    } else {
      matched = false;
    }
  } catch (...) {
    for (size_t i = 0; i < saved.size(); ++i) {
      if (saved[i].second) {
        env[saved[i].first] = saved_values[i];
      } else {
        env.erase(saved[i].first);
      }
    }
    throw;
  }
  for (size_t i = 0; i < saved.size(); ++i) {
    if (saved[i].second) {
      env[saved[i].first] = saved_values[i];
    } else {
      env.erase(saved[i].first);
    }
  }
  return matched;
}

// Shrink the loop extents to the part a block predicate allows.  pad_einsum's
// copy-back carries a predicate clipping the padded tile to the real output;
// the allowed region has to be a box anchored at the origin -- which is the
// shape a padding predicate always has -- and anything else is rejected
// rather than silently truncated.
std::unordered_map<const VarNode*, int64_t> Walker::Narrow(
    const PrimExpr& predicate, const std::vector<const VarNode*>& spatial,
    std::unordered_map<const VarNode*, int64_t> extents) {
  std::vector<bool> present(spatial.size());
  std::vector<int64_t> saved(spatial.size());
  for (size_t i = 0; i < spatial.size(); ++i) {
    auto it = env.find(spatial[i]);
    present[i] = it != env.end();
    saved[i] = present[i] ? it->second : 0;
  }
  std::vector<int64_t> limits(spatial.size(), 0);
  int64_t allowed = 0;
  Positions positions(spatial, extents);
  for (int64_t flat = 0; flat < positions.total(); ++flat) {
    const std::vector<int64_t>& position = positions.at(flat);
    for (size_t i = 0; i < spatial.size(); ++i) env[spatial[i]] = position[i];
    if (Ev(predicate)) {
      ++allowed;
      for (size_t i = 0; i < spatial.size(); ++i) {
        limits[i] = std::max(limits[i], position[i] + 1);
      }
    }
  }
  for (size_t i = 0; i < spatial.size(); ++i) {
    if (present[i]) {
      env[spatial[i]] = saved[i];
    } else {
      env.erase(spatial[i]);
    }
  }
  int64_t box = 1;
  for (int64_t limit : limits) box *= limit;
  if (box != allowed) throw Unsupported("block predicate is not a box");
  for (size_t i = 0; i < spatial.size(); ++i) extents[spatial[i]] = limits[i];
  return extents;
}

// ------------------------------------------------------------- reduction

void Walker::EmitReduction(
    const BufferStore& store, const BufferStore& /*init*/,
    const std::vector<const VarNode*>& spatial, const VarNode* axis,
    const std::unordered_map<const VarNode*, int64_t>& extents) {
  PrimExpr value = store->value;
  bool is_sum = value.as<AddNode>() != nullptr;
  bool is_max = value.as<MaxNode>() != nullptr;
  if (!is_sum && !is_max) throw Unsupported("unsupported reduction");
  PrimExpr left = is_sum ? value.as<AddNode>()->a : value.as<MaxNode>()->a;
  PrimExpr right = is_sum ? value.as<AddNode>()->b : value.as<MaxNode>()->b;

  const VarNode* accumulator = store->buffer->data.get();
  PrimExpr summand;
  bool found = false;
  for (const PrimExpr& side : {left, right}) {
    const auto* load = side.as<BufferLoadNode>();
    if (load == nullptr || load->buffer->data.get() != accumulator) {
      summand = side;
      found = true;
    }
  }
  if (!found) throw Unsupported("reduction has no summand");

  int64_t length = extents.at(axis);
  Flush();
  Positions positions(spatial, extents);
  for (int64_t flat = 0; flat < positions.total(); ++flat) {
    const std::vector<int64_t>& position = positions.at(flat);
    for (size_t i = 0; i < spatial.size(); ++i) env[spatial[i]] = position[i];
    if (const auto* load = summand.as<BufferLoadNode>()) {
      auto [base, stride] = AffineFlat(load->buffer, load->indices, axis);
      const VarNode* data = load->buffer->data.get();
      int64_t unit = InSram(data) ? Width(data) : ItemSize(data);
      if (stride != unit) throw Unsupported("reduction axis is not contiguous");
      a.vlen(length);
      stage.vector(kSrc1, base);
      a.load(0, kSrc1);
    } else {
      // compound summand (e.g. x*x): materialize it, then reduce
      depth = 0;
      int64_t nibble = Materialize(summand, axis, length);
      a.vlen(length);
      stage.vector(kSrc1, nibble);
      a.load(0, kSrc1);
    }
    if (is_sum) {
      a.v_reduce_sum();
    } else {
      a.v_reduce_max();
    }
    stage.vector(kDst, Flat(store->buffer, store->indices));
    a.save(0);
  }
}

// ------------------------------------------------------------- pointwise

struct Piece {
  PrimExpr expr;
  int64_t lo;
  int64_t count;
};

// A conditional select (if_then_else from concat, or from pad_einsum's fill)
// becomes one piece per run of the condition.  The condition is not assumed
// to hold on a prefix, since concat's holds on the tail, and the split
// recurses because an N-way concat lowers to N-1 NESTED selects -- a
// 28-layer cache stack is a 27-deep chain.
void SplitRow(Walker* w, const PrimExpr& value, const VarNode* inner,
              int64_t lo, int64_t count, std::vector<Piece>* out) {
  const auto* call = value.as<CallNode>();
  if (call == nullptr || !call->op.same_as(builtin::if_then_else())) {
    out->push_back({value, lo, count});
    return;
  }
  const PrimExpr& condition = call->args[0];
  std::vector<char> flags(count);
  for (int64_t index = 0; index < count; ++index) {
    w->env[inner] = lo + index;
    flags[index] = w->Ev(condition) ? 1 : 0;
  }
  int64_t start = 0;
  for (int64_t index = 1; index <= count; ++index) {
    if (index == count || flags[index] != flags[start]) {
      const PrimExpr& branch = flags[start] ? call->args[1] : call->args[2];
      SplitRow(w, branch, inner, lo + start, index - start, out);
      start = index;
    }
  }
}

std::vector<Piece> Pieces(Walker* w, const PrimExpr& value,
                          const VarNode* inner, int64_t length) {
  auto it = w->env.find(inner);
  bool present = it != w->env.end();
  int64_t saved = present ? it->second : 0;
  std::vector<Piece> pieces;
  try {
    SplitRow(w, value, inner, 0, length, &pieces);
  } catch (...) {
    if (present) {
      w->env[inner] = saved;
    } else {
      w->env.erase(inner);
    }
    throw;
  }
  if (present) {
    w->env[inner] = saved;
  } else {
    w->env.erase(inner);
  }
  return pieces;
}

void Walker::EmitPointwise(
    const BufferStore& store, const std::vector<const VarNode*>& spatial,
    const std::unordered_map<const VarNode*, int64_t>& extents) {
  if (store->value.as<BufferLoadNode>() != nullptr) {
    EmitMovement(store, spatial, extents);
    return;
  }
  if (spatial.empty()) throw Unsupported("pointwise block with no axes");
  const VarNode* inner = spatial.back();
  std::vector<const VarNode*> outer(spatial.begin(), spatial.end() - 1);
  int64_t length = extents.at(inner);
  Flush();
  int64_t unit = InSram(store->buffer->data.get()) ? 4 : 1;
  Positions positions(outer, extents);
  for (int64_t flat = 0; flat < positions.total(); ++flat) {
    const std::vector<int64_t>& position = positions.at(flat);
    for (size_t i = 0; i < outer.size(); ++i) env[outer[i]] = position[i];
    for (const Piece& piece : Pieces(this, store->value, inner, length)) {
      if (piece.count <= 0) continue;
      inner_base = piece.lo;
      auto [dst, dst_stride] = AffineFlat(store->buffer, store->indices, inner);
      if (dst_stride != unit) {
        throw Unsupported("pointwise destination stride " +
                          std::to_string(dst_stride));
      }
      depth = 0;
      Materialize(piece.expr, inner, piece.count, dst);
    }
    inner_base = 0;
  }
}

// -------------------------------------------------------------- movement

void Walker::EmitMovement(
    const BufferStore& store, const std::vector<const VarNode*>& spatial,
    const std::unordered_map<const VarNode*, int64_t>& extents) {
  const auto* load_node = store->value.as<BufferLoadNode>();
  if (load_node == nullptr) throw Unsupported("movement expects a buffer load");
  BufferLoad load = Downcast<BufferLoad>(store->value);
  bool dst_sram = InSram(store->buffer->data.get());
  bool src_sram = InSram(load->buffer->data.get());
  if (dst_sram != src_sram) {
    // a cache_read / cache_write stage: the DMA the hand-written backend
    // used to insert by hand
    EmitDma(store, load, spatial, extents, dst_sram);
    return;
  }
  if (spatial.empty()) throw Unsupported("movement block with no axes");
  const VarNode* inner = spatial.back();
  std::vector<const VarNode*> outer(spatial.begin(), spatial.end() - 1);
  int64_t length = extents.at(inner);
  Flush();
  const VarNode* src_data = load->buffer->data.get();
  const VarNode* dst_data = store->buffer->data.get();
  int64_t src_unit = src_sram ? Width(src_data) : ItemSize(src_data);
  int64_t dst_unit = dst_sram ? Width(dst_data) : ItemSize(dst_data);

  Positions positions(outer, extents);
  for (int64_t flat = 0; flat < positions.total(); ++flat) {
    const std::vector<int64_t>& position = positions.at(flat);
    for (size_t i = 0; i < outer.size(); ++i) env[outer[i]] = position[i];
    auto [src_base, src_stride] = AffineFlat(load->buffer, load->indices, inner);
    auto [dst_base, dst_stride] =
        AffineFlat(store->buffer, store->indices, inner);
    if (dst_stride != dst_unit) {
      throw Unsupported("movement destination stride " +
                        std::to_string(dst_stride));
    }
    if (src_stride == src_unit) {                 // contiguous copy / slice
      // the vector length field is 16-bit; longer rows go in pieces
      int64_t offset = 0;
      while (length - offset > 0xFFFF) {
        a.vlen(0xFFFF);
        stage.vector(kSrc1, src_base + offset * src_unit);
        a.load(0, kSrc1);
        a.v_copy();
        stage.vector(kDst, dst_base + offset * dst_unit);
        a.save(0);
        offset += 0xFFFF;
      }
      a.vlen(length - offset);
      stage.vector(kSrc1, src_base + offset * src_unit);
      a.load(0, kSrc1);
      a.v_copy();
      stage.vector(kDst, dst_base + offset * dst_unit);
      a.save(0);
      continue;
    }
    if (src_stride == 0) {                        // broadcast along the row
      a.vlen(length);
      stage.broadcast(src_base);
    } else {
      // non-unit source stride (transpose): the vector load only reads
      // contiguously, so use the matrix strided load, which gathers one
      // column, then copy it out as a vector
      stage.strided(kSrc1, src_base, src_stride / src_unit, length);
      a.load(1, kSrc1, 1, 1, 0);
      a.vlen(length);
      a.v_copy();
    }
    stage.vector(kDst, dst_base);
    a.save(0);
  }
}

// ------------------------------------------------------------------- DMA

struct Segment {
  int64_t glob;
  int64_t sram;
  int64_t nbytes;
};

// End of the run of segments that are adjacent in global memory.
size_t ContiguousGroup(const std::vector<Segment>& segments, size_t index) {
  size_t end = index + 1;
  while (end < segments.size() &&
         segments[end].glob == segments[end - 1].glob + segments[end - 1].nbytes) {
    ++end;
  }
  return end;
}

// Length and global row stride of a 2D block starting at ``index``.  A
// transfer addresses global memory in cells and packs SRAM rows, so a block
// needs base, row length and stride on whole 4-byte cells.
std::pair<int64_t, int64_t> RegularBlock(const std::vector<Segment>& segments,
                                         size_t index) {
  const Segment& first = segments[index];
  if (index + 1 >= segments.size() || first.glob % 4 || first.nbytes % 4) {
    return {1, 0};
  }
  int64_t stride = segments[index + 1].glob - first.glob;
  if (stride % 4 || stride < first.nbytes) return {1, 0};
  int64_t rows = 1;
  while (index + static_cast<size_t>(rows) < segments.size()) {
    const Segment& next = segments[index + static_cast<size_t>(rows)];
    if (next.nbytes != first.nbytes || next.glob != first.glob + rows * stride ||
        next.sram != first.sram + rows * first.nbytes * 2) {
      break;
    }
    ++rows;
  }
  return {rows, stride};
}

constexpr int64_t kBounceBytes = 8192;

int64_t BounceChunk(Walker* w, int64_t glob,
                    const std::vector<std::pair<int64_t, int64_t>>& pieces,
                    int64_t total, int64_t scratch, bool to_sram, int64_t lead) {
  int64_t span = total + lead;
  if (span > 2 * kBounceBytes) {
    throw Unsupported("bounce chunk of " + std::to_string(span) +
                      " exceeds the scratch slot");
  }
  if (to_sram) {
    w->stage.dma_in(glob, scratch, span);
  } else if (lead) {
    // the bytes sharing the first cell belong to someone else, so read them
    // back before the cell is written
    w->stage.dma_in(glob, scratch, lead);
  }
  int64_t offset = lead;
  for (const auto& [sram, nbytes] : pieces) {
    if (nbytes % 2 || offset % 2) {
      throw Unsupported("bounce pieces must be fp16-aligned");
    }
    int64_t src = to_sram ? scratch + offset * 2 : sram;
    int64_t dst = to_sram ? sram : scratch + offset * 2;
    w->a.vlen(nbytes / 2);            // the gather copies move fp16 lanes
    w->stage.vector(kSrc1, src);
    w->a.load(0, kSrc1);
    w->a.v_copy();
    w->stage.vector(kDst, dst);
    w->a.save(0);
    offset += nbytes;
  }
  if (!to_sram) w->stage.dma_out(glob, scratch, span);
  return glob + span;
}

// Move a globally contiguous range whose SRAM image is scattered.  A transfer
// moves whole 32-bit cells, so a row that begins mid-cell cannot be moved on
// its own, while a vector copy has element granularity; the rows are gathered
// into a contiguous scratch region and moved in aligned chunks.
void BounceDma(Walker* w, int64_t glob,
               const std::vector<std::pair<int64_t, int64_t>>& pieces,
               bool to_sram) {
  w->Flush();                         // the copies below take the PE output
  if (w->scratch_slots.empty()) throw Unsupported("no scratch slot for bounce");
  int64_t scratch = w->scratch_slots[0];
  int64_t lead = glob % 4;      // bytes sharing the first cell with someone else
  glob -= lead;
  std::vector<std::pair<int64_t, int64_t>> chunk;
  int64_t total = 0;
  for (const auto& piece : pieces) {
    chunk.push_back(piece);
    total += piece.second;
    if (total + lead >= kBounceBytes && (total + lead) % 4 == 0) {
      glob = BounceChunk(w, glob, chunk, total, scratch, to_sram, lead);
      chunk.clear();
      total = 0;
      lead = 0;
    }
  }
  if (!chunk.empty()) {
    BounceChunk(w, glob, chunk, total, scratch, to_sram, lead);
  }
}

void EmitSegments(Walker* w, const std::vector<Segment>& segments,
                  bool to_sram) {
  size_t index = 0, group = 0;
  bool aligned = false;
  while (index < segments.size()) {
    auto [rows, stride] = RegularBlock(segments, index);
    const Segment& segment = segments[index];
    if (rows > 1) {
      // a tile of a wider tensor: exactly what the 2D transfer is for
      w->stage.dma_2d(segment.glob, stride, segment.sram, rows, segment.nbytes,
                      to_sram);
      index += rows;
      group = 0;
      continue;
    }
    if (index >= group) {
      group = ContiguousGroup(segments, index);
      aligned = true;
      for (size_t i = index; i < group; ++i) {
        if (segments[i].glob % 4 || segments[i].nbytes % 4) aligned = false;
      }
    }
    if ((aligned || group == index + 1) && segment.glob % 4 == 0) {
      if (to_sram) {
        w->stage.dma_in(segment.glob, segment.sram, segment.nbytes);
      } else {
        w->stage.dma_out(segment.glob, segment.sram, segment.nbytes);
      }
      ++index;
      continue;
    }
    std::vector<std::pair<int64_t, int64_t>> pieces;
    size_t end = (group == index + 1) ? index + 1 : group;
    for (size_t i = index; i < end; ++i) {
      pieces.emplace_back(segments[i].sram, segments[i].nbytes);
    }
    BounceDma(w, segment.glob, pieces, to_sram);
    index = end;
  }
}

// Emit GLOAD/GSTORE for a cache block, one per contiguous run.  Rows adjacent
// on both sides are merged into a single transfer; merging is what makes odd
// row lengths work at all, since a transfer moves whole 32-bit cells and a row
// of odd length would leave the next row starting mid-cell.
void Walker::EmitDma(const BufferStore& store, const BufferLoad& load,
                     const std::vector<const VarNode*>& spatial,
                     const std::unordered_map<const VarNode*, int64_t>& extents,
                     bool to_sram) {
  if (spatial.empty()) throw Unsupported("DMA block with no axes");
  const VarNode* inner = spatial.back();
  std::vector<const VarNode*> outer(spatial.begin(), spatial.end() - 1);
  int64_t length = extents.at(inner);
  if (!to_sram) {
    // a write-back reads SRAM the accumulator still owns; staging an input
    // does not touch the PE, so it must not break a MAC chain
    Flush();
  }
  const VarNode* src_data = load->buffer->data.get();
  const VarNode* dst_data = store->buffer->data.get();
  int64_t src_unit = InSram(src_data) ? Width(src_data) : ItemSize(src_data);
  int64_t dst_unit = InSram(dst_data) ? Width(dst_data) : ItemSize(dst_data);
  // every row transfers the same number of bytes
  int64_t nbytes = length * ItemSize(to_sram ? src_data : dst_data);

  std::vector<Segment> segments;
  Positions positions(outer, extents);
  for (int64_t flat = 0; flat < positions.total(); ++flat) {
    const std::vector<int64_t>& position = positions.at(flat);
    for (size_t i = 0; i < outer.size(); ++i) env[outer[i]] = position[i];
    auto [src_base, src_stride] = AffineFlat(load->buffer, load->indices, inner);
    auto [dst_base, dst_stride] =
        AffineFlat(store->buffer, store->indices, inner);
    if (src_stride != src_unit || dst_stride != dst_unit) {
      throw Unsupported("DMA stage needs contiguous rows");
    }
    int64_t glob = to_sram ? src_base : dst_base;
    int64_t sram_addr = to_sram ? dst_base : src_base;
    // segment lengths are bytes; every dtype packs SRAM at two nibbles per
    // byte, so the SRAM side advances at exactly twice the bytes
    if (!segments.empty() &&
        glob == segments.back().glob + segments.back().nbytes &&
        sram_addr == segments.back().sram + segments.back().nbytes * 2) {
      segments.back().nbytes += nbytes;    // one contiguous run on both sides
    } else {
      segments.push_back({glob, sram_addr, nbytes});
    }
  }
  EmitSegments(this, segments, to_sram);
}

// ------------------------------------------------- expression materialization
// The vector unit applies one operation to a whole vector, so an expression
// tree is serialized into vector steps with SRAM temporaries.

int64_t Walker::Materialize(const PrimExpr& expr, const VarNode* inner,
                            int64_t length, int64_t into) {
  if (const auto* load = expr.as<BufferLoadNode>()) {
    auto [base, stride] = AffineFlat(load->buffer, load->indices, inner);
    const VarNode* data = load->buffer->data.get();
    int64_t unit = InSram(data) ? Width(data) : ItemSize(data);
    if (stride == 0) {
      // a per-row scalar (e.g. the reciprocal norm): broadcast it
      int64_t target = into >= 0 ? into : Slot();
      a.vlen(length);
      stage.broadcast(base);
      stage.vector(kDst, target);
      a.save(0);
      return target;
    }
    if (stride != unit) {
      throw Unsupported("operand stride " + std::to_string(stride) +
                        " is not contiguous");
    }
    if (into < 0) return base;
    a.vlen(length);
    stage.vector(kSrc1, base);
    a.load(0, kSrc1);
    a.v_copy();
    stage.vector(kDst, into);
    a.save(0);
    return into;
  }
  double literal;
  bool is_literal = false;
  if (const auto* imm = expr.as<FloatImmNode>()) {
    literal = imm->value;
    is_literal = true;
  } else if (const auto* imm = expr.as<IntImmNode>()) {
    literal = static_cast<double>(imm->value);
    is_literal = true;
  }
  if (is_literal) {
    auto it = constants.find(literal);
    if (it == constants.end()) {
      throw Unsupported("constant " + std::to_string(literal) +
                        " not in the pool");
    }
    int64_t target = into >= 0 ? into : Slot();
    a.vlen(length);
    stage.broadcast(it->second);
    stage.vector(kDst, target);
    a.save(0);
    return target;
  }

  PrimExpr left, right;
  int binary = -1;
  if (const auto* node = expr.as<AddNode>()) {
    left = node->a; right = node->b; binary = 0;
  } else if (const auto* node = expr.as<SubNode>()) {
    left = node->a; right = node->b; binary = 1;
  } else if (const auto* node = expr.as<MulNode>()) {
    left = node->a; right = node->b; binary = 2;
  } else if (const auto* node = expr.as<DivNode>()) {
    left = node->a; right = node->b; binary = 3;
  }
  if (binary >= 0) {
    int outer_depth = depth;
    int64_t lhs = Materialize(left, inner, length);
    int64_t rhs = Materialize(right, inner, length);
    depth = outer_depth;
    int64_t target = into >= 0 ? into : Slot();
    a.vlen(length);
    stage.vector(kSrc1, lhs);
    a.load(0, kSrc1);
    stage.vector(kSrc2, rhs);
    a.load(0, kSrc2);
    switch (binary) {
      case 0: a.v_add(kVector); break;
      case 1: a.v_sub(kVector); break;
      case 2: a.v_mul(kVector); break;
      default: a.v_div(kVector); break;
    }
    stage.vector(kDst, target);
    a.save(0);
    return target;
  }

  if (const auto* cast = expr.as<CastNode>()) {
    return MaterializeCast(cast, inner, length, into);
  }
  if (const auto* call = expr.as<CallNode>()) {
    const auto* op = call->op.as<OpNode>();
    if (op == nullptr) throw Unsupported("call without an op");
    const std::string& name = op->name;
    if (name == "tir.rsqrt") return MaterializeRsqrt(call, inner, length, into);
    if (name == "tir.sigmoid") {
      return MaterializeSigmoid(call, inner, length, into);
    }
    if (name == "tir.tanh") return MaterializeTanh(call, inner, length, into);
    int unary = -1;
    if (name == "tir.exp") unary = 0;
    else if (name == "tir.sqrt") unary = 1;
    else if (name == "tir.cos") unary = 2;
    else if (name == "tir.sin") unary = 3;
    if (unary < 0) throw Unsupported("unsupported intrinsic " + name);
    int outer_depth = depth;
    int64_t operand = Materialize(call->args[0], inner, length);
    depth = outer_depth;
    int64_t target = into >= 0 ? into : Slot();
    a.vlen(length);
    stage.vector(kSrc1, operand);
    a.load(0, kSrc1);
    switch (unary) {
      case 0: a.v_exp(); break;
      case 1: a.v_sqrt(); break;
      case 2: a.v_cos(); break;
      default: a.v_sin(); break;
    }
    stage.vector(kDst, target);
    a.save(0);
    return target;
  }
  throw Unsupported(std::string("unsupported expression ") + expr->GetTypeKey());
}

// int8 -> float16 via VDEQUANT; anything else has no instruction.
int64_t Walker::MaterializeCast(const CastNode* cast, const VarNode* inner,
                                int64_t length, int64_t into) {
  const auto* source = cast->value.as<BufferLoadNode>();
  if (!(cast->dtype.is_float() && cast->dtype.bits() == 16) ||
      source == nullptr || !(source->buffer->dtype.is_int() &&
                             source->buffer->dtype.bits() == 8)) {
    throw Unsupported("unsupported cast");
  }
  auto [base, stride] = AffineFlat(source->buffer, source->indices, inner);
  if (stride != Width(source->buffer->data.get())) {
    throw Unsupported("vdequant needs a contiguous int8 row");
  }
  if (one_fp32 < 0) throw Unsupported("no fp32 1.0 was staged for vdequant");
  int64_t target = into >= 0 ? into : Slot();
  a.vlen(length);
  // VDEQUANT reads SRC1's partial address with the descriptor dtype and
  // multiplies by the one scalar at the activation-scale address; a literal
  // 1.0 there leaves the conversion pure, and the per-channel scale is
  // applied by the vector multiply that follows in the TIR
  a.addr(kSrc1, base, kPartial);
  a.shape(kSrc1, 1, length, kPartial, kInt8);
  a.ascale(one_fp32);
  a.vdequant();
  stage.vector(kDst, target);
  a.save(0);
  // the INT8 dtype is sticky descriptor state: restore FP16 so the next
  // consumer of SRC1 is not misread
  a.shape(kSrc1, 1, length, kPartial, kFp16);
  return target;
}

// 1/(1+exp(-x)) from the primitives the unit provides.
int64_t Walker::MaterializeSigmoid(const CallNode* call, const VarNode* inner,
                                   int64_t length, int64_t into) {
  int outer_depth = depth;
  int64_t operand = Materialize(call->args[0], inner, length);
  depth = outer_depth;
  int64_t negated = Slot();
  a.vlen(length);
  stage.vector(kSrc1, operand);
  a.load(0, kSrc1);
  a.v_sign_inv();
  stage.vector(kDst, negated);
  a.save(0);
  int64_t exponent = Slot();
  a.vlen(length);
  stage.vector(kSrc1, negated);
  a.load(0, kSrc1);
  a.v_exp();
  stage.vector(kDst, exponent);
  a.save(0);
  int64_t one = Materialize(FloatImm(DataType::Float(16), 1.0), inner, length);
  int64_t denominator = Slot();
  a.vlen(length);
  stage.vector(kSrc1, exponent);
  a.load(0, kSrc1);
  stage.vector(kSrc2, one);
  a.load(0, kSrc2);
  a.v_add(kVector);
  stage.vector(kDst, denominator);
  a.save(0);
  int64_t target = into >= 0 ? into : Slot();
  a.vlen(length);
  stage.vector(kSrc1, one);
  a.load(0, kSrc1);
  stage.vector(kSrc2, denominator);
  a.load(0, kSrc2);
  a.v_div(kVector);
  stage.vector(kDst, target);
  a.save(0);
  return target;
}

// 1 - 2/(exp(2x)+1).  Written this way rather than (e^2x - 1)/(e^2x + 1)
// because it saturates correctly: once exp(2x) overflows FP16 the quotient is
// 0 and the result is exactly 1, where the difference form would divide
// infinity by infinity.  Large negative x needs no special case either --
// exp(2x) underflows to 0 and the result is -1.
int64_t Walker::MaterializeTanh(const CallNode* call, const VarNode* inner,
                                int64_t length, int64_t into) {
  int outer_depth = depth;
  int64_t operand = Materialize(call->args[0], inner, length);
  depth = outer_depth;
  int64_t two = Materialize(FloatImm(DataType::Float(16), 2.0), inner, length);
  int64_t doubled = Slot();
  a.vlen(length);
  stage.vector(kSrc1, operand);
  a.load(0, kSrc1);
  stage.vector(kSrc2, two);
  a.load(0, kSrc2);
  a.v_mul(kVector);
  stage.vector(kDst, doubled);
  a.save(0);
  int64_t exponent = Slot();
  a.vlen(length);
  stage.vector(kSrc1, doubled);
  a.load(0, kSrc1);
  a.v_exp();
  stage.vector(kDst, exponent);
  a.save(0);
  int64_t one = Materialize(FloatImm(DataType::Float(16), 1.0), inner, length);
  int64_t denominator = Slot();
  a.vlen(length);
  stage.vector(kSrc1, exponent);
  a.load(0, kSrc1);
  stage.vector(kSrc2, one);
  a.load(0, kSrc2);
  a.v_add(kVector);
  stage.vector(kDst, denominator);
  a.save(0);
  int64_t quotient = Slot();
  a.vlen(length);
  stage.vector(kSrc1, two);
  a.load(0, kSrc1);
  stage.vector(kSrc2, denominator);
  a.load(0, kSrc2);
  a.v_div(kVector);
  stage.vector(kDst, quotient);
  a.save(0);
  int64_t target = into >= 0 ? into : Slot();
  a.vlen(length);
  stage.vector(kSrc1, one);
  a.load(0, kSrc1);
  stage.vector(kSrc2, quotient);
  a.load(0, kSrc2);
  a.v_sub(kVector);
  stage.vector(kDst, target);
  a.save(0);
  return target;
}

// 1/sqrt(x) via the sqrt and divide the unit provides.
int64_t Walker::MaterializeRsqrt(const CallNode* call, const VarNode* inner,
                                 int64_t length, int64_t into) {
  int outer_depth = depth;
  int64_t operand = Materialize(call->args[0], inner, length);
  depth = outer_depth;
  int64_t root = Slot();
  a.vlen(length);
  stage.vector(kSrc1, operand);
  a.load(0, kSrc1);
  a.v_sqrt();
  stage.vector(kDst, root);
  a.save(0);
  int64_t one = Materialize(FloatImm(DataType::Float(16), 1.0), inner, length);
  int64_t target = into >= 0 ? into : Slot();
  a.vlen(length);
  stage.vector(kSrc1, one);
  a.load(0, kSrc1);
  stage.vector(kSrc2, root);
  a.load(0, kSrc2);
  a.v_div(kVector);
  stage.vector(kDst, target);
  a.save(0);
  return target;
}

// ------------------------------------------------------- tensorized markers

void Walker::Call(const PrimExpr& expr) {
  const auto* call = expr.as<CallNode>();
  if (call == nullptr || !call->op.same_as(builtin::call_extern())) {
    throw Unsupported("unhandled call");
  }
  const auto* name_node = call->args[0].as<StringImmNode>();
  if (name_node == nullptr) throw Unsupported("call_extern without a name");
  const std::string& name = name_node->value;
  auto starts_with = [&name](const char* prefix) {
    return name.rfind(prefix, 0) == 0;
  };
  if (name == "npu_fill_zero") {
    zeroed.insert(Ptr(call->args[1]));
    return;
  }
  if (name == "npu_gemm_acc") {
    EmitGemm(Ptr(call->args[1]), Ev(call->args[2]), Ptr(call->args[3]),
             Ev(call->args[4]), Ptr(call->args[5]), Ev(call->args[6]));
    return;
  }
  if (starts_with("npu_ew2_")) {
    Flush();
    std::string op = name.substr(std::strlen("npu_ew2_"));
    int64_t dst = Ptr(call->args[1]), lhs = Ptr(call->args[2]);
    int64_t rhs = Ptr(call->args[3]), count = Ev(call->args[4]);
    a.vlen(count);
    stage.vector(kSrc1, lhs);
    a.load(0, kSrc1);
    stage.vector(kSrc2, rhs);
    a.load(0, kSrc2);
    if (op == "add") a.v_add(kVector);
    else if (op == "subtract") a.v_sub(kVector);
    else if (op == "multiply") a.v_mul(kVector);
    else if (op == "divide") a.v_div(kVector);
    else throw Unsupported("unsupported binary op " + op);
    stage.vector(kDst, dst);
    a.save(0);
    return;
  }
  if (starts_with("npu_ew1_")) {
    Flush();
    std::string op = name.substr(std::strlen("npu_ew1_"));
    int64_t dst = Ptr(call->args[1]), src = Ptr(call->args[2]);
    int64_t count = Ev(call->args[3]);
    a.vlen(count);
    stage.vector(kSrc1, src);
    a.load(0, kSrc1);
    if (op == "sqrt") a.v_sqrt();
    else if (op == "exp") a.v_exp();
    else if (op == "negative") a.v_sign_inv();
    else if (op == "cos") a.v_cos();
    else if (op == "sin") a.v_sin();
    else throw Unsupported("unsupported unary op " + op);
    stage.vector(kDst, dst);
    a.save(0);
    return;
  }
  throw Unsupported("unknown intrinsic " + name);
}

// One 64x64x64 tile of C += A @ B.  Consecutive calls to the same C tile chain
// through the PE accumulator with the MAC bit and store once, matching the
// hand-written backend's sequence.
void Walker::EmitGemm(int64_t c, int64_t sc, int64_t a_addr, int64_t sa,
                      int64_t b_addr, int64_t sb) {
  bool first = zeroed.count(c) != 0;
  if (first) {
    zeroed.erase(c);
    Flush();
    has_pending = true;
    pending_c = c;
    pending_sc = sc;
  } else if (!has_pending || pending_c != c) {
    // the chain was interrupted (a vector op owns the PE output in between),
    // so reload the partial sum before continuing
    auto it = stored.find(c);
    if (it == stored.end()) {
      throw Unsupported("accumulate into unseen C tile");
    }
    Flush();
    stage.region(kSrc1, c, it->second, kTile, kTile);
    a.load(1, kSrc1);
    a.m_add(kImm, 0);
    has_pending = true;
    pending_c = c;
    pending_sc = sc;
  }
  stage.region(kSrc1, a_addr, sa, kTile, kTile);
  a.load(1, kSrc1);
  stage.region(kSrc2, b_addr, sb, kTile, kTile);
  a.load(1, kSrc2);
  a.m_mul(kVector, 0, kActOff, !first);
}

// A tensorized block views a tile of a root buffer through match_buffer; bind
// the view's data Var to the tile's element offset and its symbolic stride to
// the parent row width.
void Walker::BindMatch(const MatchBufferRegion& match) {
  const Buffer& view = match->buffer;
  const Buffer& source = match->source->buffer;
  // a tile lives in the last two axes; leading axes select the batch
  int rank = static_cast<int>(source->shape.size());
  int64_t offset = 0, scale = 1;
  const Array<Range>& region = match->source->region;
  for (int i = rank - 1, j = static_cast<int>(region.size()) - 1;
       i >= 0 && j >= 0; --i, --j) {
    offset += Ev(region[j]->min) * scale;
    scale *= Ev(source->shape[i]);
  }
  const VarNode* parent = source->data.get();
  const VarNode* data = view->data.get();
  // a view shares the parent's data pointer; its position is carried by the
  // symbolic elem_offset that access_ptr adds
  if (InSram(parent)) {
    sram[data] = sram[parent];
  } else if (bases.count(parent)) {
    bases[data] = bases[parent];
  } else {
    throw Unsupported("match_buffer source is not a known buffer");
  }
  // SRAM views carry their position in nibbles; global views stay in
  // elements, because Ptr applies the byte scaling itself
  int64_t unit = InSram(parent) ? Width(parent) : 1;
  if (const auto* elem_offset = view->elem_offset.as<VarNode>()) {
    env[elem_offset] = offset * unit;
  }
  int64_t values[2] = {Ev(source->shape[rank - 1]), 1};
  for (size_t i = 0; i < view->strides.size() && i < 2; ++i) {
    if (const auto* symbol = view->strides[i].as<VarNode>()) {
      env[symbol] = values[i];
    }
  }
}

// ------------------------------------------------------------ entry point

NDArray ToWords(const std::vector<Word>& words) {
  NDArray array = NDArray::Empty({static_cast<int64_t>(words.size())},
                                 DLDataType{kDLUInt, 32, 1},
                                 DLDevice{kDLCPU, 0});
  if (!words.empty()) {
    std::memcpy(static_cast<char*>(array->data) + array->byte_offset,
                words.data(), words.size() * sizeof(Word));
  }
  return array;
}

TVM_REGISTER_GLOBAL("npu.codegen_kernel")
    .set_body_typed([](PrimFunc func, Array<Integer> addresses,
                       Array<Buffer> sram_buffers, Array<Integer> sram_nibbles,
                       Array<Integer> scratch_slots,
                       Array<PrimExpr> constant_values,
                       Array<Integer> constant_addrs,
                       Integer one_fp32) -> NDArray {
      Walker walker;
      // bind by parameter order -- buffer_map is a map, and its iteration
      // order is not the signature's
      if (addresses.size() != func->params.size()) {
        throw Unsupported("address count does not match the signature");
      }
      for (size_t i = 0; i < func->params.size(); ++i) {
        auto it = func->buffer_map.find(func->params[i]);
        if (it == func->buffer_map.end()) continue;
        const Buffer& buffer = (*it).second;
        walker.bases[buffer->data.get()] = addresses[i]->value;
        walker.units[buffer->data.get()] = UnitOf(buffer->dtype);
      }
      for (size_t i = 0; i < sram_buffers.size(); ++i) {
        const Buffer& buffer = sram_buffers[i];
        walker.sram[buffer->data.get()] = sram_nibbles[i]->value;
        walker.units[buffer->data.get()] = UnitOf(buffer->dtype);
      }
      for (const Integer& slot : scratch_slots) {
        walker.scratch_slots.push_back(slot->value);
      }
      for (size_t i = 0; i < constant_values.size(); ++i) {
        const auto* imm = constant_values[i].as<FloatImmNode>();
        if (imm == nullptr) throw Unsupported("constant pool needs FloatImm");
        walker.constants[imm->value] = constant_addrs[i]->value;
      }
      walker.one_fp32 = one_fp32->value;
      walker.Run(func);
      return ToWords(walker.a.words);
    });

}  // namespace
}  // namespace npu
