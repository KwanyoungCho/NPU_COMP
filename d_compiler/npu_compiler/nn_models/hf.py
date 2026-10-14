"""The HF model itself as the compiler's input.

The other frontends re-state each architecture in ``relax.frontend.nn`` and
load checkpoint weights by name.  This one starts from the HuggingFace model:
``transformers`` builds it, ``torch.export`` traces its own forward, and TVM's
standard torch frontend converts the trace to Relax.  Structure and numerics
come from HF; nothing about the architecture is re-written by hand.

What has to happen between the trace and our standard graph pipeline, and why:

* **mask and positions are inputs, then constants.**  HF builds the causal
  mask and position ids inside ``forward`` from data-dependent code the
  machine cannot run (arange over runtime lengths).  Passing a 4D additive
  mask and explicit positions removes that code from the trace; positions are
  then bound as constants so ``FoldConstant`` computes the whole rotary
  cos/sin table at compile time and no int64 ever reaches the machine.
* **weights become constants.**  On a machine whose program is one static
  image, weights are program constants anyway.  Binding them lets
  ``FoldConstant`` do the nn.Linear transposes at compile time, which is what
  ``LiftTransformParams`` achieves for the hand-written frontends -- without
  fighting that pass's parameter bookkeeping.
* **the traced RMSNorm is folded back into the op.**  HF normalizes by
  ``x^2 -> mean`` in fp32; demoted naively to fp16 the sum overflows at real
  widths.  A pattern rewrite restores ``relax.nn.rms_norm``, so the custom
  legalization's safe expansion (scale by 1/sqrt(D) before squaring) applies
  exactly as it does for the other families.
* **fp32 islands demote to fp16.**  HF upcasts softmax and rotary to fp32 and
  casts back.  The machine stores FP16 and accumulates FP32 internally per
  op, so demoting the islands is the same decision the hand-written
  frontends made explicitly.
* **the unit batch is squeezed off matmuls.**  The trace carries a leading
  batch of one on every tensor; the matmul schedule tiles the last three
  loops and treats leading dims as real batch, so the unit dim is reshaped
  away around each matmul (the reshapes become views downstream).
"""
from __future__ import annotations

import numpy as np
import tvm
from tvm import relax
from tvm.relax.dpl import is_op, rewrite_call, wildcard
from tvm.relax.expr_functor import PyExprMutator, mutator
from tvm.relax.frontend.torch.exported_program_translator import (
    ExportedProgramImporter)

_PRECEDENCE = {"float32": 2, "float16": 1}


class HFImporter(ExportedProgramImporter):
    """The stock importer plus what a traced HF forward needs.

    The additions are aliases for ops the base map spells differently, no-ops
    for torch's metadata assertions, and fp16/fp32 promotion on binary ops --
    torch promotes implicitly, Relax requires equal dtypes.
    """

    def _nop(self, node):
        return self.env[node.args[0]]

    def _binary_op(self, relax_op, intrinsic_op):
        convert = super()._binary_op(relax_op, intrinsic_op)

        def promoted(node):
            lhs, rhs = self.retrieve_args(node)[:2]
            if (isinstance(lhs, relax.Expr) and isinstance(rhs, relax.Expr)
                    and isinstance(lhs.struct_info, relax.TensorStructInfo)
                    and isinstance(rhs.struct_info, relax.TensorStructInfo)):
                left = _PRECEDENCE.get(str(lhs.struct_info.dtype))
                right = _PRECEDENCE.get(str(rhs.struct_info.dtype))
                if left and right and left != right:
                    wide = (lhs if left > right else rhs).struct_info.dtype
                    # Promotion belongs to this use, never to the FX value:
                    # another consumer may still require the original dtype.
                    if left < right:
                        lhs = self.block_builder.emit(relax.op.astype(lhs, wide))
                    else:
                        rhs = self.block_builder.emit(relax.op.astype(rhs, wide))
                    return self.block_builder.emit(relax_op(lhs, rhs))
            return convert(node)

        return promoted

    def create_convert_map(self):
        table = super().create_convert_map()
        table.setdefault("_assert_tensor_metadata.default", self._nop)
        table.setdefault("lift_fresh_copy.default", self._nop)
        table.setdefault("alias.default", self._nop)
        table.setdefault("contiguous.default", self._nop)
        table.setdefault("clone.default", self._nop)
        table.setdefault("detach.default", self._nop)
        table.setdefault("detach_.default", self._nop)
        table.setdefault("to.dtype", self._to_copy)
        table.setdefault("to.dtype_layout", self._to_copy)
        table.setdefault("arange.default", self._arange)
        table.setdefault("arange.start", self._arange)
        table.setdefault("reshape.default", self._reshape)
        return table


def _rms_norm_pattern():
    data = wildcard()
    weight = wildcard()
    data32 = is_op("relax.astype")(data)
    exponent, epsilon = wildcard(), wildcard()
    squared = (is_op("relax.power")(data32, exponent)
               | is_op("relax.multiply")(data32, data32))
    mean = is_op("relax.mean")(squared)
    shifted = is_op("relax.add")(mean, epsilon)
    inv = is_op("relax.rsqrt")(shifted)
    normed = is_op("relax.multiply")(data32, inv)
    back = is_op("relax.astype")(normed)
    pattern = (is_op("relax.multiply")(weight, back)
               | is_op("relax.multiply")(back, weight))

    def rewrite(orig, matches):
        # A syntactic match does not establish RMSNorm semantics.  In
        # particular power(x, 3), a different reduction axis, or a runtime
        # epsilon must retain their original expressions.
        import math

        def scalar(expr):
            if not isinstance(expr, relax.Constant):
                return None
            array = expr.data.numpy()
            return float(array) if array.ndim == 0 else None

        x, gamma = matches[data], matches[weight]
        info, winfo = x.struct_info, gamma.struct_info
        if not (isinstance(info, relax.TensorStructInfo)
                and isinstance(winfo, relax.TensorStructInfo)
                and info.dtype in ("float16", "float32")
                and winfo.dtype == "float16"
                and info.ndim > 0 and winfo.ndim == 1
                and info.shape is not None and winfo.shape is not None
                and tvm.ir.structural_equal(info.shape[-1], winfo.shape[0])):
            return orig
        if (matches[data32].attrs.dtype != "float32"
                or matches[back].attrs.dtype != "float16"):
            return orig
        if matches[squared].op == tvm.ir.Op.get("relax.power"):
            if scalar(matches[exponent]) != 2.0:
                return orig
        attrs = matches[mean].attrs
        axes = list(attrs.axis) if attrs.axis is not None else list(range(info.ndim))
        if (len(axes) != 1 or int(axes[0]) not in (-1, info.ndim - 1)
                or not attrs.keepdims):
            return orig
        eps = scalar(matches[epsilon])
        if eps is None or not math.isfinite(eps) or eps < 0:
            return orig
        # TOPI's RMSNorm multiplies gamma in FP32 *before* its output cast,
        # even for FP16 input.  This pattern multiplies gamma *after* that
        # cast.  Preserve the boundary using unit-weight normalization and a
        # separate multiply.  _Demote owns the frontend's FP16 policy.
        unit_weight = relax.op.ones(relax.ShapeExpr([info.shape[-1]]), info.dtype)
        normalized = relax.op.nn.rms_norm(x, unit_weight, axes=[-1], epsilon=eps)
        return relax.op.multiply(relax.op.astype(normalized, "float16"), gamma)

    return pattern, rewrite


def _dtype_of(expr):
    info = getattr(expr, "struct_info_", None)
    return str(info.dtype) if isinstance(info, relax.TensorStructInfo) else None


@mutator
class _Demote(PyExprMutator):
    """float32 -> float16, harmonizing half-demoted argument lists before
    Relax type-checks the rebuilt call."""

    def visit_call_(self, call):
        args = [self.visit_expr(arg) for arg in call.args]
        if call.op == tvm.ir.Op.get("relax.astype") and \
                call.attrs.dtype == "float32":
            return self.builder_.normalize(
                relax.op.astype(args[0], "float16"))
        if call.op == tvm.ir.Op.get("relax.matmul") and call.attrs.out_dtype == "float32":
            # This is the explicit frontend fp16 policy, separate from the
            # shape-only squeeze pass (which must preserve all attributes).
            args = [self.builder_.normalize(relax.op.astype(arg, "float16"))
                    if _dtype_of(arg) == "float32" else arg for arg in args]
            return self.builder_.normalize(relax.op.matmul(*args, out_dtype="float16"))
        dtypes = {_dtype_of(arg) for arg in args} - {None}
        if dtypes == {"float16", "float32"}:
            args = [self.builder_.normalize(relax.op.astype(arg, "float16"))
                    if _dtype_of(arg) == "float32" else arg for arg in args]
        return self.builder_.normalize(
            relax.Call(call.op, args, call.attrs, call.sinfo_args))

    def visit_constant_(self, const):
        if str(const.struct_info.dtype) == "float32":
            return relax.const(const.data.numpy().astype("float16"))
        return const


@mutator
class _SqueezeMatmulBatch(PyExprMutator):
    def visit_call_(self, call):
        call = super().visit_call_(call)
        if call.op != tvm.ir.Op.get("relax.matmul"):
            return call
        lhs, rhs = call.args
        left = lhs.struct_info
        right = rhs.struct_info
        if not (isinstance(left, relax.TensorStructInfo) and left.ndim >= 3
                and left.shape is not None
                and isinstance(left.shape[0], tvm.tir.IntImm)
                and int(left.shape[0]) == 1
                and isinstance(right, relax.TensorStructInfo)
                and right.shape is not None and right.ndim <= left.ndim):
            return call
        # Removing a dimension shared with a non-unit RHS batch would change
        # broadcasting. Keep that matmul intact.
        if right.ndim == left.ndim and not (
                isinstance(right.shape[0], tvm.tir.IntImm)
                and int(right.shape[0]) == 1):
            return call
        block = self.builder_
        new_l = block.normalize(relax.op.reshape(
            lhs, list(left.shape)[1:]))
        new_r = rhs
        if isinstance(right, relax.TensorStructInfo) \
                and right.ndim == left.ndim and int(right.shape[0]) == 1:
            new_r = block.normalize(relax.op.reshape(
                rhs, list(right.shape)[1:]))
        out = block.normalize(relax.Call(call.op, [new_l, new_r], call.attrs,
                                        call.sinfo_args, call.span))
        return block.normalize(relax.op.reshape(
            out, call.struct_info.shape))


def _drop_output_batch(func):
    """Return the logits without the leading unit dim the trace restores.

    ``_SqueezeMatmulBatch`` puts the batch back after every matmul so the
    graph keeps HF's shapes.  ``RewriteDataflowReshape`` turns all of those
    into views except one: the reshape that produces the function's result
    cannot be a view, because a result is a real tensor.  So the last one is
    emitted as a copy of the whole logits -- 262.7 MB at seq 512, changing no
    bytes.

    Dropping it changes the returned shape from [1, S, V] to [S, V].  The
    runners already reshape the result to (seq, vocab), so nothing downstream
    moves.
    """
    body = func.body
    if not isinstance(body, relax.SeqExpr) or not body.blocks:
        return func
    result = body.body
    if not isinstance(result, relax.Var):
        return func
    blocks = list(body.blocks)
    bindings = list(blocks[-1].bindings)
    for index, binding in enumerate(bindings):
        if not (isinstance(binding, relax.VarBinding)
                and binding.var.same_as(result)):
            continue
        call = binding.value
        if not (isinstance(call, relax.Call)
                and call.op == tvm.ir.Op.get("relax.reshape")):
            return func
        source = call.args[0]
        inner = source.struct_info
        outer = result.struct_info
        if not (isinstance(inner, relax.TensorStructInfo)
                and isinstance(outer, relax.TensorStructInfo)):
            return func
        pad = outer.ndim - inner.ndim
        if pad <= 0:
            return func
        want = [1] * pad + [int(d) for d in inner.shape]
        if [int(d) for d in outer.shape] != want:
            return func          # a real reshape, not just a unit dim
        renamed = relax.Var(result.name_hint, inner)
        bindings[index] = relax.VarBinding(renamed, source)
        blocks[-1] = type(blocks[-1])(bindings)
        return relax.Function(func.params,
                              relax.SeqExpr(blocks, renamed),
                              renamed.struct_info, func.is_pure, func.attrs)
    return func


def causal_mask4d(seq, dtype=np.float16):
    """The additive 4D causal mask HF passes through untouched."""
    import torch

    fill = float(torch.finfo(torch.float16).min)
    mask = np.triu(np.full((seq, seq), fill, dtype=np.float32), k=1)
    return mask[None, None].astype(dtype)


def _fold_host_constants(mod):
    """Fold FP32 rotary/position calculations without folding device arithmetic.

    Private Relax calls are temporary barriers to standard FoldConstant. They
    keep FP16 arithmetic visible until target legalization has defined its
    reduction widths. Standard InlinePrivateFunctions removes the barriers.
    Layout operations and casts may still fold without changing arithmetic.
    """
    safe = {"relax.astype", "relax.reshape", "relax.permute_dims",
            "relax.expand_dims", "relax.squeeze", "relax.broadcast_to",
            "relax.concat", "relax.split", "relax.strided_slice"}
    @mutator
    class Protect(PyExprMutator):
        def visit_call_(self, call):
            call = super().visit_call_(call)
            def half(info):
                return (isinstance(info, relax.TensorStructInfo) and info.dtype == "float16"
                        or isinstance(info, relax.TupleStructInfo) and any(half(x) for x in info.fields))
            if (not isinstance(call.op, tvm.ir.Op) or call.op.name in safe
                    or not any(half(arg.struct_info) for arg in call.args)):
                return call
            params = [relax.Var(f"arg{i}", arg.struct_info) for i, arg in enumerate(call.args)]
            fn = relax.Function(params, relax.Call(call.op, params, call.attrs, call.sinfo_args),
                                ret_struct_info=call.struct_info)
            gv = self.builder_.add_func(fn, "defer_device_arithmetic")
            return self.builder_.normalize(relax.Call(gv, call.args))
    rewrite = Protect(mod)
    for gv, fn in mod.functions.items():
        if isinstance(fn, relax.Function):
            rewrite.builder_.update_func(gv, rewrite.visit_expr(fn))
    folded = relax.transform.FoldConstant()(rewrite.builder_.get())
    # Inlining function calls splits dataflow blocks. Re-form the pure region
    # with stock passes so RMSNorm matching can see across former barriers.
    return tvm.transform.Sequential([relax.transform.InlinePrivateFunctions(),
        relax.transform.ToNonDataflow(), relax.transform.ConvertToDataflow(min_size=1),
        relax.transform.CanonicalizeBindings()])(folded)


def import_prefill(model, seq, precision="fp16", drop_output_batch=False):
    """HF ``*ForCausalLM`` -> a Relax module ready for the graph pipeline.

    The returned function is ``main(inputs_embeds[1,S,D],
    attention_mask[1,1,S,S]) -> logits[1,S,V]``; everything else -- weights,
    positions, rotary tables -- is already a constant inside it.

    ``precision="fp16"`` explicitly demotes traced FP32 islands for v09.
    ``"preserve"`` retains them for CPU/reference use; it does not promise
    FP32 vector support on the device.
    """
    import torch
    if precision not in ("fp16", "preserve"):
        raise ValueError("HF precision must be 'fp16' or 'preserve'")

    model = model.eval()
    hidden = model.config.hidden_size
    # TVM's importer looks buffers up in state_dict; torch.export keeps
    # non-persistent buffers (rotary inv_freq) out of it
    for module in model.modules():
        module._non_persistent_buffers_set.clear()

    class Prefill(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, inputs_embeds, attention_mask, position_ids,
                    cache_position):
            return self.inner(inputs_embeds=inputs_embeds,
                              attention_mask=attention_mask,
                              position_ids=position_ids,
                              cache_position=cache_position,
                              use_cache=False).logits

    example = (torch.zeros(1, seq, hidden, dtype=torch.float16),
               torch.from_numpy(causal_mask4d(seq).astype(np.float16)),
               torch.arange(seq).unsqueeze(0),
               torch.arange(seq))
    with torch.no_grad():
        exported = torch.export.export(Prefill(model), example)
    mod = HFImporter().from_exported_program(
        exported, keep_params_as_input=True, unwrap_unit_return_tuple=True,
        no_bind_return_tuple=False)

    main = mod["main"]
    bind = {"position_ids": tvm.nd.array(example[2].numpy()),
            "cache_position": tvm.nd.array(example[3].numpy())}
    for var, value in zip(main.params[4:], list(main.attrs["params"])):
        bind[var.name_hint] = value
    mod = relax.transform.BindParams("main", bind)(mod)
    mod = (_fold_host_constants(mod) if precision == "fp16"
           else relax.transform.FoldConstant()(mod))

    pattern, rewrite = _rms_norm_pattern()
    mod["main"] = rewrite_call(pattern, rewrite, mod["main"])
    if precision == "fp16":
        mod["main"] = _Demote().visit_expr(mod["main"])
    mod["main"] = mod["main"].with_attr("npu.frontend_precision", precision)
    mod["main"] = _SqueezeMatmulBatch().visit_expr(mod["main"])
    if drop_output_batch:
        mod["main"] = _drop_output_batch(mod["main"])
    return relax.transform.Normalize()(mod)


# ---- the family interface the runners use ----------------------------------

def model_config(assets, layers=0):
    """-> (checkpoint path, truncated depth); the HF model is built lazily."""
    return {"path": str(assets.path), "layers": int(layers)}


def build_prefill(config, seq):
    """Load the HF model itself and trace it; -> (IRModule, [], meta)."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    hf_config = AutoConfig.from_pretrained(config["path"])
    if config["layers"]:
        hf_config.num_hidden_layers = config["layers"]
    hf_config.attn_implementation = "eager"
    hf_config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        config["path"], config=hf_config, torch_dtype=torch.float16,
        attn_implementation="eager")
    mod = import_prefill(model, seq)
    meta = type("HFMeta", (), {
        "num_layers": int(hf_config.num_hidden_layers),
        "hidden_size": int(hf_config.hidden_size),
        "vocab_size": int(hf_config.vocab_size),
    })()
    del model
    return mod, [], meta


def load_params(assets, params, config=None):
    """Weights are constants inside the traced module; nothing to load."""
    return []


def runtime_inputs(assets, config, token_ids):
    seq = len(token_ids)
    return {
        "inputs_embeds": assets.embedding(
            [int(i) for i in token_ids]).astype(np.float16)[None],
        "attention_mask": causal_mask4d(seq),
    }
