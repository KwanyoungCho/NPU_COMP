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
                    narrow = node.args[1] if left > right else node.args[0]
                    self.env[narrow] = self.block_builder.emit(
                        relax.op.astype(self.env[narrow], wide))
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
    squared = (is_op("relax.power")(data32, wildcard())
               | is_op("relax.multiply")(data32, data32))
    mean = is_op("relax.mean")(squared)
    shifted = is_op("relax.add")(mean, wildcard())
    inv = is_op("relax.rsqrt")(shifted)
    normed = is_op("relax.multiply")(data32, inv)
    back = is_op("relax.astype")(normed)
    pattern = (is_op("relax.multiply")(weight, back)
               | is_op("relax.multiply")(back, weight))

    def rewrite(orig, matches):
        eps = 1e-6
        add_expr = matches[shifted]
        if isinstance(add_expr.args[1], relax.Constant):
            eps = float(add_expr.args[1].data.numpy())
        gamma = relax.op.astype(matches[weight],
                                matches[data].struct_info.dtype)
        return relax.op.nn.rms_norm(matches[data], gamma, axes=[-1],
                                    epsilon=eps)

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
                and int(left.shape[0]) == 1):
            return call
        block = self.builder_
        new_l = block.normalize(relax.op.reshape(
            lhs, [int(d) for d in left.shape][1:]))
        new_r = rhs
        if isinstance(right, relax.TensorStructInfo) \
                and right.ndim == left.ndim and int(right.shape[0]) == 1:
            new_r = block.normalize(relax.op.reshape(
                rhs, [int(d) for d in right.shape][1:]))
        out = block.normalize(relax.op.matmul(new_l, new_r))
        return block.normalize(relax.op.reshape(
            out, [1] + [int(d) for d in out.struct_info.shape]))


def causal_mask4d(seq, dtype=np.float16):
    """The additive 4D causal mask HF passes through untouched."""
    import torch

    fill = float(torch.finfo(torch.float16).min)
    mask = np.triu(np.full((seq, seq), fill, dtype=np.float32), k=1)
    return mask[None, None].astype(dtype)


def import_prefill(model, seq):
    """HF ``*ForCausalLM`` -> a Relax module ready for the graph pipeline.

    The returned function is ``main(inputs_embeds[1,S,D],
    attention_mask[1,1,S,S]) -> logits[1,S,V]``; everything else -- weights,
    positions, rotary tables -- is already a constant inside it.
    """
    import torch

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
    mod = relax.transform.FoldConstant()(mod)

    pattern, rewrite = _rms_norm_pattern()
    mod["main"] = rewrite_call(pattern, rewrite, mod["main"])
    mod["main"] = _Demote().visit_expr(mod["main"])
    mod["main"] = _SqueezeMatmulBatch().visit_expr(mod["main"])
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
