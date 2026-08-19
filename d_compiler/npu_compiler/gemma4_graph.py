"""Gemma 4 E2B text graphs as Relax IR over the shared legalize primitives.

Same construction route as the Llama V3 graphs: relax.BlockBuilder emitting
legalize builders, compiled by the common 0818 backend.  No reshape is needed
for the packed PLE dimension: [S, 35*256] group operations are expressed with
last-axis slice / concat, which the backend executes as strided copies.
"""
from __future__ import annotations

import numpy as np
from tvm import relax

from . import legalize


def _c(value):
    return relax.const(np.asarray(value, dtype="float16"))


def _P(name, shape):
    return relax.Var(name, relax.TensorStructInfo(list(shape), "float16"))


def build_gemma4_ple_module(spec, S):
    """per_layer_input for S tokens: [S, num_layers*ple_dim].

    Inputs: ``se`` host-scaled embeddings [S,D], raw PLE token rows ``tok``
    [S, L*P], projection ``Wproj`` [D, L*P], group norm weight ``Wn`` [1,P].
    Computes (RMSNorm(se @ Wproj * D^-0.5) + tok * sqrt(P)) * 2^-0.5 with the
    RMSNorm applied per ple_dim group via slice/concat.
    """
    D = spec.hidden_size
    L = spec.num_layers
    Pd = spec.layers[0].ple_dim
    width = L * Pd
    op = relax.op
    se = _P("se", (S, D))
    tok = _P("tok", (S, width))
    Wproj = _P("Wproj", (D, width))
    Wn = _P("Wn", (1, Pd))
    bb = relax.BlockBuilder()
    with bb.function("main", [se, tok, Wproj, Wn]):
        with bb.dataflow():
            proj = bb.emit(op.matmul(se, Wproj))
            proj = bb.emit(op.multiply(
                proj, _c(np.full((S, width), 1.0 / np.sqrt(float(D))))))
            tok_scaled = bb.emit(op.multiply(
                tok, _c(np.full((S, width), np.sqrt(float(Pd))))))
            groups = []
            for index in range(L):
                begin, end = index * Pd, (index + 1) * Pd
                part = bb.emit(op.strided_slice(
                    proj, axes=[1], begin=[begin], end=[end]))
                normed = legalize.rms_norm(
                    bb, part, Wn, S, Pd, eps=spec.rms_norm_eps)
                token_part = bb.emit(op.strided_slice(
                    tok_scaled, axes=[1], begin=[begin], end=[end]))
                combined = bb.emit(op.add(normed, token_part))
                groups.append(bb.emit(op.multiply(
                    combined, _c(np.full((S, Pd), 1.0 / np.sqrt(2.0))))))
            gv = bb.emit_output(bb.emit(op.concat(groups, axis=1)))
        bb.emit_func_output(gv)
    return bb.finalize()
