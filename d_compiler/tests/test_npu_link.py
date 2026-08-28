"""S5 gate: a linked program runs on the C-model and computes the right values.

Single-op programs are checked against numpy; the shapes cover the exact-tile
path, the padded path (pad_einsum), and multi-K-tile accumulation.
"""
import sys
from pathlib import Path

import numpy as np
from tvm import relax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))

from npu_compiler import npu_legalize, npu_link, npu_memplan as M
from npu_compiler import tvm_pipeline as P


def _run(shapes, make, values, out_shape):
    bb = relax.BlockBuilder()
    args = [relax.Var(f"a{i}", relax.TensorStructInfo(list(s), "float16"))
            for i, s in enumerate(shapes)]
    with bb.function("prefill", args):
        with bb.dataflow():
            out = bb.emit_output(bb.emit(make(*args)))
        bb.emit_func_output(out)
    lowered = P.graph_pipeline(custom_legalize=npu_legalize.legalize_map(),
                               fuse=False, lift_params=False)(bb.finalize())
    asm, plan = npu_link.compile_program(lowered)
    planned, _ = M.assign_addresses(lowered)
    func = planned["prefill"]
    names = [p.name_hint for p in func.params]
    got, counters = npu_link.run_program(
        asm, plan, func, dict(zip(names, values)), out_shape)
    return got, len(asm.words), counters


def test_linked_matmul_shapes():
    """Exact tile, padded M and N, padded with two K tiles, and a 2x2 grid."""
    cases = [(64, 64, 64), (4, 64, 32), (7, 128, 96), (128, 128, 128)]
    for index, (m, k, n) in enumerate(cases):
        rng = np.random.default_rng(index)
        a = rng.normal(0, 0.3, (m, k)).astype(np.float16)
        b = rng.normal(0, 0.3, (k, n)).astype(np.float16)
        got, words, _ = _run([(m, k), (k, n)], relax.op.matmul, [a, b], (m, n))
        ref = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)
        error = float(np.abs(got.astype(np.float32)
                             - ref.astype(np.float32)).max())
        assert error < 0.01, (m, k, n, error)
        print(f"  [PASS] matmul [{m},{k}]x[{k},{n}] max|diff|={error:.4f} "
              f"({words:,} words)")


def test_linked_rms_norm():
    seq, width = 4, 64
    rng = np.random.default_rng(11)
    x = rng.normal(0, 0.5, (seq, width)).astype(np.float16)
    w = rng.random(width).astype(np.float16) + np.float16(0.5)
    got, words, _ = _run(
        [(seq, width), (width,)],
        lambda a, b: relax.op.nn.rms_norm(a, b, axes=[-1]), [x, w],
        (seq, width))
    xf = x.astype(np.float32)
    ref = (xf / np.sqrt((xf * xf).mean(-1, keepdims=True) + 1e-5)
           * w.astype(np.float32))
    error = float(np.abs(got.astype(np.float32) - ref).max())
    assert error < 0.02, error
    print(f"  [PASS] rms_norm max|diff|={error:.4f} ({words:,} words)")


def test_layer_ops_match_numpy():
    """Every op kind a decoder layer uses, at layer-like shapes."""
    import tvm.relax.op as R
    rng = np.random.default_rng(7)
    seq, heads, kv, head_dim, ffn = 4, 4, 2, 16, 128
    q = rng.normal(0, 0.5, (seq, heads, head_dim)).astype(np.float16)
    table = rng.normal(0, 0.5, (seq, 1, head_dim)).astype(np.float16)
    cases = [
        ("multiply broadcast", [(seq, heads, head_dim), (seq, 1, head_dim)],
         R.multiply, [q, table], (seq, heads, head_dim),
         q.astype(np.float32) * table.astype(np.float32)),
        ("permute_dims", [(seq, heads, head_dim)],
         lambda a: R.permute_dims(a, [1, 0, 2]), [q], (heads, seq, head_dim),
         q.transpose(1, 0, 2)),
    ]
    kvals = rng.normal(0, 0.5, (seq, kv, head_dim)).astype(np.float16)
    cases.append(("repeat (GQA)", [(seq, kv, head_dim)],
                  lambda a: R.repeat(a, heads // kv, axis=1), [kvals],
                  (seq, heads, head_dim), np.repeat(kvals, heads // kv, axis=1)))
    qh = rng.normal(0, 0.4, (heads, seq, head_dim)).astype(np.float16)
    kh = rng.normal(0, 0.4, (heads, head_dim, seq)).astype(np.float16)
    cases.append(("batched matmul", [(heads, seq, head_dim), (heads, head_dim, seq)],
                  R.matmul, [qh, kh], (heads, seq, seq),
                  qh.astype(np.float32) @ kh.astype(np.float32)))
    scores = rng.normal(0, 1.0, (heads, seq, seq)).astype(np.float16)
    shifted = np.exp(scores.astype(np.float32)
                     - scores.astype(np.float32).max(-1, keepdims=True))
    cases.append(("softmax", [(heads, seq, seq)],
                  lambda a: R.nn.softmax(a, axis=-1), [scores], (heads, seq, seq),
                  shifted / shifted.sum(-1, keepdims=True)))
    gate = rng.normal(0, 0.5, (seq, ffn)).astype(np.float16)
    cases.append(("silu", [(seq, ffn)], R.nn.silu, [gate], (seq, ffn),
                  gate.astype(np.float32) / (1 + np.exp(-gate.astype(np.float32)))))
    left = rng.normal(0, 0.5, (seq, head_dim)).astype(np.float16)
    right = rng.normal(0, 0.5, (seq, head_dim)).astype(np.float16)
    cases.append(("concat", [(seq, head_dim), (seq, head_dim)],
                  lambda a, b: R.concat([a, b], axis=1), [left, right],
                  (seq, 2 * head_dim), np.concatenate([left, right], axis=1)))
    half = head_dim // 2
    cases.append(("rotate_half", [(seq, head_dim)],
                  lambda a: R.concat(
                      [R.negative(R.strided_slice(a, axes=[1], begin=[half],
                                                  end=[head_dim])),
                       R.strided_slice(a, axes=[1], begin=[0], end=[half])],
                      axis=1),
                  [left], (seq, head_dim),
                  np.concatenate([-left[:, half:], left[:, :half]], axis=1)))

    for name, shapes, make, values, out_shape, reference in cases:
        got, _, _ = _run(shapes, make, values, out_shape)
        error = float(np.abs(got.astype(np.float32)
                             - np.asarray(reference, np.float32)).max())
        assert error < 0.02, (name, error)
        print(f"  [PASS] {name:20s} max|diff|={error:.5f}")


def test_padded_matmul_stages_its_inputs():
    """A padded matmul must not read global memory from a compute block."""
    lowered = None
    bb = relax.BlockBuilder()
    si = lambda s: relax.TensorStructInfo(list(s), "float16")
    a = relax.Var("a", si([4, 64]))
    b = relax.Var("b", si([64, 32]))
    with bb.function("prefill", [a, b]):
        with bb.dataflow():
            out = bb.emit_output(bb.emit(relax.op.matmul(a, b)))
        bb.emit_func_output(out)
    lowered = P.graph_pipeline(custom_legalize=npu_legalize.legalize_map(),
                               fuse=False, lift_params=False)(bb.finalize())
    asm, _ = npu_link.compile_program(lowered)
    dma = sum(1 for index, word in enumerate(asm.words)
              if (word & 0xFF) in (0xA0, 0xA8))
    assert dma > 0
    print(f"  [PASS] padded matmul stages through DMA ({dma} DMA words seen)")


def test_whole_layer_matches_the_cpu_build():
    """A one-layer model, linked and run on the C-model, against the llvm build
    of the same lowered module."""
    import tvm
    from npu_compiler.nn_models import llama

    cfg = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=1,
               num_attention_heads=4, num_key_value_heads=2, head_dim=16,
               vocab_size=32, rms_norm_eps=1e-5, rope_theta=500000.0)
    seq = 4
    mod, params, config = llama.build_prefill(cfg, seq=seq)
    rng = np.random.default_rng(5)
    weights = [tvm.nd.array(rng.normal(0, 0.15, p.shape).astype(np.float16))
               for _, p in params]
    x = rng.normal(0, 0.5, (seq, config.hidden_size)).astype(np.float16)
    cos, sin = llama.rope_inputs(config, np.arange(seq))
    mask = llama.causal_mask(config.num_heads, seq)

    lowered = P.graph_pipeline(custom_legalize=npu_legalize.legalize_map(),
                               fuse=False, lift_params=True)(mod)
    vm = relax.VirtualMachine(relax.build(lowered, "llvm"), tvm.cpu())
    transformed = vm["prefill_transform_params"]([weights])
    reference = vm["prefill"](*[tvm.nd.array(v) for v in (x, cos, sin, mask)],
                              *transformed).numpy()

    asm, plan = npu_link.compile_program(lowered)
    planned, _ = M.assign_addresses(lowered)
    func = planned["prefill"]
    values = dict(zip([p.name_hint for p in func.params],
                      [x, cos, sin, mask] + [t.numpy() for t in transformed]))
    got, _ = npu_link.run_program(asm, plan, func, values, reference.shape)

    a = got.astype(np.float64).ravel()
    b = reference.astype(np.float64).ravel()
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cosine > 0.9999, cosine
    assert int(np.argmax(a[-cfg["vocab_size"]:])) == \
        int(np.argmax(b[-cfg["vocab_size"]:]))
    print(f"  [PASS] whole layer vs llvm build: cosine {cosine:.6f}, "
          f"max|diff| {float(np.abs(a - b).max()):.5f}, {len(asm.words):,} words")


if __name__ == "__main__":
    test_linked_rms_norm()
    test_linked_matmul_shapes()
    test_layer_ops_match_numpy()
    test_padded_matmul_stages_its_inputs()
    test_whole_layer_matches_the_cpu_build()
    print("ALL NPU LINK (S5) TESTS PASSED")
