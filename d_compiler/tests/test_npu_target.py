"""S6 gate: the NPU is a target, its pipeline is registered, and build() takes
a model from an nn.Module definition to a runnable program in one call."""
import sys
from pathlib import Path

import numpy as np
import tvm
from tvm import relax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))

from npu_compiler import npu_target as T
from npu_compiler.nn_models import llama

TINY = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            vocab_size=32, rms_norm_eps=1e-5, rope_theta=500000.0)
SEQ = 4


def test_target_and_profile():
    target = T.npu_target()
    assert T.NPU_KEY in [str(key) for key in target.keys]
    profile = T.profile_of(target)
    assert (profile.sram_bytes, profile.tile) == (8 * 1024 * 1024, 64)
    assert profile.sram_nibbles == profile.sram_bytes * 2
    try:
        T.profile_of("llvm")
    except ValueError:
        pass
    else:
        raise AssertionError("a non-NPU target must be rejected")
    try:
        T.npu_target("v99")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown model must be rejected")
    print(f"  [PASS] target {target}, profile {profile.model} "
          f"({profile.sram_bytes >> 20} MiB SRAM, tile {profile.tile})")


def test_pipeline_is_registered():
    assert "npu" in relax.pipeline.PIPELINE_MAP
    assert relax.get_pipeline("npu") is not None
    print("  [PASS] relax.get_pipeline(\"npu\") resolves")


def test_build_produces_a_runnable_program():
    mod, params, cfg = llama.build_prefill(TINY, seq=SEQ)
    rng = np.random.default_rng(5)
    weights = [tvm.nd.array(rng.normal(0, 0.15, p.shape).astype(np.float16))
               for _, p in params]
    x = rng.normal(0, 0.5, (SEQ, cfg.hidden_size)).astype(np.float16)
    cos, sin = llama.rope_inputs(cfg, np.arange(SEQ))
    mask = llama.causal_mask(cfg.num_heads, SEQ)

    executable = T.build(mod, T.npu_target())
    assert executable.kernel_count > 0
    assert len(executable.words) > 0

    vm = relax.VirtualMachine(relax.build(executable.module, "llvm"), tvm.cpu())
    transformed = vm["prefill_transform_params"]([weights])
    reference = vm["prefill"](*[tvm.nd.array(v) for v in (x, cos, sin, mask)],
                              *transformed).numpy()
    values = dict(zip(executable.param_names,
                      [x, cos, sin, mask] + [t.numpy() for t in transformed]))
    got, _ = executable.run(values, reference.shape)

    a, b = got.astype(np.float64).ravel(), reference.astype(np.float64).ravel()
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cosine > 0.9999, cosine

    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = executable.save(Path(directory) / "prefill.bin")
        assert Path(path).stat().st_size == len(executable.words) * 4
    print(f"  [PASS] build -> run: cosine {cosine:.6f}, {executable!r}")


if __name__ == "__main__":
    test_target_and_profile()
    test_pipeline_is_registered()
    test_build_produces_a_runnable_program()
    print("ALL NPU TARGET (S6) TESTS PASSED")
