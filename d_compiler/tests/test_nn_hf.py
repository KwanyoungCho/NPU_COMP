"""The HF-model frontend: trace HF's own forward, compare against HF's own
output -- on llvm and on the C-model.

The reference here is the strongest available: the very ``transformers``
model whose trace we compile, run in torch.
"""
import sys
from pathlib import Path

import numpy as np
import tvm
from tvm import relax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))

from npu_compiler import npu_legalize, npu_link, npu_memplan as M
from npu_compiler import tvm_pipeline as P
from npu_compiler.nn_models import hf


def _tiny():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        vocab_size=32, rms_norm_eps=1e-5, rope_theta=500000.0,
        attn_implementation="eager", use_cache=False,
        tie_word_embeddings=False)
    torch.manual_seed(0)
    return LlamaForCausalLM(config).half(), config


def _reference(model, embeds, mask):
    import torch

    seq = embeds.shape[1]
    with torch.no_grad():
        return model(inputs_embeds=torch.from_numpy(embeds),
                     attention_mask=torch.from_numpy(mask),
                     position_ids=torch.arange(seq).unsqueeze(0),
                     cache_position=torch.arange(seq),
                     use_cache=False).logits.float().numpy()


def _cosine(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_traced_hf_llama_matches_torch_on_llvm_and_the_cmodel():
    model, config = _tiny()
    seq = 5
    rng = np.random.default_rng(0)
    embeds = rng.normal(0, 0.5, (1, seq, config.hidden_size)).astype(np.float16)
    mask = hf.causal_mask4d(seq)
    expected = _reference(model, embeds, mask)

    mod = hf.import_prefill(model, seq)
    assert str(mod["main"]).count("rms_norm") == 2 * config.num_hidden_layers + 1
    lowered = P.graph_pipeline(custom_legalize=npu_legalize.legalize_map(),
                               fuse=False, lift_params=False)(mod)

    vm = relax.VirtualMachine(relax.build(lowered, "llvm"), tvm.cpu())
    on_llvm = vm["main"](tvm.nd.array(embeds), tvm.nd.array(mask)).numpy()
    llvm_cos = _cosine(on_llvm, expected)
    assert llvm_cos > 0.9999, llvm_cos

    asm, plan = npu_link.compile_program(lowered, "main")
    planned, _ = M.assign_addresses(lowered, "main")
    func = planned["main"]
    got, _ = npu_link.run_program(
        asm, plan, func,
        dict(zip([p.name_hint for p in func.params], [embeds, mask])),
        (1, seq, config.vocab_size))
    npu_cos = _cosine(got, expected)
    assert npu_cos > 0.999, npu_cos
    assert int(np.argmax(got[0, -1])) == int(np.argmax(expected[0, -1]))
    print(f"  [PASS] traced HF llama vs torch: llvm {llvm_cos:.6f}, "
          f"npu {npu_cos:.6f}, argmax same ({len(asm.words):,} words)")


if __name__ == "__main__":
    test_traced_hf_llama_matches_torch_on_llvm_and_the_cmodel()
    print("ALL HF FRONTEND TESTS PASSED")
