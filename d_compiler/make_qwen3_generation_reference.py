#!/usr/bin/env python3
"""Create an independent Hugging Face greedy reference for Qwen3-4B."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from npu_compiler.qwen3_model import default_model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Hello, NPU compiler!")
    parser.add_argument("--tokens", type=int, default=3)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "build" / "qwen3_reference_generate_hello_3.npz")
    parser.add_argument("--layer-reference", type=Path,
                        default=Path(__file__).resolve().parent / "build" / "qwen3_layer_reference_hello.npz")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    path = default_model_path()
    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=True, clean_up_tokenization_spaces=False)
    encoded = tokenizer(args.prompt, return_tensors="pt")
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, dtype=torch.float16,
        low_cpu_mem_usage=True, attn_implementation="eager").eval()
    started = time.perf_counter()
    with torch.inference_mode():
        result = model.generate(
            **encoded, max_new_tokens=args.tokens, do_sample=False, use_cache=True,
            return_dict_in_generate=True, output_scores=True,
            pad_token_id=tokenizer.eos_token_id)
        forward = model(input_ids=encoded["input_ids"], output_hidden_states=True)
    elapsed = time.perf_counter() - started
    input_ids = encoded["input_ids"][0].cpu().numpy().astype(np.int64)
    generated = result.sequences[0, input_ids.size:].cpu().numpy().astype(np.int64)
    logits = np.stack([score[0].float().cpu().numpy() for score in result.scores])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, input_ids=input_ids, generated_ids=generated, logits=logits)
    layer_arrays = {"input_ids": input_ids}
    for index, hidden in enumerate(forward.hidden_states):
        layer_arrays[f"hidden_{index:02d}"] = hidden[0].to(torch.float16).numpy()
    np.savez(args.layer_reference, **layer_arrays)
    print(json.dumps({
        "seconds": elapsed,
        "input_ids": input_ids.tolist(),
        "generated_ids": generated.tolist(),
        "decoded_tokens": [tokenizer.decode([int(token)]) for token in generated],
        "decoded_text": tokenizer.decode(generated),
        "layers": len(forward.hidden_states) - 1,
        "logits_finite": bool(np.isfinite(logits).all()),
    }, indent=2))


if __name__ == "__main__":
    main()
