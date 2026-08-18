#!/usr/bin/env python3
"""Create an independent Hugging Face greedy-generation reference artifact."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from npu_compiler.v3_model import default_model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Hello, NPU compiler!")
    parser.add_argument("--tokens", type=int, default=3)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "build" / "v3_reference_generate_hello_3.npz")
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
    elapsed = time.perf_counter() - started
    input_ids = encoded["input_ids"][0].cpu().numpy().astype(np.int64)
    generated = result.sequences[0, input_ids.size:].cpu().numpy().astype(np.int64)
    logits = np.stack([score[0].float().cpu().numpy() for score in result.scores])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, input_ids=input_ids, generated_ids=generated, logits=logits)
    print(json.dumps({
        "seconds": elapsed,
        "input_ids": input_ids.tolist(),
        "generated_ids": generated.tolist(),
        "decoded_tokens": [tokenizer.decode([int(token)]) for token in generated],
        "decoded_text": tokenizer.decode(generated),
    }, indent=2))


if __name__ == "__main__":
    main()
