#!/usr/bin/env python3
"""Capture Gemma 4 E2B per-layer HF intermediates for the checkpoint ladder.

Saves, for one prompt (FP16 eager CPU): the scaled input embeddings, the
projected per-layer inputs [S, 35, 256], every decoder layer output, the final
norm output, and the last-position logits (with and without softcap).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from npu_compiler.gemma4_model import default_model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Hello, NPU compiler!")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "build" / "gemma4_layer_reference_hello.npz")
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    torch.set_num_threads(args.threads)
    path = default_model_path()
    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=True, clean_up_tokenization_spaces=False)
    encoded = tokenizer(args.prompt, return_tensors="pt")
    model = Gemma4ForConditionalGeneration.from_pretrained(
        path, local_files_only=True, dtype=torch.float16,
        low_cpu_mem_usage=True, attn_implementation="eager").eval()
    text_model = model.model.language_model

    started = time.perf_counter()
    with torch.inference_mode():
        input_ids = encoded["input_ids"]
        inputs_embeds = text_model.embed_tokens(input_ids)
        token_ple = text_model.get_per_layer_inputs(input_ids, inputs_embeds)
        per_layer_inputs = text_model.project_per_layer_inputs(
            inputs_embeds, token_ple)
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        final_normed = text_model.norm(hidden_states[-1])
        logits = outputs.logits[0, -1].float()
    elapsed = time.perf_counter() - started

    arrays = {
        "input_ids": input_ids[0].numpy().astype(np.int64),
        "inputs_embeds": inputs_embeds[0].to(torch.float16).numpy(),
        "per_layer_inputs": per_layer_inputs[0].to(torch.float16).numpy(),
        "final_normed": final_normed[0].to(torch.float16).numpy(),
        "logits": logits.numpy(),
    }
    for index, hidden in enumerate(hidden_states):
        arrays[f"hidden_{index:02d}"] = hidden[0].to(torch.float16).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)
    print(json.dumps({
        "seconds": elapsed,
        "input_ids": arrays["input_ids"].tolist(),
        "layers": len(hidden_states) - 1,
        "argmax": int(np.argmax(arrays["logits"])),
        "finite": bool(all(np.isfinite(a.astype(np.float64)).all()
                           for a in arrays.values())),
    }, indent=2))


if __name__ == "__main__":
    main()
