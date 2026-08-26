#!/usr/bin/env python3
"""Run/resume official Gemma 4 E2B prefill and greedy decode on source ver.08."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from npu_compiler.gemma4_model import Gemma4Assets
from npu_compiler.gemma4_source import Gemma4SourceCompiler

BUILD = Path(__file__).resolve().parent / "build"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Hello, NPU compiler!")
    parser.add_argument("--tokens", type=int, default=3)
    parser.add_argument("--output", type=Path, default=BUILD / "gemma4_source_generate_hello")
    parser.add_argument("--reference", type=Path,
                        default=BUILD / "gemma4_reference_generate_hello_3.npz")
    parser.add_argument("--no-cache-weights", action="store_true")
    parser.add_argument("--backend", default="source-0818",
                        choices=["source-0818", "v09"])
    parser.add_argument("--quant", default=None, choices=["w8a16", "w8a8"])
    return parser.parse_args()


def save_state(path, input_ids, generated, hidden, keys, values, logits):
    arrays = {
        "input_ids": np.asarray(input_ids, dtype=np.int64),
        "generated_ids": np.asarray(generated, dtype=np.int64),
        "hidden": np.asarray(hidden, dtype=np.float16),
        "logits": np.asarray(logits, dtype=np.float16),
    }
    for index, (key, value) in enumerate(zip(keys, values)):
        arrays[f"key_{index:02d}"] = key[0]
        arrays[f"value_{index:02d}"] = value[0]
    np.savez(path, **arrays)


def load_state(path, num_slots):
    state = np.load(path)
    keys = [[np.ascontiguousarray(state[f"key_{index:02d}"], dtype=np.float16)]
            for index in range(num_slots)]
    values = [[np.ascontiguousarray(state[f"value_{index:02d}"], dtype=np.float16)]
              for index in range(num_slots)]
    return state, keys, values


def softcap(logits, cap=30.0):
    return cap * np.tanh(np.asarray(logits, dtype=np.float64) / cap)


def metrics(actual, expected):
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    expected = np.asarray(expected, dtype=np.float64).reshape(-1)
    difference = actual - expected
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "cosine": float(np.dot(actual, expected) /
                        (np.linalg.norm(actual) * np.linalg.norm(expected))),
    }


def main():
    args = parse_args()
    if args.tokens < 1:
        raise ValueError("--tokens must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    assets = Gemma4Assets()
    input_ids = np.asarray(
        assets.tokenizer(args.prompt, return_tensors="np")["input_ids"][0],
        dtype=np.int64,
    )
    compiler = Gemma4SourceCompiler(
        input_ids.size, assets=assets, cache_weights=not args.no_cache_weights,
        backend=args.backend, quantize=args.quant)
    num_slots = len(compiler.cache_plan.slots)
    state_path = args.output / "state.npz"
    progress_path = args.output / "progress.jsonl"

    def progress(stage, layer, hidden, seconds):
        record = {
            "stage": stage,
            "layer": layer + 1,
            "seconds": seconds,
            "finite": bool(np.isfinite(hidden.astype(np.float32)).all()),
            "max_abs": float(np.max(np.abs(hidden.astype(np.float32)))),
        }
        with progress_path.open("a") as file:
            file.write(json.dumps(record) + "\n")
        print("LAYER " + json.dumps(record), flush=True)

    if state_path.exists():
        state, keys, values = load_state(state_path, num_slots)
        if not np.array_equal(state["input_ids"], input_ids):
            raise RuntimeError("saved input_ids do not match this prompt")
        generated = [int(value) for value in state["generated_ids"]]
        hidden = np.asarray(state["hidden"], dtype=np.float16)
        logits = np.asarray(state["logits"], dtype=np.float16)
        print(f"RESUME generated={generated} cache={keys[0][0].shape[1]}", flush=True)
    else:
        prefill = compiler.prefill(input_ids, progress=progress)
        generated = [prefill.next_token_id]
        hidden = prefill.hidden[-1:]
        keys, values, logits = prefill.keys, prefill.values, prefill.logits
        np.save(args.output / "logits_step_00.npy", logits)
        save_state(state_path, input_ids, generated, hidden, keys, values, logits)
        print(f"PREFILL token={generated[-1]}", flush=True)

    while len(generated) < args.tokens:
        position = input_ids.size + len(generated) - 1
        hidden, logits, next_id = compiler.decode_token(
            generated[-1], position, keys, values, progress=progress)
        generated.append(next_id)
        np.save(args.output / f"logits_step_{len(generated) - 1:02d}.npy", logits)
        save_state(state_path, input_ids, generated, hidden, keys, values, logits)
        print(f"DECODE position={position} token={next_id}", flush=True)

    expected = None
    logits_reference = None
    if args.reference and args.reference.exists():
        reference = np.load(args.reference)
        if np.array_equal(reference["input_ids"], input_ids):
            expected = [int(value) for value in reference["generated_ids"][:args.tokens]]
            if "logits" in reference and len(reference["logits"]) >= args.tokens:
                # HF logits carry the final softcap; ours elide it for greedy,
                # so apply it on the comparison side only.
                logits_reference = metrics(
                    softcap(logits), reference["logits"][args.tokens - 1])
    summary = {
        "prompt": args.prompt,
        "input_ids": input_ids.tolist(),
        "generated_ids": generated,
        "decoded_tokens": [assets.tokenizer.decode([token]) for token in generated],
        "decoded_text": assets.tokenizer.decode(generated),
        "expected_ids": expected,
        "match": expected is None or generated == expected,
        "logits_reference": logits_reference,
        "cache_length": int(keys[0][0].shape[1]),
        "stats": compiler.stats(),
    }
    (args.output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("RESULT " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
