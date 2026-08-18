#!/usr/bin/env python3
"""Run/resume official Llama 3.2 3B prefill on the 0818 vendor C-model."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from npu_compiler.v3_llama import Llama32PrefillCompiler
from npu_compiler.v3_runtime import ParallelVendorSession

BUILD = Path(__file__).resolve().parent / "build"


def metrics(actual, expected):
    actual = actual.astype(np.float32)
    expected = expected.astype(np.float32)
    difference = actual - expected
    flat_a = actual.reshape(-1).astype(np.float64)
    flat_b = expected.reshape(-1).astype(np.float64)
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "cosine": float(np.dot(flat_a, flat_b) /
                        (np.linalg.norm(flat_a) * np.linalg.norm(flat_b))),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Hello, NPU compiler!")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path,
                        default=BUILD / "v3_prefill_hello")
    parser.add_argument("--reference", type=Path,
                        default=BUILD / "v3_reference_hello.npz")
    return parser.parse_args()


def main():
    args = parse_args()
    # Tokenizer comes from the same validated local checkpoint as the weights.
    from npu_compiler.v3_model import Llama32Assets
    compiler_assets = Llama32Assets()
    input_ids = np.asarray(
        compiler_assets.tokenizer(args.prompt, return_tensors="np")["input_ids"][0],
        dtype=np.int64)
    compiler = Llama32PrefillCompiler(len(input_ids), assets=compiler_assets)
    args.checkpoint.mkdir(parents=True, exist_ok=True)

    start_layer, hidden = 0, None
    state_path = args.checkpoint / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["input_ids"] != input_ids.tolist():
            raise RuntimeError("checkpoint input_ids do not match this prompt")
        start_layer = int(state["layers_completed"])
        if start_layer:
            hidden = np.load(
                args.checkpoint / f"hidden_after_layer_{start_layer:02d}.npy")
        print(f"RESUME layer={start_layer}", flush=True)

    reference = None
    if args.reference.exists():
        candidate = np.load(args.reference)
        if np.array_equal(candidate["input_ids"], input_ids):
            reference = candidate

    progress_path = args.checkpoint / "progress.jsonl"

    def progress(layer, layer_hidden, delta):
        record = {"layer": layer + 1, **delta}
        # HF exposes pre-layer states 0..27 and the final normalized state at
        # index 28; it does not expose raw output of layer 27.
        if reference is not None and layer + 1 < compiler.assets.config["num_hidden_layers"]:
            record["reference"] = metrics(
                layer_hidden, reference["hidden_states"][layer + 1])
        with progress_path.open("a") as file:
            file.write(json.dumps(record) + "\n")
        print("LAYER " + json.dumps(record), flush=True)

    with ParallelVendorSession(args.workers) as vendor:
        result = compiler.run(
            input_ids, vendor=vendor, start_layer=start_layer, hidden=hidden,
            checkpoint_dir=args.checkpoint, progress=progress)

    final_metrics = None
    normalized_metrics = None
    expected_token = None
    if reference is not None:
        final_metrics = metrics(result.logits, reference["last_logits"])
        normalized_metrics = metrics(
            result.normalized, reference["hidden_states"][-1])
        expected_token = int(reference["next_token_id"])
    np.savez(
        args.checkpoint / "final.npz", input_ids=result.input_ids,
        hidden=result.hidden, normalized=result.normalized, logits=result.logits,
        next_token_id=np.asarray(result.next_token_id, dtype=np.int64))
    decoded = compiler.assets.tokenizer.decode([result.next_token_id])
    summary = {
        "next_token_id": result.next_token_id,
        "decoded": decoded,
        "expected_token_id": expected_token,
        "normalized_reference": normalized_metrics,
        "logits_reference": final_metrics,
        "stats": result.stats,
    }
    (args.checkpoint / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("RESULT " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
