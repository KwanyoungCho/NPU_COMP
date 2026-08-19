#!/usr/bin/env python3
"""Precompute the Gemma 4 E2B per-layer-input (PLE) table for all vocab tokens.

The table stores exactly what the NPU PLE graph (gemma4_graph.build_gemma4_ple_module)
would compute, using the same FP16 G-buffer step boundaries: FP16 storage after
every op, float32 arithmetic inside each op.  Sampled rows are proven bit-exact
against the actual source C-model program by tests/test_gemma4_ple.py.

Output: [vocab_size, num_layers*ple_dim] FP16 row-major, written next to the
real checkpoint file (ple_table_f16.bin), ~4.7GB.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from npu_compiler.gemma4_model import Gemma4Assets
from npu_compiler.gemma4_ple import compute_rows, f16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--limit", type=int, help="only this many tokens (debug)")
    args = parser.parse_args()

    assets = Gemma4Assets()
    spec = assets.spec
    vocab = spec.vocab_size if args.limit is None else int(args.limit)
    width = spec.num_layers * spec.layers[0].ple_dim
    output = assets.ple_table_path()
    projection = assets.per_layer_model_projection()
    norm_weight = assets.per_layer_projection_norm()
    embed_scale = np.float16(np.sqrt(float(spec.hidden_size)))

    row_bytes = width * 2
    done_rows = 0
    if output.exists():
        done_rows = output.stat().st_size // row_bytes
        with open(output, "r+b") as file:
            file.truncate(done_rows * row_bytes)
        print(f"RESUME from row {done_rows}", flush=True)
    started = time.perf_counter()
    with open(output, "ab") as file:
        for begin in range(done_rows, vocab, args.chunk):
            end = min(vocab, begin + args.chunk)
            span = (slice(begin, end), slice(None))
            scaled_embed = f16(assets._slice("embed_tokens.weight", span) * embed_scale)
            tok_rows = assets._slice("embed_tokens_per_layer.weight", span)
            rows = compute_rows(spec, scaled_embed, tok_rows, projection, norm_weight)
            if not np.isfinite(rows.astype(np.float32)).all():
                raise RuntimeError(f"non-finite PLE rows in [{begin},{end})")
            file.write(rows.tobytes())
            print(f"  rows {end}/{vocab}  {time.perf_counter() - started:.1f}s",
                  flush=True)
    size = output.stat().st_size
    print(f"DONE {output} bytes={size} expected={vocab * width * 2}")


if __name__ == "__main__":
    main()
