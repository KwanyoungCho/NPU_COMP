"""Exit-code based v09 regression gate, with per-test logs and JSON results."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TESTS = """meta_schedule compile_speed compiler_refinement schedule_memory architecture_contracts pipeline_extensions optimization_contracts completion_contracts
sram_lifetime priority_contracts compiler_contracts npu_memplan npu_target npu_link
npu_quantize native_codegen nn_hf nn_decode cmodel_v09_compute cmodel_v09 peephole
long_prompt nn_frontend nn_gemma nn_qwen3 tir_codegen_v09 tvm_pipeline isa_compute params_cache""".split()
# Checkpoint-dependent checks are opt-in, so the default gate needs no model downloads.
ASSET_TESTS = "llama_assets gemma4_assets qwen3_assets model_spec".split()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "build/validation")
    parser.add_argument("--tests", nargs="*")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--require-no-skips", action="store_true")
    args = parser.parse_args()
    names = args.tests if args.tests is not None else TESTS
    unknown = sorted(set(names) - set(TESTS + ASSET_TESTS))
    if unknown:
        parser.error(f"unknown test suites: {', '.join(unknown)}")
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    from npu_compiler.npu_provenance import describe
    (output/"provenance.json").write_text(json.dumps(describe(),indent=2)+"\n")
    jobs = [(name, [sys.executable, str(root/"d_compiler/tests"/f"test_{name}.py")]) for name in names]
    results = []
    for name, command in jobs:
        path = output/f"test_{name}.log"
        start = time.perf_counter()
        with path.open("w") as log:
            try:
                result = subprocess.run(command, cwd=root, env=dict(os.environ, OMP_NUM_THREADS="1"),
                                        stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
                log.write(f"\nGATE TIMEOUT after {args.timeout}s\n")
        content = path.read_text(errors="replace")
        skipped = "[SKIP]" in content or "skipped=" in content or " ... skipped " in content
        status = "failed" if code else ("skipped" if skipped else "passed")
        record = dict(name=name, exit_code=code, status=status, seconds=time.perf_counter()-start, log=str(path))
        results.append(record)
        (output/"validation.json").write_text(json.dumps(results,indent=2)+"\n")
        print(f"{status:7s} {name}: exit={code} ({record['seconds']:.1f}s)",flush=True)
    return int(any(r["exit_code"] or (args.require_no_skips and r["status"]=="skipped") for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
