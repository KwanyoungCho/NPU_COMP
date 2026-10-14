"""Compile full 28-layer Llama prefill without a graph time cutoff.

This runs the production graph selector, including region probes, scheduling,
joint memory planning and final codegen. The search remains bounded by its
supported graph alternatives and explicit per-kernel schedule budgets.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
from time import perf_counter

COMPILER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(COMPILER))


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


@contextmanager
def compile_memory_limit(gib):
    """Bound host address space for compilation; restore before validation.

    This Linux process limit is a last resort against an expanding instruction
    stream, not a schedule search budget or an estimate of device SRAM usage.
    """
    import math
    import resource
    if not math.isfinite(gib) or gib <= 0:
        raise ValueError('compile-memory-gib must be finite and positive')
    previous = resource.getrlimit(resource.RLIMIT_AS)
    limit = int(gib * 1024**3)
    for bound in previous:
        if bound != resource.RLIM_INFINITY:
            limit = min(limit, bound)
    resource.setrlimit(resource.RLIMIT_AS, (limit, previous[1]))
    try:
        yield limit
    finally:
        resource.setrlimit(resource.RLIMIT_AS, previous)


def compile_model(args):
    import numpy as np
    import tvm
    from npu_compiler.assets.llama import Llama32Assets, MODEL_ID, MODEL_REVISION
    from npu_compiler.nn_models import llama
    from npu_compiler.npu_build import lower_and_select
    from npu_compiler.npu_compilation import compilation
    from npu_compiler.npu_provenance import describe, sha256
    from npu_compiler.device import PROFILES
    from npu_compiler.v09_stats import analyze

    assets = Llama32Assets()
    prompt = 'Hello, NPU compiler! ' * (args.tokens // 5 + 10)
    tokens = [int(t) for t in assets.tokenizer(prompt)['input_ids'][:args.tokens]]
    if len(tokens) != args.tokens:
        raise ValueError('prompt did not provide the requested token count')
    module, params, cfg = llama.build_generate(assets.config, len(tokens), len(tokens), last_logits=True)
    profile = replace(PROFILES['v09'], graph_search_seconds=None,
        schedule_space_trials=args.trials, schedule_max_attempts=args.attempts,
        schedule_probe_word_budget=10**9)
    source = describe()
    request = dict(model=MODEL_ID, revision=MODEL_REVISION, layers=assets.config['num_hidden_layers'],
        prompt_ids=tokens, prefill_tokens=len(tokens), requested_entries=['prefill_cache'],
        prompt_description='Repeated Hello, NPU compiler! text, tokenizer output truncated to requested length',
        capacity=len(tokens), profile=asdict(profile), provenance=source,
        runner_sha256=sha256(__file__), graph_time_limit=None,
        compile_address_space_limit_gib=args.compile_memory_gib,
        effective_compile_address_space_limit_bytes=args.compile_address_space_limit_bytes)
    write_json(args.out / 'request.json', request)
    events = []
    last = 0.
    with (args.out / 'events.jsonl').open('w', buffering=1) as log:
        def progress(event):
            nonlocal last
            kind = event['event']
            if kind in ('candidate_begin', 'candidate_end', 'entry_compiled', 'entry_reused'):
                events.append(event)
                print(json.dumps(event), flush=True)
            elif event['elapsed_seconds'] - last >= 20:
                last = event['elapsed_seconds']
                print(json.dumps(event), flush=True)
            log.write(json.dumps(event) + '\n')
        start = perf_counter()
        with compilation(progress=progress) as context:
            selected, programs, selection = lower_and_select(module, profile,
                ['prefill_cache'], fusion='auto')
            metrics = context.describe()
            # Include searches performed in rejected graphs and temporary region
            # probes, not only the winning program's schedules.
            searches = []
            for records in context.recipes.values():
                for prim, _, diagnostics in records:
                    if diagnostics:
                        searches.append(dict(workload_hash=str(tvm.ir.structural_hash(prim)),
                                             diagnostics=diagnostics))
        elapsed = perf_counter() - start
    if selection['search_budget']['exhausted']:
        raise AssertionError('unlimited graph search unexpectedly exhausted its time budget')
    write_json(args.out / 'selection.json', selection)
    write_json(args.out / 'schedule_searches.json', searches)
    entries = {}
    for entry, (asm, plan) in programs.items():
        words = np.asarray(asm.words, dtype='<u4')
        words.tofile(args.out / f'{entry}.bin')
        entries[entry] = dict(statistics=analyze(asm.words, plan, detail=False),
            word_sha256=hashlib.sha256(words.tobytes()).hexdigest(),
            kernels=len(plan.kernel_span), schedules=plan.schedules,
            joint_memory=getattr(plan, 'joint_memory', None))
    (args.out / 'selected_module.py').write_text(selected.script() + '\n')
    # Local trusted checkpoint permits validation without another compile.
    with (args.out / 'compiled.pkl').open('wb') as file:
        pickle.dump((selected, programs, params, cfg), file, protocol=pickle.HIGHEST_PROTOCOL)
    summary = dict(status='compiled', **request, compile_seconds=elapsed,
        selection=selection, compilation=metrics, entries=entries,
        candidate_events=events,
        schedule_search_count=len(searches),
        schedule_stop_reasons=dict(Counter(s['diagnostics'].get('stop_reason', 'not_matmul')
                                          for s in searches)))
    write_json(args.out / 'result.json', summary)
    print(f'COMPILE COMPLETE: {elapsed:.1f}s; selected={selection["selected"]}', flush=True)


def prepare_reference(args, assets, tokens):
    import numpy as np
    path = args.out / 'hf_prefill_reference.npz'
    if path.exists():
        with np.load(path) as old:
            if {'input_ids', 'logits', 'k', 'v'} <= set(old.files):
                np.testing.assert_array_equal(old['input_ids'], tokens)
                return path
    import torch
    from transformers import AutoModelForCausalLM
    print(f'REFERENCE: Hugging Face {len(tokens)}-token prefill with K/V outputs', flush=True)
    torch.set_num_threads(args.reference_threads)
    model = AutoModelForCausalLM.from_pretrained(assets.path, local_files_only=True,
        dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation='eager').eval()
    with torch.inference_mode():
        output = model(input_ids=torch.tensor([tokens]), use_cache=True)
        logits = output.logits[0, -1].float().cpu().numpy().copy()
        cache = list(output.past_key_values)
        k = np.stack([row[0][0].cpu().numpy().transpose(0, 2, 1) for row in cache])
        v = np.stack([row[1][0].cpu().numpy() for row in cache])
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as file:
        np.savez(file, input_ids=np.asarray(tokens), logits=logits, k=k, v=v)
    temporary.replace(path)
    print(f'REFERENCE COMPLETE: token={int(np.argmax(logits))}, K={k.shape}, V={v.shape}', flush=True)
    return path


def validate_model(args):
    import numpy as np
    from npu_compiler import npu_generate as gen, npu_link
    from npu_compiler.assets.llama import Llama32Assets
    from npu_compiler.nn_models import llama
    from npu_compiler.npu_provenance import describe, sha256
    from npu_compiler.v09_stats import check_counters

    summary = json.loads((args.out / 'result.json').read_text())
    if summary['provenance']['compiler_source_sha256'] != describe()['compiler_source_sha256']:
        raise RuntimeError('compiler changed since compile; refusing unlabelled mixed-source validation')
    with (args.out / 'compiled.pkl').open('rb') as file:
        lowered, programs, params, cfg = pickle.load(file)
    assets = Llama32Assets()
    start = perf_counter()
    tokens = summary['prompt_ids']
    reference_path = prepare_reference(args, assets, tokens)
    reference = np.load(reference_path)
    np.testing.assert_array_equal(reference['input_ids'], tokens)
    expected = reference['logits'].astype(np.float64)
    print('VALIDATE: load checkpoint and transform parameters', flush=True)
    weights = [gen._nd(value) for value in llama.load_params(assets, params, cfg)]
    inputs = llama.runtime_inputs(assets, cfg, tokens)
    cos, sin = llama.rope_inputs(cfg, np.arange(len(tokens)))
    runtime = dict(input_embeds=inputs['input_embeds'], cos=cos, sin=sin, mask=inputs['mask'])
    transformed = gen._transform(lowered, 'prefill_cache', weights, args.params_cache)
    asm, plan = programs['prefill_cache']
    print('VALIDATE: C-model prefill, all 28 layers', flush=True)
    outputs, counters = npu_link.run_program(asm, plan, plan.func,
        gen._bind_inputs(plan.func, runtime, transformed))
    check_counters(asm.words, counters)
    logits, k_rows, v_rows = outputs
    np.savez(args.out / 'npu_prefill_outputs.npz', logits=logits, k=k_rows, v=v_rows)
    got = np.asarray(logits[-1], dtype=np.float64)
    cosine = float(got @ expected / (np.linalg.norm(got) * np.linalg.norm(expected)))
    token = int(np.argmax(got))
    expected_token = int(np.argmax(expected))
    cache_checks = []
    for name, values in (('k', k_rows), ('v', v_rows)):
        target = reference[name]
        if values.shape != target.shape:
            raise AssertionError(f'{name} cache shape {values.shape} != {target.shape}')
        for layer, (value, baseline) in enumerate(zip(values, target)):
            a, b = value.astype(np.float64).ravel(), baseline.astype(np.float64).ravel()
            similarity = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
            cache_checks.append(dict(kind=name, layer=layer, cosine_vs_hf=similarity,
                max_abs_diff=float(np.max(np.abs(a-b))), finite=bool(np.isfinite(a).all())))
    cache_pass = all(r['finite'] and r['cosine_vs_hf'] >= .999 for r in cache_checks)
    validation = dict(full_model=True, layers=summary['layers'], prefill_tokens=len(tokens),
        checkpoint_weights=True, predicted_token=token, expected_token=expected_token,
        cosine_vs_hf=cosine, max_abs_diff=float(np.max(np.abs(got-expected))),
        counters_match=True, counters=counters, cosine_threshold=.999,
        seconds=perf_counter()-start, reused_compiled_programs=True,
        k_shape=list(k_rows.shape), v_shape=list(v_rows.shape),
        decode_executed=False, hf_reference=str(reference_path), hf_reference_sha256=sha256(reference_path),
        outputs_sha256=sha256(args.out / 'npu_prefill_outputs.npz'),
        cache_checks=cache_checks, cache_passed=cache_pass,
        validation_runner_sha256=sha256(__file__))
    summary.update(status='passed' if token == expected_token and cosine >= .999 and cache_pass else 'validation_failed',
                   validation=validation)
    write_json(args.out / 'result.json', summary)
    print(f'VALIDATION: token={token}, expected={expected_token}, cosine={cosine:.9f}', flush=True)
    if summary['status'] != 'passed':
        raise AssertionError('full-model output differs from Hugging Face reference')


def recover_result(args):
    """Recreate a summary from a completed local build; never recompile it."""
    from npu_compiler.npu_provenance import sha256
    request = json.loads((args.out / 'request.json').read_text())
    if 'entries' in request:
        request['requested_entries'] = request.pop('entries')
    selection = json.loads((args.out / 'selection.json').read_text())
    searches = json.loads((args.out / 'schedule_searches.json').read_text())
    events = [json.loads(line) for line in (args.out / 'events.jsonl').read_text().splitlines()]
    events = [e for e in events if e['event'] in
              ('candidate_begin', 'candidate_end', 'entry_compiled', 'entry_reused')]
    with (args.out / 'compiled.pkl').open('rb') as file:
        _, programs, _, _ = pickle.load(file)
    chosen = next(r for r in selection['candidates'] if r.get('selected') and 'costs' in r)
    entries = {name: dict(statistics=chosen['costs'][name],
        word_sha256=sha256(args.out / f'{name}.bin'), kernels=len(plan.kernel_span),
        schedules=plan.schedules, joint_memory=getattr(plan, 'joint_memory', None))
        for name, (_, plan) in programs.items()}
    metrics = next(iter(programs.values()))[1].compilation
    result = dict(status='compiled', **request, compile_seconds=metrics['elapsed_seconds'],
        compile_time_definition='elapsed compilation context at final graph selection',
        selection=selection, compilation=metrics, entries=entries, candidate_events=events,
        schedule_search_count=len(searches), schedule_stop_reasons=dict(Counter(
            s['diagnostics'].get('stop_reason', 'not_matmul') for s in searches)),
        summary_recovered_from_completed_artifacts=True, summary_writer_sha256=sha256(__file__))
    write_json(args.out / 'result.json', result)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=256)
    parser.add_argument('--reference-threads', type=int, default=16)
    parser.add_argument('--trials', type=int, default=16)
    parser.add_argument('--attempts', type=int, default=128)
    parser.add_argument('--compile-memory-gib', type=float, default=24,
                        help='host address-space limit during compile only (GiB, default 24)')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--compile-only', action='store_true')
    parser.add_argument('--recover-result', action='store_true',
                        help='recreate summary from a completed local compiled.pkl, without compiling')
    parser.add_argument('--reference-only', action='store_true')
    parser.add_argument('--params-cache', default='/data2/chokwans99/npu_params_full_search')
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('NPU_V09_TMPDIR', '/data2/chokwans99/npu_tmp')
    Path(os.environ['NPU_V09_TMPDIR']).mkdir(parents=True, exist_ok=True)
    if args.reference_only:
        from npu_compiler.assets.llama import Llama32Assets
        request = json.loads((args.out / 'request.json').read_text())
        prepare_reference(args, Llama32Assets(), request['prompt_ids'])
        return
    if args.recover_result:
        recover_result(args)
    elif not args.validate_only:
        if (args.out / 'request.json').exists():
            raise RuntimeError('use a fresh output directory for each measured compile')
        with compile_memory_limit(args.compile_memory_gib) as limit:
            args.compile_address_space_limit_bytes = limit
            try:
                compile_model(args)
            except MemoryError:
                write_json(args.out / 'failure.json', dict(status='compile_memory_limit',
                    limit_gib=args.compile_memory_gib,
                    message='Compilation stopped without a validated final artifact.'))
                raise
    if not args.compile_only:
        validate_model(args)


if __name__ == '__main__':
    main()
