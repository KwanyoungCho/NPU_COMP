"""Bounded ordered-vs-MetaSchedule search on checkpoint-width Llama kernels.

Extracts kernels from the real frontend graph, uses layer-0 checkpoint weights
and seeded activation inputs. This is NOT end-to-end Llama token generation.
Each worker has a wall-clock limit, including compilation and C-model runs.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import perf_counter

COMPILER=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(COMPILER))
CASES=('decode_gate','prefill_gate_silu_multiply','prefill_down','prefill_down_512')


def write_json(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def kernel_module(prim):
    import tvm
    from tvm import relax
    buffers=[prim.buffer_map[p] for p in prim.params]
    bb=relax.BlockBuilder()
    kernel=bb.add_func(prim.with_attr('global_symbol','kernel'),'kernel')
    args=[relax.Var(f'p{i}',relax.TensorStructInfo(b.shape,b.dtype))
          for i,b in enumerate(buffers[:-1])]
    with bb.function('prefill',args):
        with bb.dataflow():
            y=bb.emit_output(relax.call_tir(kernel,relax.Tuple(args),
                out_sinfo=relax.TensorStructInfo(buffers[-1].shape,buffers[-1].dtype)))
        bb.emit_func_output(y)
    return bb.get()


def worker(args):
    import numpy as np
    import tvm
    from tvm import tir,meta_schedule as ms
    from npu_compiler import npu_link,npu_legalize,tvm_pipeline
    from npu_compiler.assets.llama import Llama32Assets,MODEL_ID,MODEL_REVISION
    from npu_compiler.nn_models.llama import build_generate
    from npu_compiler.npu_kernel import single_matmul
    from npu_compiler.npu_compilation import compilation,reachable_module
    from npu_compiler.device import PROFILES,npu_target
    from npu_compiler.v09_stats import analyze,cost,check_counters
    from npu_compiler.npu_provenance import describe
    start=perf_counter()
    folder=args.out/args.worker;folder.mkdir(parents=True,exist_ok=True)
    assets=Llama32Assets()
    config=dict(assets.config,num_hidden_layers=1)
    tokens=512 if args.worker=='prefill_down_512' else 64
    authored,_,_=build_generate(config,tokens,tokens,last_logits=True)
    prefix=tvm_pipeline.graph_prefix()(authored)
    fused=args.worker=='prefill_gate_silu_multiply'
    semantic=tvm_pipeline.graph_pipeline(custom_legalize=npu_legalize.legalize_map(),
        fuse=fused,fuse_attention=False,prepared=True)(prefix)
    entry='decode' if args.worker=='decode_gate' else 'prefill_cache'
    semantic=reachable_module(semantic,[entry])
    hidden=config['hidden_size']; intermediate=config['intermediate_size']
    extents=(1,intermediate,hidden) if entry=='decode' else ((tokens,hidden,intermediate)
             if args.worker.startswith('prefill_down') else (tokens,intermediate,hidden))
    matches=[]
    for gv,fn in semantic.functions.items():
        if not isinstance(fn,tir.PrimFunc):continue
        info=single_matmul(fn)
        if info and tuple(info.extents)==extents:
            if not fused or len(fn.params)==4:matches.append((gv,fn))
    if not matches:raise RuntimeError(f'no actual Llama kernel for {args.worker}: {extents}')
    gv,prim=matches[0]
    isolated=kernel_module(prim)
    (folder/'semantic_tir.py').write_text(isolated.script()+'\n')
    tasks=ms.relax_integration.extract_tasks(semantic,npu_target(),module_equality='structural')
    extraction=dict(source_function=gv.name_hint,entry=entry,extents=list(extents),
        model_layers=assets.config['num_hidden_layers'],exported_layers=1,
        entry_unique_tasks=len(tasks),entry_call_sites=sum(int(t.weight) for t in tasks))
    profile=replace(PROFILES['v09'],schedule_candidates=1,schedule_space_trials=args.trials,
        schedule_max_attempts=args.attempts,schedule_probe_word_budget=250000,
        joint_memory=False,graph_search_seconds=0,schedule_seed=args.seed)
    programs={};rows=[]
    for mode in ('space','meta_schedule'):
        print(f'{args.worker}: compile {mode}, at most {args.attempts} attempts, {args.trials} legal candidates',flush=True)
        before=perf_counter()
        database=folder/f'db_{mode}' if args.reuse_check else None
        if database is not None and database.exists():
            raise RuntimeError('reuse comparison needs a fresh --out directory')
        with compilation(schedule_database=database,
                         schedule_database_mode='tune' if database is not None else None) as context:
            asm,plan=npu_link.compile_program(isolated,profile=replace(profile,schedule_search=mode),
                residency=False,native_mode='use')
            metrics=context.describe()
        stats=analyze(asm.words,plan,detail=False)
        programs[mode]=(asm,plan)
        words=np.asarray(asm.words,dtype='<u4')
        words.tofile(folder/f'{mode}.bin')
        row=dict(mode=mode,compile_seconds=perf_counter()-before,statistics=stats,
            word_sha256=hashlib.sha256(words.tobytes()).hexdigest(),
            schedules=plan.schedules,compilation=metrics)
        rows.append(row)
        write_json(folder/'progress.json',dict(extraction=extraction,rows=rows))
        print(f'{args.worker}: {mode} {row["compile_seconds"]:.2f}s, cost={cost(stats)}',flush=True)
        if database is not None:
            before=perf_counter()
            fast=replace(profile,schedule_search='meta_schedule' if mode=='space' else 'space',
                schedule_seed=args.seed+1,schedule_space_trials=1,schedule_max_attempts=1,
                schedule_probe_word_budget=1)
            with compilation(schedule_database=database) as context:
                replay,replay_plan=npu_link.compile_program(isolated,profile=fast,
                    residency=False,native_mode='use')
                metrics=context.describe()
            if replay.words!=asm.words or metrics['meta_schedule']['hits']<1:
                raise AssertionError('tuned recipe did not survive a changed search policy')
            if any(d['examined_candidates'] for d in replay_plan.schedules.values()):
                raise AssertionError('fast replay searched new candidates')
            row['reuse_check']=dict(seconds=perf_counter()-before,instructions_identical=True,
                replay_profile=asdict(fast),compilation=metrics)
            print(f'{args.worker}: {mode} DB reuse {row["reuse_check"]["seconds"]:.2f}s, no new candidates',flush=True)

    # The only model tensors loaded are the necessary first-layer MLP weights.
    # Synthetic activations are normalized with the actual post-attention norm.
    m=extents[0];rng=np.random.default_rng(929)
    x=rng.normal(0,1,(m,3072)).astype('float32')
    x=(x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+assets.config['rms_norm_eps'])
       *assets.norm(0,post_attention=True).astype('float32')).astype('float16')
    gate=assets.linear(0,'mlp.gate_proj',(3072,8192))
    g=(x.astype('float32')@gate.astype('float32')).astype('float16')
    inputs=[x,gate];reference=g
    if args.worker!='decode_gate':
        up=assets.linear(0,'mlp.up_proj',(3072,8192))
        u=(x.astype('float32')@up.astype('float32')).astype('float16')
        s=(g.astype('float32')/(1+np.exp(-g.astype('float32')))).astype('float16')
        h=(s*u).astype('float16')
        if fused:
            inputs=[x,gate,u];reference=h
        else:
            down=assets.linear(0,'mlp.down_proj',(8192,3072))
            inputs=[h,down]
            reference=(h.astype('float32')@down.astype('float32')).astype('float16')
    expected_shapes=[tuple(int(d) for d in prim.buffer_map[p].shape) for p in prim.params[:-1]]
    if [v.shape for v in inputs]!=expected_shapes:
        raise RuntimeError(f'input ABI differs from expected Llama kernel: {expected_shapes}')
    values={f'p{i}':value for i,value in enumerate(inputs)}
    outputs={};executed={};checks=[]
    for row in rows:
        mode=row['mode'];asm,plan=programs[mode]
        fingerprint=row['word_sha256']
        if fingerprint in executed:
            outputs[mode]=outputs[executed[fingerprint]]
            checks.append(dict(mode=mode,same_program_as=executed[fingerprint],executed=False))
            continue
        print(f'{args.worker}: C-model {mode} with checkpoint weights',flush=True)
        before=perf_counter()
        got,counters=npu_link.run_program(asm,plan,plan.func,values)
        check_counters(asm.words,counters)
        error=np.abs(got.astype('float32')-reference.astype('float32'))
        np.testing.assert_allclose(got,reference,rtol=.02,atol=.005)
        outputs[mode]=got;executed[fingerprint]=mode
        checks.append(dict(mode=mode,executed=True,seconds=perf_counter()-before,
            max_abs=float(error.max()),rtol=.02,atol=.005,counters_match=True))
    bit_exact=bool(np.array_equal(outputs['space'].view('uint16'),outputs['meta_schedule'].view('uint16')))
    if not bit_exact:raise AssertionError('ordered and MetaSchedule outputs differ in FP16 bits')
    a,b=[cost(row['statistics']) for row in rows]
    result=dict(status='passed',case=args.worker,model=MODEL_ID,revision=MODEL_REVISION,
        checkpoint_path=str(assets.path),checkpoint_weights=True,activation_input='seeded synthetic, actual layer-0 norm',
        end_to_end_generation=False,full_layer_executed=False,extraction=extraction,
        profile=asdict(profile),scope='isolated kernels; DRAM input/output ABI; residency disabled; fixed fusion boundaries',
        meta_strategy='TVM ReplayFunc; uniform configuration sampling; not evolutionary/learned',
        fairness='same normalized and capacity-filtered candidate pool, shared legacy incumbent, equal adaptive budgets; independent build contexts',
        database_reuse_checked=args.reuse_check,
        rows=rows,validation=checks,outputs_bit_exact=bit_exact,
        meta_result='better' if b<a else 'equal' if b==a else 'worse',
        total_seconds=perf_counter()-start,provenance=describe())
    write_json(folder/'result.json',result)
    print(f'{args.worker}: PASS, MetaSchedule {result["meta_result"]}, {result["total_seconds"]:.1f}s',flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--out',type=Path,default=COMPILER/'build/llama_meta_schedule')
    parser.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES[:3]))
    parser.add_argument('--worker',choices=CASES)
    parser.add_argument('--attempts',type=int,default=16)
    parser.add_argument('--trials',type=int,default=4)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--timeout',type=float,default=90)
    parser.add_argument('--reuse-check',action='store_true',
                        help='save tuning results and verify replay with different search settings')
    args=parser.parse_args();args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    if args.worker:return worker(args)
    results=[]
    for case in args.cases:
        print(f'START {case}: two search methods, timeout {args.timeout}s including execution',flush=True)
        command=[sys.executable,str(Path(__file__).resolve()),'--worker',case,'--out',str(args.out),
            '--attempts',str(args.attempts),'--trials',str(args.trials),'--seed',str(args.seed)]
        if args.reuse_check: command.append('--reuse-check')
        with (args.out/f'{case}.log').open('w') as log:
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait();code=124
            except BaseException:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGKILL)
                proc.wait()
                raise
        path=args.out/case/'result.json'
        if code==0 and path.exists():
            result=json.loads(path.read_text());results.append(result)
            print(f'PASS {case}: {result["meta_result"]}, {result["total_seconds"]:.1f}s',flush=True)
        else:
            results.append(dict(case=case,status='timeout' if code==124 else 'failed',exit_code=code))
            print((args.out/f'{case}.log').read_text()[-6000:],flush=True)
        complete=len(results)==len(args.cases)
        write_json(args.out/'comparison.json',dict(measurement='single cold-context samples; no NPU latency measurement',
            requested_cases=args.cases,complete=complete,
            cases=results,passed=complete and all(r['status']=='passed' for r in results)))
    return int(any(r['status']!='passed' for r in results))


if __name__=='__main__':sys.exit(main())
