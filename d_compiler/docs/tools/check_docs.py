"""Check the manual's links, preserved evidence and example consistency.

No compiler build, C-model run, checkpoint download or network access.
Run reproduce.py separately when execution evidence needs refreshing.
"""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
from urllib.parse import unquote
import xml.etree.ElementTree as ET

DOCS = Path(__file__).resolve().parents[1]
REPO = DOCS.parents[1]
RESULT = DOCS/'evidence/checks.json'
checks = []


def record(name, passed, details):
    checks.append(dict(name=name, passed=bool(passed), details=details))


def read(path):
    return json.loads((DOCS/path).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    documents=sorted(DOCS.rglob('*.md'))+[REPO/'d_compiler/README.md',
        REPO/'d_compiler/npu_codegen/README.md', REPO/'d_compiler/ISA_V09.md',
        REPO/'report/README.md']
    missing=[];count=0
    for path in documents:
        if not path.exists():
            missing.append(str(path));continue
        # Code spans as well as fenced examples contain tensor indexing and
        # formulae that resemble Markdown links, e.g. mask[S](-65504).
        text=re.sub(r'```.*?```','',path.read_text(),flags=re.S)
        text=re.sub(r'`[^`\n]*`','',text)
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
            target=target.strip().split(' "',1)[0].strip('<>')
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',target):continue
            target=unquote(target.split('#',1)[0])
            if not target:continue
            count+=1
            if not (path.parent/target).exists():missing.append(f'{path.relative_to(REPO)} -> {target}')
    record('local_markdown_targets',not missing,dict(documents=len(documents),links=count,missing=missing,
        scope='file/directory targets; external URLs and heading anchors are not checked'))

    failures=[];totals={}
    for extension,parser in [('json',json.loads),('py',ast.parse),('svg',ET.fromstring)]:
        paths=sorted(DOCS.rglob('*.'+extension));totals[extension]=len(paths)
        for p in paths:
            try:parser(p.read_text())
            except Exception as error:failures.append(f'{p.relative_to(DOCS)}: {error}')
    record('artifact_syntax',not failures,dict(counts=totals,failures=failures,
        scope='Python AST only; TVM IR dumps are not executed as scripts'))

    retained=read('evidence/retained_sources.json')['files'];bad=[]
    for f in retained:
        p=DOCS/f['retained'];expected=f.get('retained_sha256',f['sha256'])
        if not p.is_file() or sha(p)!=expected:bad.append(f['retained'])
    record('retained_artifact_hashes',not bad,dict(files=len(retained),mismatches=bad))

    pipeline=read('examples/pipeline/stage_summary.json')
    memory=read('examples/memory/measurements.json')
    completion=read('evidence/current/completion.json')
    provenance=pipeline['provenance']
    source_paths=sorted([*REPO.joinpath('d_compiler/npu_compiler').rglob('*.py'),
        *REPO.joinpath('d_compiler/npu_codegen').glob('*.cc'),
        *REPO.joinpath('d_compiler/npu_codegen').glob('*.h')])
    digest=hashlib.sha256()
    for p in source_paths:digest.update((str(p.relative_to(REPO))+'\0'+sha(p)+'\n').encode())
    actual=digest.hexdigest()
    record('historical_example_source_consistency',provenance['compiler_source_sha256']==
        memory['provenance']['compiler_source_sha256']==completion['compiler_source_sha256'],
        dict(current=actual,example=provenance['compiler_source_sha256'],
             historical_final_validation=completion['compiler_source_sha256'],
             scope='historical examples retain their original source identity; not current-source execution'))

    latest=DOCS/'evidence/current/llama_prefill256_full_search.json'
    if latest.exists():
        full=json.loads(latest.read_text())
        validation=full.get('validation',{})
        selection=full['selection']
        events=full['candidate_events']
        begins=[e for e in events if e['event']=='candidate_begin']
        ends=[e for e in events if e['event']=='candidate_end']
        labels=lambda rows: [e['label'] for e in rows]
        record('current_full_prefill_evidence',full['status']=='passed' and
            full['provenance']['compiler_source_sha256']==actual and
            full['layers']==28 and full['prefill_tokens']==256 and
            full['requested_entries']==['prefill_cache'] and
            selection['search_budget']['additional_seconds'] is None and
            not selection['search_budget']['exhausted'] and labels(begins)==labels(ends) and
            validation.get('predicted_token')==validation.get('expected_token') and
            validation.get('cosine_vs_hf',0)>=.999 and validation.get('counters_match') and
            validation.get('cache_passed') and len(validation.get('cache_checks',[]))==56 and
            not validation.get('decode_executed',True),
            dict(source=full['provenance']['compiler_source_sha256'],
                 current=actual,candidates_started=len(begins),candidates_finished=len(ends),
                 scope='recorded whole-model prefill execution; checker does not rerun model'))

    a,b=[pipeline['records'][k]['statistics'] for k in ('off','auto')]
    record('pipeline_evidence',a['words']==2812 and b['words']==1577 and
        a['dma_bytes']==b['dma_bytes']==41100 and pipeline['validation']['fused_unfused_bit_exact']
        and pipeline['validation']['cmodel_counters_match'] and pipeline['validation']['llvm_max_abs']==0
        and bool(pipeline['actual_recipe_exports']),dict(validation=pipeline['validation'],
            actual_recipe_exports=pipeline['actual_recipe_exports']))
    expected=[(32768,False,88,81932),(32768,True,88,81932),
        (45056,False,87,81932),(45056,True,77,49164),(65536,True,77,49164)]
    observed=[(r['sram_capacity_bytes'],r['joint_memory'],r['statistics']['words'],r['statistics']['dma_bytes'])
        for r in memory['records']]
    record('memory_evidence',observed==expected and all(r['bit_exact_to_baseline'] and r['counters_match']
        for r in memory['records']),dict(rows=observed,scope='recorded results of reproduce.py; not rerun by this checker'))
    binary=DOCS/'examples/two_matmuls_recorded/program_memory.bin'
    words=list(struct.unpack('<'+'I'*(binary.stat().st_size//4),binary.read_bytes()))
    fresh=next(r for r in memory['records'] if r['sram_capacity_bytes']==45056 and r['joint_memory'])
    record('reproduced_77_word_program',len(words)==77 and sha(binary)==fresh['word_sha256'] and
        words[8:12]==[0x000c18a0,0,0x800,0x10800] and words[76]==0xff,
        dict(words=len(words),sha256=sha(binary),fresh_sha256=fresh['word_sha256']))
    bundle=read('examples/pipeline/bundle/program.json')
    record('pipeline_bundle_binary',sha(DOCS/'examples/pipeline/bundle/program_memory.bin')==
        bundle['program_sha256']==pipeline['records']['auto']['word_sha256'],bundle['program_sha256'])

    llama=read('evidence/current/llama_validation.json')
    suites=read('evidence/current/regression_validation.json')
    logs=[]
    for suite in suites:
        p=DOCS/('evidence/current/quantization.log' if suite['name']=='npu_quantize' else
                f"evidence/current/gate_logs/test_{suite['name']}.log")
        if not p.is_file():logs.append(str(p))
    record('historical_full_model_and_gates',llama['status']=='passed' and
        llama['generated_ids']==completion['generated_ids']==[358,2846,4560] and
        len(suites)==completion['regression_suites']==17 and
        all(s['status']=='passed' and s['exit_code']==0 for s in suites) and not logs and
        all(s['token']==s['expected_token'] and s['cosine_vs_hf']>=.999 for s in llama['steps']),
        dict(scope='preserved evidence only; no new whole-model/gate execution',missing_logs=logs))
    ir=read('examples/llama_ir/summary.json')
    record('llama_ir_scope',ir['config']['num_hidden_layers']==1 and ir['config']['hidden_size']==3072
        and ir['config']['rope_scaling']['factor']==32 and not ir['checkpoint_loaded']
        and not ir['npu_scheduled'] and not ir['npu_executed'],ir['scope'])
    schedule=read('examples/schedule_api/checks.json')
    record('schedule_api_execution_record','exact match' in schedule['compute_at_llvm'] and
        'exact match' in schedule['reverse_compute_at_llvm'],schedule)
    cleanup=read('evidence/cleanup.json')
    remaining=[f['path'] for f in cleanup['removed'] if (REPO/f['path']).exists()]
    record('obsolete_material_removed',cleanup['status']=='completed' and not remaining,
        dict(count=len(cleanup['removed']),bytes=cleanup['removed_bytes'],remaining=remaining))


if __name__=='__main__':
    try:run()
    except Exception as error:record('checker_exception',False,f'{type(error).__name__}: {error}')
    ok=all(c['passed'] for c in checks)
    report=dict(status='passed' if ok else 'failed',checked_at=datetime.now(timezone.utc).isoformat(),
        scope='documentation integrity and recorded-result consistency; no compilation or model execution',
        checks=checks)
    RESULT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for check in checks:print(('PASS' if check['passed'] else 'FAIL')+' '+check['name'])
    print(RESULT)
    sys.exit(0 if ok else 1)
