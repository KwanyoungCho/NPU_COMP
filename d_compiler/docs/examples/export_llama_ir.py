"""Export one actual-width Llama layer's graph/TIR; no weights or NPU execution."""
import argparse
import json
from pathlib import Path
import sys

COMPILER=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(COMPILER))
from npu_compiler.nn_models.llama import build_generate
from npu_compiler import tvm_pipeline,npu_legalize
from npu_compiler.npu_provenance import describe


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=COMPILER/'build/docs_llama_ir')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    config=dict(hidden_size=3072,intermediate_size=8192,num_hidden_layers=1,
        num_attention_heads=24,num_key_value_heads=8,head_dim=128,vocab_size=128256,
        rms_norm_eps=1e-5,rope_theta=500000.0,
        rope_scaling=dict(rope_type='llama3',factor=32.0,low_freq_factor=1.0,
                          high_freq_factor=4.0,original_max_position_embeddings=8192))
    module,params,_=build_generate(config,7,9,last_logits=True)
    prefix=tvm_pipeline.graph_prefix()(module)
    semantic=tvm_pipeline.graph_pipeline(custom_legalize=npu_legalize.legalize_map(),
        fuse=False,fuse_attention=False,prepared=True)(prefix)
    for name,ir in [('01_authored_relax',module),('02_graph_prefix',prefix),('03_semantic_tir',semantic)]:
        (args.out/f'{name}.py').write_text(ir.script()+'\n')
    summary=dict(scope='IR export/legalization only; actual 3B widths with layer count reduced 28→1',
        checkpoint_loaded=False,npu_scheduled=False,npu_executed=False,config=config,
        prefill_tokens=7,decode_capacity=9,last_logits=True,
        parameters=[dict(name=name,shape=[int(x) for x in p.shape],dtype=str(p.dtype)) for name,p in params],
        provenance=describe())
    (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print('Exported one-layer actual-width Llama IR, no NPU compilation/execution.',flush=True)


if __name__=='__main__':main()
