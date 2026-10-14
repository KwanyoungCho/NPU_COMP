"""Compare installed TVM attention lowering with the current NPU graph pass.

Run from the repository root in npu-tvm. This inspects IR and compilation;
it does not run numerical inference or benchmark device performance.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "d_compiler"))

import tvm
from tvm import relax, tir
from npu_compiler import npu_legalize, npu_target
from npu_compiler.npu_attention import CanonicalizeAttention
from npu_compiler.npu_errors import LinkError


def source_module():
    args = [relax.Var(name, relax.TensorStructInfo((1, 4, 2, 8), "float16"))
            for name in ("q", "k", "v")]
    bb = relax.BlockBuilder()
    with bb.function("prefill", args):
        with bb.dataflow():
            result = bb.emit_output(relax.op.nn.attention(*args))
        bb.emit_func_output(result)
    return bb.finalize()


def describe(module):
    calls = Counter()
    reductions = []

    def call(expr):
        if isinstance(expr, relax.Call):
            name = getattr(expr.op, "name", None) or getattr(expr.op, "name_hint", str(expr.op))
            calls[str(name)] += 1

    def block(node):
        if isinstance(node, tir.Block) and any(
                axis.iter_type == tir.IterVar.CommReduce for axis in node.iter_vars):
            reductions.append(dict(block=node.name_hint,
                                   output_dtypes=[str(r.buffer.dtype) for r in node.writes]))

    for _, function in module.functions.items():
        if isinstance(function, relax.Function):
            relax.analysis.post_order_visit(function.body, call)
        elif isinstance(function, tir.PrimFunc):
            tir.stmt_functor.post_order_visit(function.body, block)
    return dict(calls=dict(calls), reductions=reductions)


def main():
    directory = Path(__file__).resolve().parent
    source = source_module()
    decomposed = relax.transform.DecomposeOpsForInference()(source)
    canonical = CanonicalizeAttention()(source)
    # Even with our matmul/softmax callbacks installed, the stock attention
    # legalizer builds its internal computations directly through TE/TOPI.
    legalize = relax.transform.LegalizeOps(npu_legalize.legalize_map())
    variants = {
        "01_input": source,
        "02_standard_decompose": decomposed,
        "03_npu_canonical": canonical,
        "04_standard_attention_legalize": legalize(source),
        "05_canonical_then_legalize": legalize(canonical),
    }
    report = dict(tvm_version=tvm.__version__,
                  tvm_commit=tvm.support.libinfo().get("GIT_COMMIT_HASH"),
                  input_shape=[1, 4, 2, 8], dtype="float16",
                  standard_decompose_unchanged=tvm.ir.structural_equal(source, decomposed),
                  variants={name: describe(mod) for name, mod in variants.items()},
                  validation_scope="IR inspection and compilation; no numerical run or timing")
    for name, module in variants.items():
        (directory / (name + ".py")).write_text(module.script() + "\n")
    try:
        npu_target.build(source, pipeline=legalize, fusion="off")
    except LinkError as error:
        report["stock_attention_npu_build"] = dict(status="rejected", reason=str(error))
    else:
        report["stock_attention_npu_build"] = dict(status="passed")
    executable = npu_target.build(source, fusion="off")
    report["current_path_npu_build"] = dict(status="passed", kernels=executable.kernel_count,
                                           words=len(executable.words))
    (directory / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
