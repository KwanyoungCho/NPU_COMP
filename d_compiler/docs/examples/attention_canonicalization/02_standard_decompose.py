# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def prefill(q: R.Tensor((1, 4, 2, 8), dtype="float16"), k: R.Tensor((1, 4, 2, 8), dtype="float16"), v: R.Tensor((1, 4, 2, 8), dtype="float16")) -> R.Tensor((1, 4, 2, 8), dtype="float16"):
        with R.dataflow():
            gv: R.Tensor((1, 4, 2, 8), dtype="float16") = R.nn.attention(q, k, v, scale=None, causal_mask=None, window_size=None)
            R.output(gv)
        return gv
