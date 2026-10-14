# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def prefill(q: R.Tensor((1, 4, 2, 8), dtype="float16"), k: R.Tensor((1, 4, 2, 8), dtype="float16"), v: R.Tensor((1, 4, 2, 8), dtype="float16")) -> R.Tensor((1, 4, 2, 8), dtype="float16"):
        with R.dataflow():
            lv: R.Tensor((1, 2, 4, 8), dtype="float16") = R.permute_dims(q, axes=[0, 2, 1, 3])
            lv1: R.Tensor((2, 4, 8), dtype="float16") = R.reshape(lv, R.shape([2, 4, 8]))
            lv2: R.Tensor((1, 2, 8, 4), dtype="float16") = R.permute_dims(k, axes=[0, 2, 3, 1])
            lv3: R.Tensor((2, 8, 4), dtype="float16") = R.reshape(lv2, R.shape([2, 8, 4]))
            lv4: R.Tensor((1, 2, 4, 8), dtype="float16") = R.permute_dims(v, axes=[0, 2, 1, 3])
            lv5: R.Tensor((2, 4, 8), dtype="float16") = R.reshape(lv4, R.shape([2, 4, 8]))
            lv6: R.Tensor((2, 4, 4), dtype="float16") = R.matmul(lv1, lv3, out_dtype="void")
            lv7: R.Tensor((2, 4, 4), dtype="float16") = R.divide(lv6, R.const(2.828125, "float16"))
            lv8: R.Tensor((2, 4, 4), dtype="float16") = R.nn.softmax(lv7, axis=-1)
            lv9: R.Tensor((2, 4, 8), dtype="float16") = R.matmul(lv8, lv5, out_dtype="void")
            lv10: R.Tensor((1, 2, 4, 8), dtype="float16") = R.reshape(lv9, R.shape([1, 2, 4, 8]))
            gv: R.Tensor((1, 4, 2, 8), dtype="float16") = R.permute_dims(lv10, axes=[0, 2, 1, 3])
            R.output(gv)
        return gv
