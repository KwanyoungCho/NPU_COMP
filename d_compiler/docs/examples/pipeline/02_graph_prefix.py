# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def prefill(x: R.Tensor((64, 128), dtype="float16"), bias: R.Tensor((64,), dtype="float16"), lv: R.Tensor((128, 64), dtype="float16")) -> R.Tensor((64, 64), dtype="float16"):
        R.func_attr({"num_input": 1})
        with R.dataflow():
            lv1: R.Tensor((64, 64), dtype="float16") = R.matmul(x, lv, out_dtype="void")
            lv2: R.Tensor((64, 64), dtype="float16") = R.add(lv1, bias)
            gv: R.Tensor((64, 64), dtype="float16") = R.nn.silu(lv2)
            R.output(gv)
        return gv

    @R.function
    def prefill_transform_params(model_params: R.Tuple(R.Tensor((64, 128), dtype="float16"), R.Tensor((64,), dtype="float16"))) -> R.Tuple(R.Tensor((64,), dtype="float16"), R.Tensor((128, 64), dtype="float16")):
        R.func_attr({"num_input": 0})
        with R.dataflow():
            weight: R.Tensor((64, 128), dtype="float16") = model_params[0]
            lv: R.Tensor((128, 64), dtype="float16") = R.permute_dims(weight, axes=[1, 0])
            bias: R.Tensor((64,), dtype="float16") = model_params[1]
            output_tuple: R.Tuple(R.Tensor((64,), dtype="float16"), R.Tensor((128, 64), dtype="float16")) = bias, lv
            R.output(output_tuple)
        return output_tuple
