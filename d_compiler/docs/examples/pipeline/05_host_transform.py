# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def transpose(weight: T.Buffer((T.int64(64), T.int64(128)), "float16"), T_transpose: T.Buffer((T.int64(128), T.int64(64)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(128), T.int64(64)):
            with T.block("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(weight[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = weight[v_ax1, v_ax0]

    @R.function
    def prefill_transform_params(model_params: R.Tuple(R.Tensor((64, 128), dtype="float16"), R.Tensor((64,), dtype="float16"))) -> R.Tuple(R.Tensor((64,), dtype="float16"), R.Tensor((128, 64), dtype="float16")):
        R.func_attr({"num_input": 0})
        cls = Module
        with R.dataflow():
            weight: R.Tensor((64, 128), dtype="float16") = model_params[0]
            lv = R.call_tir(cls.transpose, (weight,), out_sinfo=R.Tensor((128, 64), dtype="float16"))
            bias: R.Tensor((64,), dtype="float16") = model_params[1]
            output_tuple: R.Tuple(R.Tensor((64,), dtype="float16"), R.Tensor((128, 64), dtype="float16")) = bias, lv
            R.output(output_tuple)
        return output_tuple
