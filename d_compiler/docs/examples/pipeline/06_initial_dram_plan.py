# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def fused_matmul_add_silu(x: T.Buffer((T.int64(64), T.int64(128)), "float16"), lv: T.Buffer((T.int64(128), T.int64(64)), "float16"), bias: T.Buffer((T.int64(64),), "float16"), T_multiply_intermediate: T.Buffer((T.int64(64), T.int64(64)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul = T.alloc_buffer((T.int64(64), T.int64(64)))
        compute_intermediate = T.alloc_buffer((T.int64(64), T.int64(64)), "float16")
        T_add_intermediate = T.alloc_buffer((T.int64(64), T.int64(64)), "float16")
        compute = T.alloc_buffer((T.int64(64), T.int64(64)), "float16")
        for i0, i1, k in T.grid(T.int64(64), T.int64(64), T.int64(128)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(x[v_i0, v_k], lv[v_k, v_i1])
                T.writes(matmul[v_i0, v_i1])
                with T.init():
                    matmul[v_i0, v_i1] = T.float32(0.0)
                matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + T.Cast("float32", x[v_i0, v_k]) * T.Cast("float32", lv[v_k, v_i1])
        for i0, i1 in T.grid(T.int64(64), T.int64(64)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(matmul[v_i0, v_i1])
                T.writes(compute_intermediate[v_i0, v_i1])
                compute_intermediate[v_i0, v_i1] = T.Cast("float16", matmul[v_i0, v_i1])
        for ax0, ax1 in T.grid(T.int64(64), T.int64(64)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(compute_intermediate[v_ax0, v_ax1], bias[v_ax1])
                T.writes(T_add_intermediate[v_ax0, v_ax1])
                T_add_intermediate[v_ax0, v_ax1] = compute_intermediate[v_ax0, v_ax1] + bias[v_ax1]
        for i0, i1 in T.grid(T.int64(64), T.int64(64)):
            with T.block("compute_1"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(T_add_intermediate[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.sigmoid(T_add_intermediate[v_i0, v_i1])
        for ax0, ax1 in T.grid(T.int64(64), T.int64(64)):
            with T.block("T_multiply"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(T_add_intermediate[v_ax0, v_ax1], compute[v_ax0, v_ax1])
                T.writes(T_multiply_intermediate[v_ax0, v_ax1])
                T_multiply_intermediate[v_ax0, v_ax1] = T_add_intermediate[v_ax0, v_ax1] * compute[v_ax0, v_ax1]

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
    def prefill(x: R.Tensor((64, 128), dtype="float16"), bias: R.Tensor((64,), dtype="float16"), lv: R.Tensor((128, 64), dtype="float16")) -> R.Tensor((64, 64), dtype="float16"):
        R.func_attr({"num_input": 1, "relax.force_pure": True})
        cls = Module
        alloc1: R.Tensor((64, 64), dtype="float16") = R.builtin.alloc_tensor(R.shape([64, 64]), R.dtype("float16"), R.prim_value(0), R.str("global"))
        cls.fused_matmul_add_silu(x, lv, bias, alloc1)
        gv: R.Tensor((64, 64), dtype="float16") = alloc1
        return gv

    @R.function
    def prefill_transform_params(model_params: R.Tuple(R.Tensor((64, 128), dtype="float16"), R.Tensor((64,), dtype="float16"))) -> R.Tuple(R.Tensor((64,), dtype="float16"), R.Tensor((128, 64), dtype="float16")):
        R.func_attr({"num_input": 0, "relax.force_pure": True})
        cls = Module
        weight: R.Tensor((64, 128), dtype="float16") = model_params[0]
        alloc: R.Tensor((128, 64), dtype="float16") = R.builtin.alloc_tensor(R.shape([128, 64]), R.dtype("float16"), R.prim_value(0), R.str("global"))
        cls.transpose(weight, alloc)
        lv: R.Tensor((128, 64), dtype="float16") = alloc
        bias: R.Tensor((64,), dtype="float16") = model_params[1]
        output_tuple: R.Tuple(R.Tensor((64,), dtype="float16"), R.Tensor((128, 64), dtype="float16")) = bias, lv
        return output_tuple
