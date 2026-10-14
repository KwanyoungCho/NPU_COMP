# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(lv1: T.Buffer((T.int64(64), T.int64(64)), "float16"), bias: T.Buffer((T.int64(64),), "float16"), T_add: T.Buffer((T.int64(64), T.int64(64)), "float16")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(64), T.int64(64)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv1[v_ax0, v_ax1], bias[v_ax1])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = lv1[v_ax0, v_ax1] + bias[v_ax1]

    @T.prim_func(private=True)
    def matmul(x: T.Buffer((T.int64(64), T.int64(128)), "float16"), lv: T.Buffer((T.int64(128), T.int64(64)), "float16"), compute: T.Buffer((T.int64(64), T.int64(64)), "float16")):
        T.func_attr({"npu.numeric_policy": "fp16_storage_fp32_accum", "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul = T.alloc_buffer((T.int64(64), T.int64(64)))
        matmul_rf = T.alloc_buffer((T.int64(2), T.int64(64), T.int64(64)))
        for i0, i1, k_0, k_1 in T.grid(T.int64(64), T.int64(64), T.int64(2), T.int64(64)):
            with T.block("matmul_rf"):
                vk_0, v_i0, v_i1, vk_1 = T.axis.remap("SSSR", [k_0, i0, i1, k_1])
                T.reads(x[v_i0, vk_0 * T.int64(64) + vk_1], lv[vk_0 * T.int64(64) + vk_1, v_i1])
                T.writes(matmul_rf[vk_0, v_i0, v_i1])
                with T.init():
                    matmul_rf[vk_0, v_i0, v_i1] = T.float32(0.0)
                matmul_rf[vk_0, v_i0, v_i1] = matmul_rf[vk_0, v_i0, v_i1] + T.Cast("float32", x[v_i0, vk_0 * T.int64(64) + vk_1]) * T.Cast("float32", lv[vk_0 * T.int64(64) + vk_1, v_i1])
        for i0, i1, k_0 in T.grid(T.int64(64), T.int64(64), T.int64(2)):
            with T.block("matmul"):
                vk_0, v_i0, v_i1 = T.axis.remap("RSS", [k_0, i0, i1])
                T.reads(matmul_rf[vk_0, v_i0, v_i1])
                T.writes(matmul[v_i0, v_i1])
                with T.init():
                    matmul[v_i0, v_i1] = T.float32(0.0)
                matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + matmul_rf[vk_0, v_i0, v_i1]
        for i0, i1 in T.grid(T.int64(64), T.int64(64)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(matmul[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.Cast("float16", matmul[v_i0, v_i1])

    @T.prim_func(private=True)
    def silu(lv2: T.Buffer((T.int64(64), T.int64(64)), "float16"), T_multiply: T.Buffer((T.int64(64), T.int64(64)), "float16")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        compute = T.alloc_buffer((T.int64(64), T.int64(64)), "float16")
        for i0, i1 in T.grid(T.int64(64), T.int64(64)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(lv2[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.sigmoid(lv2[v_i0, v_i1])
        for ax0, ax1 in T.grid(T.int64(64), T.int64(64)):
            with T.block("T_multiply"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv2[v_ax0, v_ax1], compute[v_ax0, v_ax1])
                T.writes(T_multiply[v_ax0, v_ax1])
                T_multiply[v_ax0, v_ax1] = lv2[v_ax0, v_ax1] * compute[v_ax0, v_ax1]

    @R.function
    def prefill(x: R.Tensor((64, 128), dtype="float16"), bias: R.Tensor((64,), dtype="float16"), lv: R.Tensor((128, 64), dtype="float16")) -> R.Tensor((64, 64), dtype="float16"):
        R.func_attr({"num_input": 1})
        cls = Module
        with R.dataflow():
            lv1 = R.call_tir(cls.matmul, (x, lv), out_sinfo=R.Tensor((64, 64), dtype="float16"))
            lv2 = R.call_tir(cls.add, (lv1, bias), out_sinfo=R.Tensor((64, 64), dtype="float16"))
            gv = R.call_tir(cls.silu, (lv2,), out_sinfo=R.Tensor((64, 64), dtype="float16"))
            R.output(gv)
        return gv
