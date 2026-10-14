# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def matmul(x: T.Buffer((T.int64(64), T.int64(64)), "float16"), w: T.Buffer((T.int64(64), T.int64(128)), "float16"), compute: T.Buffer((T.int64(64), T.int64(128)), "float16")):
        T.func_attr({"npu.numeric_policy": "fp16_storage_fp32_accum", "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul = T.alloc_buffer((T.int64(64), T.int64(128)))
        for i0, i1, k in T.grid(T.int64(64), T.int64(128), T.int64(64)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(x[v_i0, v_k], w[v_k, v_i1])
                T.writes(matmul[v_i0, v_i1])
                with T.init():
                    matmul[v_i0, v_i1] = T.float32(0.0)
                matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + T.Cast("float32", x[v_i0, v_k]) * T.Cast("float32", w[v_k, v_i1])
        for i0, i1 in T.grid(T.int64(64), T.int64(128)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(matmul[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.Cast("float16", matmul[v_i0, v_i1])

    @T.prim_func(private=True)
    def matmul1(lv: T.Buffer((T.int64(64), T.int64(128)), "float16"), v: T.Buffer((T.int64(128), T.int64(64)), "float16"), compute: T.Buffer((T.int64(64), T.int64(64)), "float16")):
        T.func_attr({"npu.numeric_policy": "fp16_storage_fp32_accum", "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul = T.alloc_buffer((T.int64(64), T.int64(64)))
        for i0, i1, k in T.grid(T.int64(64), T.int64(64), T.int64(128)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(lv[v_i0, v_k], v[v_k, v_i1])
                T.writes(matmul[v_i0, v_i1])
                with T.init():
                    matmul[v_i0, v_i1] = T.float32(0.0)
                matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + T.Cast("float32", lv[v_i0, v_k]) * T.Cast("float32", v[v_k, v_i1])
        for i0, i1 in T.grid(T.int64(64), T.int64(64)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(matmul[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.Cast("float16", matmul[v_i0, v_i1])

    @R.function
    def prefill(x: R.Tensor((64, 64), dtype="float16"), w: R.Tensor((64, 128), dtype="float16"), v: R.Tensor((128, 64), dtype="float16")) -> R.Tensor((64, 64), dtype="float16"):
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.matmul, (x, w), out_sinfo=R.Tensor((64, 128), dtype="float16"))
            gv = R.call_tir(cls.matmul1, (lv, v), out_sinfo=R.Tensor((64, 64), dtype="float16"))
            R.output(gv)
        return gv
