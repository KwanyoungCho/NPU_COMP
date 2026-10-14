# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def divide(lv6: T.Buffer((T.int64(2), T.int64(4), T.int64(4)), "float16"), T_divide: T.Buffer((T.int64(2), T.int64(4), T.int64(4)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4), T.int64(4)):
            with T.block("T_divide"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv6[v_ax0, v_ax1, v_ax2])
                T.writes(T_divide[v_ax0, v_ax1, v_ax2])
                T_divide[v_ax0, v_ax1, v_ax2] = lv6[v_ax0, v_ax1, v_ax2] * T.float16(0.35359116022099446)

    @T.prim_func(private=True)
    def matmul(lv1: T.Buffer((T.int64(2), T.int64(4), T.int64(8)), "float16"), lv3: T.Buffer((T.int64(2), T.int64(8), T.int64(4)), "float16"), compute: T.Buffer((T.int64(2), T.int64(4), T.int64(4)), "float16")):
        T.func_attr({"npu.numeric_policy": "fp16_storage_fp32_accum", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul = T.alloc_buffer((T.int64(2), T.int64(4), T.int64(4)))
        for i0, i1, i2, k in T.grid(T.int64(2), T.int64(4), T.int64(4), T.int64(8)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv1[v_i0, v_i1, v_k], lv3[v_i0, v_k, v_i2])
                T.writes(matmul[v_i0, v_i1, v_i2])
                with T.init():
                    matmul[v_i0, v_i1, v_i2] = T.float32(0.0)
                matmul[v_i0, v_i1, v_i2] = matmul[v_i0, v_i1, v_i2] + T.Cast("float32", lv1[v_i0, v_i1, v_k]) * T.Cast("float32", lv3[v_i0, v_k, v_i2])
        for i0, i1, i2 in T.grid(T.int64(2), T.int64(4), T.int64(4)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(matmul[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = T.Cast("float16", matmul[v_i0, v_i1, v_i2])

    @T.prim_func(private=True)
    def matmul1(lv8: T.Buffer((T.int64(2), T.int64(4), T.int64(4)), "float16"), lv5: T.Buffer((T.int64(2), T.int64(4), T.int64(8)), "float16"), compute: T.Buffer((T.int64(2), T.int64(4), T.int64(8)), "float16")):
        T.func_attr({"npu.numeric_policy": "fp16_storage_fp32_accum", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul = T.alloc_buffer((T.int64(2), T.int64(4), T.int64(8)))
        for i0, i1, i2, k in T.grid(T.int64(2), T.int64(4), T.int64(8), T.int64(4)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv8[v_i0, v_i1, v_k], lv5[v_i0, v_k, v_i2])
                T.writes(matmul[v_i0, v_i1, v_i2])
                with T.init():
                    matmul[v_i0, v_i1, v_i2] = T.float32(0.0)
                matmul[v_i0, v_i1, v_i2] = matmul[v_i0, v_i1, v_i2] + T.Cast("float32", lv8[v_i0, v_i1, v_k]) * T.Cast("float32", lv5[v_i0, v_k, v_i2])
        for i0, i1, i2 in T.grid(T.int64(2), T.int64(4), T.int64(8)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(matmul[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = T.Cast("float16", matmul[v_i0, v_i1, v_i2])

    @T.prim_func(private=True)
    def reshape(lv: T.Buffer((T.int64(1), T.int64(2), T.int64(4), T.int64(8)), "float16"), T_reshape: T.Buffer((T.int64(2), T.int64(4), T.int64(8)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4), T.int64(8)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv[T.int64(0), ((v_ax2 // T.int64(8) + v_ax1) // T.int64(4) + v_ax0) % T.int64(2), (v_ax2 // T.int64(8) + v_ax1) % T.int64(4), v_ax2 % T.int64(8)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = lv[T.int64(0), ((v_ax2 // T.int64(8) + v_ax1) // T.int64(4) + v_ax0) % T.int64(2), (v_ax2 // T.int64(8) + v_ax1) % T.int64(4), v_ax2 % T.int64(8)]

    @T.prim_func(private=True)
    def reshape1(lv2: T.Buffer((T.int64(1), T.int64(2), T.int64(8), T.int64(4)), "float16"), T_reshape: T.Buffer((T.int64(2), T.int64(8), T.int64(4)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(8), T.int64(4)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv2[T.int64(0), ((v_ax2 // T.int64(4) + v_ax1) // T.int64(8) + v_ax0) % T.int64(2), (v_ax2 // T.int64(4) + v_ax1) % T.int64(8), v_ax2 % T.int64(4)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = lv2[T.int64(0), ((v_ax2 // T.int64(4) + v_ax1) // T.int64(8) + v_ax0) % T.int64(2), (v_ax2 // T.int64(4) + v_ax1) % T.int64(8), v_ax2 % T.int64(4)]

    @T.prim_func(private=True)
    def reshape2(lv9: T.Buffer((T.int64(2), T.int64(4), T.int64(8)), "float16"), T_reshape: T.Buffer((T.int64(1), T.int64(2), T.int64(4), T.int64(8)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(2), T.int64(4), T.int64(8)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(lv9[((v_ax3 // T.int64(8) + v_ax2) // T.int64(4) + v_ax1) % T.int64(2), (v_ax3 // T.int64(8) + v_ax2) % T.int64(4), v_ax3 % T.int64(8)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = lv9[((v_ax3 // T.int64(8) + v_ax2) // T.int64(4) + v_ax1) % T.int64(2), (v_ax3 // T.int64(8) + v_ax2) % T.int64(4), v_ax3 % T.int64(8)]

    @T.prim_func(private=True)
    def softmax(lv7: T.Buffer((T.int64(2), T.int64(4), T.int64(4)), "float16"), T_softmax_norm: T.Buffer((T.int64(2), T.int64(4), T.int64(4)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        T_softmax_maxelem = T.alloc_buffer((T.int64(2), T.int64(4)), "float16")
        T_softmax_exp = T.alloc_buffer((T.int64(2), T.int64(4), T.int64(4)), "float16")
        softmax_sum32 = T.alloc_buffer((T.int64(2), T.int64(4)))
        T_softmax_expsum = T.alloc_buffer((T.int64(2), T.int64(4)), "float16")
        for i0, i1, k in T.grid(T.int64(2), T.int64(4), T.int64(4)):
            with T.block("T_softmax_maxelem"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(lv7[v_i0, v_i1, v_k])
                T.writes(T_softmax_maxelem[v_i0, v_i1])
                with T.init():
                    T_softmax_maxelem[v_i0, v_i1] = T.float16(-65504.0)
                T_softmax_maxelem[v_i0, v_i1] = T.max(T_softmax_maxelem[v_i0, v_i1], lv7[v_i0, v_i1, v_k])
        for i0, i1, i2 in T.grid(T.int64(2), T.int64(4), T.int64(4)):
            with T.block("T_softmax_exp"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(lv7[v_i0, v_i1, v_i2], T_softmax_maxelem[v_i0, v_i1])
                T.writes(T_softmax_exp[v_i0, v_i1, v_i2])
                T_softmax_exp[v_i0, v_i1, v_i2] = T.exp(lv7[v_i0, v_i1, v_i2] - T_softmax_maxelem[v_i0, v_i1])
        for i0, i1, k in T.grid(T.int64(2), T.int64(4), T.int64(4)):
            with T.block("softmax_sum32"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(T_softmax_exp[v_i0, v_i1, v_k])
                T.writes(softmax_sum32[v_i0, v_i1])
                with T.init():
                    softmax_sum32[v_i0, v_i1] = T.float32(0.0)
                softmax_sum32[v_i0, v_i1] = softmax_sum32[v_i0, v_i1] + T.Cast("float32", T_softmax_exp[v_i0, v_i1, v_k])
        for i0, i1 in T.grid(T.int64(2), T.int64(4)):
            with T.block("T_softmax_expsum"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(softmax_sum32[v_i0, v_i1])
                T.writes(T_softmax_expsum[v_i0, v_i1])
                T_softmax_expsum[v_i0, v_i1] = T.Cast("float16", softmax_sum32[v_i0, v_i1])
        for i0, i1, i2 in T.grid(T.int64(2), T.int64(4), T.int64(4)):
            with T.block("T_softmax_norm"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(T_softmax_exp[v_i0, v_i1, v_i2], T_softmax_expsum[v_i0, v_i1])
                T.writes(T_softmax_norm[v_i0, v_i1, v_i2])
                T_softmax_norm[v_i0, v_i1, v_i2] = T_softmax_exp[v_i0, v_i1, v_i2] / T_softmax_expsum[v_i0, v_i1]

    @T.prim_func(private=True)
    def transpose(q: T.Buffer((T.int64(1), T.int64(4), T.int64(2), T.int64(8)), "float16"), T_transpose: T.Buffer((T.int64(1), T.int64(2), T.int64(4), T.int64(8)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(2), T.int64(4), T.int64(8)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(q[v_ax0, v_ax2, v_ax1, v_ax3])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
                T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = q[v_ax0, v_ax2, v_ax1, v_ax3]

    @T.prim_func(private=True)
    def transpose1(k: T.Buffer((T.int64(1), T.int64(4), T.int64(2), T.int64(8)), "float16"), T_transpose: T.Buffer((T.int64(1), T.int64(2), T.int64(8), T.int64(4)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(2), T.int64(8), T.int64(4)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(k[v_ax0, v_ax3, v_ax1, v_ax2])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
                T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = k[v_ax0, v_ax3, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def transpose2(lv10: T.Buffer((T.int64(1), T.int64(2), T.int64(4), T.int64(8)), "float16"), T_transpose: T.Buffer((T.int64(1), T.int64(4), T.int64(2), T.int64(8)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(4), T.int64(2), T.int64(8)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(lv10[v_ax0, v_ax2, v_ax1, v_ax3])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
                T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = lv10[v_ax0, v_ax2, v_ax1, v_ax3]

    @R.function
    def prefill(q: R.Tensor((1, 4, 2, 8), dtype="float16"), k: R.Tensor((1, 4, 2, 8), dtype="float16"), v: R.Tensor((1, 4, 2, 8), dtype="float16")) -> R.Tensor((1, 4, 2, 8), dtype="float16"):
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.transpose, (q,), out_sinfo=R.Tensor((1, 2, 4, 8), dtype="float16"))
            lv1 = R.call_tir(cls.reshape, (lv,), out_sinfo=R.Tensor((2, 4, 8), dtype="float16"))
            lv2 = R.call_tir(cls.transpose1, (k,), out_sinfo=R.Tensor((1, 2, 8, 4), dtype="float16"))
            lv3 = R.call_tir(cls.reshape1, (lv2,), out_sinfo=R.Tensor((2, 8, 4), dtype="float16"))
            lv4 = R.call_tir(cls.transpose, (v,), out_sinfo=R.Tensor((1, 2, 4, 8), dtype="float16"))
            lv5 = R.call_tir(cls.reshape, (lv4,), out_sinfo=R.Tensor((2, 4, 8), dtype="float16"))
            lv6 = R.call_tir(cls.matmul, (lv1, lv3), out_sinfo=R.Tensor((2, 4, 4), dtype="float16"))
            lv7 = R.call_tir(cls.divide, (lv6,), out_sinfo=R.Tensor((2, 4, 4), dtype="float16"))
            lv8 = R.call_tir(cls.softmax, (lv7,), out_sinfo=R.Tensor((2, 4, 4), dtype="float16"))
            lv9 = R.call_tir(cls.matmul1, (lv8, lv5), out_sinfo=R.Tensor((2, 4, 8), dtype="float16"))
            lv10 = R.call_tir(cls.reshape2, (lv9,), out_sinfo=R.Tensor((1, 2, 4, 8), dtype="float16"))
            gv = R.call_tir(cls.transpose2, (lv10,), out_sinfo=R.Tensor((1, 4, 2, 8), dtype="float16"))
            R.output(gv)
        return gv
