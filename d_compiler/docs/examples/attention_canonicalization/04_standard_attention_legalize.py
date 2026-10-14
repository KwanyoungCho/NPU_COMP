# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def attention(q: T.Buffer((T.int64(1), T.int64(4), T.int64(2), T.int64(8)), "float16"), k: T.Buffer((T.int64(1), T.int64(4), T.int64(2), T.int64(8)), "float16"), v: T.Buffer((T.int64(1), T.int64(4), T.int64(2), T.int64(8)), "float16"), T_transpose: T.Buffer((T.int64(1), T.int64(4), T.int64(2), T.int64(8)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        T_transpose_1 = T.alloc_buffer((T.int64(1), T.int64(2), T.int64(4), T.int64(8)), "float16")
        T_reshape = T.alloc_buffer((T.int64(2), T.int64(4), T.int64(8)), "float16")
        T_transpose_2 = T.alloc_buffer((T.int64(1), T.int64(2), T.int64(4), T.int64(8)), "float16")
        T_reshape_1 = T.alloc_buffer((T.int64(2), T.int64(4), T.int64(8)), "float16")
        T_batch_matmul_NT = T.alloc_buffer((T.int64(2), T.int64(4), T.int64(4)), "float16")
        T_divide = T.alloc_buffer((T.int64(2), T.int64(4), T.int64(4)), "float16")
        T_softmax_maxelem = T.alloc_buffer((T.int64(2), T.int64(4)), "float16")
        T_softmax_exp = T.alloc_buffer((T.int64(2), T.int64(4), T.int64(4)), "float16")
        T_softmax_expsum = T.alloc_buffer((T.int64(2), T.int64(4)), "float16")
        T_softmax_norm = T.alloc_buffer((T.int64(2), T.int64(4), T.int64(4)), "float16")
        T_transpose_3 = T.alloc_buffer((T.int64(1), T.int64(2), T.int64(4), T.int64(8)), "float16")
        T_reshape_2 = T.alloc_buffer((T.int64(2), T.int64(4), T.int64(8)), "float16")
        T_batch_matmul_NN = T.alloc_buffer((T.int64(2), T.int64(4), T.int64(8)), "float16")
        T_reshape_3 = T.alloc_buffer((T.int64(1), T.int64(2), T.int64(4), T.int64(8)), "float16")
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(2), T.int64(4), T.int64(8)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(q[v_ax0, v_ax2, v_ax1, v_ax3])
                T.writes(T_transpose_1[v_ax0, v_ax1, v_ax2, v_ax3])
                T_transpose_1[v_ax0, v_ax1, v_ax2, v_ax3] = q[v_ax0, v_ax2, v_ax1, v_ax3]
        for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4), T.int64(8)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_transpose_1[T.int64(0), ((v_ax2 // T.int64(8) + v_ax1) // T.int64(4) + v_ax0) % T.int64(2), (v_ax2 // T.int64(8) + v_ax1) % T.int64(4), v_ax2 % T.int64(8)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = T_transpose_1[T.int64(0), ((v_ax2 // T.int64(8) + v_ax1) // T.int64(4) + v_ax0) % T.int64(2), (v_ax2 // T.int64(8) + v_ax1) % T.int64(4), v_ax2 % T.int64(8)]
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(2), T.int64(4), T.int64(8)):
            with T.block("T_transpose_1"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(k[v_ax0, v_ax2, v_ax1, v_ax3])
                T.writes(T_transpose_2[v_ax0, v_ax1, v_ax2, v_ax3])
                T_transpose_2[v_ax0, v_ax1, v_ax2, v_ax3] = k[v_ax0, v_ax2, v_ax1, v_ax3]
        for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4), T.int64(8)):
            with T.block("T_reshape_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_transpose_2[T.int64(0), ((v_ax2 // T.int64(8) + v_ax1) // T.int64(4) + v_ax0) % T.int64(2), (v_ax2 // T.int64(8) + v_ax1) % T.int64(4), v_ax2 % T.int64(8)])
                T.writes(T_reshape_1[v_ax0, v_ax1, v_ax2])
                T_reshape_1[v_ax0, v_ax1, v_ax2] = T_transpose_2[T.int64(0), ((v_ax2 // T.int64(8) + v_ax1) // T.int64(4) + v_ax0) % T.int64(2), (v_ax2 // T.int64(8) + v_ax1) % T.int64(4), v_ax2 % T.int64(8)]
        for b, i, j, k_1 in T.grid(T.int64(2), T.int64(4), T.int64(4), T.int64(8)):
            with T.block("T_batch_matmul_NT"):
                v_b, v_i, v_j, v_k = T.axis.remap("SSSR", [b, i, j, k_1])
                T.reads(T_reshape[v_b, v_i, v_k], T_reshape_1[v_b, v_j, v_k])
                T.writes(T_batch_matmul_NT[v_b, v_i, v_j])
                T.block_attr({"layout_free_placeholders": [T_reshape_1]})
                with T.init():
                    T_batch_matmul_NT[v_b, v_i, v_j] = T.float16(0.0)
                T_batch_matmul_NT[v_b, v_i, v_j] = T_batch_matmul_NT[v_b, v_i, v_j] + T_reshape[v_b, v_i, v_k] * T_reshape_1[v_b, v_j, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4), T.int64(4)):
            with T.block("T_divide"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_batch_matmul_NT[v_ax0, v_ax1, v_ax2])
                T.writes(T_divide[v_ax0, v_ax1, v_ax2])
                T_divide[v_ax0, v_ax1, v_ax2] = T_batch_matmul_NT[v_ax0, v_ax1, v_ax2] / T.sqrt(T.float16(8.0))
        for i0, i1, k_1 in T.grid(T.int64(2), T.int64(4), T.int64(4)):
            with T.block("T_softmax_maxelem"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k_1])
                T.reads(T_divide[v_i0, v_i1, v_k])
                T.writes(T_softmax_maxelem[v_i0, v_i1])
                with T.init():
                    T_softmax_maxelem[v_i0, v_i1] = T.float16(-65504.0)
                T_softmax_maxelem[v_i0, v_i1] = T.max(T_softmax_maxelem[v_i0, v_i1], T_divide[v_i0, v_i1, v_k])
        for i0, i1, i2 in T.grid(T.int64(2), T.int64(4), T.int64(4)):
            with T.block("T_softmax_exp"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(T_divide[v_i0, v_i1, v_i2], T_softmax_maxelem[v_i0, v_i1])
                T.writes(T_softmax_exp[v_i0, v_i1, v_i2])
                T_softmax_exp[v_i0, v_i1, v_i2] = T.exp(T_divide[v_i0, v_i1, v_i2] - T_softmax_maxelem[v_i0, v_i1])
        for i0, i1, k_1 in T.grid(T.int64(2), T.int64(4), T.int64(4)):
            with T.block("T_softmax_expsum"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k_1])
                T.reads(T_softmax_exp[v_i0, v_i1, v_k])
                T.writes(T_softmax_expsum[v_i0, v_i1])
                with T.init():
                    T_softmax_expsum[v_i0, v_i1] = T.float16(0.0)
                T_softmax_expsum[v_i0, v_i1] = T_softmax_expsum[v_i0, v_i1] + T_softmax_exp[v_i0, v_i1, v_k]
        for i0, i1, i2 in T.grid(T.int64(2), T.int64(4), T.int64(4)):
            with T.block("T_softmax_norm"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(T_softmax_exp[v_i0, v_i1, v_i2], T_softmax_expsum[v_i0, v_i1])
                T.writes(T_softmax_norm[v_i0, v_i1, v_i2])
                T.block_attr({"axis": 2})
                T_softmax_norm[v_i0, v_i1, v_i2] = T_softmax_exp[v_i0, v_i1, v_i2] / T_softmax_expsum[v_i0, v_i1]
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(2), T.int64(4), T.int64(8)):
            with T.block("T_transpose_2"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(v[v_ax0, v_ax2, v_ax1, v_ax3])
                T.writes(T_transpose_3[v_ax0, v_ax1, v_ax2, v_ax3])
                T_transpose_3[v_ax0, v_ax1, v_ax2, v_ax3] = v[v_ax0, v_ax2, v_ax1, v_ax3]
        for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4), T.int64(8)):
            with T.block("T_reshape_2"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_transpose_3[T.int64(0), ((v_ax2 // T.int64(8) + v_ax1) // T.int64(4) + v_ax0) % T.int64(2), (v_ax2 // T.int64(8) + v_ax1) % T.int64(4), v_ax2 % T.int64(8)])
                T.writes(T_reshape_2[v_ax0, v_ax1, v_ax2])
                T_reshape_2[v_ax0, v_ax1, v_ax2] = T_transpose_3[T.int64(0), ((v_ax2 // T.int64(8) + v_ax1) // T.int64(4) + v_ax0) % T.int64(2), (v_ax2 // T.int64(8) + v_ax1) % T.int64(4), v_ax2 % T.int64(8)]
        for b, i, j, k_1 in T.grid(T.int64(2), T.int64(4), T.int64(8), T.int64(4)):
            with T.block("T_batch_matmul_NN"):
                v_b, v_i, v_j, v_k = T.axis.remap("SSSR", [b, i, j, k_1])
                T.reads(T_softmax_norm[v_b, v_i, v_k], T_reshape_2[v_b, v_k, v_j])
                T.writes(T_batch_matmul_NN[v_b, v_i, v_j])
                T.block_attr({"layout_free_placeholders": [T_reshape_2]})
                with T.init():
                    T_batch_matmul_NN[v_b, v_i, v_j] = T.float16(0.0)
                T_batch_matmul_NN[v_b, v_i, v_j] = T_batch_matmul_NN[v_b, v_i, v_j] + T_softmax_norm[v_b, v_i, v_k] * T_reshape_2[v_b, v_k, v_j]
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(2), T.int64(4), T.int64(8)):
            with T.block("T_reshape_3"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(T_batch_matmul_NN[((v_ax3 // T.int64(8) + v_ax2) // T.int64(4) + v_ax1) % T.int64(2), (v_ax3 // T.int64(8) + v_ax2) % T.int64(4), v_ax3 % T.int64(8)])
                T.writes(T_reshape_3[v_ax0, v_ax1, v_ax2, v_ax3])
                T_reshape_3[v_ax0, v_ax1, v_ax2, v_ax3] = T_batch_matmul_NN[((v_ax3 // T.int64(8) + v_ax2) // T.int64(4) + v_ax1) % T.int64(2), (v_ax3 // T.int64(8) + v_ax2) % T.int64(4), v_ax3 % T.int64(8)]
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(4), T.int64(2), T.int64(8)):
            with T.block("T_transpose_3"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(T_reshape_3[v_ax0, v_ax2, v_ax1, v_ax3])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
                T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = T_reshape_3[v_ax0, v_ax2, v_ax1, v_ax3]

    @R.function
    def prefill(q: R.Tensor((1, 4, 2, 8), dtype="float16"), k: R.Tensor((1, 4, 2, 8), dtype="float16"), v: R.Tensor((1, 4, 2, 8), dtype="float16")) -> R.Tensor((1, 4, 2, 8), dtype="float16"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.attention, (q, k, v), out_sinfo=R.Tensor((1, 4, 2, 8), dtype="float16"))
            R.output(gv)
        return gv
