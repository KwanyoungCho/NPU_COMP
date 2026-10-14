# from tvm.script import tir as T

@T.prim_func(private=True)
def main(x: T.Buffer((T.int64(64), T.int64(128)), "float16"), lv: T.Buffer((T.int64(128), T.int64(64)), "float16"), bias: T.Buffer((T.int64(64),), "float16"), T_multiply_intermediate: T.Buffer((T.int64(64), T.int64(64)), "float16")):
    T.func_attr({"npu.accumulator_store_lowered": 1, "npu.pointwise_vector_width": 128, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i0_0 in range(T.int64(1)):
        with T.block(""):
            T.reads(x[T.int64(0):T.int64(64), T.int64(0):T.int64(128)], lv[T.int64(0):T.int64(128), T.int64(0):T.int64(64)], bias[T.int64(0):T.int64(64)])
            T.writes(T_multiply_intermediate[T.int64(0):T.int64(64), T.int64(0):T.int64(64)])
            x_global_sram = T.alloc_buffer((T.int64(64), T.int64(128)), "float16", scope="global.sram")
            lv_global_sram = T.alloc_buffer((T.int64(128), T.int64(64)), "float16", scope="global.sram")
            for ax0, ax1 in T.grid(T.int64(64), T.int64(128)):
                with T.block("x_global.sram"):
                    T.reads(x[ax0, ax1])
                    T.writes(x_global_sram[ax0, ax1])
                    x_global_sram[ax0, ax1] = x[ax0, ax1]
            for ax0, ax1 in T.grid(T.int64(128), T.int64(64)):
                with T.block("lv_global.sram"):
                    T.reads(lv[ax0, ax1])
                    T.writes(lv_global_sram[ax0, ax1])
                    lv_global_sram[ax0, ax1] = lv[ax0, ax1]
            for i1_0, i0_1_0, i1_1_0 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                with T.block(""):
                    T.reads(x_global_sram[T.int64(0):T.int64(64), T.int64(0):T.int64(128)], lv_global_sram[T.int64(0):T.int64(128), T.int64(0):T.int64(64)], bias[T.int64(0):T.int64(64)])
                    T.writes(T_multiply_intermediate[T.int64(0):T.int64(64), T.int64(0):T.int64(64)])
                    compute_intermediate_global_sram = T.alloc_buffer((T.int64(64), T.int64(64)), "float16", scope="global.sram")
                    T_add_intermediate_global_sram = T.alloc_buffer((T.int64(64), T.int64(64)), "float16", scope="global.sram")
                    compute_global_sram = T.alloc_buffer((T.int64(64), T.int64(64)), "float16", scope="global.sram")
                    matmul_global_sram = T.alloc_buffer((T.int64(64), T.int64(64)), "float16", scope="global.sram")
                    bias_global_sram = T.alloc_buffer((T.int64(64),), "float16", scope="global.sram")
                    T_multiply_intermediate_global_sram = T.alloc_buffer((T.int64(64), T.int64(64)), "float16", scope="global.sram")
                    with T.block("matmul_init_o"):
                        T.reads()
                        T.writes(matmul_global_sram[T.int64(0):T.int64(64), T.int64(0):T.int64(64)])
                        C = T.match_buffer(matmul_global_sram[T.int64(0):T.int64(64), T.int64(0):T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("C_s0", T.int64(1)), scope="global.sram", offset_factor=1)
                        T.call_extern("int32", "npu_fill_zero", T.tvm_access_ptr(T.type_annotation("float16"), C.data, C.elem_offset, C.strides[0] * T.int64(64), 2), C.strides[0])
                    for k_0, k_1_0 in T.grid(T.int64(2), T.int64(1)):
                        with T.block("matmul_update_o"):
                            T.reads(matmul_global_sram[T.int64(0):T.int64(64), T.int64(0):T.int64(64)], x_global_sram[T.int64(0):T.int64(64), (k_0 + k_1_0) * T.int64(64):(k_0 + k_1_0) * T.int64(64) + T.int64(64)], lv_global_sram[(k_0 + k_1_0) * T.int64(64):(k_0 + k_1_0) * T.int64(64) + T.int64(64), T.int64(0):T.int64(64)])
                            T.writes(matmul_global_sram[T.int64(0):T.int64(64), T.int64(0):T.int64(64)])
                            A = T.match_buffer(x_global_sram[T.int64(0):T.int64(64), (k_0 + k_1_0) * T.int64(64):(k_0 + k_1_0) * T.int64(64) + T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("A_s0", T.int64(1)), scope="global.sram", offset_factor=1)
                            B = T.match_buffer(lv_global_sram[(k_0 + k_1_0) * T.int64(64):(k_0 + k_1_0) * T.int64(64) + T.int64(64), T.int64(0):T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("B_s0", T.int64(1)), scope="global.sram", offset_factor=1)
                            C = T.match_buffer(matmul_global_sram[T.int64(0):T.int64(64), T.int64(0):T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("C_s0", T.int64(1)), scope="global.sram", offset_factor=1)
                            T.call_extern("int32", "npu_gemm_acc", T.tvm_access_ptr(T.type_annotation("float16"), C.data, C.elem_offset, C.strides[0] * T.int64(64), 3), C.strides[0], T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * T.int64(64), 1), A.strides[0], T.tvm_access_ptr(T.type_annotation("float16"), B.data, B.elem_offset, B.strides[0] * T.int64(64), 1), B.strides[0])
                    for ax0, ax1 in T.grid(T.int64(64), T.int64(64)):
                        with T.block("compute"):
                            T.reads(matmul_global_sram[ax0, ax1])
                            T.writes(compute_intermediate_global_sram[ax0, ax1])
                            compute_intermediate_global_sram[ax0, ax1] = matmul_global_sram[ax0, ax1]
                    for ax0 in range(T.int64(64)):
                        with T.block("bias_global.sram"):
                            T.reads(bias[ax0])
                            T.writes(bias_global_sram[ax0])
                            bias_global_sram[ax0] = bias[ax0]
                    for ax0, ax1 in T.grid(T.int64(64), T.int64(64)):
                        with T.block("T_add"):
                            T.reads(compute_intermediate_global_sram[ax0, ax1], bias_global_sram[ax1])
                            T.writes(T_add_intermediate_global_sram[ax0, ax1])
                            T_add_intermediate_global_sram[ax0, ax1] = compute_intermediate_global_sram[ax0, ax1] + bias_global_sram[ax1]
                    for ax0_ax1_fused_0, ax0_ax1_fused_1 in T.grid(T.int64(32), T.int64(128)):
                        with T.block("compute_1"):
                            T.reads(T_add_intermediate_global_sram[(ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1) // T.int64(64), (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1) % T.int64(64)])
                            T.writes(compute_global_sram[(ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1) // T.int64(64), (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1) % T.int64(64)])
                            compute_global_sram[(ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1) // T.int64(64), (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1) % T.int64(64)] = T.sigmoid(T_add_intermediate_global_sram[(ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1) // T.int64(64), (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1) % T.int64(64)])
                    for ax0_ax1_fused_0, ax0_ax1_fused_1 in T.grid(T.int64(32), T.int64(128)):
                        with T.block("T_multiply"):
                            T.reads(T_add_intermediate_global_sram[(ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1) // T.int64(64), (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1) % T.int64(64)], compute_global_sram[(ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1) // T.int64(64), (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1) % T.int64(64)])
                            T.writes(T_multiply_intermediate_global_sram[(ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1) // T.int64(64), (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1) % T.int64(64)])
                            T_multiply_intermediate_global_sram[(ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1) // T.int64(64), (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1) % T.int64(64)] = T_add_intermediate_global_sram[(ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1) // T.int64(64), (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1) % T.int64(64)] * compute_global_sram[(ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1) // T.int64(64), (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1) % T.int64(64)]
                    for ax0, ax1 in T.grid(T.int64(64), T.int64(64)):
                        with T.block("T_multiply_intermediate_global.sram"):
                            T.reads(T_multiply_intermediate_global_sram[ax0, ax1])
                            T.writes(T_multiply_intermediate[ax0, ax1])
                            T_multiply_intermediate[ax0, ax1] = T_multiply_intermediate_global_sram[ax0, ax1]
