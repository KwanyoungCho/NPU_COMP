# from tvm.script import tir as T

@T.prim_func(private=True)
def main(var_lv: T.handle, v: T.Buffer((T.int64(128), T.int64(64)), "float16"), compute: T.Buffer((T.int64(64), T.int64(64)), "float16")):
    T.func_attr({"npu.accumulator_store_lowered": 1, "npu.numeric_policy": "fp16_storage_fp32_accum", "op_pattern": 4, "tir.noalias": T.bool(True)})
    lv = T.match_buffer(var_lv, (T.int64(64), T.int64(128)), "float16", scope="global.sram")
    # with T.block("root"):
    for i1_0 in range(T.int64(1)):
        with T.block(""):
            T.reads(v[T.int64(0):T.int64(128), T.int64(0):T.int64(64)], lv[T.int64(0):T.int64(64), T.int64(0):T.int64(128)])
            T.writes(compute[T.int64(0):T.int64(64), T.int64(0):T.int64(64)])
            v_global_sram = T.alloc_buffer((T.int64(128), T.int64(64)), "float16", scope="global.sram")
            for ax0, ax1 in T.grid(T.int64(128), T.int64(64)):
                with T.block("v_global.sram"):
                    T.reads(v[ax0, ax1])
                    T.writes(v_global_sram[ax0, ax1])
                    v_global_sram[ax0, ax1] = v[ax0, ax1]
            for i0_0, i1_1_0 in T.grid(T.int64(1), T.int64(1)):
                with T.block(""):
                    T.reads(lv[T.int64(0):T.int64(64), T.int64(0):T.int64(128)], v_global_sram[T.int64(0):T.int64(128), T.int64(0):T.int64(64)])
                    T.writes(compute[T.int64(0):T.int64(64), T.int64(0):T.int64(64)])
                    matmul_global_sram = T.alloc_buffer((T.int64(64), T.int64(64)), "float16", scope="global.sram")
                    with T.block("matmul_init_o"):
                        T.reads()
                        T.writes(matmul_global_sram[T.int64(0):T.int64(64), T.int64(0):T.int64(64)])
                        C = T.match_buffer(matmul_global_sram[T.int64(0):T.int64(64), T.int64(0):T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("C_s0", T.int64(1)), scope="global.sram", offset_factor=1)
                        T.call_extern("int32", "npu_fill_zero", T.tvm_access_ptr(T.type_annotation("float16"), C.data, C.elem_offset, C.strides[0] * T.int64(64), 2), C.strides[0])
                    for k_0 in range(T.int64(2)):
                        with T.block("matmul_update_o"):
                            T.reads(matmul_global_sram[T.int64(0):T.int64(64), T.int64(0):T.int64(64)], lv[T.int64(0):T.int64(64), k_0 * T.int64(64):k_0 * T.int64(64) + T.int64(64)], v_global_sram[k_0 * T.int64(64):k_0 * T.int64(64) + T.int64(64), T.int64(0):T.int64(64)])
                            T.writes(matmul_global_sram[T.int64(0):T.int64(64), T.int64(0):T.int64(64)])
                            A = T.match_buffer(lv[T.int64(0):T.int64(64), k_0 * T.int64(64):k_0 * T.int64(64) + T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("A_s0", T.int64(1)), scope="global.sram", offset_factor=1)
                            B = T.match_buffer(v_global_sram[k_0 * T.int64(64):k_0 * T.int64(64) + T.int64(64), T.int64(0):T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("B_s0", T.int64(1)), scope="global.sram", offset_factor=1)
                            C = T.match_buffer(matmul_global_sram[T.int64(0):T.int64(64), T.int64(0):T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("C_s0", T.int64(1)), scope="global.sram", offset_factor=1)
                            T.call_extern("int32", "npu_gemm_acc", T.tvm_access_ptr(T.type_annotation("float16"), C.data, C.elem_offset, C.strides[0] * T.int64(64), 3), C.strides[0], T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * T.int64(64), 1), A.strides[0], T.tvm_access_ptr(T.type_annotation("float16"), B.data, B.elem_offset, B.strides[0] * T.int64(64), 1), B.strides[0])
                    for ax0, ax1 in T.grid(T.int64(64), T.int64(64)):
                        with T.block("compute"):
                            T.reads(matmul_global_sram[ax0, ax1])
                            T.writes(compute[ax0, ax1])
                            compute[ax0, ax1] = matmul_global_sram[ax0, ax1]
