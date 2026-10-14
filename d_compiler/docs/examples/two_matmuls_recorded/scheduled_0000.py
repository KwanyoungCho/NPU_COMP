# from tvm.script import tir as T

@T.prim_func(private=True)
def main(x: T.Buffer((T.int64(64), T.int64(64)), "float16"), w: T.Buffer((T.int64(64), T.int64(128)), "float16"), var_compute: T.handle):
    T.func_attr({"npu.accumulator_store_lowered": 1, "npu.forwarded_output_tiles": 1, "npu.numeric_policy": "fp16_storage_fp32_accum", "op_pattern": 4, "tir.noalias": T.bool(True)})
    compute = T.match_buffer(var_compute, (T.int64(64), T.int64(128)), "float16", scope="global.sram")
    # with T.block("root"):
    for i1_0 in range(T.int64(1)):
        with T.block(""):
            T.reads(w[T.int64(0):T.int64(64), T.int64(0):T.int64(128)], x[T.int64(0):T.int64(64), T.int64(0):T.int64(64)])
            T.writes(compute[T.int64(0):T.int64(64), T.int64(0):T.int64(128)])
            w_global_sram = T.alloc_buffer((T.int64(64), T.int64(128)), "float16", scope="global.sram")
            for ax0, ax1 in T.grid(T.int64(64), T.int64(128)):
                with T.block("w_global.sram"):
                    T.reads(w[ax0, ax1])
                    T.writes(w_global_sram[ax0, ax1])
                    w_global_sram[ax0, ax1] = w[ax0, ax1]
            for i0_0 in range(T.int64(1)):
                with T.block(""):
                    T.reads(x[T.int64(0):T.int64(64), T.int64(0):T.int64(64)], w_global_sram[T.int64(0):T.int64(64), i1_0 * T.int64(128):i1_0 * T.int64(128) + T.int64(128)])
                    T.writes(compute[T.int64(0):T.int64(64), T.int64(0):T.int64(128)])
                    x_global_sram = T.alloc_buffer((T.int64(64), T.int64(64)), "float16", scope="global.sram")
                    for ax0, ax1 in T.grid(T.int64(64), T.int64(64)):
                        with T.block("x_global.sram"):
                            T.reads(x[ax0, ax1])
                            T.writes(x_global_sram[ax0, ax1])
                            x_global_sram[ax0, ax1] = x[ax0, ax1]
                    for i1_1_0 in range(T.int64(2)):
                        with T.block(""):
                            T.reads(x_global_sram[T.int64(0):T.int64(64), T.int64(0):T.int64(64)], w_global_sram[T.int64(0):T.int64(64), i1_0 * T.int64(128) + i1_1_0 * T.int64(64):i1_0 * T.int64(128) + i1_1_0 * T.int64(64) + T.int64(64)])
                            T.writes(compute[T.int64(0):T.int64(64), i1_1_0 * T.int64(64):i1_1_0 * T.int64(64) + T.int64(64)])
                            with T.block("matmul_init_o"):
                                T.reads()
                                T.writes(compute[T.int64(0):T.int64(64), i1_0 * T.int64(128) + i1_1_0 * T.int64(64):i1_0 * T.int64(128) + i1_1_0 * T.int64(64) + T.int64(64)])
                                C = T.match_buffer(compute[T.int64(0):T.int64(64), i1_0 * T.int64(128) + i1_1_0 * T.int64(64):i1_0 * T.int64(128) + i1_1_0 * T.int64(64) + T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("C_s0", T.int64(1)), scope="global.sram", offset_factor=1)
                                T.call_extern("int32", "npu_fill_zero", T.tvm_access_ptr(T.type_annotation("float16"), C.data, C.elem_offset, C.strides[0] * T.int64(64), 2), C.strides[0])
                            for k_0 in range(T.int64(1)):
                                with T.block("matmul_update_o"):
                                    T.reads(compute[T.int64(0):T.int64(64), i1_0 * T.int64(128) + i1_1_0 * T.int64(64):i1_0 * T.int64(128) + i1_1_0 * T.int64(64) + T.int64(64)], x_global_sram[T.int64(0):T.int64(64), T.int64(0):T.int64(64)], w_global_sram[T.int64(0):T.int64(64), (i1_0 * T.int64(2) + i1_1_0) * T.int64(64):(i1_0 * T.int64(2) + i1_1_0) * T.int64(64) + T.int64(64)])
                                    T.writes(compute[T.int64(0):T.int64(64), i1_0 * T.int64(128) + i1_1_0 * T.int64(64):i1_0 * T.int64(128) + i1_1_0 * T.int64(64) + T.int64(64)])
                                    A = T.match_buffer(x_global_sram[T.int64(0):T.int64(64), T.int64(0):T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("A_s0", T.int64(1)), scope="global.sram", offset_factor=1)
                                    B = T.match_buffer(w_global_sram[T.int64(0):T.int64(64), (i1_0 * T.int64(2) + i1_1_0) * T.int64(64):(i1_0 * T.int64(2) + i1_1_0) * T.int64(64) + T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("B_s0", T.int64(1)), scope="global.sram", offset_factor=1)
                                    C = T.match_buffer(compute[T.int64(0):T.int64(64), i1_0 * T.int64(128) + i1_1_0 * T.int64(64):i1_0 * T.int64(128) + i1_1_0 * T.int64(64) + T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("C_s0", T.int64(1)), scope="global.sram", offset_factor=1)
                                    T.call_extern("int32", "npu_gemm_acc", T.tvm_access_ptr(T.type_annotation("float16"), C.data, C.elem_offset, C.strides[0] * T.int64(64), 3), C.strides[0], T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * T.int64(64), 1), A.strides[0], T.tvm_access_ptr(T.type_annotation("float16"), B.data, B.elem_offset, B.strides[0] * T.int64(64), 1), B.strides[0])
