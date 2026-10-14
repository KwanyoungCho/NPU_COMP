# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((4, 4), "float32"), C: T.Buffer((4, 4), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i_0 in range(2):
            with T.block(""):
                T.reads(A[i_0 * 2:i_0 * 2 + 2, 0:4])
                T.writes(C[i_0 * 2:i_0 * 2 + 2, 0:4])
                A_global_sram = T.alloc_buffer((2, 4), scope="global.sram")
                for ax0, ax1 in T.grid(2, 4):
                    with T.block("A_global.sram"):
                        T.reads(A[i_0 * 2 + ax0, ax1])
                        T.writes(A_global_sram[i_0 * 2 + ax0 - i_0 * 2, ax1])
                        A_global_sram[i_0 * 2 + ax0 - i_0 * 2, ax1] = A[i_0 * 2 + ax0, ax1]
                for i_1, j in T.grid(2, 4):
                    with T.block("C"):
                        T.reads(A_global_sram[i_0 * 2 + i_1 - i_0 * 2, j])
                        T.writes(C[i_0 * 2 + i_1, j])
                        C[i_0 * 2 + i_1, j] = A_global_sram[i_0 * 2 + i_1 - i_0 * 2, j] + T.float32(1.0)
