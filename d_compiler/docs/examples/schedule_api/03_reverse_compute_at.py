# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((4, 4), "float32"), C: T.Buffer((4, 4), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        B = T.alloc_buffer((4, 4))
        for i_0 in range(2):
            for i_1, j in T.grid(2, 4):
                with T.block("B"):
                    v_i = T.axis.spatial(4, i_0 * 2 + i_1)
                    v_j = T.axis.spatial(4, j)
                    T.reads(A[v_i, v_j])
                    T.writes(B[v_i, v_j])
                    B[v_i, v_j] = A[v_i, v_j] + T.float32(1.0)
            for ax0, ax1 in T.grid(2, 4):
                with T.block("C"):
                    v_i = T.axis.spatial(4, i_0 * 2 + ax0)
                    v_j = T.axis.spatial(4, ax1)
                    T.reads(B[v_i, v_j])
                    T.writes(C[v_i, v_j])
                    C[v_i, v_j] = B[v_i, v_j] * T.float32(2.0)
