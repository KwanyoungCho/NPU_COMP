# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((4, 4), "float32"), C: T.Buffer((4, 4), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        A_global_sram = T.alloc_buffer((4, 4), scope="global.sram")
        for ax0, ax1 in T.grid(4, 4):
            with T.block("A_global.sram"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A[v0, v1])
                T.writes(A_global_sram[v0, v1])
                A_global_sram[v0, v1] = A[v0, v1]
        for i, j in T.grid(4, 4):
            with T.block("C"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(A_global_sram[v_i, v_j])
                T.writes(C[v_i, v_j])
                C[v_i, v_j] = A_global_sram[v_i, v_j] + T.float32(1.0)
