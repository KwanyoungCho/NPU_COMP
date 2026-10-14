# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((4, 4), "float32"), C: T.Buffer((4, 4), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        B_global_sram = T.alloc_buffer((4, 4), scope="global.sram")
        for i, j in T.grid(4, 4):
            with T.block("B"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(A[v_i, v_j])
                T.writes(B_global_sram[v_i, v_j])
                B_global_sram[v_i, v_j] = A[v_i, v_j] + T.float32(1.0)
        for i, j in T.grid(4, 4):
            with T.block("C"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(B_global_sram[v_i, v_j])
                T.writes(C[v_i, v_j])
                C[v_i, v_j] = B_global_sram[v_i, v_j] * T.float32(2.0)
