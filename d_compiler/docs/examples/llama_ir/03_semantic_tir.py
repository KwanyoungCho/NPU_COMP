# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(mul12: T.Buffer((T.int64(1), T.int64(24), T.int64(128)), "float16"), mul13: T.Buffer((T.int64(1), T.int64(24), T.int64(128)), "float16"), T_add: T.Buffer((T.int64(1), T.int64(24), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(24), T.int64(128)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(mul12[v_ax0, v_ax1, v_ax2], mul13[v_ax0, v_ax1, v_ax2])
                T.writes(T_add[v_ax0, v_ax1, v_ax2])
                T_add[v_ax0, v_ax1, v_ax2] = mul12[v_ax0, v_ax1, v_ax2] + mul13[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def add1(mul14: T.Buffer((T.int64(1), T.int64(8), T.int64(128)), "float16"), mul15: T.Buffer((T.int64(1), T.int64(8), T.int64(128)), "float16"), T_add: T.Buffer((T.int64(1), T.int64(8), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(8), T.int64(128)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(mul14[v_ax0, v_ax1, v_ax2], mul15[v_ax0, v_ax1, v_ax2])
                T.writes(T_add[v_ax0, v_ax1, v_ax2])
                T_add[v_ax0, v_ax1, v_ax2] = mul14[v_ax0, v_ax1, v_ax2] + mul15[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def add2(mul16: T.Buffer((T.int64(24), T.int64(1), T.int64(10)), "float16"), mask: T.Buffer((T.int64(1), T.int64(1), T.int64(10)), "float16"), T_add: T.Buffer((T.int64(24), T.int64(1), T.int64(10)), "float16")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(24), T.int64(1), T.int64(10)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(mul16[v_ax0, v_ax1, v_ax2], mask[T.int64(0), v_ax1, v_ax2])
                T.writes(T_add[v_ax0, v_ax1, v_ax2])
                T_add[v_ax0, v_ax1, v_ax2] = mul16[v_ax0, v_ax1, v_ax2] + mask[T.int64(0), v_ax1, v_ax2]

    @T.prim_func(private=True)
    def add3(input_embeds: T.Buffer((T.int64(1), T.int64(3072)), "float16"), matmul25: T.Buffer((T.int64(1), T.int64(3072)), "float16"), T_add: T.Buffer((T.int64(1), T.int64(3072)), "float16")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(3072)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(input_embeds[v_ax0, v_ax1], matmul25[v_ax0, v_ax1])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = input_embeds[v_ax0, v_ax1] + matmul25[v_ax0, v_ax1]

    @T.prim_func(private=True)
    def add4(mul: T.Buffer((T.int64(7), T.int64(24), T.int64(128)), "float16"), mul1: T.Buffer((T.int64(7), T.int64(24), T.int64(128)), "float16"), T_add: T.Buffer((T.int64(7), T.int64(24), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(7), T.int64(24), T.int64(128)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(mul[v_ax0, v_ax1, v_ax2], mul1[v_ax0, v_ax1, v_ax2])
                T.writes(T_add[v_ax0, v_ax1, v_ax2])
                T_add[v_ax0, v_ax1, v_ax2] = mul[v_ax0, v_ax1, v_ax2] + mul1[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def add5(mul2: T.Buffer((T.int64(7), T.int64(8), T.int64(128)), "float16"), mul3: T.Buffer((T.int64(7), T.int64(8), T.int64(128)), "float16"), T_add: T.Buffer((T.int64(7), T.int64(8), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(7), T.int64(8), T.int64(128)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(mul2[v_ax0, v_ax1, v_ax2], mul3[v_ax0, v_ax1, v_ax2])
                T.writes(T_add[v_ax0, v_ax1, v_ax2])
                T_add[v_ax0, v_ax1, v_ax2] = mul2[v_ax0, v_ax1, v_ax2] + mul3[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def add6(mul4: T.Buffer((T.int64(24), T.int64(7), T.int64(7)), "float16"), mask: T.Buffer((T.int64(1), T.int64(7), T.int64(7)), "float16"), T_add: T.Buffer((T.int64(24), T.int64(7), T.int64(7)), "float16")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(24), T.int64(7), T.int64(7)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(mul4[v_ax0, v_ax1, v_ax2], mask[T.int64(0), v_ax1, v_ax2])
                T.writes(T_add[v_ax0, v_ax1, v_ax2])
                T_add[v_ax0, v_ax1, v_ax2] = mul4[v_ax0, v_ax1, v_ax2] + mask[T.int64(0), v_ax1, v_ax2]

    @T.prim_func(private=True)
    def add7(input_embeds: T.Buffer((T.int64(7), T.int64(3072)), "float16"), matmul5: T.Buffer((T.int64(7), T.int64(3072)), "float16"), T_add: T.Buffer((T.int64(7), T.int64(3072)), "float16")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(7), T.int64(3072)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(input_embeds[v_ax0, v_ax1], matmul5[v_ax0, v_ax1])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = input_embeds[v_ax0, v_ax1] + matmul5[v_ax0, v_ax1]

    @T.prim_func(private=True)
    def concatenate(neg4: T.Buffer((T.int64(1), T.int64(24), T.int64(64)), "float16"), split_04: T.Buffer((T.int64(1), T.int64(24), T.int64(64)), "float16"), T_concat: T.Buffer((T.int64(1), T.int64(24), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(24), T.int64(128)):
            with T.block("T_concat"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(split_04[v_ax0, v_ax1, v_ax2 - T.int64(64)], neg4[v_ax0, v_ax1, v_ax2])
                T.writes(T_concat[v_ax0, v_ax1, v_ax2])
                T_concat[v_ax0, v_ax1, v_ax2] = T.if_then_else(T.int64(64) <= v_ax2, split_04[v_ax0, v_ax1, v_ax2 - T.int64(64)], neg4[v_ax0, v_ax1, v_ax2])

    @T.prim_func(private=True)
    def concatenate1(neg5: T.Buffer((T.int64(1), T.int64(8), T.int64(64)), "float16"), split_05: T.Buffer((T.int64(1), T.int64(8), T.int64(64)), "float16"), T_concat: T.Buffer((T.int64(1), T.int64(8), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(8), T.int64(128)):
            with T.block("T_concat"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(split_05[v_ax0, v_ax1, v_ax2 - T.int64(64)], neg5[v_ax0, v_ax1, v_ax2])
                T.writes(T_concat[v_ax0, v_ax1, v_ax2])
                T_concat[v_ax0, v_ax1, v_ax2] = T.if_then_else(T.int64(64) <= v_ax2, split_05[v_ax0, v_ax1, v_ax2 - T.int64(64)], neg5[v_ax0, v_ax1, v_ax2])

    @T.prim_func(private=True)
    def concatenate2(reshape12: T.Buffer((T.int64(8), T.int64(128), T.int64(9)), "float16"), permute_dims31: T.Buffer((T.int64(8), T.int64(128), T.int64(1)), "float16"), T_concat: T.Buffer((T.int64(8), T.int64(128), T.int64(10)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(8), T.int64(128), T.int64(10)):
            with T.block("T_concat"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(permute_dims31[v_ax0, v_ax1, v_ax2 - T.int64(9)], reshape12[v_ax0, v_ax1, v_ax2])
                T.writes(T_concat[v_ax0, v_ax1, v_ax2])
                T_concat[v_ax0, v_ax1, v_ax2] = T.if_then_else(T.int64(9) <= v_ax2, permute_dims31[v_ax0, v_ax1, v_ax2 - T.int64(9)], reshape12[v_ax0, v_ax1, v_ax2])

    @T.prim_func(private=True)
    def concatenate3(reshape13: T.Buffer((T.int64(8), T.int64(9), T.int64(128)), "float16"), permute_dims32: T.Buffer((T.int64(8), T.int64(1), T.int64(128)), "float16"), T_concat: T.Buffer((T.int64(8), T.int64(10), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(8), T.int64(10), T.int64(128)):
            with T.block("T_concat"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(permute_dims32[v_ax0, v_ax1 - T.int64(9), v_ax2], reshape13[v_ax0, v_ax1, v_ax2])
                T.writes(T_concat[v_ax0, v_ax1, v_ax2])
                T_concat[v_ax0, v_ax1, v_ax2] = T.if_then_else(T.int64(9) <= v_ax1, permute_dims32[v_ax0, v_ax1 - T.int64(9), v_ax2], reshape13[v_ax0, v_ax1, v_ax2])

    @T.prim_func(private=True)
    def concatenate4(unsqueeze2: T.Buffer((T.int64(1), T.int64(8), T.int64(128), T.int64(1)), "float16"), T_concat: T.Buffer((T.int64(1), T.int64(8), T.int64(128), T.int64(1)), "float16")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(8), T.int64(128), T.int64(1)):
            with T.block("T_concat"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(unsqueeze2[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_concat[v_ax0, v_ax1, v_ax2, v_ax3])
                T_concat[v_ax0, v_ax1, v_ax2, v_ax3] = unsqueeze2[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func(private=True)
    def concatenate5(unsqueeze3: T.Buffer((T.int64(1), T.int64(8), T.int64(1), T.int64(128)), "float16"), T_concat: T.Buffer((T.int64(1), T.int64(8), T.int64(1), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(8), T.int64(1), T.int64(128)):
            with T.block("T_concat"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(unsqueeze3[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_concat[v_ax0, v_ax1, v_ax2, v_ax3])
                T_concat[v_ax0, v_ax1, v_ax2, v_ax3] = unsqueeze3[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func(private=True)
    def concatenate6(neg: T.Buffer((T.int64(7), T.int64(24), T.int64(64)), "float16"), split_0: T.Buffer((T.int64(7), T.int64(24), T.int64(64)), "float16"), T_concat: T.Buffer((T.int64(7), T.int64(24), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(7), T.int64(24), T.int64(128)):
            with T.block("T_concat"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(split_0[v_ax0, v_ax1, v_ax2 - T.int64(64)], neg[v_ax0, v_ax1, v_ax2])
                T.writes(T_concat[v_ax0, v_ax1, v_ax2])
                T_concat[v_ax0, v_ax1, v_ax2] = T.if_then_else(T.int64(64) <= v_ax2, split_0[v_ax0, v_ax1, v_ax2 - T.int64(64)], neg[v_ax0, v_ax1, v_ax2])

    @T.prim_func(private=True)
    def concatenate7(neg1: T.Buffer((T.int64(7), T.int64(8), T.int64(64)), "float16"), split_01: T.Buffer((T.int64(7), T.int64(8), T.int64(64)), "float16"), T_concat: T.Buffer((T.int64(7), T.int64(8), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(7), T.int64(8), T.int64(128)):
            with T.block("T_concat"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(split_01[v_ax0, v_ax1, v_ax2 - T.int64(64)], neg1[v_ax0, v_ax1, v_ax2])
                T.writes(T_concat[v_ax0, v_ax1, v_ax2])
                T_concat[v_ax0, v_ax1, v_ax2] = T.if_then_else(T.int64(64) <= v_ax2, split_01[v_ax0, v_ax1, v_ax2 - T.int64(64)], neg1[v_ax0, v_ax1, v_ax2])

    @T.prim_func(private=True)
    def concatenate8(unsqueeze: T.Buffer((T.int64(1), T.int64(8), T.int64(128), T.int64(7)), "float16"), T_concat: T.Buffer((T.int64(1), T.int64(8), T.int64(128), T.int64(7)), "float16")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(8), T.int64(128), T.int64(7)):
            with T.block("T_concat"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(unsqueeze[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_concat[v_ax0, v_ax1, v_ax2, v_ax3])
                T_concat[v_ax0, v_ax1, v_ax2, v_ax3] = unsqueeze[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func(private=True)
    def concatenate9(unsqueeze1: T.Buffer((T.int64(1), T.int64(8), T.int64(7), T.int64(128)), "float16"), T_concat: T.Buffer((T.int64(1), T.int64(8), T.int64(7), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(8), T.int64(7), T.int64(128)):
            with T.block("T_concat"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(unsqueeze1[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_concat[v_ax0, v_ax1, v_ax2, v_ax3])
                T_concat[v_ax0, v_ax1, v_ax2, v_ax3] = unsqueeze1[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func(private=True)
    def expand_dims(permute_dims31: T.Buffer((T.int64(8), T.int64(128), T.int64(1)), "float16"), expand_dims: T.Buffer((T.int64(1), T.int64(8), T.int64(128), T.int64(1)), "float16")):
        T.func_attr({"op_pattern": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(8), T.int64(128), T.int64(1)):
            with T.block("expand_dims"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(permute_dims31[v_i1, v_i2, v_i3])
                T.writes(expand_dims[v_i0, v_i1, v_i2, v_i3])
                expand_dims[v_i0, v_i1, v_i2, v_i3] = permute_dims31[v_i1, v_i2, v_i3]

    @T.prim_func(private=True)
    def expand_dims1(permute_dims32: T.Buffer((T.int64(8), T.int64(1), T.int64(128)), "float16"), expand_dims: T.Buffer((T.int64(1), T.int64(8), T.int64(1), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(8), T.int64(1), T.int64(128)):
            with T.block("expand_dims"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(permute_dims32[v_i1, v_i2, v_i3])
                T.writes(expand_dims[v_i0, v_i1, v_i2, v_i3])
                expand_dims[v_i0, v_i1, v_i2, v_i3] = permute_dims32[v_i1, v_i2, v_i3]

    @T.prim_func(private=True)
    def expand_dims2(permute_dims25: T.Buffer((T.int64(8), T.int64(128), T.int64(7)), "float16"), expand_dims: T.Buffer((T.int64(1), T.int64(8), T.int64(128), T.int64(7)), "float16")):
        T.func_attr({"op_pattern": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(8), T.int64(128), T.int64(7)):
            with T.block("expand_dims"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(permute_dims25[v_i1, v_i2, v_i3])
                T.writes(expand_dims[v_i0, v_i1, v_i2, v_i3])
                expand_dims[v_i0, v_i1, v_i2, v_i3] = permute_dims25[v_i1, v_i2, v_i3]

    @T.prim_func(private=True)
    def expand_dims3(permute_dims26: T.Buffer((T.int64(8), T.int64(7), T.int64(128)), "float16"), expand_dims: T.Buffer((T.int64(1), T.int64(8), T.int64(7), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(8), T.int64(7), T.int64(128)):
            with T.block("expand_dims"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(permute_dims26[v_i1, v_i2, v_i3])
                T.writes(expand_dims[v_i0, v_i1, v_i2, v_i3])
                expand_dims[v_i0, v_i1, v_i2, v_i3] = permute_dims26[v_i1, v_i2, v_i3]

    @T.prim_func(private=True)
    def last_hidden(add9: T.Buffer((T.int64(7), T.int64(3072)), "float16"), last_hidden: T.Buffer((T.int64(1), T.int64(3072)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i, j in T.grid(T.int64(1), T.int64(3072)):
            with T.block("last_hidden"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(add9[T.int64(6), v_j])
                T.writes(last_hidden[v_i, v_j])
                last_hidden[v_i, v_j] = add9[T.int64(6), v_j]

    @T.prim_func(private=True)
    def matmul(rms_norm6: T.Buffer((T.int64(1), T.int64(3072)), "float16"), permute_dims28: T.Buffer((T.int64(3072), T.int64(3072)), "float16"), compute: T.Buffer((T.int64(1), T.int64(3072)), "float16")):
        T.func_attr({"npu.numeric_policy": "fp16_storage_fp32_accum", "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul = T.alloc_buffer((T.int64(1), T.int64(3072)))
        for i0, i1, k in T.grid(T.int64(1), T.int64(3072), T.int64(3072)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(rms_norm6[v_i0, v_k], permute_dims28[v_k, v_i1])
                T.writes(matmul[v_i0, v_i1])
                with T.init():
                    matmul[v_i0, v_i1] = T.float32(0.0)
                matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + T.Cast("float32", rms_norm6[v_i0, v_k]) * T.Cast("float32", permute_dims28[v_k, v_i1])
        for i0, i1 in T.grid(T.int64(1), T.int64(3072)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(matmul[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.Cast("float16", matmul[v_i0, v_i1])

    @T.prim_func(private=True)
    def matmul1(rms_norm6: T.Buffer((T.int64(1), T.int64(3072)), "float16"), permute_dims29: T.Buffer((T.int64(3072), T.int64(1024)), "float16"), compute: T.Buffer((T.int64(1), T.int64(1024)), "float16")):
        T.func_attr({"npu.numeric_policy": "fp16_storage_fp32_accum", "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul = T.alloc_buffer((T.int64(1), T.int64(1024)))
        for i0, i1, k in T.grid(T.int64(1), T.int64(1024), T.int64(3072)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(rms_norm6[v_i0, v_k], permute_dims29[v_k, v_i1])
                T.writes(matmul[v_i0, v_i1])
                with T.init():
                    matmul[v_i0, v_i1] = T.float32(0.0)
                matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + T.Cast("float32", rms_norm6[v_i0, v_k]) * T.Cast("float32", permute_dims29[v_k, v_i1])
        for i0, i1 in T.grid(T.int64(1), T.int64(1024)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(matmul[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.Cast("float16", matmul[v_i0, v_i1])

    @T.prim_func(private=True)
    def matmul10(softmax: T.Buffer((T.int64(24), T.int64(7), T.int64(7)), "float16"), permute_dims5: T.Buffer((T.int64(24), T.int64(7), T.int64(128)), "float16"), compute: T.Buffer((T.int64(24), T.int64(7), T.int64(128)), "float16")):
        T.func_attr({"npu.numeric_policy": "fp16_storage_fp32_accum", "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul = T.alloc_buffer((T.int64(24), T.int64(7), T.int64(128)))
        for i0, i1, i2, k in T.grid(T.int64(24), T.int64(7), T.int64(128), T.int64(7)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(softmax[v_i0, v_i1, v_k], permute_dims5[v_i0, v_k, v_i2])
                T.writes(matmul[v_i0, v_i1, v_i2])
                with T.init():
                    matmul[v_i0, v_i1, v_i2] = T.float32(0.0)
                matmul[v_i0, v_i1, v_i2] = matmul[v_i0, v_i1, v_i2] + T.Cast("float32", softmax[v_i0, v_i1, v_k]) * T.Cast("float32", permute_dims5[v_i0, v_k, v_i2])
        for i0, i1, i2 in T.grid(T.int64(24), T.int64(7), T.int64(128)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(matmul[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = T.Cast("float16", matmul[v_i0, v_i1, v_i2])

    @T.prim_func(private=True)
    def matmul11(rms_norm1: T.Buffer((T.int64(7), T.int64(3072)), "float16"), permute_dims9: T.Buffer((T.int64(3072), T.int64(8192)), "float16"), compute: T.Buffer((T.int64(7), T.int64(8192)), "float16")):
        T.func_attr({"npu.numeric_policy": "fp16_storage_fp32_accum", "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul = T.alloc_buffer((T.int64(7), T.int64(8192)))
        for i0, i1, k in T.grid(T.int64(7), T.int64(8192), T.int64(3072)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(rms_norm1[v_i0, v_k], permute_dims9[v_k, v_i1])
                T.writes(matmul[v_i0, v_i1])
                with T.init():
                    matmul[v_i0, v_i1] = T.float32(0.0)
                matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + T.Cast("float32", rms_norm1[v_i0, v_k]) * T.Cast("float32", permute_dims9[v_k, v_i1])
        for i0, i1 in T.grid(T.int64(7), T.int64(8192)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(matmul[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.Cast("float16", matmul[v_i0, v_i1])

    @T.prim_func(private=True)
    def matmul12(mul5: T.Buffer((T.int64(7), T.int64(8192)), "float16"), permute_dims11: T.Buffer((T.int64(8192), T.int64(3072)), "float16"), compute: T.Buffer((T.int64(7), T.int64(3072)), "float16")):
        T.func_attr({"npu.numeric_policy": "fp16_storage_fp32_accum", "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul = T.alloc_buffer((T.int64(7), T.int64(3072)))
        for i0, i1, k in T.grid(T.int64(7), T.int64(3072), T.int64(8192)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(mul5[v_i0, v_k], permute_dims11[v_k, v_i1])
                T.writes(matmul[v_i0, v_i1])
                with T.init():
                    matmul[v_i0, v_i1] = T.float32(0.0)
                matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + T.Cast("float32", mul5[v_i0, v_k]) * T.Cast("float32", permute_dims11[v_k, v_i1])
        for i0, i1 in T.grid(T.int64(7), T.int64(3072)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(matmul[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.Cast("float16", matmul[v_i0, v_i1])

    @T.prim_func(private=True)
    def matmul13(rms_norm2: T.Buffer((T.int64(7), T.int64(3072)), "float16"), permute_dims12: T.Buffer((T.int64(3072), T.int64(128256)), "float16"), compute: T.Buffer((T.int64(7), T.int64(128256)), "float16")):
        T.func_attr({"npu.numeric_policy": "fp16_storage_fp32_accum", "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul = T.alloc_buffer((T.int64(7), T.int64(128256)))
        for i0, i1, k in T.grid(T.int64(7), T.int64(128256), T.int64(3072)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(rms_norm2[v_i0, v_k], permute_dims12[v_k, v_i1])
                T.writes(matmul[v_i0, v_i1])
                with T.init():
                    matmul[v_i0, v_i1] = T.float32(0.0)
                matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + T.Cast("float32", rms_norm2[v_i0, v_k]) * T.Cast("float32", permute_dims12[v_k, v_i1])
        for i0, i1 in T.grid(T.int64(7), T.int64(128256)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(matmul[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.Cast("float16", matmul[v_i0, v_i1])

    @T.prim_func(private=True)
    def matmul2(permute_dims33: T.Buffer((T.int64(24), T.int64(1), T.int64(128)), "float16"), repeat4: T.Buffer((T.int64(24), T.int64(128), T.int64(10)), "float16"), compute: T.Buffer((T.int64(24), T.int64(1), T.int64(10)), "float16")):
        T.func_attr({"npu.numeric_policy": "fp16_storage_fp32_accum", "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul = T.alloc_buffer((T.int64(24), T.int64(1), T.int64(10)))
        for i0, i1, i2, k in T.grid(T.int64(24), T.int64(1), T.int64(10), T.int64(128)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(permute_dims33[v_i0, v_i1, v_k], repeat4[v_i0, v_k, v_i2])
                T.writes(matmul[v_i0, v_i1, v_i2])
                with T.init():
                    matmul[v_i0, v_i1, v_i2] = T.float32(0.0)
                matmul[v_i0, v_i1, v_i2] = matmul[v_i0, v_i1, v_i2] + T.Cast("float32", permute_dims33[v_i0, v_i1, v_k]) * T.Cast("float32", repeat4[v_i0, v_k, v_i2])
        for i0, i1, i2 in T.grid(T.int64(24), T.int64(1), T.int64(10)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(matmul[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = T.Cast("float16", matmul[v_i0, v_i1, v_i2])

    @T.prim_func(private=True)
    def matmul3(softmax2: T.Buffer((T.int64(24), T.int64(1), T.int64(10)), "float16"), repeat5: T.Buffer((T.int64(24), T.int64(10), T.int64(128)), "float16"), compute: T.Buffer((T.int64(24), T.int64(1), T.int64(128)), "float16")):
        T.func_attr({"npu.numeric_policy": "fp16_storage_fp32_accum", "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul = T.alloc_buffer((T.int64(24), T.int64(1), T.int64(128)))
        for i0, i1, i2, k in T.grid(T.int64(24), T.int64(1), T.int64(128), T.int64(10)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(softmax2[v_i0, v_i1, v_k], repeat5[v_i0, v_k, v_i2])
                T.writes(matmul[v_i0, v_i1, v_i2])
                with T.init():
                    matmul[v_i0, v_i1, v_i2] = T.float32(0.0)
                matmul[v_i0, v_i1, v_i2] = matmul[v_i0, v_i1, v_i2] + T.Cast("float32", softmax2[v_i0, v_i1, v_k]) * T.Cast("float32", repeat5[v_i0, v_k, v_i2])
        for i0, i1, i2 in T.grid(T.int64(24), T.int64(1), T.int64(128)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(matmul[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = T.Cast("float16", matmul[v_i0, v_i1, v_i2])

    @T.prim_func(private=True)
    def matmul4(rms_norm7: T.Buffer((T.int64(1), T.int64(3072)), "float16"), permute_dims36: T.Buffer((T.int64(3072), T.int64(8192)), "float16"), compute: T.Buffer((T.int64(1), T.int64(8192)), "float16")):
        T.func_attr({"npu.numeric_policy": "fp16_storage_fp32_accum", "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul = T.alloc_buffer((T.int64(1), T.int64(8192)))
        for i0, i1, k in T.grid(T.int64(1), T.int64(8192), T.int64(3072)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(rms_norm7[v_i0, v_k], permute_dims36[v_k, v_i1])
                T.writes(matmul[v_i0, v_i1])
                with T.init():
                    matmul[v_i0, v_i1] = T.float32(0.0)
                matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + T.Cast("float32", rms_norm7[v_i0, v_k]) * T.Cast("float32", permute_dims36[v_k, v_i1])
        for i0, i1 in T.grid(T.int64(1), T.int64(8192)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(matmul[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.Cast("float16", matmul[v_i0, v_i1])

    @T.prim_func(private=True)
    def matmul5(mul17: T.Buffer((T.int64(1), T.int64(8192)), "float16"), permute_dims38: T.Buffer((T.int64(8192), T.int64(3072)), "float16"), compute: T.Buffer((T.int64(1), T.int64(3072)), "float16")):
        T.func_attr({"npu.numeric_policy": "fp16_storage_fp32_accum", "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul = T.alloc_buffer((T.int64(1), T.int64(3072)))
        for i0, i1, k in T.grid(T.int64(1), T.int64(3072), T.int64(8192)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(mul17[v_i0, v_k], permute_dims38[v_k, v_i1])
                T.writes(matmul[v_i0, v_i1])
                with T.init():
                    matmul[v_i0, v_i1] = T.float32(0.0)
                matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + T.Cast("float32", mul17[v_i0, v_k]) * T.Cast("float32", permute_dims38[v_k, v_i1])
        for i0, i1 in T.grid(T.int64(1), T.int64(3072)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(matmul[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.Cast("float16", matmul[v_i0, v_i1])

    @T.prim_func(private=True)
    def matmul6(rms_norm8: T.Buffer((T.int64(1), T.int64(3072)), "float16"), permute_dims39: T.Buffer((T.int64(3072), T.int64(128256)), "float16"), compute: T.Buffer((T.int64(1), T.int64(128256)), "float16")):
        T.func_attr({"npu.numeric_policy": "fp16_storage_fp32_accum", "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul = T.alloc_buffer((T.int64(1), T.int64(128256)))
        for i0, i1, k in T.grid(T.int64(1), T.int64(128256), T.int64(3072)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(rms_norm8[v_i0, v_k], permute_dims39[v_k, v_i1])
                T.writes(matmul[v_i0, v_i1])
                with T.init():
                    matmul[v_i0, v_i1] = T.float32(0.0)
                matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + T.Cast("float32", rms_norm8[v_i0, v_k]) * T.Cast("float32", permute_dims39[v_k, v_i1])
        for i0, i1 in T.grid(T.int64(1), T.int64(128256)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(matmul[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.Cast("float16", matmul[v_i0, v_i1])

    @T.prim_func(private=True)
    def matmul7(rms_norm: T.Buffer((T.int64(7), T.int64(3072)), "float16"), permute_dims: T.Buffer((T.int64(3072), T.int64(3072)), "float16"), compute: T.Buffer((T.int64(7), T.int64(3072)), "float16")):
        T.func_attr({"npu.numeric_policy": "fp16_storage_fp32_accum", "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul = T.alloc_buffer((T.int64(7), T.int64(3072)))
        for i0, i1, k in T.grid(T.int64(7), T.int64(3072), T.int64(3072)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(rms_norm[v_i0, v_k], permute_dims[v_k, v_i1])
                T.writes(matmul[v_i0, v_i1])
                with T.init():
                    matmul[v_i0, v_i1] = T.float32(0.0)
                matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + T.Cast("float32", rms_norm[v_i0, v_k]) * T.Cast("float32", permute_dims[v_k, v_i1])
        for i0, i1 in T.grid(T.int64(7), T.int64(3072)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(matmul[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.Cast("float16", matmul[v_i0, v_i1])

    @T.prim_func(private=True)
    def matmul8(rms_norm: T.Buffer((T.int64(7), T.int64(3072)), "float16"), permute_dims1: T.Buffer((T.int64(3072), T.int64(1024)), "float16"), compute: T.Buffer((T.int64(7), T.int64(1024)), "float16")):
        T.func_attr({"npu.numeric_policy": "fp16_storage_fp32_accum", "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul = T.alloc_buffer((T.int64(7), T.int64(1024)))
        for i0, i1, k in T.grid(T.int64(7), T.int64(1024), T.int64(3072)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(rms_norm[v_i0, v_k], permute_dims1[v_k, v_i1])
                T.writes(matmul[v_i0, v_i1])
                with T.init():
                    matmul[v_i0, v_i1] = T.float32(0.0)
                matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + T.Cast("float32", rms_norm[v_i0, v_k]) * T.Cast("float32", permute_dims1[v_k, v_i1])
        for i0, i1 in T.grid(T.int64(7), T.int64(1024)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(matmul[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.Cast("float16", matmul[v_i0, v_i1])

    @T.prim_func(private=True)
    def matmul9(permute_dims3: T.Buffer((T.int64(24), T.int64(7), T.int64(128)), "float16"), permute_dims6: T.Buffer((T.int64(24), T.int64(128), T.int64(7)), "float16"), compute: T.Buffer((T.int64(24), T.int64(7), T.int64(7)), "float16")):
        T.func_attr({"npu.numeric_policy": "fp16_storage_fp32_accum", "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul = T.alloc_buffer((T.int64(24), T.int64(7), T.int64(7)))
        for i0, i1, i2, k in T.grid(T.int64(24), T.int64(7), T.int64(7), T.int64(128)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(permute_dims3[v_i0, v_i1, v_k], permute_dims6[v_i0, v_k, v_i2])
                T.writes(matmul[v_i0, v_i1, v_i2])
                with T.init():
                    matmul[v_i0, v_i1, v_i2] = T.float32(0.0)
                matmul[v_i0, v_i1, v_i2] = matmul[v_i0, v_i1, v_i2] + T.Cast("float32", permute_dims3[v_i0, v_i1, v_k]) * T.Cast("float32", permute_dims6[v_i0, v_k, v_i2])
        for i0, i1, i2 in T.grid(T.int64(24), T.int64(7), T.int64(7)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(matmul[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = T.Cast("float16", matmul[v_i0, v_i1, v_i2])

    @T.prim_func(private=True)
    def multiply(reshape14: T.Buffer((T.int64(1), T.int64(24), T.int64(128)), "float16"), cos: T.Buffer((T.int64(1), T.int64(1), T.int64(128)), "float16"), T_multiply: T.Buffer((T.int64(1), T.int64(24), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(24), T.int64(128)):
            with T.block("T_multiply"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(reshape14[v_ax0, v_ax1, v_ax2], cos[v_ax0, T.int64(0), v_ax2])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                T_multiply[v_ax0, v_ax1, v_ax2] = reshape14[v_ax0, v_ax1, v_ax2] * cos[v_ax0, T.int64(0), v_ax2]

    @T.prim_func(private=True)
    def multiply1(reshape15: T.Buffer((T.int64(1), T.int64(8), T.int64(128)), "float16"), cos: T.Buffer((T.int64(1), T.int64(1), T.int64(128)), "float16"), T_multiply: T.Buffer((T.int64(1), T.int64(8), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(8), T.int64(128)):
            with T.block("T_multiply"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(reshape15[v_ax0, v_ax1, v_ax2], cos[v_ax0, T.int64(0), v_ax2])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                T_multiply[v_ax0, v_ax1, v_ax2] = reshape15[v_ax0, v_ax1, v_ax2] * cos[v_ax0, T.int64(0), v_ax2]

    @T.prim_func(private=True)
    def multiply2(matmul23: T.Buffer((T.int64(24), T.int64(1), T.int64(10)), "float16"), T_multiply: T.Buffer((T.int64(24), T.int64(1), T.int64(10)), "float16")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(24), T.int64(1), T.int64(10)):
            with T.block("T_multiply"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(matmul23[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                T_multiply[v_ax0, v_ax1, v_ax2] = matmul23[v_ax0, v_ax1, v_ax2] * T.float16(0.08837890625)

    @T.prim_func(private=True)
    def multiply3(silu2: T.Buffer((T.int64(1), T.int64(8192)), "float16"), matmul27: T.Buffer((T.int64(1), T.int64(8192)), "float16"), T_multiply: T.Buffer((T.int64(1), T.int64(8192)), "float16")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(8192)):
            with T.block("T_multiply"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(silu2[v_ax0, v_ax1], matmul27[v_ax0, v_ax1])
                T.writes(T_multiply[v_ax0, v_ax1])
                T_multiply[v_ax0, v_ax1] = silu2[v_ax0, v_ax1] * matmul27[v_ax0, v_ax1]

    @T.prim_func(private=True)
    def multiply4(reshape: T.Buffer((T.int64(7), T.int64(24), T.int64(128)), "float16"), cos: T.Buffer((T.int64(7), T.int64(1), T.int64(128)), "float16"), T_multiply: T.Buffer((T.int64(7), T.int64(24), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(7), T.int64(24), T.int64(128)):
            with T.block("T_multiply"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(reshape[v_ax0, v_ax1, v_ax2], cos[v_ax0, T.int64(0), v_ax2])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                T_multiply[v_ax0, v_ax1, v_ax2] = reshape[v_ax0, v_ax1, v_ax2] * cos[v_ax0, T.int64(0), v_ax2]

    @T.prim_func(private=True)
    def multiply5(reshape1: T.Buffer((T.int64(7), T.int64(8), T.int64(128)), "float16"), cos: T.Buffer((T.int64(7), T.int64(1), T.int64(128)), "float16"), T_multiply: T.Buffer((T.int64(7), T.int64(8), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(7), T.int64(8), T.int64(128)):
            with T.block("T_multiply"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(reshape1[v_ax0, v_ax1, v_ax2], cos[v_ax0, T.int64(0), v_ax2])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                T_multiply[v_ax0, v_ax1, v_ax2] = reshape1[v_ax0, v_ax1, v_ax2] * cos[v_ax0, T.int64(0), v_ax2]

    @T.prim_func(private=True)
    def multiply6(matmul3: T.Buffer((T.int64(24), T.int64(7), T.int64(7)), "float16"), T_multiply: T.Buffer((T.int64(24), T.int64(7), T.int64(7)), "float16")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(24), T.int64(7), T.int64(7)):
            with T.block("T_multiply"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(matmul3[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                T_multiply[v_ax0, v_ax1, v_ax2] = matmul3[v_ax0, v_ax1, v_ax2] * T.float16(0.08837890625)

    @T.prim_func(private=True)
    def multiply7(silu: T.Buffer((T.int64(7), T.int64(8192)), "float16"), matmul7: T.Buffer((T.int64(7), T.int64(8192)), "float16"), T_multiply: T.Buffer((T.int64(7), T.int64(8192)), "float16")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(7), T.int64(8192)):
            with T.block("T_multiply"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(silu[v_ax0, v_ax1], matmul7[v_ax0, v_ax1])
                T.writes(T_multiply[v_ax0, v_ax1])
                T_multiply[v_ax0, v_ax1] = silu[v_ax0, v_ax1] * matmul7[v_ax0, v_ax1]

    @T.prim_func(private=True)
    def npu_rms_norm(input_embeds: T.Buffer((T.int64(1), T.int64(3072)), "float16"), layers_0_input_layernorm_weight2: T.Buffer((T.int64(3072),), "float16"), rms_norm: T.Buffer((T.int64(1), T.int64(3072)), "float16")):
        T.func_attr({"npu.numeric_policy": "fp16_storage_fp32_accum", "npu.rms_algorithm": "scaled_absmax_v1", "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        rms_absmax = T.alloc_buffer((T.int64(1),), "float16")
        rms_divisor = T.alloc_buffer((T.int64(1),), "float16")
        rms_unit = T.alloc_buffer((T.int64(1), T.int64(3072)), "float16")
        rms_scaled = T.alloc_buffer((T.int64(1), T.int64(3072)), "float16")
        rms_squared = T.alloc_buffer((T.int64(1), T.int64(3072)), "float16")
        rms_sum32 = T.alloc_buffer((T.int64(1),))
        rms_square_sum = T.alloc_buffer((T.int64(1),), "float16")
        rms_eps_ratio = T.alloc_buffer((T.int64(1),), "float16")
        rms_inv = T.alloc_buffer((T.int64(1),), "float16")
        rms_normalized = T.alloc_buffer((T.int64(1), T.int64(3072)), "float16")
        for i0, km in T.grid(T.int64(1), T.int64(3072)):
            with T.block("rms_absmax"):
                v_i0, v_km = T.axis.remap("SR", [i0, km])
                T.reads(input_embeds[v_i0, v_km])
                T.writes(rms_absmax[v_i0])
                with T.init():
                    rms_absmax[v_i0] = T.float16(-65504.0)
                rms_absmax[v_i0] = T.max(rms_absmax[v_i0], T.fabs(input_embeds[v_i0, v_km]))
        for i0 in range(T.int64(1)):
            with T.block("rms_divisor"):
                v_i0 = T.axis.spatial(T.int64(1), i0)
                T.reads(rms_absmax[v_i0])
                T.writes(rms_divisor[v_i0])
                rms_divisor[v_i0] = T.max(rms_absmax[v_i0], T.float16(0.0031622776601683794))
        for i0, i1 in T.grid(T.int64(1), T.int64(3072)):
            with T.block("rms_unit"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(input_embeds[v_i0, v_i1], rms_divisor[v_i0])
                T.writes(rms_unit[v_i0, v_i1])
                rms_unit[v_i0, v_i1] = input_embeds[v_i0, v_i1] / rms_divisor[v_i0]
        for i0, i1 in T.grid(T.int64(1), T.int64(3072)):
            with T.block("rms_scaled"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(rms_unit[v_i0, v_i1])
                T.writes(rms_scaled[v_i0, v_i1])
                rms_scaled[v_i0, v_i1] = rms_unit[v_i0, v_i1] * T.float16(0.018042195912175808)
        for i0, i1 in T.grid(T.int64(1), T.int64(3072)):
            with T.block("rms_squared"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(rms_scaled[v_i0, v_i1])
                T.writes(rms_squared[v_i0, v_i1])
                rms_squared[v_i0, v_i1] = rms_scaled[v_i0, v_i1] * rms_scaled[v_i0, v_i1]
        for i0, k in T.grid(T.int64(1), T.int64(3072)):
            with T.block("rms_sum32"):
                v_i0, v_k = T.axis.remap("SR", [i0, k])
                T.reads(rms_squared[v_i0, v_k])
                T.writes(rms_sum32[v_i0])
                with T.init():
                    rms_sum32[v_i0] = T.float32(0.0)
                rms_sum32[v_i0] = rms_sum32[v_i0] + T.Cast("float32", rms_squared[v_i0, v_k])
        for i0 in range(T.int64(1)):
            with T.block("rms_square_sum"):
                v_i0 = T.axis.spatial(T.int64(1), i0)
                T.reads(rms_sum32[v_i0])
                T.writes(rms_square_sum[v_i0])
                rms_square_sum[v_i0] = T.Cast("float16", rms_sum32[v_i0])
        for i0 in range(T.int64(1)):
            with T.block("rms_eps_ratio"):
                v_i0 = T.axis.spatial(T.int64(1), i0)
                T.reads(rms_divisor[v_i0])
                T.writes(rms_eps_ratio[v_i0])
                rms_eps_ratio[v_i0] = T.float16(0.0031622776601683794) / rms_divisor[v_i0]
        for i0 in range(T.int64(1)):
            with T.block("rms_inv"):
                v_i0 = T.axis.spatial(T.int64(1), i0)
                T.reads(rms_square_sum[v_i0], rms_eps_ratio[v_i0])
                T.writes(rms_inv[v_i0])
                rms_inv[v_i0] = T.float16(1.0) / T.sqrt(rms_square_sum[v_i0] + rms_eps_ratio[v_i0] * rms_eps_ratio[v_i0])
        for i0, i1 in T.grid(T.int64(1), T.int64(3072)):
            with T.block("rms_normalized"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(rms_unit[v_i0, v_i1], rms_inv[v_i0])
                T.writes(rms_normalized[v_i0, v_i1])
                rms_normalized[v_i0, v_i1] = rms_unit[v_i0, v_i1] * rms_inv[v_i0]
        for i0, i1 in T.grid(T.int64(1), T.int64(3072)):
            with T.block("rms_norm"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(rms_normalized[v_i0, v_i1], layers_0_input_layernorm_weight2[v_i1])
                T.writes(rms_norm[v_i0, v_i1])
                rms_norm[v_i0, v_i1] = rms_normalized[v_i0, v_i1] * layers_0_input_layernorm_weight2[v_i1]

    @T.prim_func(private=True)
    def npu_rms_norm1(input_embeds: T.Buffer((T.int64(7), T.int64(3072)), "float16"), layers_0_input_layernorm_weight: T.Buffer((T.int64(3072),), "float16"), rms_norm: T.Buffer((T.int64(7), T.int64(3072)), "float16")):
        T.func_attr({"npu.numeric_policy": "fp16_storage_fp32_accum", "npu.rms_algorithm": "scaled_absmax_v1", "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        rms_absmax = T.alloc_buffer((T.int64(7),), "float16")
        rms_divisor = T.alloc_buffer((T.int64(7),), "float16")
        rms_unit = T.alloc_buffer((T.int64(7), T.int64(3072)), "float16")
        rms_scaled = T.alloc_buffer((T.int64(7), T.int64(3072)), "float16")
        rms_squared = T.alloc_buffer((T.int64(7), T.int64(3072)), "float16")
        rms_sum32 = T.alloc_buffer((T.int64(7),))
        rms_square_sum = T.alloc_buffer((T.int64(7),), "float16")
        rms_eps_ratio = T.alloc_buffer((T.int64(7),), "float16")
        rms_inv = T.alloc_buffer((T.int64(7),), "float16")
        rms_normalized = T.alloc_buffer((T.int64(7), T.int64(3072)), "float16")
        for i0, km in T.grid(T.int64(7), T.int64(3072)):
            with T.block("rms_absmax"):
                v_i0, v_km = T.axis.remap("SR", [i0, km])
                T.reads(input_embeds[v_i0, v_km])
                T.writes(rms_absmax[v_i0])
                with T.init():
                    rms_absmax[v_i0] = T.float16(-65504.0)
                rms_absmax[v_i0] = T.max(rms_absmax[v_i0], T.fabs(input_embeds[v_i0, v_km]))
        for i0 in range(T.int64(7)):
            with T.block("rms_divisor"):
                v_i0 = T.axis.spatial(T.int64(7), i0)
                T.reads(rms_absmax[v_i0])
                T.writes(rms_divisor[v_i0])
                rms_divisor[v_i0] = T.max(rms_absmax[v_i0], T.float16(0.0031622776601683794))
        for i0, i1 in T.grid(T.int64(7), T.int64(3072)):
            with T.block("rms_unit"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(input_embeds[v_i0, v_i1], rms_divisor[v_i0])
                T.writes(rms_unit[v_i0, v_i1])
                rms_unit[v_i0, v_i1] = input_embeds[v_i0, v_i1] / rms_divisor[v_i0]
        for i0, i1 in T.grid(T.int64(7), T.int64(3072)):
            with T.block("rms_scaled"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(rms_unit[v_i0, v_i1])
                T.writes(rms_scaled[v_i0, v_i1])
                rms_scaled[v_i0, v_i1] = rms_unit[v_i0, v_i1] * T.float16(0.018042195912175808)
        for i0, i1 in T.grid(T.int64(7), T.int64(3072)):
            with T.block("rms_squared"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(rms_scaled[v_i0, v_i1])
                T.writes(rms_squared[v_i0, v_i1])
                rms_squared[v_i0, v_i1] = rms_scaled[v_i0, v_i1] * rms_scaled[v_i0, v_i1]
        for i0, k in T.grid(T.int64(7), T.int64(3072)):
            with T.block("rms_sum32"):
                v_i0, v_k = T.axis.remap("SR", [i0, k])
                T.reads(rms_squared[v_i0, v_k])
                T.writes(rms_sum32[v_i0])
                with T.init():
                    rms_sum32[v_i0] = T.float32(0.0)
                rms_sum32[v_i0] = rms_sum32[v_i0] + T.Cast("float32", rms_squared[v_i0, v_k])
        for i0 in range(T.int64(7)):
            with T.block("rms_square_sum"):
                v_i0 = T.axis.spatial(T.int64(7), i0)
                T.reads(rms_sum32[v_i0])
                T.writes(rms_square_sum[v_i0])
                rms_square_sum[v_i0] = T.Cast("float16", rms_sum32[v_i0])
        for i0 in range(T.int64(7)):
            with T.block("rms_eps_ratio"):
                v_i0 = T.axis.spatial(T.int64(7), i0)
                T.reads(rms_divisor[v_i0])
                T.writes(rms_eps_ratio[v_i0])
                rms_eps_ratio[v_i0] = T.float16(0.0031622776601683794) / rms_divisor[v_i0]
        for i0 in range(T.int64(7)):
            with T.block("rms_inv"):
                v_i0 = T.axis.spatial(T.int64(7), i0)
                T.reads(rms_square_sum[v_i0], rms_eps_ratio[v_i0])
                T.writes(rms_inv[v_i0])
                rms_inv[v_i0] = T.float16(1.0) / T.sqrt(rms_square_sum[v_i0] + rms_eps_ratio[v_i0] * rms_eps_ratio[v_i0])
        for i0, i1 in T.grid(T.int64(7), T.int64(3072)):
            with T.block("rms_normalized"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(rms_unit[v_i0, v_i1], rms_inv[v_i0])
                T.writes(rms_normalized[v_i0, v_i1])
                rms_normalized[v_i0, v_i1] = rms_unit[v_i0, v_i1] * rms_inv[v_i0]
        for i0, i1 in T.grid(T.int64(7), T.int64(3072)):
            with T.block("rms_norm"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(rms_normalized[v_i0, v_i1], layers_0_input_layernorm_weight[v_i1])
                T.writes(rms_norm[v_i0, v_i1])
                rms_norm[v_i0, v_i1] = rms_normalized[v_i0, v_i1] * layers_0_input_layernorm_weight[v_i1]

    @T.prim_func(private=True)
    def repeat(concat8: T.Buffer((T.int64(8), T.int64(128), T.int64(10)), "float16"), T_repeat: T.Buffer((T.int64(24), T.int64(128), T.int64(10)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(24), T.int64(128), T.int64(10)):
            with T.block("T_repeat"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(concat8[v_ax0 // T.int64(3), v_ax1, v_ax2])
                T.writes(T_repeat[v_ax0, v_ax1, v_ax2])
                T_repeat[v_ax0, v_ax1, v_ax2] = concat8[v_ax0 // T.int64(3), v_ax1, v_ax2]

    @T.prim_func(private=True)
    def repeat1(concat9: T.Buffer((T.int64(8), T.int64(10), T.int64(128)), "float16"), T_repeat: T.Buffer((T.int64(24), T.int64(10), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(24), T.int64(10), T.int64(128)):
            with T.block("T_repeat"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(concat9[v_ax0 // T.int64(3), v_ax1, v_ax2])
                T.writes(T_repeat[v_ax0, v_ax1, v_ax2])
                T_repeat[v_ax0, v_ax1, v_ax2] = concat9[v_ax0 // T.int64(3), v_ax1, v_ax2]

    @T.prim_func(private=True)
    def repeat2(add1: T.Buffer((T.int64(7), T.int64(8), T.int64(128)), "float16"), T_repeat: T.Buffer((T.int64(7), T.int64(24), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(7), T.int64(24), T.int64(128)):
            with T.block("T_repeat"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(add1[v_ax0, v_ax1 // T.int64(3), v_ax2])
                T.writes(T_repeat[v_ax0, v_ax1, v_ax2])
                T_repeat[v_ax0, v_ax1, v_ax2] = add1[v_ax0, v_ax1 // T.int64(3), v_ax2]

    @T.prim_func(private=True)
    def reshape(k_cache: T.Buffer((T.int64(1), T.int64(8), T.int64(128), T.int64(9)), "float16"), T_reshape: T.Buffer((T.int64(8), T.int64(128), T.int64(9)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(8), T.int64(128), T.int64(9)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(k_cache[T.int64(0), ((v_ax2 // T.int64(9) + v_ax1) // T.int64(128) + v_ax0) % T.int64(8), (v_ax2 // T.int64(9) + v_ax1) % T.int64(128), v_ax2 % T.int64(9)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = k_cache[T.int64(0), ((v_ax2 // T.int64(9) + v_ax1) // T.int64(128) + v_ax0) % T.int64(8), (v_ax2 // T.int64(9) + v_ax1) % T.int64(128), v_ax2 % T.int64(9)]

    @T.prim_func(private=True)
    def reshape1(v_cache: T.Buffer((T.int64(1), T.int64(8), T.int64(9), T.int64(128)), "float16"), T_reshape: T.Buffer((T.int64(8), T.int64(9), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(8), T.int64(9), T.int64(128)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(v_cache[T.int64(0), ((v_ax2 // T.int64(128) + v_ax1) // T.int64(9) + v_ax0) % T.int64(8), (v_ax2 // T.int64(128) + v_ax1) % T.int64(9), v_ax2 % T.int64(128)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = v_cache[T.int64(0), ((v_ax2 // T.int64(128) + v_ax1) // T.int64(9) + v_ax0) % T.int64(8), (v_ax2 // T.int64(128) + v_ax1) % T.int64(9), v_ax2 % T.int64(128)]

    @T.prim_func(private=True)
    def reshape2(matmul20: T.Buffer((T.int64(1), T.int64(3072)), "float16"), T_reshape: T.Buffer((T.int64(1), T.int64(24), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(24), T.int64(128)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(matmul20[T.int64(0), (v_ax1 * T.int64(128) + v_ax2) % T.int64(3072)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = matmul20[T.int64(0), (v_ax1 * T.int64(128) + v_ax2) % T.int64(3072)]

    @T.prim_func(private=True)
    def reshape3(matmul21: T.Buffer((T.int64(1), T.int64(1024)), "float16"), T_reshape: T.Buffer((T.int64(1), T.int64(8), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(8), T.int64(128)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(matmul21[T.int64(0), (v_ax1 * T.int64(128) + v_ax2) % T.int64(1024)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = matmul21[T.int64(0), (v_ax1 * T.int64(128) + v_ax2) % T.int64(1024)]

    @T.prim_func(private=True)
    def reshape4(permute_dims34: T.Buffer((T.int64(1), T.int64(24), T.int64(128)), "float16"), T_reshape: T.Buffer((T.int64(1), T.int64(3072)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(3072)):
            with T.block("T_reshape"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(permute_dims34[T.int64(0), v_ax1 % T.int64(3072) // T.int64(128), v_ax1 % T.int64(128)])
                T.writes(T_reshape[v_ax0, v_ax1])
                T_reshape[v_ax0, v_ax1] = permute_dims34[T.int64(0), v_ax1 % T.int64(3072) // T.int64(128), v_ax1 % T.int64(128)]

    @T.prim_func(private=True)
    def reshape5(matmul: T.Buffer((T.int64(7), T.int64(3072)), "float16"), T_reshape: T.Buffer((T.int64(7), T.int64(24), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(7), T.int64(24), T.int64(128)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(matmul[((v_ax1 * T.int64(128) + v_ax2) // T.int64(3072) + v_ax0) % T.int64(7), (v_ax1 * T.int64(128) + v_ax2) % T.int64(3072)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = matmul[((v_ax1 * T.int64(128) + v_ax2) // T.int64(3072) + v_ax0) % T.int64(7), (v_ax1 * T.int64(128) + v_ax2) % T.int64(3072)]

    @T.prim_func(private=True)
    def reshape6(matmul1: T.Buffer((T.int64(7), T.int64(1024)), "float16"), T_reshape: T.Buffer((T.int64(7), T.int64(8), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(7), T.int64(8), T.int64(128)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(matmul1[((v_ax1 * T.int64(128) + v_ax2) // T.int64(1024) + v_ax0) % T.int64(7), (v_ax1 * T.int64(128) + v_ax2) % T.int64(1024)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = matmul1[((v_ax1 * T.int64(128) + v_ax2) // T.int64(1024) + v_ax0) % T.int64(7), (v_ax1 * T.int64(128) + v_ax2) % T.int64(1024)]

    @T.prim_func(private=True)
    def reshape7(repeat: T.Buffer((T.int64(7), T.int64(24), T.int64(128)), "float16"), T_reshape: T.Buffer((T.int64(7), T.int64(24), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(7), T.int64(24), T.int64(128)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(repeat[((v_ax2 // T.int64(128) + v_ax1) // T.int64(24) + v_ax0) % T.int64(7), (v_ax2 // T.int64(128) + v_ax1) % T.int64(24), v_ax2 % T.int64(128)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = repeat[((v_ax2 // T.int64(128) + v_ax1) // T.int64(24) + v_ax0) % T.int64(7), (v_ax2 // T.int64(128) + v_ax1) % T.int64(24), v_ax2 % T.int64(128)]

    @T.prim_func(private=True)
    def reshape8(permute_dims7: T.Buffer((T.int64(7), T.int64(24), T.int64(128)), "float16"), T_reshape: T.Buffer((T.int64(7), T.int64(3072)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(7), T.int64(3072)):
            with T.block("T_reshape"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(permute_dims7[(v_ax1 // T.int64(3072) + v_ax0) % T.int64(7), v_ax1 % T.int64(3072) // T.int64(128), v_ax1 % T.int64(128)])
                T.writes(T_reshape[v_ax0, v_ax1])
                T_reshape[v_ax0, v_ax1] = permute_dims7[(v_ax1 // T.int64(3072) + v_ax0) % T.int64(7), v_ax1 % T.int64(3072) // T.int64(128), v_ax1 % T.int64(128)]

    @T.prim_func(private=True)
    def silu(matmul26: T.Buffer((T.int64(1), T.int64(8192)), "float16"), T_multiply: T.Buffer((T.int64(1), T.int64(8192)), "float16")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        compute = T.alloc_buffer((T.int64(1), T.int64(8192)), "float16")
        for i0, i1 in T.grid(T.int64(1), T.int64(8192)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(matmul26[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.sigmoid(matmul26[v_i0, v_i1])
        for ax0, ax1 in T.grid(T.int64(1), T.int64(8192)):
            with T.block("T_multiply"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(matmul26[v_ax0, v_ax1], compute[v_ax0, v_ax1])
                T.writes(T_multiply[v_ax0, v_ax1])
                T_multiply[v_ax0, v_ax1] = matmul26[v_ax0, v_ax1] * compute[v_ax0, v_ax1]

    @T.prim_func(private=True)
    def silu1(matmul6: T.Buffer((T.int64(7), T.int64(8192)), "float16"), T_multiply: T.Buffer((T.int64(7), T.int64(8192)), "float16")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        compute = T.alloc_buffer((T.int64(7), T.int64(8192)), "float16")
        for i0, i1 in T.grid(T.int64(7), T.int64(8192)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(matmul6[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.sigmoid(matmul6[v_i0, v_i1])
        for ax0, ax1 in T.grid(T.int64(7), T.int64(8192)):
            with T.block("T_multiply"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(matmul6[v_ax0, v_ax1], compute[v_ax0, v_ax1])
                T.writes(T_multiply[v_ax0, v_ax1])
                T_multiply[v_ax0, v_ax1] = matmul6[v_ax0, v_ax1] * compute[v_ax0, v_ax1]

    @T.prim_func(private=True)
    def softmax(add12: T.Buffer((T.int64(24), T.int64(1), T.int64(10)), "float16"), T_softmax_norm: T.Buffer((T.int64(24), T.int64(1), T.int64(10)), "float16")):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        T_softmax_maxelem = T.alloc_buffer((T.int64(24), T.int64(1)), "float16")
        T_softmax_exp = T.alloc_buffer((T.int64(24), T.int64(1), T.int64(10)), "float16")
        softmax_sum32 = T.alloc_buffer((T.int64(24), T.int64(1)))
        T_softmax_expsum = T.alloc_buffer((T.int64(24), T.int64(1)), "float16")
        for i0, i1, k in T.grid(T.int64(24), T.int64(1), T.int64(10)):
            with T.block("T_softmax_maxelem"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(add12[v_i0, v_i1, v_k])
                T.writes(T_softmax_maxelem[v_i0, v_i1])
                with T.init():
                    T_softmax_maxelem[v_i0, v_i1] = T.float16(-65504.0)
                T_softmax_maxelem[v_i0, v_i1] = T.max(T_softmax_maxelem[v_i0, v_i1], add12[v_i0, v_i1, v_k])
        for i0, i1, i2 in T.grid(T.int64(24), T.int64(1), T.int64(10)):
            with T.block("T_softmax_exp"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(add12[v_i0, v_i1, v_i2], T_softmax_maxelem[v_i0, v_i1])
                T.writes(T_softmax_exp[v_i0, v_i1, v_i2])
                T_softmax_exp[v_i0, v_i1, v_i2] = T.exp(add12[v_i0, v_i1, v_i2] - T_softmax_maxelem[v_i0, v_i1])
        for i0, i1, k in T.grid(T.int64(24), T.int64(1), T.int64(10)):
            with T.block("softmax_sum32"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(T_softmax_exp[v_i0, v_i1, v_k])
                T.writes(softmax_sum32[v_i0, v_i1])
                with T.init():
                    softmax_sum32[v_i0, v_i1] = T.float32(0.0)
                softmax_sum32[v_i0, v_i1] = softmax_sum32[v_i0, v_i1] + T.Cast("float32", T_softmax_exp[v_i0, v_i1, v_k])
        for i0, i1 in T.grid(T.int64(24), T.int64(1)):
            with T.block("T_softmax_expsum"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(softmax_sum32[v_i0, v_i1])
                T.writes(T_softmax_expsum[v_i0, v_i1])
                T_softmax_expsum[v_i0, v_i1] = T.Cast("float16", softmax_sum32[v_i0, v_i1])
        for i0, i1, i2 in T.grid(T.int64(24), T.int64(1), T.int64(10)):
            with T.block("T_softmax_norm"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(T_softmax_exp[v_i0, v_i1, v_i2], T_softmax_expsum[v_i0, v_i1])
                T.writes(T_softmax_norm[v_i0, v_i1, v_i2])
                T_softmax_norm[v_i0, v_i1, v_i2] = T_softmax_exp[v_i0, v_i1, v_i2] / T_softmax_expsum[v_i0, v_i1]

    @T.prim_func(private=True)
    def softmax1(add2: T.Buffer((T.int64(24), T.int64(7), T.int64(7)), "float16"), T_softmax_norm: T.Buffer((T.int64(24), T.int64(7), T.int64(7)), "float16")):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        T_softmax_maxelem = T.alloc_buffer((T.int64(24), T.int64(7)), "float16")
        T_softmax_exp = T.alloc_buffer((T.int64(24), T.int64(7), T.int64(7)), "float16")
        softmax_sum32 = T.alloc_buffer((T.int64(24), T.int64(7)))
        T_softmax_expsum = T.alloc_buffer((T.int64(24), T.int64(7)), "float16")
        for i0, i1, k in T.grid(T.int64(24), T.int64(7), T.int64(7)):
            with T.block("T_softmax_maxelem"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(add2[v_i0, v_i1, v_k])
                T.writes(T_softmax_maxelem[v_i0, v_i1])
                with T.init():
                    T_softmax_maxelem[v_i0, v_i1] = T.float16(-65504.0)
                T_softmax_maxelem[v_i0, v_i1] = T.max(T_softmax_maxelem[v_i0, v_i1], add2[v_i0, v_i1, v_k])
        for i0, i1, i2 in T.grid(T.int64(24), T.int64(7), T.int64(7)):
            with T.block("T_softmax_exp"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(add2[v_i0, v_i1, v_i2], T_softmax_maxelem[v_i0, v_i1])
                T.writes(T_softmax_exp[v_i0, v_i1, v_i2])
                T_softmax_exp[v_i0, v_i1, v_i2] = T.exp(add2[v_i0, v_i1, v_i2] - T_softmax_maxelem[v_i0, v_i1])
        for i0, i1, k in T.grid(T.int64(24), T.int64(7), T.int64(7)):
            with T.block("softmax_sum32"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(T_softmax_exp[v_i0, v_i1, v_k])
                T.writes(softmax_sum32[v_i0, v_i1])
                with T.init():
                    softmax_sum32[v_i0, v_i1] = T.float32(0.0)
                softmax_sum32[v_i0, v_i1] = softmax_sum32[v_i0, v_i1] + T.Cast("float32", T_softmax_exp[v_i0, v_i1, v_k])
        for i0, i1 in T.grid(T.int64(24), T.int64(7)):
            with T.block("T_softmax_expsum"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(softmax_sum32[v_i0, v_i1])
                T.writes(T_softmax_expsum[v_i0, v_i1])
                T_softmax_expsum[v_i0, v_i1] = T.Cast("float16", softmax_sum32[v_i0, v_i1])
        for i0, i1, i2 in T.grid(T.int64(24), T.int64(7), T.int64(7)):
            with T.block("T_softmax_norm"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(T_softmax_exp[v_i0, v_i1, v_i2], T_softmax_expsum[v_i0, v_i1])
                T.writes(T_softmax_norm[v_i0, v_i1, v_i2])
                T_softmax_norm[v_i0, v_i1, v_i2] = T_softmax_exp[v_i0, v_i1, v_i2] / T_softmax_expsum[v_i0, v_i1]

    @T.prim_func(private=True)
    def split(reshape14: T.Buffer((T.int64(1), T.int64(24), T.int64(128)), "float16"), T_split: T.Buffer((T.int64(1), T.int64(24), T.int64(64)), "float16"), T_split_1: T.Buffer((T.int64(1), T.int64(24), T.int64(64)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(24), T.int64(64)):
            with T.block("T_split"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(reshape14[v_ax0, v_ax1, v_ax2])
                T.writes(T_split[v_ax0, v_ax1, v_ax2])
                T_split[v_ax0, v_ax1, v_ax2] = reshape14[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(24), T.int64(64)):
            with T.block("T_split_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(reshape14[v_ax0, v_ax1, v_ax2 + T.int64(64)])
                T.writes(T_split_1[v_ax0, v_ax1, v_ax2])
                T_split_1[v_ax0, v_ax1, v_ax2] = reshape14[v_ax0, v_ax1, v_ax2 + T.int64(64)]

    @T.prim_func(private=True)
    def split1(reshape15: T.Buffer((T.int64(1), T.int64(8), T.int64(128)), "float16"), T_split: T.Buffer((T.int64(1), T.int64(8), T.int64(64)), "float16"), T_split_1: T.Buffer((T.int64(1), T.int64(8), T.int64(64)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(8), T.int64(64)):
            with T.block("T_split"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(reshape15[v_ax0, v_ax1, v_ax2])
                T.writes(T_split[v_ax0, v_ax1, v_ax2])
                T_split[v_ax0, v_ax1, v_ax2] = reshape15[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(8), T.int64(64)):
            with T.block("T_split_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(reshape15[v_ax0, v_ax1, v_ax2 + T.int64(64)])
                T.writes(T_split_1[v_ax0, v_ax1, v_ax2])
                T_split_1[v_ax0, v_ax1, v_ax2] = reshape15[v_ax0, v_ax1, v_ax2 + T.int64(64)]

    @T.prim_func(private=True)
    def split2(reshape: T.Buffer((T.int64(7), T.int64(24), T.int64(128)), "float16"), T_split: T.Buffer((T.int64(7), T.int64(24), T.int64(64)), "float16"), T_split_1: T.Buffer((T.int64(7), T.int64(24), T.int64(64)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(7), T.int64(24), T.int64(64)):
            with T.block("T_split"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(reshape[v_ax0, v_ax1, v_ax2])
                T.writes(T_split[v_ax0, v_ax1, v_ax2])
                T_split[v_ax0, v_ax1, v_ax2] = reshape[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(7), T.int64(24), T.int64(64)):
            with T.block("T_split_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(reshape[v_ax0, v_ax1, v_ax2 + T.int64(64)])
                T.writes(T_split_1[v_ax0, v_ax1, v_ax2])
                T_split_1[v_ax0, v_ax1, v_ax2] = reshape[v_ax0, v_ax1, v_ax2 + T.int64(64)]

    @T.prim_func(private=True)
    def split3(reshape1: T.Buffer((T.int64(7), T.int64(8), T.int64(128)), "float16"), T_split: T.Buffer((T.int64(7), T.int64(8), T.int64(64)), "float16"), T_split_1: T.Buffer((T.int64(7), T.int64(8), T.int64(64)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(7), T.int64(8), T.int64(64)):
            with T.block("T_split"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(reshape1[v_ax0, v_ax1, v_ax2])
                T.writes(T_split[v_ax0, v_ax1, v_ax2])
                T_split[v_ax0, v_ax1, v_ax2] = reshape1[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(7), T.int64(8), T.int64(64)):
            with T.block("T_split_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(reshape1[v_ax0, v_ax1, v_ax2 + T.int64(64)])
                T.writes(T_split_1[v_ax0, v_ax1, v_ax2])
                T_split_1[v_ax0, v_ax1, v_ax2] = reshape1[v_ax0, v_ax1, v_ax2 + T.int64(64)]

    @T.prim_func(private=True)
    def tir_negative(split_14: T.Buffer((T.int64(1), T.int64(24), T.int64(64)), "float16"), compute: T.Buffer((T.int64(1), T.int64(24), T.int64(64)), "float16")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(24), T.int64(64)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(split_14[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = split_14[v_i0, v_i1, v_i2] * T.float16(-1.0)

    @T.prim_func(private=True)
    def tir_negative1(split_15: T.Buffer((T.int64(1), T.int64(8), T.int64(64)), "float16"), compute: T.Buffer((T.int64(1), T.int64(8), T.int64(64)), "float16")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(8), T.int64(64)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(split_15[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = split_15[v_i0, v_i1, v_i2] * T.float16(-1.0)

    @T.prim_func(private=True)
    def tir_negative2(split_1: T.Buffer((T.int64(7), T.int64(24), T.int64(64)), "float16"), compute: T.Buffer((T.int64(7), T.int64(24), T.int64(64)), "float16")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2 in T.grid(T.int64(7), T.int64(24), T.int64(64)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(split_1[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = split_1[v_i0, v_i1, v_i2] * T.float16(-1.0)

    @T.prim_func(private=True)
    def tir_negative3(split_11: T.Buffer((T.int64(7), T.int64(8), T.int64(64)), "float16"), compute: T.Buffer((T.int64(7), T.int64(8), T.int64(64)), "float16")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2 in T.grid(T.int64(7), T.int64(8), T.int64(64)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(split_11[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = split_11[v_i0, v_i1, v_i2] * T.float16(-1.0)

    @T.prim_func(private=True)
    def transpose(add11: T.Buffer((T.int64(1), T.int64(8), T.int64(128)), "float16"), T_transpose: T.Buffer((T.int64(8), T.int64(128), T.int64(1)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(8), T.int64(128), T.int64(1)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(add11[v_ax2, v_ax0, v_ax1])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2])
                T_transpose[v_ax0, v_ax1, v_ax2] = add11[v_ax2, v_ax0, v_ax1]

    @T.prim_func(private=True)
    def transpose1(reshape16: T.Buffer((T.int64(1), T.int64(8), T.int64(128)), "float16"), T_transpose: T.Buffer((T.int64(8), T.int64(1), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(8), T.int64(1), T.int64(128)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(reshape16[v_ax1, v_ax0, v_ax2])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2])
                T_transpose[v_ax0, v_ax1, v_ax2] = reshape16[v_ax1, v_ax0, v_ax2]

    @T.prim_func(private=True)
    def transpose10(permute_dims4: T.Buffer((T.int64(24), T.int64(7), T.int64(128)), "float16"), T_transpose: T.Buffer((T.int64(24), T.int64(128), T.int64(7)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(24), T.int64(128), T.int64(7)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(permute_dims4[v_ax0, v_ax2, v_ax1])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2])
                T_transpose[v_ax0, v_ax1, v_ax2] = permute_dims4[v_ax0, v_ax2, v_ax1]

    @T.prim_func(private=True)
    def transpose11(matmul4: T.Buffer((T.int64(24), T.int64(7), T.int64(128)), "float16"), T_transpose: T.Buffer((T.int64(7), T.int64(24), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(7), T.int64(24), T.int64(128)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(matmul4[v_ax1, v_ax0, v_ax2])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2])
                T_transpose[v_ax0, v_ax1, v_ax2] = matmul4[v_ax1, v_ax0, v_ax2]

    @T.prim_func(private=True)
    def transpose12(add6: T.Buffer((T.int64(7), T.int64(8), T.int64(128)), "float16"), T_transpose: T.Buffer((T.int64(8), T.int64(128), T.int64(7)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(8), T.int64(128), T.int64(7)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(add6[v_ax2, v_ax0, v_ax1])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2])
                T_transpose[v_ax0, v_ax1, v_ax2] = add6[v_ax2, v_ax0, v_ax1]

    @T.prim_func(private=True)
    def transpose13(reshape8: T.Buffer((T.int64(7), T.int64(8), T.int64(128)), "float16"), T_transpose: T.Buffer((T.int64(8), T.int64(7), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(8), T.int64(7), T.int64(128)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(reshape8[v_ax1, v_ax0, v_ax2])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2])
                T_transpose[v_ax0, v_ax1, v_ax2] = reshape8[v_ax1, v_ax0, v_ax2]

    @T.prim_func(private=True)
    def transpose2(add10: T.Buffer((T.int64(1), T.int64(24), T.int64(128)), "float16"), T_transpose: T.Buffer((T.int64(24), T.int64(1), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(24), T.int64(1), T.int64(128)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(add10[v_ax1, v_ax0, v_ax2])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2])
                T_transpose[v_ax0, v_ax1, v_ax2] = add10[v_ax1, v_ax0, v_ax2]

    @T.prim_func(private=True)
    def transpose3(matmul24: T.Buffer((T.int64(24), T.int64(1), T.int64(128)), "float16"), T_transpose: T.Buffer((T.int64(1), T.int64(24), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(24), T.int64(128)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(matmul24[v_ax1, v_ax0, v_ax2])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2])
                T_transpose[v_ax0, v_ax1, v_ax2] = matmul24[v_ax1, v_ax0, v_ax2]

    @T.prim_func(private=True)
    def transpose4(layers_0_self_attn_q_proj_weight2: T.Buffer((T.int64(3072), T.int64(3072)), "float16"), T_transpose: T.Buffer((T.int64(3072), T.int64(3072)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(3072), T.int64(3072)):
            with T.block("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(layers_0_self_attn_q_proj_weight2[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = layers_0_self_attn_q_proj_weight2[v_ax1, v_ax0]

    @T.prim_func(private=True)
    def transpose5(layers_0_self_attn_k_proj_weight2: T.Buffer((T.int64(1024), T.int64(3072)), "float16"), T_transpose: T.Buffer((T.int64(3072), T.int64(1024)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(3072), T.int64(1024)):
            with T.block("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(layers_0_self_attn_k_proj_weight2[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = layers_0_self_attn_k_proj_weight2[v_ax1, v_ax0]

    @T.prim_func(private=True)
    def transpose6(layers_0_mlp_gate_proj_weight2: T.Buffer((T.int64(8192), T.int64(3072)), "float16"), T_transpose: T.Buffer((T.int64(3072), T.int64(8192)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(3072), T.int64(8192)):
            with T.block("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(layers_0_mlp_gate_proj_weight2[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = layers_0_mlp_gate_proj_weight2[v_ax1, v_ax0]

    @T.prim_func(private=True)
    def transpose7(layers_0_mlp_down_proj_weight2: T.Buffer((T.int64(3072), T.int64(8192)), "float16"), T_transpose: T.Buffer((T.int64(8192), T.int64(3072)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(8192), T.int64(3072)):
            with T.block("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(layers_0_mlp_down_proj_weight2[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = layers_0_mlp_down_proj_weight2[v_ax1, v_ax0]

    @T.prim_func(private=True)
    def transpose8(lm_head_weight2: T.Buffer((T.int64(128256), T.int64(3072)), "float16"), T_transpose: T.Buffer((T.int64(3072), T.int64(128256)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(3072), T.int64(128256)):
            with T.block("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lm_head_weight2[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = lm_head_weight2[v_ax1, v_ax0]

    @T.prim_func(private=True)
    def transpose9(add: T.Buffer((T.int64(7), T.int64(24), T.int64(128)), "float16"), T_transpose: T.Buffer((T.int64(24), T.int64(7), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(24), T.int64(7), T.int64(128)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(add[v_ax1, v_ax0, v_ax2])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2])
                T_transpose[v_ax0, v_ax1, v_ax2] = add[v_ax1, v_ax0, v_ax2]

    @R.function
    def decode(input_embeds: R.Tensor((1, 3072), dtype="float16"), cos: R.Tensor((1, 1, 128), dtype="float16"), sin: R.Tensor((1, 1, 128), dtype="float16"), k_cache: R.Tensor((1, 8, 128, 9), dtype="float16"), v_cache: R.Tensor((1, 8, 9, 128), dtype="float16"), mask: R.Tensor((1, 1, 10), dtype="float16"), layers_0_input_layernorm_weight2: R.Tensor((3072,), dtype="float16"), layers_0_post_attention_layernorm_weight2: R.Tensor((3072,), dtype="float16"), norm_weight2: R.Tensor((3072,), dtype="float16"), permute_dims28: R.Tensor((3072, 3072), dtype="float16"), permute_dims29: R.Tensor((3072, 1024), dtype="float16"), permute_dims30: R.Tensor((3072, 1024), dtype="float16"), permute_dims35: R.Tensor((3072, 3072), dtype="float16"), permute_dims36: R.Tensor((3072, 8192), dtype="float16"), permute_dims37: R.Tensor((3072, 8192), dtype="float16"), permute_dims38: R.Tensor((8192, 3072), dtype="float16"), permute_dims39: R.Tensor((3072, 128256), dtype="float16")) -> R.Tuple(R.Tensor((1, 128256), dtype="float16"), R.Tensor((1, 8, 128, 1), dtype="float16"), R.Tensor((1, 8, 1, 128), dtype="float16")):
        R.func_attr({"npu.cache_append": "{\"entries\": [{\"axis\": 3, \"output\": 1, \"parameter\": \"k_cache\"}, {\"axis\": 2, \"output\": 2, \"parameter\": \"v_cache\"}], \"mask\": \"mask\", \"version\": 1}", "num_input": 6})
        cls = Module
        with R.dataflow():
            reshape12 = R.call_tir(cls.reshape, (k_cache,), out_sinfo=R.Tensor((8, 128, 9), dtype="float16"))
            reshape13 = R.call_tir(cls.reshape1, (v_cache,), out_sinfo=R.Tensor((8, 9, 128), dtype="float16"))
            rms_norm6 = R.call_tir(cls.npu_rms_norm, (input_embeds, layers_0_input_layernorm_weight2), out_sinfo=R.Tensor((1, 3072), dtype="float16"))
            matmul20 = R.call_tir(cls.matmul, (rms_norm6, permute_dims28), out_sinfo=R.Tensor((1, 3072), dtype="float16"))
            reshape14 = R.call_tir(cls.reshape2, (matmul20,), out_sinfo=R.Tensor((1, 24, 128), dtype="float16"))
            matmul21 = R.call_tir(cls.matmul1, (rms_norm6, permute_dims29), out_sinfo=R.Tensor((1, 1024), dtype="float16"))
            reshape15 = R.call_tir(cls.reshape3, (matmul21,), out_sinfo=R.Tensor((1, 8, 128), dtype="float16"))
            matmul22 = R.call_tir(cls.matmul1, (rms_norm6, permute_dims30), out_sinfo=R.Tensor((1, 1024), dtype="float16"))
            reshape16 = R.call_tir(cls.reshape3, (matmul22,), out_sinfo=R.Tensor((1, 8, 128), dtype="float16"))
            mul12 = R.call_tir(cls.multiply, (reshape14, cos), out_sinfo=R.Tensor((1, 24, 128), dtype="float16"))
            split4 = R.call_tir(cls.split, (reshape14,), out_sinfo=[R.Tensor((1, 24, 64), dtype="float16"), R.Tensor((1, 24, 64), dtype="float16")])
            split_04: R.Tensor((1, 24, 64), dtype="float16") = split4[0]
            split_14: R.Tensor((1, 24, 64), dtype="float16") = split4[1]
            neg4 = R.call_tir(cls.tir_negative, (split_14,), out_sinfo=R.Tensor((1, 24, 64), dtype="float16"))
            concat6 = R.call_tir(cls.concatenate, (neg4, split_04), out_sinfo=R.Tensor((1, 24, 128), dtype="float16"))
            mul13 = R.call_tir(cls.multiply, (concat6, sin), out_sinfo=R.Tensor((1, 24, 128), dtype="float16"))
            add10 = R.call_tir(cls.add, (mul12, mul13), out_sinfo=R.Tensor((1, 24, 128), dtype="float16"))
            mul14 = R.call_tir(cls.multiply1, (reshape15, cos), out_sinfo=R.Tensor((1, 8, 128), dtype="float16"))
            split5 = R.call_tir(cls.split1, (reshape15,), out_sinfo=[R.Tensor((1, 8, 64), dtype="float16"), R.Tensor((1, 8, 64), dtype="float16")])
            split_05: R.Tensor((1, 8, 64), dtype="float16") = split5[0]
            split_15: R.Tensor((1, 8, 64), dtype="float16") = split5[1]
            neg5 = R.call_tir(cls.tir_negative1, (split_15,), out_sinfo=R.Tensor((1, 8, 64), dtype="float16"))
            concat7 = R.call_tir(cls.concatenate1, (neg5, split_05), out_sinfo=R.Tensor((1, 8, 128), dtype="float16"))
            mul15 = R.call_tir(cls.multiply1, (concat7, sin), out_sinfo=R.Tensor((1, 8, 128), dtype="float16"))
            add11 = R.call_tir(cls.add1, (mul14, mul15), out_sinfo=R.Tensor((1, 8, 128), dtype="float16"))
            permute_dims31 = R.call_tir(cls.transpose, (add11,), out_sinfo=R.Tensor((8, 128, 1), dtype="float16"))
            permute_dims32 = R.call_tir(cls.transpose1, (reshape16,), out_sinfo=R.Tensor((8, 1, 128), dtype="float16"))
            concat8 = R.call_tir(cls.concatenate2, (reshape12, permute_dims31), out_sinfo=R.Tensor((8, 128, 10), dtype="float16"))
            concat9 = R.call_tir(cls.concatenate3, (reshape13, permute_dims32), out_sinfo=R.Tensor((8, 10, 128), dtype="float16"))
            repeat4 = R.call_tir(cls.repeat, (concat8,), out_sinfo=R.Tensor((24, 128, 10), dtype="float16"))
            repeat5 = R.call_tir(cls.repeat1, (concat9,), out_sinfo=R.Tensor((24, 10, 128), dtype="float16"))
            permute_dims33 = R.call_tir(cls.transpose2, (add10,), out_sinfo=R.Tensor((24, 1, 128), dtype="float16"))
            matmul23 = R.call_tir(cls.matmul2, (permute_dims33, repeat4), out_sinfo=R.Tensor((24, 1, 10), dtype="float16"))
            mul16 = R.call_tir(cls.multiply2, (matmul23,), out_sinfo=R.Tensor((24, 1, 10), dtype="float16"))
            add12 = R.call_tir(cls.add2, (mul16, mask), out_sinfo=R.Tensor((24, 1, 10), dtype="float16"))
            softmax2 = R.call_tir(cls.softmax, (add12,), out_sinfo=R.Tensor((24, 1, 10), dtype="float16"))
            matmul24 = R.call_tir(cls.matmul3, (softmax2, repeat5), out_sinfo=R.Tensor((24, 1, 128), dtype="float16"))
            permute_dims34 = R.call_tir(cls.transpose3, (matmul24,), out_sinfo=R.Tensor((1, 24, 128), dtype="float16"))
            reshape17 = R.call_tir(cls.reshape4, (permute_dims34,), out_sinfo=R.Tensor((1, 3072), dtype="float16"))
            matmul25 = R.call_tir(cls.matmul, (reshape17, permute_dims35), out_sinfo=R.Tensor((1, 3072), dtype="float16"))
            add13 = R.call_tir(cls.add3, (input_embeds, matmul25), out_sinfo=R.Tensor((1, 3072), dtype="float16"))
            rms_norm7 = R.call_tir(cls.npu_rms_norm, (add13, layers_0_post_attention_layernorm_weight2), out_sinfo=R.Tensor((1, 3072), dtype="float16"))
            matmul26 = R.call_tir(cls.matmul4, (rms_norm7, permute_dims36), out_sinfo=R.Tensor((1, 8192), dtype="float16"))
            silu2 = R.call_tir(cls.silu, (matmul26,), out_sinfo=R.Tensor((1, 8192), dtype="float16"))
            matmul27 = R.call_tir(cls.matmul4, (rms_norm7, permute_dims37), out_sinfo=R.Tensor((1, 8192), dtype="float16"))
            mul17 = R.call_tir(cls.multiply3, (silu2, matmul27), out_sinfo=R.Tensor((1, 8192), dtype="float16"))
            matmul28 = R.call_tir(cls.matmul5, (mul17, permute_dims38), out_sinfo=R.Tensor((1, 3072), dtype="float16"))
            add14 = R.call_tir(cls.add3, (add13, matmul28), out_sinfo=R.Tensor((1, 3072), dtype="float16"))
            unsqueeze2 = R.call_tir(cls.expand_dims, (permute_dims31,), out_sinfo=R.Tensor((1, 8, 128, 1), dtype="float16"))
            unsqueeze3 = R.call_tir(cls.expand_dims1, (permute_dims32,), out_sinfo=R.Tensor((1, 8, 1, 128), dtype="float16"))
            rms_norm8 = R.call_tir(cls.npu_rms_norm, (add14, norm_weight2), out_sinfo=R.Tensor((1, 3072), dtype="float16"))
            matmul29 = R.call_tir(cls.matmul6, (rms_norm8, permute_dims39), out_sinfo=R.Tensor((1, 128256), dtype="float16"))
            concat10 = R.call_tir(cls.concatenate4, (unsqueeze2,), out_sinfo=R.Tensor((1, 8, 128, 1), dtype="float16"))
            concat11 = R.call_tir(cls.concatenate5, (unsqueeze3,), out_sinfo=R.Tensor((1, 8, 1, 128), dtype="float16"))
            gv2: R.Tuple(R.Tensor((1, 128256), dtype="float16"), R.Tensor((1, 8, 128, 1), dtype="float16"), R.Tensor((1, 8, 1, 128), dtype="float16")) = matmul29, concat10, concat11
            R.output(gv2)
        return gv2

    @R.function
    def decode_transform_params(model_params: R.Tuple(R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")))) -> R.Tuple(R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((3072, 1024), dtype="float16"), R.Tensor((3072, 1024), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 128256), dtype="float16")):
        R.func_attr({"num_input": 0})
        cls = Module
        with R.dataflow():
            packed_params: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            layers_0_self_attn_q_proj_weight2: R.Tensor((3072, 3072), dtype="float16") = packed_params[0]
            packed_params1: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            layers_0_self_attn_k_proj_weight2: R.Tensor((1024, 3072), dtype="float16") = packed_params1[1]
            packed_params2: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            layers_0_self_attn_v_proj_weight2: R.Tensor((1024, 3072), dtype="float16") = packed_params2[2]
            packed_params3: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            layers_0_self_attn_o_proj_weight2: R.Tensor((3072, 3072), dtype="float16") = packed_params3[3]
            packed_params4: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            layers_0_mlp_gate_proj_weight2: R.Tensor((8192, 3072), dtype="float16") = packed_params4[4]
            packed_params5: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            layers_0_mlp_up_proj_weight2: R.Tensor((8192, 3072), dtype="float16") = packed_params5[5]
            packed_params6: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            layers_0_mlp_down_proj_weight2: R.Tensor((3072, 8192), dtype="float16") = packed_params6[6]
            packed_params7: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            layers_0_input_layernorm_weight2: R.Tensor((3072,), dtype="float16") = packed_params7[7]
            packed_params8: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            layers_0_post_attention_layernorm_weight2: R.Tensor((3072,), dtype="float16") = packed_params8[8]
            packed_params9: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            norm_weight2: R.Tensor((3072,), dtype="float16") = packed_params9[9]
            packed_params10: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            lm_head_weight2: R.Tensor((128256, 3072), dtype="float16") = packed_params10[10]
            permute_dims28 = R.call_tir(cls.transpose4, (layers_0_self_attn_q_proj_weight2,), out_sinfo=R.Tensor((3072, 3072), dtype="float16"))
            permute_dims29 = R.call_tir(cls.transpose5, (layers_0_self_attn_k_proj_weight2,), out_sinfo=R.Tensor((3072, 1024), dtype="float16"))
            permute_dims30 = R.call_tir(cls.transpose5, (layers_0_self_attn_v_proj_weight2,), out_sinfo=R.Tensor((3072, 1024), dtype="float16"))
            permute_dims35 = R.call_tir(cls.transpose4, (layers_0_self_attn_o_proj_weight2,), out_sinfo=R.Tensor((3072, 3072), dtype="float16"))
            permute_dims36 = R.call_tir(cls.transpose6, (layers_0_mlp_gate_proj_weight2,), out_sinfo=R.Tensor((3072, 8192), dtype="float16"))
            permute_dims37 = R.call_tir(cls.transpose6, (layers_0_mlp_up_proj_weight2,), out_sinfo=R.Tensor((3072, 8192), dtype="float16"))
            permute_dims38 = R.call_tir(cls.transpose7, (layers_0_mlp_down_proj_weight2,), out_sinfo=R.Tensor((8192, 3072), dtype="float16"))
            permute_dims39 = R.call_tir(cls.transpose8, (lm_head_weight2,), out_sinfo=R.Tensor((3072, 128256), dtype="float16"))
            output_tuple: R.Tuple(R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((3072, 1024), dtype="float16"), R.Tensor((3072, 1024), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 128256), dtype="float16")) = layers_0_input_layernorm_weight2, layers_0_post_attention_layernorm_weight2, norm_weight2, permute_dims28, permute_dims29, permute_dims30, permute_dims35, permute_dims36, permute_dims37, permute_dims38, permute_dims39
            R.output(output_tuple)
        return output_tuple

    @R.function
    def prefill(input_embeds: R.Tensor((7, 3072), dtype="float16"), cos: R.Tensor((7, 1, 128), dtype="float16"), sin: R.Tensor((7, 1, 128), dtype="float16"), mask: R.Tensor((1, 7, 7), dtype="float16"), layers_0_input_layernorm_weight: R.Tensor((3072,), dtype="float16"), layers_0_post_attention_layernorm_weight: R.Tensor((3072,), dtype="float16"), norm_weight: R.Tensor((3072,), dtype="float16"), permute_dims: R.Tensor((3072, 3072), dtype="float16"), permute_dims1: R.Tensor((3072, 1024), dtype="float16"), permute_dims2: R.Tensor((3072, 1024), dtype="float16"), permute_dims8: R.Tensor((3072, 3072), dtype="float16"), permute_dims9: R.Tensor((3072, 8192), dtype="float16"), permute_dims10: R.Tensor((3072, 8192), dtype="float16"), permute_dims11: R.Tensor((8192, 3072), dtype="float16"), permute_dims12: R.Tensor((3072, 128256), dtype="float16")) -> R.Tensor((7, 128256), dtype="float16"):
        R.func_attr({"num_input": 4})
        cls = Module
        with R.dataflow():
            rms_norm = R.call_tir(cls.npu_rms_norm1, (input_embeds, layers_0_input_layernorm_weight), out_sinfo=R.Tensor((7, 3072), dtype="float16"))
            matmul = R.call_tir(cls.matmul7, (rms_norm, permute_dims), out_sinfo=R.Tensor((7, 3072), dtype="float16"))
            reshape = R.call_tir(cls.reshape5, (matmul,), out_sinfo=R.Tensor((7, 24, 128), dtype="float16"))
            matmul1 = R.call_tir(cls.matmul8, (rms_norm, permute_dims1), out_sinfo=R.Tensor((7, 1024), dtype="float16"))
            reshape1 = R.call_tir(cls.reshape6, (matmul1,), out_sinfo=R.Tensor((7, 8, 128), dtype="float16"))
            matmul2 = R.call_tir(cls.matmul8, (rms_norm, permute_dims2), out_sinfo=R.Tensor((7, 1024), dtype="float16"))
            reshape2 = R.call_tir(cls.reshape6, (matmul2,), out_sinfo=R.Tensor((7, 8, 128), dtype="float16"))
            mul = R.call_tir(cls.multiply4, (reshape, cos), out_sinfo=R.Tensor((7, 24, 128), dtype="float16"))
            split = R.call_tir(cls.split2, (reshape,), out_sinfo=[R.Tensor((7, 24, 64), dtype="float16"), R.Tensor((7, 24, 64), dtype="float16")])
            split_0: R.Tensor((7, 24, 64), dtype="float16") = split[0]
            split_1: R.Tensor((7, 24, 64), dtype="float16") = split[1]
            neg = R.call_tir(cls.tir_negative2, (split_1,), out_sinfo=R.Tensor((7, 24, 64), dtype="float16"))
            concat = R.call_tir(cls.concatenate6, (neg, split_0), out_sinfo=R.Tensor((7, 24, 128), dtype="float16"))
            mul1 = R.call_tir(cls.multiply4, (concat, sin), out_sinfo=R.Tensor((7, 24, 128), dtype="float16"))
            add = R.call_tir(cls.add4, (mul, mul1), out_sinfo=R.Tensor((7, 24, 128), dtype="float16"))
            mul2 = R.call_tir(cls.multiply5, (reshape1, cos), out_sinfo=R.Tensor((7, 8, 128), dtype="float16"))
            split1 = R.call_tir(cls.split3, (reshape1,), out_sinfo=[R.Tensor((7, 8, 64), dtype="float16"), R.Tensor((7, 8, 64), dtype="float16")])
            split_01: R.Tensor((7, 8, 64), dtype="float16") = split1[0]
            split_11: R.Tensor((7, 8, 64), dtype="float16") = split1[1]
            neg1 = R.call_tir(cls.tir_negative3, (split_11,), out_sinfo=R.Tensor((7, 8, 64), dtype="float16"))
            concat1 = R.call_tir(cls.concatenate7, (neg1, split_01), out_sinfo=R.Tensor((7, 8, 128), dtype="float16"))
            mul3 = R.call_tir(cls.multiply5, (concat1, sin), out_sinfo=R.Tensor((7, 8, 128), dtype="float16"))
            add1 = R.call_tir(cls.add5, (mul2, mul3), out_sinfo=R.Tensor((7, 8, 128), dtype="float16"))
            repeat = R.call_tir(cls.repeat2, (add1,), out_sinfo=R.Tensor((7, 24, 128), dtype="float16"))
            reshape3 = R.call_tir(cls.reshape7, (repeat,), out_sinfo=R.Tensor((7, 24, 128), dtype="float16"))
            repeat1 = R.call_tir(cls.repeat2, (reshape2,), out_sinfo=R.Tensor((7, 24, 128), dtype="float16"))
            reshape4 = R.call_tir(cls.reshape7, (repeat1,), out_sinfo=R.Tensor((7, 24, 128), dtype="float16"))
            permute_dims3 = R.call_tir(cls.transpose9, (add,), out_sinfo=R.Tensor((24, 7, 128), dtype="float16"))
            permute_dims4 = R.call_tir(cls.transpose9, (reshape3,), out_sinfo=R.Tensor((24, 7, 128), dtype="float16"))
            permute_dims5 = R.call_tir(cls.transpose9, (reshape4,), out_sinfo=R.Tensor((24, 7, 128), dtype="float16"))
            permute_dims6 = R.call_tir(cls.transpose10, (permute_dims4,), out_sinfo=R.Tensor((24, 128, 7), dtype="float16"))
            matmul3 = R.call_tir(cls.matmul9, (permute_dims3, permute_dims6), out_sinfo=R.Tensor((24, 7, 7), dtype="float16"))
            mul4 = R.call_tir(cls.multiply6, (matmul3,), out_sinfo=R.Tensor((24, 7, 7), dtype="float16"))
            add2 = R.call_tir(cls.add6, (mul4, mask), out_sinfo=R.Tensor((24, 7, 7), dtype="float16"))
            softmax = R.call_tir(cls.softmax1, (add2,), out_sinfo=R.Tensor((24, 7, 7), dtype="float16"))
            matmul4 = R.call_tir(cls.matmul10, (softmax, permute_dims5), out_sinfo=R.Tensor((24, 7, 128), dtype="float16"))
            permute_dims7 = R.call_tir(cls.transpose11, (matmul4,), out_sinfo=R.Tensor((7, 24, 128), dtype="float16"))
            reshape5 = R.call_tir(cls.reshape8, (permute_dims7,), out_sinfo=R.Tensor((7, 3072), dtype="float16"))
            matmul5 = R.call_tir(cls.matmul7, (reshape5, permute_dims8), out_sinfo=R.Tensor((7, 3072), dtype="float16"))
            add3 = R.call_tir(cls.add7, (input_embeds, matmul5), out_sinfo=R.Tensor((7, 3072), dtype="float16"))
            rms_norm1 = R.call_tir(cls.npu_rms_norm1, (add3, layers_0_post_attention_layernorm_weight), out_sinfo=R.Tensor((7, 3072), dtype="float16"))
            matmul6 = R.call_tir(cls.matmul11, (rms_norm1, permute_dims9), out_sinfo=R.Tensor((7, 8192), dtype="float16"))
            silu = R.call_tir(cls.silu1, (matmul6,), out_sinfo=R.Tensor((7, 8192), dtype="float16"))
            matmul7 = R.call_tir(cls.matmul11, (rms_norm1, permute_dims10), out_sinfo=R.Tensor((7, 8192), dtype="float16"))
            mul5 = R.call_tir(cls.multiply7, (silu, matmul7), out_sinfo=R.Tensor((7, 8192), dtype="float16"))
            matmul8 = R.call_tir(cls.matmul12, (mul5, permute_dims11), out_sinfo=R.Tensor((7, 3072), dtype="float16"))
            add4 = R.call_tir(cls.add7, (add3, matmul8), out_sinfo=R.Tensor((7, 3072), dtype="float16"))
            rms_norm2 = R.call_tir(cls.npu_rms_norm1, (add4, norm_weight), out_sinfo=R.Tensor((7, 3072), dtype="float16"))
            gv = R.call_tir(cls.matmul13, (rms_norm2, permute_dims12), out_sinfo=R.Tensor((7, 128256), dtype="float16"))
            R.output(gv)
        return gv

    @R.function
    def prefill_cache(input_embeds: R.Tensor((7, 3072), dtype="float16"), cos: R.Tensor((7, 1, 128), dtype="float16"), sin: R.Tensor((7, 1, 128), dtype="float16"), mask: R.Tensor((1, 7, 7), dtype="float16"), layers_0_input_layernorm_weight1: R.Tensor((3072,), dtype="float16"), layers_0_post_attention_layernorm_weight1: R.Tensor((3072,), dtype="float16"), norm_weight1: R.Tensor((3072,), dtype="float16"), permute_dims13: R.Tensor((3072, 3072), dtype="float16"), permute_dims14: R.Tensor((3072, 1024), dtype="float16"), permute_dims15: R.Tensor((3072, 1024), dtype="float16"), permute_dims21: R.Tensor((3072, 3072), dtype="float16"), permute_dims22: R.Tensor((3072, 8192), dtype="float16"), permute_dims23: R.Tensor((3072, 8192), dtype="float16"), permute_dims24: R.Tensor((8192, 3072), dtype="float16"), permute_dims27: R.Tensor((3072, 128256), dtype="float16")) -> R.Tuple(R.Tensor((1, 128256), dtype="float16"), R.Tensor((1, 8, 128, 7), dtype="float16"), R.Tensor((1, 8, 7, 128), dtype="float16")):
        R.func_attr({"num_input": 4})
        cls = Module
        with R.dataflow():
            rms_norm3 = R.call_tir(cls.npu_rms_norm1, (input_embeds, layers_0_input_layernorm_weight1), out_sinfo=R.Tensor((7, 3072), dtype="float16"))
            matmul10 = R.call_tir(cls.matmul7, (rms_norm3, permute_dims13), out_sinfo=R.Tensor((7, 3072), dtype="float16"))
            reshape6 = R.call_tir(cls.reshape5, (matmul10,), out_sinfo=R.Tensor((7, 24, 128), dtype="float16"))
            matmul11 = R.call_tir(cls.matmul8, (rms_norm3, permute_dims14), out_sinfo=R.Tensor((7, 1024), dtype="float16"))
            reshape7 = R.call_tir(cls.reshape6, (matmul11,), out_sinfo=R.Tensor((7, 8, 128), dtype="float16"))
            matmul12 = R.call_tir(cls.matmul8, (rms_norm3, permute_dims15), out_sinfo=R.Tensor((7, 1024), dtype="float16"))
            reshape8 = R.call_tir(cls.reshape6, (matmul12,), out_sinfo=R.Tensor((7, 8, 128), dtype="float16"))
            mul6 = R.call_tir(cls.multiply4, (reshape6, cos), out_sinfo=R.Tensor((7, 24, 128), dtype="float16"))
            split2 = R.call_tir(cls.split2, (reshape6,), out_sinfo=[R.Tensor((7, 24, 64), dtype="float16"), R.Tensor((7, 24, 64), dtype="float16")])
            split_02: R.Tensor((7, 24, 64), dtype="float16") = split2[0]
            split_12: R.Tensor((7, 24, 64), dtype="float16") = split2[1]
            neg2 = R.call_tir(cls.tir_negative2, (split_12,), out_sinfo=R.Tensor((7, 24, 64), dtype="float16"))
            concat2 = R.call_tir(cls.concatenate6, (neg2, split_02), out_sinfo=R.Tensor((7, 24, 128), dtype="float16"))
            mul7 = R.call_tir(cls.multiply4, (concat2, sin), out_sinfo=R.Tensor((7, 24, 128), dtype="float16"))
            add5 = R.call_tir(cls.add4, (mul6, mul7), out_sinfo=R.Tensor((7, 24, 128), dtype="float16"))
            mul8 = R.call_tir(cls.multiply5, (reshape7, cos), out_sinfo=R.Tensor((7, 8, 128), dtype="float16"))
            split3 = R.call_tir(cls.split3, (reshape7,), out_sinfo=[R.Tensor((7, 8, 64), dtype="float16"), R.Tensor((7, 8, 64), dtype="float16")])
            split_03: R.Tensor((7, 8, 64), dtype="float16") = split3[0]
            split_13: R.Tensor((7, 8, 64), dtype="float16") = split3[1]
            neg3 = R.call_tir(cls.tir_negative3, (split_13,), out_sinfo=R.Tensor((7, 8, 64), dtype="float16"))
            concat3 = R.call_tir(cls.concatenate7, (neg3, split_03), out_sinfo=R.Tensor((7, 8, 128), dtype="float16"))
            mul9 = R.call_tir(cls.multiply5, (concat3, sin), out_sinfo=R.Tensor((7, 8, 128), dtype="float16"))
            add6 = R.call_tir(cls.add5, (mul8, mul9), out_sinfo=R.Tensor((7, 8, 128), dtype="float16"))
            repeat2 = R.call_tir(cls.repeat2, (add6,), out_sinfo=R.Tensor((7, 24, 128), dtype="float16"))
            reshape9 = R.call_tir(cls.reshape7, (repeat2,), out_sinfo=R.Tensor((7, 24, 128), dtype="float16"))
            repeat3 = R.call_tir(cls.repeat2, (reshape8,), out_sinfo=R.Tensor((7, 24, 128), dtype="float16"))
            reshape10 = R.call_tir(cls.reshape7, (repeat3,), out_sinfo=R.Tensor((7, 24, 128), dtype="float16"))
            permute_dims16 = R.call_tir(cls.transpose9, (add5,), out_sinfo=R.Tensor((24, 7, 128), dtype="float16"))
            permute_dims17 = R.call_tir(cls.transpose9, (reshape9,), out_sinfo=R.Tensor((24, 7, 128), dtype="float16"))
            permute_dims18 = R.call_tir(cls.transpose9, (reshape10,), out_sinfo=R.Tensor((24, 7, 128), dtype="float16"))
            permute_dims19 = R.call_tir(cls.transpose10, (permute_dims17,), out_sinfo=R.Tensor((24, 128, 7), dtype="float16"))
            matmul13 = R.call_tir(cls.matmul9, (permute_dims16, permute_dims19), out_sinfo=R.Tensor((24, 7, 7), dtype="float16"))
            mul10 = R.call_tir(cls.multiply6, (matmul13,), out_sinfo=R.Tensor((24, 7, 7), dtype="float16"))
            add7 = R.call_tir(cls.add6, (mul10, mask), out_sinfo=R.Tensor((24, 7, 7), dtype="float16"))
            softmax1 = R.call_tir(cls.softmax1, (add7,), out_sinfo=R.Tensor((24, 7, 7), dtype="float16"))
            matmul14 = R.call_tir(cls.matmul10, (softmax1, permute_dims18), out_sinfo=R.Tensor((24, 7, 128), dtype="float16"))
            permute_dims20 = R.call_tir(cls.transpose11, (matmul14,), out_sinfo=R.Tensor((7, 24, 128), dtype="float16"))
            reshape11 = R.call_tir(cls.reshape8, (permute_dims20,), out_sinfo=R.Tensor((7, 3072), dtype="float16"))
            matmul15 = R.call_tir(cls.matmul7, (reshape11, permute_dims21), out_sinfo=R.Tensor((7, 3072), dtype="float16"))
            add8 = R.call_tir(cls.add7, (input_embeds, matmul15), out_sinfo=R.Tensor((7, 3072), dtype="float16"))
            rms_norm4 = R.call_tir(cls.npu_rms_norm1, (add8, layers_0_post_attention_layernorm_weight1), out_sinfo=R.Tensor((7, 3072), dtype="float16"))
            matmul16 = R.call_tir(cls.matmul11, (rms_norm4, permute_dims22), out_sinfo=R.Tensor((7, 8192), dtype="float16"))
            silu1 = R.call_tir(cls.silu1, (matmul16,), out_sinfo=R.Tensor((7, 8192), dtype="float16"))
            matmul17 = R.call_tir(cls.matmul11, (rms_norm4, permute_dims23), out_sinfo=R.Tensor((7, 8192), dtype="float16"))
            mul11 = R.call_tir(cls.multiply7, (silu1, matmul17), out_sinfo=R.Tensor((7, 8192), dtype="float16"))
            matmul18 = R.call_tir(cls.matmul12, (mul11, permute_dims24), out_sinfo=R.Tensor((7, 3072), dtype="float16"))
            add9 = R.call_tir(cls.add7, (add8, matmul18), out_sinfo=R.Tensor((7, 3072), dtype="float16"))
            permute_dims25 = R.call_tir(cls.transpose12, (add6,), out_sinfo=R.Tensor((8, 128, 7), dtype="float16"))
            unsqueeze = R.call_tir(cls.expand_dims2, (permute_dims25,), out_sinfo=R.Tensor((1, 8, 128, 7), dtype="float16"))
            permute_dims26 = R.call_tir(cls.transpose13, (reshape8,), out_sinfo=R.Tensor((8, 7, 128), dtype="float16"))
            unsqueeze1 = R.call_tir(cls.expand_dims3, (permute_dims26,), out_sinfo=R.Tensor((1, 8, 7, 128), dtype="float16"))
            lv = R.call_tir(cls.last_hidden, (add9,), out_sinfo=R.Tensor((1, 3072), dtype="float16"))
            rms_norm5 = R.call_tir(cls.npu_rms_norm, (lv, norm_weight1), out_sinfo=R.Tensor((1, 3072), dtype="float16"))
            matmul19 = R.call_tir(cls.matmul6, (rms_norm5, permute_dims27), out_sinfo=R.Tensor((1, 128256), dtype="float16"))
            concat4 = R.call_tir(cls.concatenate8, (unsqueeze,), out_sinfo=R.Tensor((1, 8, 128, 7), dtype="float16"))
            concat5 = R.call_tir(cls.concatenate9, (unsqueeze1,), out_sinfo=R.Tensor((1, 8, 7, 128), dtype="float16"))
            gv1: R.Tuple(R.Tensor((1, 128256), dtype="float16"), R.Tensor((1, 8, 128, 7), dtype="float16"), R.Tensor((1, 8, 7, 128), dtype="float16")) = matmul19, concat4, concat5
            R.output(gv1)
        return gv1

    @R.function
    def prefill_cache_transform_params(model_params: R.Tuple(R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")))) -> R.Tuple(R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((3072, 1024), dtype="float16"), R.Tensor((3072, 1024), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 128256), dtype="float16")):
        R.func_attr({"num_input": 0})
        cls = Module
        with R.dataflow():
            packed_params: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            layers_0_self_attn_q_proj_weight1: R.Tensor((3072, 3072), dtype="float16") = packed_params[0]
            packed_params1: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            layers_0_self_attn_k_proj_weight1: R.Tensor((1024, 3072), dtype="float16") = packed_params1[1]
            packed_params2: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            layers_0_self_attn_v_proj_weight1: R.Tensor((1024, 3072), dtype="float16") = packed_params2[2]
            packed_params3: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            layers_0_self_attn_o_proj_weight1: R.Tensor((3072, 3072), dtype="float16") = packed_params3[3]
            packed_params4: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            layers_0_mlp_gate_proj_weight1: R.Tensor((8192, 3072), dtype="float16") = packed_params4[4]
            packed_params5: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            layers_0_mlp_up_proj_weight1: R.Tensor((8192, 3072), dtype="float16") = packed_params5[5]
            packed_params6: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            layers_0_mlp_down_proj_weight1: R.Tensor((3072, 8192), dtype="float16") = packed_params6[6]
            packed_params7: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            layers_0_input_layernorm_weight1: R.Tensor((3072,), dtype="float16") = packed_params7[7]
            packed_params8: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            layers_0_post_attention_layernorm_weight1: R.Tensor((3072,), dtype="float16") = packed_params8[8]
            packed_params9: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            norm_weight1: R.Tensor((3072,), dtype="float16") = packed_params9[9]
            packed_params10: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            lm_head_weight1: R.Tensor((128256, 3072), dtype="float16") = packed_params10[10]
            permute_dims13 = R.call_tir(cls.transpose4, (layers_0_self_attn_q_proj_weight1,), out_sinfo=R.Tensor((3072, 3072), dtype="float16"))
            permute_dims14 = R.call_tir(cls.transpose5, (layers_0_self_attn_k_proj_weight1,), out_sinfo=R.Tensor((3072, 1024), dtype="float16"))
            permute_dims15 = R.call_tir(cls.transpose5, (layers_0_self_attn_v_proj_weight1,), out_sinfo=R.Tensor((3072, 1024), dtype="float16"))
            permute_dims21 = R.call_tir(cls.transpose4, (layers_0_self_attn_o_proj_weight1,), out_sinfo=R.Tensor((3072, 3072), dtype="float16"))
            permute_dims22 = R.call_tir(cls.transpose6, (layers_0_mlp_gate_proj_weight1,), out_sinfo=R.Tensor((3072, 8192), dtype="float16"))
            permute_dims23 = R.call_tir(cls.transpose6, (layers_0_mlp_up_proj_weight1,), out_sinfo=R.Tensor((3072, 8192), dtype="float16"))
            permute_dims24 = R.call_tir(cls.transpose7, (layers_0_mlp_down_proj_weight1,), out_sinfo=R.Tensor((8192, 3072), dtype="float16"))
            permute_dims27 = R.call_tir(cls.transpose8, (lm_head_weight1,), out_sinfo=R.Tensor((3072, 128256), dtype="float16"))
            output_tuple: R.Tuple(R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((3072, 1024), dtype="float16"), R.Tensor((3072, 1024), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 128256), dtype="float16")) = layers_0_input_layernorm_weight1, layers_0_post_attention_layernorm_weight1, norm_weight1, permute_dims13, permute_dims14, permute_dims15, permute_dims21, permute_dims22, permute_dims23, permute_dims24, permute_dims27
            R.output(output_tuple)
        return output_tuple

    @R.function
    def prefill_transform_params(model_params: R.Tuple(R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")))) -> R.Tuple(R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((3072, 1024), dtype="float16"), R.Tensor((3072, 1024), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 128256), dtype="float16")):
        R.func_attr({"num_input": 0})
        cls = Module
        with R.dataflow():
            packed_params: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            layers_0_self_attn_q_proj_weight: R.Tensor((3072, 3072), dtype="float16") = packed_params[0]
            packed_params1: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            layers_0_self_attn_k_proj_weight: R.Tensor((1024, 3072), dtype="float16") = packed_params1[1]
            packed_params2: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            layers_0_self_attn_v_proj_weight: R.Tensor((1024, 3072), dtype="float16") = packed_params2[2]
            packed_params3: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            layers_0_self_attn_o_proj_weight: R.Tensor((3072, 3072), dtype="float16") = packed_params3[3]
            packed_params4: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            layers_0_mlp_gate_proj_weight: R.Tensor((8192, 3072), dtype="float16") = packed_params4[4]
            packed_params5: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            layers_0_mlp_up_proj_weight: R.Tensor((8192, 3072), dtype="float16") = packed_params5[5]
            packed_params6: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            layers_0_mlp_down_proj_weight: R.Tensor((3072, 8192), dtype="float16") = packed_params6[6]
            packed_params7: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            layers_0_input_layernorm_weight: R.Tensor((3072,), dtype="float16") = packed_params7[7]
            packed_params8: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            layers_0_post_attention_layernorm_weight: R.Tensor((3072,), dtype="float16") = packed_params8[8]
            packed_params9: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            norm_weight: R.Tensor((3072,), dtype="float16") = packed_params9[9]
            packed_params10: R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")) = model_params[0]
            lm_head_weight: R.Tensor((128256, 3072), dtype="float16") = packed_params10[10]
            permute_dims = R.call_tir(cls.transpose4, (layers_0_self_attn_q_proj_weight,), out_sinfo=R.Tensor((3072, 3072), dtype="float16"))
            permute_dims1 = R.call_tir(cls.transpose5, (layers_0_self_attn_k_proj_weight,), out_sinfo=R.Tensor((3072, 1024), dtype="float16"))
            permute_dims2 = R.call_tir(cls.transpose5, (layers_0_self_attn_v_proj_weight,), out_sinfo=R.Tensor((3072, 1024), dtype="float16"))
            permute_dims8 = R.call_tir(cls.transpose4, (layers_0_self_attn_o_proj_weight,), out_sinfo=R.Tensor((3072, 3072), dtype="float16"))
            permute_dims9 = R.call_tir(cls.transpose6, (layers_0_mlp_gate_proj_weight,), out_sinfo=R.Tensor((3072, 8192), dtype="float16"))
            permute_dims10 = R.call_tir(cls.transpose6, (layers_0_mlp_up_proj_weight,), out_sinfo=R.Tensor((3072, 8192), dtype="float16"))
            permute_dims11 = R.call_tir(cls.transpose7, (layers_0_mlp_down_proj_weight,), out_sinfo=R.Tensor((8192, 3072), dtype="float16"))
            permute_dims12 = R.call_tir(cls.transpose8, (lm_head_weight,), out_sinfo=R.Tensor((3072, 128256), dtype="float16"))
            output_tuple: R.Tuple(R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((3072, 1024), dtype="float16"), R.Tensor((3072, 1024), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 128256), dtype="float16")) = layers_0_input_layernorm_weight, layers_0_post_attention_layernorm_weight, norm_weight, permute_dims, permute_dims1, permute_dims2, permute_dims8, permute_dims9, permute_dims10, permute_dims11, permute_dims12
            R.output(output_tuple)
        return output_tuple
