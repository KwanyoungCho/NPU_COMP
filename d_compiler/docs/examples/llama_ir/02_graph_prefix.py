# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def last_hidden(add9: T.Buffer((T.int64(7), T.int64(3072)), "float16"), last_hidden: T.Buffer((T.int64(1), T.int64(3072)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i, j in T.grid(T.int64(1), T.int64(3072)):
            with T.block("last_hidden"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(add9[T.int64(6), v_j])
                T.writes(last_hidden[v_i, v_j])
                last_hidden[v_i, v_j] = add9[T.int64(6), v_j]

    @R.function
    def decode(input_embeds: R.Tensor((1, 3072), dtype="float16"), cos: R.Tensor((1, 1, 128), dtype="float16"), sin: R.Tensor((1, 1, 128), dtype="float16"), k_cache: R.Tensor((1, 8, 128, 9), dtype="float16"), v_cache: R.Tensor((1, 8, 9, 128), dtype="float16"), mask: R.Tensor((1, 1, 10), dtype="float16"), layers_0_input_layernorm_weight2: R.Tensor((3072,), dtype="float16"), layers_0_post_attention_layernorm_weight2: R.Tensor((3072,), dtype="float16"), norm_weight2: R.Tensor((3072,), dtype="float16"), permute_dims28: R.Tensor((3072, 3072), dtype="float16"), permute_dims29: R.Tensor((3072, 1024), dtype="float16"), permute_dims30: R.Tensor((3072, 1024), dtype="float16"), permute_dims35: R.Tensor((3072, 3072), dtype="float16"), permute_dims36: R.Tensor((3072, 8192), dtype="float16"), permute_dims37: R.Tensor((3072, 8192), dtype="float16"), permute_dims38: R.Tensor((8192, 3072), dtype="float16"), permute_dims39: R.Tensor((3072, 128256), dtype="float16")) -> R.Tuple(R.Tensor((1, 128256), dtype="float16"), R.Tensor((1, 8, 128, 1), dtype="float16"), R.Tensor((1, 8, 1, 128), dtype="float16")):
        R.func_attr({"npu.cache_append": "{\"entries\": [{\"axis\": 3, \"output\": 1, \"parameter\": \"k_cache\"}, {\"axis\": 2, \"output\": 2, \"parameter\": \"v_cache\"}], \"mask\": \"mask\", \"version\": 1}", "num_input": 6})
        with R.dataflow():
            reshape12: R.Tensor((8, 128, 9), dtype="float16") = R.reshape(k_cache, R.shape([8, 128, 9]))
            reshape13: R.Tensor((8, 9, 128), dtype="float16") = R.reshape(v_cache, R.shape([8, 9, 128]))
            rms_norm6: R.Tensor((1, 3072), dtype="float16") = R.nn.rms_norm(input_embeds, layers_0_input_layernorm_weight2, axes=[-1], epsilon=1.0000000000000001e-05)
            matmul20: R.Tensor((1, 3072), dtype="float16") = R.matmul(rms_norm6, permute_dims28, out_dtype="void")
            reshape14: R.Tensor((1, 24, 128), dtype="float16") = R.reshape(matmul20, R.shape([1, 24, 128]))
            matmul21: R.Tensor((1, 1024), dtype="float16") = R.matmul(rms_norm6, permute_dims29, out_dtype="void")
            reshape15: R.Tensor((1, 8, 128), dtype="float16") = R.reshape(matmul21, R.shape([1, 8, 128]))
            matmul22: R.Tensor((1, 1024), dtype="float16") = R.matmul(rms_norm6, permute_dims30, out_dtype="void")
            reshape16: R.Tensor((1, 8, 128), dtype="float16") = R.reshape(matmul22, R.shape([1, 8, 128]))
            mul12: R.Tensor((1, 24, 128), dtype="float16") = R.multiply(reshape14, cos)
            split4: R.Tuple(R.Tensor((1, 24, 64), dtype="float16"), R.Tensor((1, 24, 64), dtype="float16")) = R.split(reshape14, indices_or_sections=[64], axis=-1)
            split_04: R.Tensor((1, 24, 64), dtype="float16") = split4[0]
            split_14: R.Tensor((1, 24, 64), dtype="float16") = split4[1]
            neg4: R.Tensor((1, 24, 64), dtype="float16") = R.negative(split_14)
            concat6: R.Tensor((1, 24, 128), dtype="float16") = R.concat((neg4, split_04), axis=-1)
            mul13: R.Tensor((1, 24, 128), dtype="float16") = R.multiply(concat6, sin)
            add10: R.Tensor((1, 24, 128), dtype="float16") = R.add(mul12, mul13)
            mul14: R.Tensor((1, 8, 128), dtype="float16") = R.multiply(reshape15, cos)
            split5: R.Tuple(R.Tensor((1, 8, 64), dtype="float16"), R.Tensor((1, 8, 64), dtype="float16")) = R.split(reshape15, indices_or_sections=[64], axis=-1)
            split_05: R.Tensor((1, 8, 64), dtype="float16") = split5[0]
            split_15: R.Tensor((1, 8, 64), dtype="float16") = split5[1]
            neg5: R.Tensor((1, 8, 64), dtype="float16") = R.negative(split_15)
            concat7: R.Tensor((1, 8, 128), dtype="float16") = R.concat((neg5, split_05), axis=-1)
            mul15: R.Tensor((1, 8, 128), dtype="float16") = R.multiply(concat7, sin)
            add11: R.Tensor((1, 8, 128), dtype="float16") = R.add(mul14, mul15)
            permute_dims31: R.Tensor((8, 128, 1), dtype="float16") = R.permute_dims(add11, axes=[1, 2, 0])
            permute_dims32: R.Tensor((8, 1, 128), dtype="float16") = R.permute_dims(reshape16, axes=[1, 0, 2])
            concat8: R.Tensor((8, 128, 10), dtype="float16") = R.concat((reshape12, permute_dims31), axis=2)
            concat9: R.Tensor((8, 10, 128), dtype="float16") = R.concat((reshape13, permute_dims32), axis=1)
            repeat4: R.Tensor((24, 128, 10), dtype="float16") = R.repeat(concat8, repeats=3, axis=0)
            repeat5: R.Tensor((24, 10, 128), dtype="float16") = R.repeat(concat9, repeats=3, axis=0)
            permute_dims33: R.Tensor((24, 1, 128), dtype="float16") = R.permute_dims(add10, axes=[1, 0, 2])
            matmul23: R.Tensor((24, 1, 10), dtype="float16") = R.matmul(permute_dims33, repeat4, out_dtype="void")
            mul16: R.Tensor((24, 1, 10), dtype="float16") = R.multiply(matmul23, R.const(0.08837890625, "float16"))
            add12: R.Tensor((24, 1, 10), dtype="float16") = R.add(mul16, mask)
            softmax2: R.Tensor((24, 1, 10), dtype="float16") = R.nn.softmax(add12, axis=-1)
            matmul24: R.Tensor((24, 1, 128), dtype="float16") = R.matmul(softmax2, repeat5, out_dtype="void")
            permute_dims34: R.Tensor((1, 24, 128), dtype="float16") = R.permute_dims(matmul24, axes=[1, 0, 2])
            reshape17: R.Tensor((1, 3072), dtype="float16") = R.reshape(permute_dims34, R.shape([1, 3072]))
            matmul25: R.Tensor((1, 3072), dtype="float16") = R.matmul(reshape17, permute_dims35, out_dtype="void")
            add13: R.Tensor((1, 3072), dtype="float16") = R.add(input_embeds, matmul25)
            rms_norm7: R.Tensor((1, 3072), dtype="float16") = R.nn.rms_norm(add13, layers_0_post_attention_layernorm_weight2, axes=[-1], epsilon=1.0000000000000001e-05)
            matmul26: R.Tensor((1, 8192), dtype="float16") = R.matmul(rms_norm7, permute_dims36, out_dtype="void")
            silu2: R.Tensor((1, 8192), dtype="float16") = R.nn.silu(matmul26)
            matmul27: R.Tensor((1, 8192), dtype="float16") = R.matmul(rms_norm7, permute_dims37, out_dtype="void")
            mul17: R.Tensor((1, 8192), dtype="float16") = R.multiply(silu2, matmul27)
            matmul28: R.Tensor((1, 3072), dtype="float16") = R.matmul(mul17, permute_dims38, out_dtype="void")
            add14: R.Tensor((1, 3072), dtype="float16") = R.add(add13, matmul28)
            unsqueeze2: R.Tensor((1, 8, 128, 1), dtype="float16") = R.expand_dims(permute_dims31, axis=[0])
            unsqueeze3: R.Tensor((1, 8, 1, 128), dtype="float16") = R.expand_dims(permute_dims32, axis=[0])
            rms_norm8: R.Tensor((1, 3072), dtype="float16") = R.nn.rms_norm(add14, norm_weight2, axes=[-1], epsilon=1.0000000000000001e-05)
            matmul29: R.Tensor((1, 128256), dtype="float16") = R.matmul(rms_norm8, permute_dims39, out_dtype="void")
            concat10: R.Tensor((1, 8, 128, 1), dtype="float16") = R.concat((unsqueeze2,), axis=0)
            concat11: R.Tensor((1, 8, 1, 128), dtype="float16") = R.concat((unsqueeze3,), axis=0)
            gv2: R.Tuple(R.Tensor((1, 128256), dtype="float16"), R.Tensor((1, 8, 128, 1), dtype="float16"), R.Tensor((1, 8, 1, 128), dtype="float16")) = matmul29, concat10, concat11
            R.output(gv2)
        return gv2

    @R.function
    def decode_transform_params(model_params: R.Tuple(R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")))) -> R.Tuple(R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((3072, 1024), dtype="float16"), R.Tensor((3072, 1024), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 128256), dtype="float16")):
        R.func_attr({"num_input": 0})
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
            permute_dims28: R.Tensor((3072, 3072), dtype="float16") = R.permute_dims(layers_0_self_attn_q_proj_weight2, axes=None)
            permute_dims29: R.Tensor((3072, 1024), dtype="float16") = R.permute_dims(layers_0_self_attn_k_proj_weight2, axes=None)
            permute_dims30: R.Tensor((3072, 1024), dtype="float16") = R.permute_dims(layers_0_self_attn_v_proj_weight2, axes=None)
            permute_dims35: R.Tensor((3072, 3072), dtype="float16") = R.permute_dims(layers_0_self_attn_o_proj_weight2, axes=None)
            permute_dims36: R.Tensor((3072, 8192), dtype="float16") = R.permute_dims(layers_0_mlp_gate_proj_weight2, axes=None)
            permute_dims37: R.Tensor((3072, 8192), dtype="float16") = R.permute_dims(layers_0_mlp_up_proj_weight2, axes=None)
            permute_dims38: R.Tensor((8192, 3072), dtype="float16") = R.permute_dims(layers_0_mlp_down_proj_weight2, axes=None)
            permute_dims39: R.Tensor((3072, 128256), dtype="float16") = R.permute_dims(lm_head_weight2, axes=None)
            output_tuple: R.Tuple(R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((3072, 1024), dtype="float16"), R.Tensor((3072, 1024), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 128256), dtype="float16")) = layers_0_input_layernorm_weight2, layers_0_post_attention_layernorm_weight2, norm_weight2, permute_dims28, permute_dims29, permute_dims30, permute_dims35, permute_dims36, permute_dims37, permute_dims38, permute_dims39
            R.output(output_tuple)
        return output_tuple

    @R.function
    def prefill(input_embeds: R.Tensor((7, 3072), dtype="float16"), cos: R.Tensor((7, 1, 128), dtype="float16"), sin: R.Tensor((7, 1, 128), dtype="float16"), mask: R.Tensor((1, 7, 7), dtype="float16"), layers_0_input_layernorm_weight: R.Tensor((3072,), dtype="float16"), layers_0_post_attention_layernorm_weight: R.Tensor((3072,), dtype="float16"), norm_weight: R.Tensor((3072,), dtype="float16"), permute_dims: R.Tensor((3072, 3072), dtype="float16"), permute_dims1: R.Tensor((3072, 1024), dtype="float16"), permute_dims2: R.Tensor((3072, 1024), dtype="float16"), permute_dims8: R.Tensor((3072, 3072), dtype="float16"), permute_dims9: R.Tensor((3072, 8192), dtype="float16"), permute_dims10: R.Tensor((3072, 8192), dtype="float16"), permute_dims11: R.Tensor((8192, 3072), dtype="float16"), permute_dims12: R.Tensor((3072, 128256), dtype="float16")) -> R.Tensor((7, 128256), dtype="float16"):
        R.func_attr({"num_input": 4})
        with R.dataflow():
            rms_norm: R.Tensor((7, 3072), dtype="float16") = R.nn.rms_norm(input_embeds, layers_0_input_layernorm_weight, axes=[-1], epsilon=1.0000000000000001e-05)
            matmul: R.Tensor((7, 3072), dtype="float16") = R.matmul(rms_norm, permute_dims, out_dtype="void")
            reshape: R.Tensor((7, 24, 128), dtype="float16") = R.reshape(matmul, R.shape([7, 24, 128]))
            matmul1: R.Tensor((7, 1024), dtype="float16") = R.matmul(rms_norm, permute_dims1, out_dtype="void")
            reshape1: R.Tensor((7, 8, 128), dtype="float16") = R.reshape(matmul1, R.shape([7, 8, 128]))
            matmul2: R.Tensor((7, 1024), dtype="float16") = R.matmul(rms_norm, permute_dims2, out_dtype="void")
            reshape2: R.Tensor((7, 8, 128), dtype="float16") = R.reshape(matmul2, R.shape([7, 8, 128]))
            mul: R.Tensor((7, 24, 128), dtype="float16") = R.multiply(reshape, cos)
            split: R.Tuple(R.Tensor((7, 24, 64), dtype="float16"), R.Tensor((7, 24, 64), dtype="float16")) = R.split(reshape, indices_or_sections=[64], axis=-1)
            split_0: R.Tensor((7, 24, 64), dtype="float16") = split[0]
            split_1: R.Tensor((7, 24, 64), dtype="float16") = split[1]
            neg: R.Tensor((7, 24, 64), dtype="float16") = R.negative(split_1)
            concat: R.Tensor((7, 24, 128), dtype="float16") = R.concat((neg, split_0), axis=-1)
            mul1: R.Tensor((7, 24, 128), dtype="float16") = R.multiply(concat, sin)
            add: R.Tensor((7, 24, 128), dtype="float16") = R.add(mul, mul1)
            mul2: R.Tensor((7, 8, 128), dtype="float16") = R.multiply(reshape1, cos)
            split1: R.Tuple(R.Tensor((7, 8, 64), dtype="float16"), R.Tensor((7, 8, 64), dtype="float16")) = R.split(reshape1, indices_or_sections=[64], axis=-1)
            split_01: R.Tensor((7, 8, 64), dtype="float16") = split1[0]
            split_11: R.Tensor((7, 8, 64), dtype="float16") = split1[1]
            neg1: R.Tensor((7, 8, 64), dtype="float16") = R.negative(split_11)
            concat1: R.Tensor((7, 8, 128), dtype="float16") = R.concat((neg1, split_01), axis=-1)
            mul3: R.Tensor((7, 8, 128), dtype="float16") = R.multiply(concat1, sin)
            add1: R.Tensor((7, 8, 128), dtype="float16") = R.add(mul2, mul3)
            repeat: R.Tensor((7, 24, 128), dtype="float16") = R.repeat(add1, repeats=3, axis=1)
            reshape3: R.Tensor((7, 24, 128), dtype="float16") = R.reshape(repeat, R.shape([7, 24, 128]))
            repeat1: R.Tensor((7, 24, 128), dtype="float16") = R.repeat(reshape2, repeats=3, axis=1)
            reshape4: R.Tensor((7, 24, 128), dtype="float16") = R.reshape(repeat1, R.shape([7, 24, 128]))
            permute_dims3: R.Tensor((24, 7, 128), dtype="float16") = R.permute_dims(add, axes=[1, 0, 2])
            permute_dims4: R.Tensor((24, 7, 128), dtype="float16") = R.permute_dims(reshape3, axes=[1, 0, 2])
            permute_dims5: R.Tensor((24, 7, 128), dtype="float16") = R.permute_dims(reshape4, axes=[1, 0, 2])
            permute_dims6: R.Tensor((24, 128, 7), dtype="float16") = R.permute_dims(permute_dims4, axes=[0, 2, 1])
            matmul3: R.Tensor((24, 7, 7), dtype="float16") = R.matmul(permute_dims3, permute_dims6, out_dtype="void")
            mul4: R.Tensor((24, 7, 7), dtype="float16") = R.multiply(matmul3, R.const(0.08837890625, "float16"))
            add2: R.Tensor((24, 7, 7), dtype="float16") = R.add(mul4, mask)
            softmax: R.Tensor((24, 7, 7), dtype="float16") = R.nn.softmax(add2, axis=-1)
            matmul4: R.Tensor((24, 7, 128), dtype="float16") = R.matmul(softmax, permute_dims5, out_dtype="void")
            permute_dims7: R.Tensor((7, 24, 128), dtype="float16") = R.permute_dims(matmul4, axes=[1, 0, 2])
            reshape5: R.Tensor((7, 3072), dtype="float16") = R.reshape(permute_dims7, R.shape([7, 3072]))
            matmul5: R.Tensor((7, 3072), dtype="float16") = R.matmul(reshape5, permute_dims8, out_dtype="void")
            add3: R.Tensor((7, 3072), dtype="float16") = R.add(input_embeds, matmul5)
            rms_norm1: R.Tensor((7, 3072), dtype="float16") = R.nn.rms_norm(add3, layers_0_post_attention_layernorm_weight, axes=[-1], epsilon=1.0000000000000001e-05)
            matmul6: R.Tensor((7, 8192), dtype="float16") = R.matmul(rms_norm1, permute_dims9, out_dtype="void")
            silu: R.Tensor((7, 8192), dtype="float16") = R.nn.silu(matmul6)
            matmul7: R.Tensor((7, 8192), dtype="float16") = R.matmul(rms_norm1, permute_dims10, out_dtype="void")
            mul5: R.Tensor((7, 8192), dtype="float16") = R.multiply(silu, matmul7)
            matmul8: R.Tensor((7, 3072), dtype="float16") = R.matmul(mul5, permute_dims11, out_dtype="void")
            add4: R.Tensor((7, 3072), dtype="float16") = R.add(add3, matmul8)
            rms_norm2: R.Tensor((7, 3072), dtype="float16") = R.nn.rms_norm(add4, norm_weight, axes=[-1], epsilon=1.0000000000000001e-05)
            gv: R.Tensor((7, 128256), dtype="float16") = R.matmul(rms_norm2, permute_dims12, out_dtype="void")
            R.output(gv)
        return gv

    @R.function
    def prefill_cache(input_embeds: R.Tensor((7, 3072), dtype="float16"), cos: R.Tensor((7, 1, 128), dtype="float16"), sin: R.Tensor((7, 1, 128), dtype="float16"), mask: R.Tensor((1, 7, 7), dtype="float16"), layers_0_input_layernorm_weight1: R.Tensor((3072,), dtype="float16"), layers_0_post_attention_layernorm_weight1: R.Tensor((3072,), dtype="float16"), norm_weight1: R.Tensor((3072,), dtype="float16"), permute_dims13: R.Tensor((3072, 3072), dtype="float16"), permute_dims14: R.Tensor((3072, 1024), dtype="float16"), permute_dims15: R.Tensor((3072, 1024), dtype="float16"), permute_dims21: R.Tensor((3072, 3072), dtype="float16"), permute_dims22: R.Tensor((3072, 8192), dtype="float16"), permute_dims23: R.Tensor((3072, 8192), dtype="float16"), permute_dims24: R.Tensor((8192, 3072), dtype="float16"), permute_dims27: R.Tensor((3072, 128256), dtype="float16")) -> R.Tuple(R.Tensor((1, 128256), dtype="float16"), R.Tensor((1, 8, 128, 7), dtype="float16"), R.Tensor((1, 8, 7, 128), dtype="float16")):
        R.func_attr({"num_input": 4})
        cls = Module
        with R.dataflow():
            rms_norm3: R.Tensor((7, 3072), dtype="float16") = R.nn.rms_norm(input_embeds, layers_0_input_layernorm_weight1, axes=[-1], epsilon=1.0000000000000001e-05)
            matmul10: R.Tensor((7, 3072), dtype="float16") = R.matmul(rms_norm3, permute_dims13, out_dtype="void")
            reshape6: R.Tensor((7, 24, 128), dtype="float16") = R.reshape(matmul10, R.shape([7, 24, 128]))
            matmul11: R.Tensor((7, 1024), dtype="float16") = R.matmul(rms_norm3, permute_dims14, out_dtype="void")
            reshape7: R.Tensor((7, 8, 128), dtype="float16") = R.reshape(matmul11, R.shape([7, 8, 128]))
            matmul12: R.Tensor((7, 1024), dtype="float16") = R.matmul(rms_norm3, permute_dims15, out_dtype="void")
            reshape8: R.Tensor((7, 8, 128), dtype="float16") = R.reshape(matmul12, R.shape([7, 8, 128]))
            mul6: R.Tensor((7, 24, 128), dtype="float16") = R.multiply(reshape6, cos)
            split2: R.Tuple(R.Tensor((7, 24, 64), dtype="float16"), R.Tensor((7, 24, 64), dtype="float16")) = R.split(reshape6, indices_or_sections=[64], axis=-1)
            split_02: R.Tensor((7, 24, 64), dtype="float16") = split2[0]
            split_12: R.Tensor((7, 24, 64), dtype="float16") = split2[1]
            neg2: R.Tensor((7, 24, 64), dtype="float16") = R.negative(split_12)
            concat2: R.Tensor((7, 24, 128), dtype="float16") = R.concat((neg2, split_02), axis=-1)
            mul7: R.Tensor((7, 24, 128), dtype="float16") = R.multiply(concat2, sin)
            add5: R.Tensor((7, 24, 128), dtype="float16") = R.add(mul6, mul7)
            mul8: R.Tensor((7, 8, 128), dtype="float16") = R.multiply(reshape7, cos)
            split3: R.Tuple(R.Tensor((7, 8, 64), dtype="float16"), R.Tensor((7, 8, 64), dtype="float16")) = R.split(reshape7, indices_or_sections=[64], axis=-1)
            split_03: R.Tensor((7, 8, 64), dtype="float16") = split3[0]
            split_13: R.Tensor((7, 8, 64), dtype="float16") = split3[1]
            neg3: R.Tensor((7, 8, 64), dtype="float16") = R.negative(split_13)
            concat3: R.Tensor((7, 8, 128), dtype="float16") = R.concat((neg3, split_03), axis=-1)
            mul9: R.Tensor((7, 8, 128), dtype="float16") = R.multiply(concat3, sin)
            add6: R.Tensor((7, 8, 128), dtype="float16") = R.add(mul8, mul9)
            repeat2: R.Tensor((7, 24, 128), dtype="float16") = R.repeat(add6, repeats=3, axis=1)
            reshape9: R.Tensor((7, 24, 128), dtype="float16") = R.reshape(repeat2, R.shape([7, 24, 128]))
            repeat3: R.Tensor((7, 24, 128), dtype="float16") = R.repeat(reshape8, repeats=3, axis=1)
            reshape10: R.Tensor((7, 24, 128), dtype="float16") = R.reshape(repeat3, R.shape([7, 24, 128]))
            permute_dims16: R.Tensor((24, 7, 128), dtype="float16") = R.permute_dims(add5, axes=[1, 0, 2])
            permute_dims17: R.Tensor((24, 7, 128), dtype="float16") = R.permute_dims(reshape9, axes=[1, 0, 2])
            permute_dims18: R.Tensor((24, 7, 128), dtype="float16") = R.permute_dims(reshape10, axes=[1, 0, 2])
            permute_dims19: R.Tensor((24, 128, 7), dtype="float16") = R.permute_dims(permute_dims17, axes=[0, 2, 1])
            matmul13: R.Tensor((24, 7, 7), dtype="float16") = R.matmul(permute_dims16, permute_dims19, out_dtype="void")
            mul10: R.Tensor((24, 7, 7), dtype="float16") = R.multiply(matmul13, R.const(0.08837890625, "float16"))
            add7: R.Tensor((24, 7, 7), dtype="float16") = R.add(mul10, mask)
            softmax1: R.Tensor((24, 7, 7), dtype="float16") = R.nn.softmax(add7, axis=-1)
            matmul14: R.Tensor((24, 7, 128), dtype="float16") = R.matmul(softmax1, permute_dims18, out_dtype="void")
            permute_dims20: R.Tensor((7, 24, 128), dtype="float16") = R.permute_dims(matmul14, axes=[1, 0, 2])
            reshape11: R.Tensor((7, 3072), dtype="float16") = R.reshape(permute_dims20, R.shape([7, 3072]))
            matmul15: R.Tensor((7, 3072), dtype="float16") = R.matmul(reshape11, permute_dims21, out_dtype="void")
            add8: R.Tensor((7, 3072), dtype="float16") = R.add(input_embeds, matmul15)
            rms_norm4: R.Tensor((7, 3072), dtype="float16") = R.nn.rms_norm(add8, layers_0_post_attention_layernorm_weight1, axes=[-1], epsilon=1.0000000000000001e-05)
            matmul16: R.Tensor((7, 8192), dtype="float16") = R.matmul(rms_norm4, permute_dims22, out_dtype="void")
            silu1: R.Tensor((7, 8192), dtype="float16") = R.nn.silu(matmul16)
            matmul17: R.Tensor((7, 8192), dtype="float16") = R.matmul(rms_norm4, permute_dims23, out_dtype="void")
            mul11: R.Tensor((7, 8192), dtype="float16") = R.multiply(silu1, matmul17)
            matmul18: R.Tensor((7, 3072), dtype="float16") = R.matmul(mul11, permute_dims24, out_dtype="void")
            add9: R.Tensor((7, 3072), dtype="float16") = R.add(add8, matmul18)
            permute_dims25: R.Tensor((8, 128, 7), dtype="float16") = R.permute_dims(add6, axes=[1, 2, 0])
            unsqueeze: R.Tensor((1, 8, 128, 7), dtype="float16") = R.expand_dims(permute_dims25, axis=[0])
            permute_dims26: R.Tensor((8, 7, 128), dtype="float16") = R.permute_dims(reshape8, axes=[1, 0, 2])
            unsqueeze1: R.Tensor((1, 8, 7, 128), dtype="float16") = R.expand_dims(permute_dims26, axis=[0])
            lv = R.call_tir(cls.last_hidden, (add9,), out_sinfo=R.Tensor((1, 3072), dtype="float16"))
            rms_norm5: R.Tensor((1, 3072), dtype="float16") = R.nn.rms_norm(lv, norm_weight1, axes=[-1], epsilon=1.0000000000000001e-05)
            matmul19: R.Tensor((1, 128256), dtype="float16") = R.matmul(rms_norm5, permute_dims27, out_dtype="void")
            concat4: R.Tensor((1, 8, 128, 7), dtype="float16") = R.concat((unsqueeze,), axis=0)
            concat5: R.Tensor((1, 8, 7, 128), dtype="float16") = R.concat((unsqueeze1,), axis=0)
            gv1: R.Tuple(R.Tensor((1, 128256), dtype="float16"), R.Tensor((1, 8, 128, 7), dtype="float16"), R.Tensor((1, 8, 7, 128), dtype="float16")) = matmul19, concat4, concat5
            R.output(gv1)
        return gv1

    @R.function
    def prefill_cache_transform_params(model_params: R.Tuple(R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")))) -> R.Tuple(R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((3072, 1024), dtype="float16"), R.Tensor((3072, 1024), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 128256), dtype="float16")):
        R.func_attr({"num_input": 0})
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
            permute_dims13: R.Tensor((3072, 3072), dtype="float16") = R.permute_dims(layers_0_self_attn_q_proj_weight1, axes=None)
            permute_dims14: R.Tensor((3072, 1024), dtype="float16") = R.permute_dims(layers_0_self_attn_k_proj_weight1, axes=None)
            permute_dims15: R.Tensor((3072, 1024), dtype="float16") = R.permute_dims(layers_0_self_attn_v_proj_weight1, axes=None)
            permute_dims21: R.Tensor((3072, 3072), dtype="float16") = R.permute_dims(layers_0_self_attn_o_proj_weight1, axes=None)
            permute_dims22: R.Tensor((3072, 8192), dtype="float16") = R.permute_dims(layers_0_mlp_gate_proj_weight1, axes=None)
            permute_dims23: R.Tensor((3072, 8192), dtype="float16") = R.permute_dims(layers_0_mlp_up_proj_weight1, axes=None)
            permute_dims24: R.Tensor((8192, 3072), dtype="float16") = R.permute_dims(layers_0_mlp_down_proj_weight1, axes=None)
            permute_dims27: R.Tensor((3072, 128256), dtype="float16") = R.permute_dims(lm_head_weight1, axes=None)
            output_tuple: R.Tuple(R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((3072, 1024), dtype="float16"), R.Tensor((3072, 1024), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 128256), dtype="float16")) = layers_0_input_layernorm_weight1, layers_0_post_attention_layernorm_weight1, norm_weight1, permute_dims13, permute_dims14, permute_dims15, permute_dims21, permute_dims22, permute_dims23, permute_dims24, permute_dims27
            R.output(output_tuple)
        return output_tuple

    @R.function
    def prefill_transform_params(model_params: R.Tuple(R.Tuple(R.Tensor((3072, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((1024, 3072), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((128256, 3072), dtype="float16")))) -> R.Tuple(R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((3072, 1024), dtype="float16"), R.Tensor((3072, 1024), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 128256), dtype="float16")):
        R.func_attr({"num_input": 0})
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
            permute_dims: R.Tensor((3072, 3072), dtype="float16") = R.permute_dims(layers_0_self_attn_q_proj_weight, axes=None)
            permute_dims1: R.Tensor((3072, 1024), dtype="float16") = R.permute_dims(layers_0_self_attn_k_proj_weight, axes=None)
            permute_dims2: R.Tensor((3072, 1024), dtype="float16") = R.permute_dims(layers_0_self_attn_v_proj_weight, axes=None)
            permute_dims8: R.Tensor((3072, 3072), dtype="float16") = R.permute_dims(layers_0_self_attn_o_proj_weight, axes=None)
            permute_dims9: R.Tensor((3072, 8192), dtype="float16") = R.permute_dims(layers_0_mlp_gate_proj_weight, axes=None)
            permute_dims10: R.Tensor((3072, 8192), dtype="float16") = R.permute_dims(layers_0_mlp_up_proj_weight, axes=None)
            permute_dims11: R.Tensor((8192, 3072), dtype="float16") = R.permute_dims(layers_0_mlp_down_proj_weight, axes=None)
            permute_dims12: R.Tensor((3072, 128256), dtype="float16") = R.permute_dims(lm_head_weight, axes=None)
            output_tuple: R.Tuple(R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072,), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((3072, 1024), dtype="float16"), R.Tensor((3072, 1024), dtype="float16"), R.Tensor((3072, 3072), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((3072, 8192), dtype="float16"), R.Tensor((8192, 3072), dtype="float16"), R.Tensor((3072, 128256), dtype="float16")) = layers_0_input_layernorm_weight, layers_0_post_attention_layernorm_weight, norm_weight, permute_dims, permute_dims1, permute_dims2, permute_dims8, permute_dims9, permute_dims10, permute_dims11, permute_dims12
            R.output(output_tuple)
        return output_tuple
