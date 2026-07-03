"""End-to-end driver: Relax IRModule -> NPU ISA -> run on given mysim -> output.

Ties together memplan + codegen + runtime. B0 (logical) path.
"""
import numpy as np
from . import memplan as _memplan
from . import codegen as _codegen
from . import runtime as _runtime


def compile_func(func, tile=None):
    """Relax function -> (Asm program, MemPlan). tile=64 enables B0.5 K-tiling."""
    mp = _memplan.plan(func)
    asm = _codegen.compile_func(func, mp, tile=tile)
    return asm, mp


def compile_module(mod, func_name="main", tile=None, backend="direct", fuse_oproj=True,
                   pack_params=False):
    """Relax module -> (asm, mp). Split out so a caller can compile ONCE and reuse
    the compiled program across many runs — e.g. the SAME decode kernel serves all
    28 layers and every token (only the weight *inputs* differ). This is what makes
    full-size generation practical: compile ~once, then just re-run on mysim.

    pack_params=True pre-packs static weight PARAMS (tile-blocked) so the reused
    kernel reads weights contiguously (no gather); run_compiled packs the fed data."""
    if backend == "tir":                      # matmul-only modules via pure TIR path
        from . import tir_backend
        return tir_backend.compile_func(mod, func_name)
    if backend == "hybrid":                   # whole graph: matmul->TIR, rest->direct
        func = mod[func_name]
        mp = _memplan.plan(func, pack_params=pack_params)
        asm = _codegen.compile_func(func, mp, tile=64, mm_backend="tir", fuse_oproj=fuse_oproj)
        return asm, mp
    func = mod[func_name]
    return compile_func(func, tile=tile)


def _pack2d(arr, K, N, T=64):
    """Tile-block a [K,N] weight -> [Kt,Nt,T,T] flattened (same layout as
    memplan.alloc_const_packed), so a packed-param kernel reads it with no gather."""
    return arr.reshape(K, N).reshape(K // T, T, N // T, T).transpose(0, 2, 1, 3).reshape(-1)


def run_compiled(asm, mp, inputs, maxrun=None):
    """Run an already-compiled (asm, mp). Rebuilds the G-buffer from `inputs` each
    call (cheap) and runs mysim — NO recompilation. inputs: dict {name->array} or
    positional list. Returns the output (float32, FP16-rounded)."""
    gbuf = np.zeros(mp.top, dtype=np.float32)
    for c in mp.constants:                         # bake constant data into initial G-buffer
        data = mp.const_data[c].astype(np.float32).reshape(-1)
        gbuf[mp.offset[c]:mp.offset[c] + data.size] = data
    for i, p in enumerate(mp.params):
        if isinstance(inputs, dict):
            if p.name_hint not in inputs:
                raise KeyError(f"missing input for param '{p.name_hint}'")
            src = inputs[p.name_hint]
        else:
            src = inputs[i]
        arr = np.asarray(src, dtype=np.float32).reshape(-1)
        off = mp.offset[p]
        if arr.size != _numel(mp.shape[p]):
            raise ValueError(f"param #{i} '{p.name_hint}' size {arr.size} != {mp.shape[p]}")
        if p in mp.packed_meta:                    # packed weight param: reorder to tile-blocked
            K, N = mp.shape[p]
            arr = _pack2d(arr, K, N)
        gbuf[off:off + arr.size] = arr
    out_var = mp.output
    out_off = mp.offset[out_var]; n_out = _numel(mp.shape[out_var])
    full = _runtime.run(asm, gbuf, gn=mp.top, maxrun=maxrun)
    return full[out_off:out_off + n_out].reshape(mp.shape[out_var])


def run_module(mod, inputs, func_name="main", maxrun=None, tile=None, backend="direct",
               fuse_oproj=True, pack_params=False):
    """Convenience: compile + run a Relax module on mysim (recompiles each call).
    For loops that reuse one kernel, prefer compile_module()+run_compiled()."""
    asm, mp = compile_module(mod, func_name, tile, backend, fuse_oproj, pack_params)
    return run_compiled(asm, mp, inputs, maxrun)


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


# ============================ prefill -> decode generation ============================
NEG = -1e4          # additive attention mask for invalid/future cache slots (exp -> 0)


def generate(cfg, W, X_full, MAX, cos, sin, rot, backend="hybrid", pack_weights=True):
    """Host generation loop with a static max-length KV cache.

    Processes tokens X_full[p] one at a time (prefill prompt + decode), chaining the
    two decode kernels (model.build_kv_proj_module / build_attn_ffn_module) and
    managing the cache in host memory. Only the dynamic step — writing the new
    token's K/V at slot pos — happens here, so both kernels stay static/unrollable.

    pack_weights=True pre-packs static weight params (no weight gather; byte-exact).
    Cache: Kt[k]=[HD,MAX] (K transposed -> append column), Vc[k]=[MAX,HD] (append row).
    Returns the layer output hidden states [N,D]. cos/sin/rot are rope tables (>=MAX rows).
    """
    from . import model
    HD, KV = cfg.HD, cfg.KV
    N = X_full.shape[0]
    assert N <= MAX, f"sequence {N} exceeds cache MAX {MAX}"
    kv_c = compile_module(model.build_kv_proj_module(cfg), backend=backend, pack_params=pack_weights)
    attn_c = compile_module(model.build_attn_ffn_module(cfg, MAX), backend=backend, pack_params=pack_weights)
    Kt = [np.zeros((HD, MAX), np.float16) for _ in range(KV)]
    Vc = [np.zeros((MAX, HD), np.float16) for _ in range(KV)]
    f16 = lambda a: np.asarray(a, np.float16)
    outs = []
    for p in range(N):
        x_new = f16(X_full[p:p + 1]); cosp, sinp = f16(cos[p:p + 1]), f16(sin[p:p + 1])
        outs.append(_decode_layer(cfg, kv_c, attn_c, W, x_new, Kt, Vc, p, MAX, cosp, sinp))
    return np.concatenate(outs, axis=0)


def _decode_layer(cfg, kv_c, attn_c, W, x, Kt, Vc, p, MAX, cosp, sinp):
    """Run one decoder layer for token at position p using PRECOMPILED kernels
    (kv_c/attn_c = (asm, mp)): project+append K/V, then attention(masked to <=p)+FFN.
    x[1,D] -> next hidden[1,D]."""
    HD, KV, H = cfg.HD, cfg.KV, cfg.H
    f16 = lambda a: np.asarray(a, np.float16)
    ins1 = {"x": x, "Wn1": W["Wn1"], "cos": cosp, "sin": sinp}
    for k in range(KV):
        ins1[f"Wk{k}"] = W[f"Wk{k}"]; ins1[f"Wv{k}"] = W[f"Wv{k}"]
    o1 = run_compiled(*kv_c, ins1)
    for k in range(KV):
        Kt[k][:, p] = o1[0, 2 * k * HD:(2 * k + 1) * HD]
        Vc[k][p, :] = o1[0, (2 * k + 1) * HD:(2 * k + 2) * HD]
    mask = np.zeros((1, MAX), np.float16); mask[0, p + 1:] = NEG
    ins2 = {"x": x, "Wn1": W["Wn1"], "Wn2": W["Wn2"], "cos": cosp, "sin": sinp,
            "mask": mask, "Wg": W["Wg"], "Wu": W["Wu"], "Wd": W["Wd"]}
    for h in range(H):
        ins2[f"Wq{h}"] = W[f"Wq{h}"]; ins2[f"Wo{h}"] = W[f"Wo{h}"]
    for k in range(KV):
        ins2[f"Kt{k}"] = Kt[k]; ins2[f"Vc{k}"] = Vc[k]
    return f16(run_compiled(*attn_c, ins2))


def _decode_all_layers(cfg, kv_c, attn_c, layer_Ws, Kt, Vc, x, p, MAX, cosp, sinp):
    """One token through all N decoder layers at position p. x[1,D] -> hidden[1,D]."""
    for l in range(len(layer_Ws)):
        x = _decode_layer(cfg, kv_c, attn_c, layer_Ws[l], x, Kt[l], Vc[l], p, MAX, cosp, sinp)
    return x


def _batched_prefill(cfg, prefill_c, layer_Ws, top, prompt_ids, Kt, Vc, cos, sin):
    """Run all layers over the whole prompt at once (PRECOMPILED prefill_c, one call
    per layer), seeding every layer's cache (Kk.T -> columns, Vk -> rows) and returning
    the prompt hidden [P,D]."""
    HD, KV, H, D = cfg.HD, cfg.KV, cfg.H, cfg.D
    P = len(prompt_ids)
    f16 = lambda a: np.asarray(a, np.float16)
    x = f16(top["Wemb"][np.asarray(prompt_ids)])                    # [P,D]
    cosP, sinP = f16(cos[:P]), f16(sin[:P])
    for l in range(len(layer_Ws)):
        W = layer_Ws[l]
        ins = {"x": x, "Wn1": W["Wn1"], "Wn2": W["Wn2"], "cos": cosP, "sin": sinP,
               "Wg": W["Wg"], "Wu": W["Wu"], "Wd": W["Wd"]}
        for h in range(H):
            ins[f"Wq{h}"] = W[f"Wq{h}"]; ins[f"Wo{h}"] = W[f"Wo{h}"]
        for k in range(KV):
            ins[f"Wk{k}"] = W[f"Wk{k}"]; ins[f"Wv{k}"] = W[f"Wv{k}"]
        out = run_compiled(*prefill_c, ins)                        # [P, D + 2*KV*HD]
        x = f16(out[:, :D]); off = D
        for k in range(KV):
            Kt[l][k][:, :P] = out[:, off:off + HD].T; off += HD     # K -> columns 0..P-1
            Vc[l][k][:P, :] = out[:, off:off + HD]; off += HD       # V -> rows 0..P-1
    return x


def generate_tokens(cfg, layer_Ws, top, prompt_ids, n_gen, MAX, cos, sin, rot,
                    backend="hybrid", batched_prefill=False, verbose=False, pack_weights=True):
    """Full greedy autoregressive generation: [CPU] embed -> [NPU] N layers + lm_head
    -> [CPU] argmax -> loop. Returns the generated token ids.

    **Kernels are compiled ONCE** (kv/attn/lm/prefill) and reused for every layer and
    every token — the graphs are identical across layers, only the weight *inputs*
    differ. This is what makes full-size (e.g. 3B) generation practical.

    batched_prefill=False: prompt processed token-by-token; True: one kernel/layer."""
    from . import model
    import time
    HD, KV = cfg.HD, cfg.KV
    nL = len(layer_Ws); P = len(prompt_ids)
    assert P >= 1 and P + n_gen <= MAX
    t0 = time.time()
    pk = pack_weights
    kv_c = compile_module(model.build_kv_proj_module(cfg), backend=backend, pack_params=pk)
    attn_c = compile_module(model.build_attn_ffn_module(cfg, MAX), backend=backend, pack_params=pk)
    lm_c = compile_module(model.build_lm_head_module(cfg, top["Wlm"].shape[1]), backend=backend, pack_params=pk)
    prefill_c = (compile_module(model.build_prefill_layer_module(cfg, P), backend=backend, pack_params=pk)
                 if batched_prefill else None)
    if verbose:
        print(f"  [compile] {'4' if batched_prefill else '3'} kernels compiled once in "
              f"{time.time()-t0:.1f}s (reused for all {nL} layers x {n_gen} tokens)")
    Kt = [[np.zeros((HD, MAX), np.float16) for _ in range(KV)] for _ in range(nL)]
    Vc = [[np.zeros((MAX, HD), np.float16) for _ in range(KV)] for _ in range(nL)]
    emb = top["Wemb"]; f16 = lambda a: np.asarray(a, np.float16)

    def argmax_h(h):
        return int(np.argmax(run_compiled(*lm_c, {"h": h, "Wn_f": top["Wn_f"], "Wlm": top["Wlm"]})[0]))

    tokens = list(prompt_ids)
    if batched_prefill:
        x = _batched_prefill(cfg, prefill_c, layer_Ws, top, prompt_ids, Kt, Vc, cos, sin)
        tokens.append(argmax_h(x[-1:]))                            # 1st gen from last prompt hidden
        p = P
    else:
        p = 0
    while len(tokens) < P + n_gen:
        ts = time.time()
        xt = f16(emb[tokens[p]:tokens[p] + 1])                     # [CPU] embedding lookup
        xt = _decode_all_layers(cfg, kv_c, attn_c, layer_Ws, Kt, Vc, xt, p, MAX,
                                f16(cos[p:p + 1]), f16(sin[p:p + 1]))
        if p >= P - 1:                                             # [NPU] lm_head, [CPU] argmax
            tokens.append(argmax_h(xt))
        if verbose:
            tok = tokens[-1] if p >= P - 1 else "(prefill)"
            print(f"  [step p={p}] {nL} layers on mysim in {time.time()-ts:.1f}s -> {tok}")
        p += 1
    return tokens[P:]
