"""F4 (Stage 4 Phase 4.3): _build_packed(mod) — a PRE-LEGALIZE Relax transform that
rewrites the high-level layer graph (model.build_layer_module) into TILE-BLOCKED
(packed 4D [Rt,Ct,64,64]) form, so that after relax.transform.LegalizeOps() the tile
ops become native TIR (packed einsum matmul, layout-transparent 4D elementwise/reduce/
broadcast) and the residual stream stays tile end-to-end.

Design (validated below):
  * Params keep their ORIGINAL 2D shape in the signature; every layout conversion is an
    in-graph relax.op.layout_transform. => feeds are BYTE-IDENTICAL to the row module,
    output is unpacked back to [seq,d] row (compare straight to model.ref_layer).
  * Generic pack map (r,c)->(r//64,c//64,r%64,c%64) covers every 2D shape:
        [R,C]->[Rt,Ct,64,64]   [R,1]->[Rt,1,64,1]   [1,C]->[1,Ct,1,64]   [1,1]->[1,1,1,1]
    and its inverse (a,b,i,j)->(a*64+i,b*64+j) unpacks any of them.
  * Per-op layout policy:
      matmul                           -> TILE  (packed einsum "acik,cdkj->adij")
      strided_slice/concat/permute_dims-> ROW   (rebuilt verbatim on row-coerced inputs)
      everything else (ew/unary/cos/sin/sum/max/broadcast_to) -> LAYOUT-FOLLOWING:
          output = row iff any INTERMEDIATE-var input is row, else tile
          (param/const leaves do not vote; they are coerced to the chosen layout)
    => the rotate-half (slice/negative/concat) + K permute_dims form a ROW ISLAND reached
       by unpacking Q/K at the slice/add boundary; the residual/RMSNorm/FFN/proj/o-proj
       stream stays TILE with zero interior unpack.
"""
import numpy as np
import tvm
from tvm import relax
from tvm.relax.op import layout_transform, einsum, reshape
import tvm.topi  # noqa: register legalization

T = 64
_PACK = lambda r, c: (r // T, c // T, r % T, c % T)
_UNPACK = lambda a, b, i, j: (a * T + i, b * T + j)


def _t4(shape):
    """2D logical [R,C] -> its packed 4D shape [Rt,Ct,ri,ci]. A size-1 axis keeps
    extent 1 (NOT padded to 64) — it is a contiguous reshape, not a tile-interleave."""
    R, C = shape
    rt, ri = (R // T, T) if R > 1 else (1, 1)
    ct, ci = (C // T, T) if C > 1 else (1, 1)
    return [rt, ct, ri, ci]


def _is_full(shape):
    """True iff BOTH logical dims are >1 (needs the tile-transpose layout_transform);
    a row/col/scalar vector is packed by a pure contiguous reshape instead (a %64 index
    map would WRONGLY pad the size-1 axis to extent 64 -> garbage)."""
    return shape[0] > 1 and shape[1] > 1

_ROW_ONLY = {"relax.strided_slice", "relax.concat", "relax.permute_dims"}
_MATMUL = "relax.matmul"
# ops the packed pass knows how to run TILE (layout-transparent / packed-reduce); anything
# else (incl _ROW_ONLY and any op from a different model) falls back to a ROW island.
_LAYOUT_FOLLOWING = {"relax.add", "relax.subtract", "relax.multiply", "relax.divide",
                     "relax.sqrt", "relax.exp", "relax.negative", "relax.cos", "relax.sin",
                     "relax.nn.silu", "relax.sum", "relax.max", "relax.broadcast_to"}


def _shape2(sinfo_holder):
    return [int(d) for d in sinfo_holder.struct_info.shape]


class _Info:
    """One rewritten tensor: its 2D logical shape + materializations per layout."""
    __slots__ = ("shape", "mat")

    def __init__(self, shape):
        self.shape = shape          # logical 2D [R,C] (or [R,1]/[1,C]/[1,1])
        self.mat = {}               # layout -> relax expr; leaf seeded {"leaf":expr},
                                    # intermediates tagged mat["__interm__"] in {"row","tile"}


def _emit_pack(bb, e, shape):
    if _is_full(shape):
        if isinstance(e, relax.Constant):                      # host-pack the constant (free):
            from . import memplan                              # emit it already tile-blocked, so
            arr = np.asarray(e.data.numpy())                   # no on-device layout_transform is
            packed = memplan.pack_tiled(arr, T).reshape(_t4(shape))  # needed to pack it (like weights)
            return relax.const(packed, str(e.struct_info.dtype))
        return bb.emit(layout_transform(e, index_map=_PACK))
    return bb.emit(reshape(e, relax.ShapeExpr(_t4(shape))))     # vector: contiguous reshape


def _emit_unpack(bb, e, shape):
    if _is_full(shape):
        return bb.emit(layout_transform(e, index_map=_UNPACK))
    return bb.emit(reshape(e, relax.ShapeExpr(list(shape))))    # vector: contiguous reshape


def _coerce(bb, info, want):
    """Return `info` materialized in layout `want` ('row' | 'tile'), inserting one
    transform if needed. Leaves (only 'leaf' present) pack/unpack from the raw 2D.
    Full 2D uses layout_transform (tile-transpose); a row/col/scalar vector uses a
    contiguous reshape (a %64 index map would pad its size-1 axis to 64 = garbage)."""
    if want in info.mat:
        return info.mat[want]
    if "leaf" in info.mat and "row" not in info.mat:
        info.mat["row"] = info.mat["leaf"]      # raw 2D expr is the row form
    if want == "row":
        src = info.mat.get("tile")
        assert src is not None
        e = _emit_unpack(bb, src, info.shape)
    else:  # tile
        src = info.mat.get("row")
        assert src is not None
        e = _emit_pack(bb, src, info.shape)
    info.mat[want] = e
    return e


def _build_packed(mod, target_name=None):
    """Returns (packed_module, pack_names). pack_names = weight params (used ONLY as matmul
    operands, 64-mult 2D) declared with the packed 4D shape and seeded already-tile, so they
    are fed PRE-PACKED (host, free) instead of packed on-device via layout_transform."""
    old = mod["main"]
    params = list(old.params)
    uses = {}                                   # param Var -> [used-as-matmul-operand? ...]
    for b in old.body.blocks[0].bindings:
        c = b.value
        if isinstance(c, relax.Call):
            mm = getattr(c.op, "name", "") == _MATMUL
            for a in c.args:
                if isinstance(a, relax.Var):
                    uses.setdefault(a, []).append(mm)
                elif isinstance(a, relax.Tuple):
                    for f in a.fields:
                        if isinstance(f, relax.Var):
                            uses.setdefault(f, []).append(False)
    pack_names, packed_of, new_params = set(), {}, []
    for p in params:
        sh = _shape2(p); u = uses.get(p, [])
        if u and all(u) and len(sh) == 2 and sh[0] % 64 == 0 and sh[1] % 64 == 0:
            pv = relax.Var(p.name_hint, relax.TensorStructInfo(_t4(sh), p.struct_info.dtype))
            pack_names.add(p.name_hint); packed_of[p] = pv; new_params.append(pv)
        else:
            new_params.append(p)

    bb = relax.BlockBuilder()
    env = {}                                    # old Var -> _Info

    with bb.function("main", new_params):
        with bb.dataflow():
            for p in params:
                inf = _Info(_shape2(p))
                if p in packed_of:
                    inf.mat["tile"] = packed_of[p]   # fed pre-packed -> already tile (no transform)
                else:
                    inf.mat["leaf"] = p
                env[p] = inf

            target_info = None
            for binding in old.body.blocks[0].bindings:
                var = binding.value
                dst = binding.var
                # alias binding (gv = somevar)
                if isinstance(var, relax.Var):
                    env[dst] = env[var]; continue
                assert isinstance(var, relax.Call), type(var)
                name = getattr(var.op, "name", "")
                out_shape = _shape2(dst)

                def arg_info(a):
                    if isinstance(a, relax.Var):
                        return env[a]
                    if isinstance(a, relax.Constant):
                        inf = _Info([int(d) for d in a.struct_info.shape]); inf.mat["leaf"] = a
                        return inf
                    return None

                if name == _MATMUL:
                    ai = arg_info(var.args[0]); bi = arg_info(var.args[1])
                    A = _coerce(bb, ai, "tile"); B = _coerce(bb, bi, "tile")
                    out = bb.emit(einsum((A, B), subscripts="acik,cdkj->adij"))
                    ninf = _Info(out_shape); ninf.mat["tile"] = out; ninf.mat["__interm__"] = "tile"
                    env[dst] = ninf
                    if target_name is not None and dst.name_hint == target_name:
                        target_info = ninf
                    continue

                # ---- RoPE/attention structural ops: TILE-native when tile-aligned (hd/2 a
                # 64-multiple), else fall through to the row island below. ----
                _tiled = None
                if name == "relax.strided_slice":
                    _tiled = _tile_slice(bb, var, arg_info)
                elif name == "relax.concat":
                    _tiled = _tile_concat(bb, var, arg_info, out_shape)
                elif name == "relax.permute_dims":
                    _tiled = _tile_transpose(bb, var, arg_info, out_shape)
                if _tiled is not None:
                    env[dst] = _mk_tile(out_shape, _tiled)
                    if target_name is not None and dst.name_hint == target_name:
                        target_info = env[dst]
                    continue

                if name in _LAYOUT_FOLLOWING:               # ew/unary/reduce/broadcast -> follow inputs
                    tensor_args = [a for a in _iter_tensor_args(var)]
                    infos = [arg_info(a) for a in tensor_args]
                    want = "row" if any(i.mat.get("__interm__") == "row" for i in infos) else "tile"
                    coerced = [_coerce(bb, i, want) for i in infos]
                    out = _emit_following(bb, name, var, coerced, out_shape, want)
                    ninf = _Info(out_shape); ninf.mat[want] = out; ninf.mat["__interm__"] = want
                    env[dst] = ninf
                    if target_name is not None and dst.name_hint == target_name:
                        target_info = env[dst]
                    continue

                # ---- ROW-ONLY / UNKNOWN op (model-agnostic fallback): rebuild the SAME call
                # verbatim on row-coerced tensor inputs. Any op the pass can't tile stays a row
                # island -> a different model's ops flow through (correct, just not tile-opt). ----
                new_args = []
                for a in var.args:
                    if isinstance(a, (relax.Var, relax.Constant)):
                        new_args.append(_coerce(bb, arg_info(a), "row"))
                    elif isinstance(a, relax.Tuple):
                        new_args.append(relax.Tuple(
                            [_coerce(bb, arg_info(f), "row") if isinstance(f, (relax.Var, relax.Constant)) else f
                             for f in a.fields]))
                    else:
                        new_args.append(a)
                out = bb.emit(relax.Call(var.op, new_args, var.attrs, var.sinfo_args))
                ninf = _Info(out_shape); ninf.mat["row"] = out; ninf.mat["__interm__"] = "row"
                env[dst] = ninf
                if target_name is not None and dst.name_hint == target_name:
                    target_info = ninf

            ret = old.body.body                 # returned Var
            chosen = target_info if target_name is not None else env[ret]
            y = _coerce(bb, chosen, "row")
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize(), pack_names


def _iter_tensor_args(call):
    """Tensor-typed args of a layout-following op (skips ShapeExpr/attrs)."""
    name = call.op.name
    if name == "relax.broadcast_to":
        return [call.args[0]]
    # sum/max: single tensor; elementwise/unary/cos/sin/silu: all args that are tensors
    out = []
    for a in call.args:
        if isinstance(a, (relax.Var, relax.Constant)):
            out.append(a)
    return out


def _mk_tile(shape, expr):
    inf = _Info(shape); inf.mat["tile"] = expr; inf.mat["__interm__"] = "tile"
    return inf


def _tile_slice(bb, call, arg_info):
    """RoPE half-slice in TILE layout: q[:, b:e] on a single axis whose begin/end are 64-
    multiples becomes a whole-tile-column (or row) slice of the packed [St,Ct,64,64] tensor
    (index_map-free, no unpack). Returns the tile expr, or None to fall back to the row island
    (sub-tile begin/end, e.g. hd/2 not a multiple of 64)."""
    axes = [int(x.value) for x in call.args[1]]              # args are Tuples of PrimValue
    begin = [int(x.value) for x in call.args[2]]
    end = [int(x.value) for x in call.args[3]]
    if len(axes) != 1 or axes[0] not in (0, 1) or begin[0] % T or end[0] % T:
        return None
    ai = arg_info(call.args[0])
    if not _is_full(ai.shape):
        return None
    A = _coerce(bb, ai, "tile")
    return bb.emit(relax.op.strided_slice(A, axes=[axes[0]], begin=[begin[0] // T], end=[end[0] // T]))


def _tile_concat(bb, call, arg_info, out_shape):
    """Concat in TILE layout along the row-tile or col-tile axis (every input aligned to 64
    on that axis). Returns the tile expr, or None to fall back to the row island."""
    axis = int(call.attrs.axis)
    if axis not in (0, 1) or not _is_full(out_shape):
        return None
    infos = [arg_info(f) for f in call.args[0].fields]
    for i in infos:
        if not _is_full(i.shape) or i.shape[axis] % T:
            return None
    coerced = [_coerce(bb, i, "tile") for i in infos]
    return bb.emit(relax.op.concat(coerced, axis=axis))


def _tile_transpose(bb, call, arg_info, out_shape):
    """permute_dims([1,0]) in TILE layout = permute_dims([1,0,3,2]) on the packed tensor:
    [St,Ct,64,64] -> [Ct,St,64,64] transposing BOTH the tile grid ([1,0]) and each 64x64 tile
    ([3,2]). out_tile[a,b,i,j] = in_tile[b,a,j,i] == the logical transpose. Returns the tile
    expr, or None (non-[1,0] permute) to fall back to the row island."""
    axes = [int(x) for x in call.attrs.axes]
    if axes != [1, 0] or not _is_full(out_shape):
        return None
    ai = arg_info(call.args[0])
    if not _is_full(ai.shape):
        return None
    A = _coerce(bb, ai, "tile")
    return bb.emit(relax.op.permute_dims(A, axes=[1, 0, 3, 2]))


def _emit_following(bb, name, call, coerced, out_shape, want):
    op = relax.op
    if name in ("relax.add", "relax.subtract", "relax.multiply", "relax.divide"):
        f = {"relax.add": op.add, "relax.subtract": op.subtract,
             "relax.multiply": op.multiply, "relax.divide": op.divide}[name]
        return bb.emit(f(coerced[0], coerced[1]))
    if name in ("relax.sqrt", "relax.exp", "relax.negative", "relax.cos", "relax.sin"):
        f = {"relax.sqrt": op.sqrt, "relax.exp": op.exp, "relax.negative": op.negative,
             "relax.cos": op.cos, "relax.sin": op.sin}[name]
        return bb.emit(f(coerced[0]))
    if name == "relax.nn.silu":
        return bb.emit(op.nn.silu(coerced[0]))
    if name in ("relax.sum", "relax.max"):
        f = op.sum if name == "relax.sum" else op.max
        axis = [1, 3] if want == "tile" else [-1]
        return bb.emit(f(coerced[0], axis=axis, keepdims=True))
    if name == "relax.broadcast_to":
        tgt = relax.ShapeExpr(_t4(out_shape) if want == "tile" else list(out_shape))
        return bb.emit(op.broadcast_to(coerced[0], tgt))
    raise NotImplementedError(name)


# ============================ validation harness ============================
def _count_lt(mod):
    """(layout_transform count, reshape count) surviving in the pre-legalize packed graph."""
    c = {"relax.layout_transform": 0, "relax.reshape": 0}

    def visit(e):
        if isinstance(e, relax.Call):
            nm = getattr(e.op, "name", "")
            if nm in c:
                c[nm] += 1
    for gv, fn in mod.functions.items():
        if isinstance(fn, relax.Function):
            relax.analysis.post_order_visit(fn, visit)
    return c["relax.layout_transform"], c["relax.reshape"]


def _param_order(cfg):
    return (["x", "Wn1", "Wn2"] + [f"Wq{h}" for h in range(cfg.H)] +
            [f"Wo{h}" for h in range(cfg.H)] + [f"Wk{k}" for k in range(cfg.KV)] +
            [f"Wv{k}" for k in range(cfg.KV)] + ["Wg", "Wu", "Wd"])


def _run_vm(mod, cfg, W):
    from . import memplan
    feeds = []
    for p in mod["main"].params:                    # feed by param rank: 4D weight -> pre-packed
        arr = np.asarray(W[p.name_hint], np.float16)
        if len(p.struct_info.shape) == 4:
            R, C = W[p.name_hint].shape
            arr = memplan.pack_tiled(arr, 64).reshape(R // 64, C // 64, 64, 64)
        feeds.append(tvm.nd.array(arr))
    ex = relax.build(relax.transform.LegalizeOps()(mod), target="llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())
    return vm["main"](*feeds).numpy()


def _rel(a, b):
    a = np.asarray(a, np.float32).reshape(-1); b = np.asarray(b, np.float32).reshape(-1)
    return float(np.max(np.abs(a - b))) / (float(np.max(np.abs(b))) + 1e-9)


def validate(cfg):
    from npu_compiler import model
    cos, sin, rot = model.rope_tables(cfg)
    W = model.make_weights(cfg, seed=7)
    hi = model.build_layer_module(cfg)

    packed, _pack_names = _build_packed(hi)
    n_lt, n_rs = _count_lt(packed)

    out_packed = _run_vm(packed, cfg, W)
    out_row = _run_vm(hi, cfg, W)                         # row module (same VM), fp16 oracle
    exp = np.asarray(model.ref_layer(cfg, W, cos, sin), np.float32)

    rel_ref = _rel(out_packed, exp)
    rel_row = _rel(out_packed, out_row)

    # also compare vs current tile compile_module (mysim)
    rel_cm = None
    try:
        rel_cm = _run_compile_module(cfg, W, out_packed)
    except Exception as e:
        rel_cm = f"SKIP({e})"
    return dict(cfg=cfg.name, n_lt=n_lt, n_rs=n_rs, rel_ref=rel_ref, rel_row=rel_row,
                rel_cm=rel_cm, shape=out_packed.shape)


def _run_compile_module(cfg, W, out_packed):
    """Run the CURRENT v2.compile_module(tile=True) on mysim and rel vs packed VM output."""
    import sys, os
    sys.path.insert(0, "/home/chokwans99/NPU_cmodel/d_compiler/tests")
    from npu_compiler import v2_backend as v2, runtime, memplan, model
    from test_v2 import _feed, _read_out
    mod = relax.transform.LegalizeOps()(model.build_layer_module(cfg))
    asm, off, shp, top, outn, consts, tf, lay = v2.compile_module(mod, tile=True)
    g = np.zeros(top + 200000, np.float32)
    for co, arr in consts:
        g[co:co + arr.size] = np.asarray(arr, np.float32).reshape(-1)
    _feed(g, off, tf, list(W.items()))
    gbuf = runtime.run(asm.words, g, gn=top)
    out_cm = _read_out(gbuf, off, shp, outn, lay)
    return _rel(out_packed, out_cm)


if __name__ == "__main__":
    from npu_compiler import model
    CFGS = [
        model.MEDIUM,
        model.LayerConfig("s128", SEQ=128, D=128, H=2, KV=2, HD=64, F=256,
                          eps=1e-5, rope_base=1e4, rope_scale=False),
        model.LayerConfig("hd128", SEQ=128, D=512, H=4, KV=2, HD=128, F=256,
                          eps=1e-5, rope_base=1e4, rope_scale=False),
    ]
    for cfg in CFGS:
        r = validate(cfg)
        print(f"[{r['cfg']:>9}] out{r['shape']} lt={r['n_lt']:3d} reshape={r['n_rs']:3d}  "
              f"rel_vs_ref={r['rel_ref']:.4f}  rel_vs_rowVM={r['rel_row']:.4f}  "
              f"rel_vs_compile_module={r['rel_cm']}")
