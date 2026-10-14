"""W8A8 schedules with explicit staging and a tensorized row quantizer.

The original semantic PrimFunc stays available for CPU execution. Its versioned
contract is lowered into SRAM buffers and loops; only hardware primitives remain
opaque. The matrix accumulator is a logical FP32 buffer whose save is explicit.
"""
import tvm
from tvm import tir
from tvm.script import tir as T

from .npu_w8a8 import is_w8a8

_REGISTERED = set()


def workspace_bytes(panel, k, n):
    """Exact bump footprint of the version-1 explicit workspace below.

    K and N are multiples of 64, so every allocation is DMA-cell aligned.
    All eight SRAM buffers are pinned by the current external-call contract.
    Schedule selection checks this against the actual TIR allocation layout.
    """
    return (3 * k + 4) * panel + 66 * k + 4 * n + 8196


def _quantizer(width):
    name = f"npu.quantize_row.{width}"
    if name in _REGISTERED:
        return name

    @T.prim_func
    def desc(x: T.handle, q: T.handle, s: T.handle, temp: T.handle, acc: T.handle):
        X = T.match_buffer(x, (width,), "float16", scope="global.sram", offset_factor=1)
        Q = T.match_buffer(q, (width,), "int8", scope="global.sram", offset_factor=1)
        S = T.match_buffer(s, (1,), "float32", scope="global.sram", offset_factor=1)
        Temp = T.match_buffer(temp, (width,), "float16", scope="global.sram", offset_factor=1)
        Acc = T.match_buffer(acc, (2,), "float16", scope="global.sram", offset_factor=1)
        with T.block("root"):
            T.reads(X[0:width])
            T.writes(Q[0:width], S[0:1], Temp[0:width], Acc[0:2])
            for k in range(width):
                with T.block("absmax"):
                    vk = T.axis.reduce(width, k)
                    with T.init():
                        S[0] = T.float32(0)
                    S[0] = T.max(S[0], T.abs(T.Cast("float32", X[vk])))
            for i in range(1):
                with T.block("scale"):
                    vi = T.axis.spatial(1, i)
                    S[vi] = T.if_then_else(S[vi] == T.float32(0), T.float32(1), S[vi] / T.float32(127))
            for k in range(width):
                with T.block("quant"):
                    vk = T.axis.spatial(width, k)
                    Q[vk] = T.Cast("int8", T.max(T.min(T.round(T.Cast("float32", X[vk]) / S[0]), T.float32(127)), T.float32(-127)))

    @T.prim_func
    def impl(x: T.handle, q: T.handle, s: T.handle, temp: T.handle, acc: T.handle):
        X = T.match_buffer(x, (width,), "float16", scope="global.sram", offset_factor=1)
        Q = T.match_buffer(q, (width,), "int8", scope="global.sram", offset_factor=1)
        S = T.match_buffer(s, (1,), "float32", scope="global.sram", offset_factor=1)
        Temp = T.match_buffer(temp, (width,), "float16", scope="global.sram", offset_factor=1)
        Acc = T.match_buffer(acc, (2,), "float16", scope="global.sram", offset_factor=1)
        with T.block("root"):
            T.reads(X[0:width])
            T.writes(Q[0:width], S[0:1], Temp[0:width], Acc[0:2])
            T.evaluate(T.call_extern("int32", "npu_quantize_row", X.access_ptr("r"), Q.access_ptr("w"), S.access_ptr("w"), Temp.access_ptr("rw"), Acc.access_ptr("rw"), width))

    tir.TensorIntrin.register(name, desc, impl)
    _REGISTERED.add(name)
    return name


def schedule(prim, panel):
    if not is_w8a8(prim) or int(prim.attrs.get("npu.kernel_version", 0)) != 1:
        raise ValueError("unsupported W8A8 semantic contract")
    bufs = [prim.buffer_map[p] for p in prim.params]
    if len(bufs) != 4 or tuple(str(b.dtype) for b in bufs) != ("float16", "int8", "float32", "float16"):
        raise ValueError("W8A8 requires FP16 X, INT8 W, FP32 scales, FP16 output")
    if tuple(len(b.shape) for b in bufs) != (2, 2, 1, 2):
        raise ValueError("invalid W8A8 ranks")
    m, k = map(int, bufs[0].shape)
    n = int(bufs[1].shape[1])
    if min(m, k, n) <= 0 or k % 64 or n % 64 or k > 65535:
        raise ValueError("W8A8 requires positive M, K/N multiples of 64 and K <= 65535")
    if tuple(map(int, bufs[1].shape)) != (k, n) or tuple(map(int, bufs[2].shape)) != (n,) or tuple(map(int, bufs[3].shape)) != (m, n):
        raise ValueError("inconsistent W8A8 shapes")
    # Panels divide a padded M domain; only valid rows are loaded/quantized.
    # Matrix primitive accepts the exact tail row count.
    tiles = (panel + 63) // 64
    blocks = (m + panel - 1) // panel

    @T.prim_func
    def kernel(x: T.Buffer((m, k), "float16"), w: T.Buffer((k, n), "int8"),
               ws: T.Buffer((n,), "float32"), out: T.Buffer((m, n), "float16")):
        T.func_attr({"tir.noalias": True, "npu.w8a8_lowered": 1})
        Temp = T.alloc_buffer((k,), "float16", scope="global.sram")
        Acc = T.alloc_buffer((2,), "float16", scope="global.sram")
        X = T.alloc_buffer((panel, k), "float16", scope="global.sram")
        Q = T.alloc_buffer((panel, k), "int8", scope="global.sram")
        AS = T.alloc_buffer((panel,), "float32", scope="global.sram")
        WS = T.alloc_buffer((n,), "float32", scope="global.sram")
        W = T.alloc_buffer((k, 64), "int8", scope="global.sram")
        C = T.alloc_buffer((64, 64), "float32", scope="local.accumulator")
        D = T.alloc_buffer((64, 64), "float16", scope="global.sram")
        for j in range(n):
            with T.block("load_scale"):
                vj = T.axis.spatial(n, j)
                WS[vj] = ws[vj]
        for p in range(blocks):
            for i, j in T.grid(T.min(panel, m - p * panel), k):
                with T.block("load_x"):
                    vp = T.axis.spatial(blocks, p)
                    vi, vj = T.axis.remap("SS", [i, j])
                    X[vi, vj] = x[vp * panel + vi, vj]
            for row in range(T.min(panel, m - p * panel)):
                with T.block("quantize"):
                    vr = T.axis.spatial(panel, row)
                    T.reads(X[vr, 0:k])
                    T.writes(Q[vr, 0:k], AS[vr:vr+1], Temp[0:k], Acc[0:2])
                    for j in range(k):
                        with T.block("absmax"):
                            vj = T.axis.reduce(k, j)
                            with T.init():
                                AS[vr] = T.float32(0)
                            AS[vr] = T.max(AS[vr], T.abs(T.Cast("float32", X[vr, vj])))
                    for i in range(1):
                        with T.block("scale"):
                            vi = T.axis.spatial(1, i)
                            AS[vr + vi] = T.if_then_else(AS[vr + vi] == T.float32(0), T.float32(1), AS[vr + vi] / T.float32(127))
                    for j in range(k):
                        with T.block("quant"):
                            vj = T.axis.spatial(k, j)
                            Q[vr, vj] = T.Cast("int8", T.max(T.min(T.round(T.Cast("float32", X[vr, vj]) / AS[vr]), T.float32(127)), T.float32(-127)))
            for col in range(n // 64):
                for i, j in T.grid(k, 64):
                    with T.block("load_w"):
                        vc = T.axis.spatial(n // 64, col)
                        vi, vj = T.axis.remap("SS", [i, j])
                        W[vi, vj] = w[vi, vc * 64 + vj]
                for tile in range(tiles):
                    if tile * 64 < T.min(panel, m - p * panel):
                        for kt in range(k // 64):
                            with T.block("scaled_mac"):
                                T.reads(Q[tile*64:T.min((tile+1)*64, T.min(panel, m-p*panel)), kt*64:(kt+1)*64], W[kt*64:(kt+1)*64, 0:64], AS[tile*64:T.min((tile+1)*64, T.min(panel, m-p*panel))], WS[col*64:(col+1)*64], C[0:64, 0:64])
                                T.writes(C[0:64, 0:64])
                                T.evaluate(T.call_extern("int32", "npu_scaled_mac", C.access_ptr("rw"), Q.access_ptr("r", offset=tile*64*k+kt*64), k, W.access_ptr("r", offset=kt*64*64), AS.access_ptr("r", offset=tile*64), WS.access_ptr("r", offset=col*64), T.min(64, T.min(panel, m-p*panel)-tile*64), kt == 0))
                        with T.block("store_accumulator"):
                            T.reads(C[0:64, 0:64])
                            T.writes(D[0:64, 0:64])
                            T.evaluate(T.call_extern("int32", "npu_store_scaled", C.access_ptr("r"), D.access_ptr("w"), T.min(64, T.min(panel, m-p*panel)-tile*64)))
                        for i, j in T.grid(T.min(64, T.min(panel, m-p*panel)-tile*64), 64):
                            with T.block("store_out"):
                                vp = T.axis.spatial(blocks, p)
                                vt = T.axis.spatial(tiles, tile)
                                vc = T.axis.spatial(n // 64, col)
                                vi, vj = T.axis.remap("SS", [i, j])
                                out[vp*panel + vt*64 + vi, vc*64 + vj] = D[vi, vj]
        T.evaluate(T.call_extern("int32", "npu_reset_quant_descriptors"))

    kernel = tir.transform.Simplify()(tvm.IRModule({"main": kernel}))["main"]
    sch = tir.Schedule(kernel)
    sch.tensorize(sch.get_block("quantize"), _quantizer(k))
    return sch.mod["main"]
