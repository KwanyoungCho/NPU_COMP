"""Relax-level legalization helpers — decompose ops the NPU ISA lacks into
NPU-supported primitives (matmul + elementwise + sqrt/exp), per report.md §5.

These are *builders* (emit decomposed ops into a relax.BlockBuilder), reusable
when constructing the model graph. The key tricks:
  - reduce-sum over last dim  ->  relax.sum (dedicated reduce op)
  - broadcast                 ->  relax.broadcast_to (dedicated op)
  - 1/x                       ->  divide(ones, x)
reduce/broadcast are NOT matmul nodes (codegen lowers them on the matrix engine),
so relax.matmul stays exclusively true GEMM -> the TIR-only backend.
softmax max-subtraction is intentionally omitted.
"""
import numpy as np
from tvm import relax


def _c(arr):
    """fp16 relax constant from a numpy-able array."""
    return relax.const(np.asarray(arr, dtype="float16"))


def reduce_sum_lastdim(bb, x, rows, k):
    """x[rows,k] -> [rows,1], sum over last dim. Dedicated reduce op (relax.sum),
    NOT a matmul: codegen lowers it on the matrix engine (ones-matmul under the
    hood) so matmul-the-op stays TIR-only."""
    return bb.emit(relax.op.sum(x, axis=[-1], keepdims=True))


def broadcast_col(bb, x, rows, n):
    """x[rows,1] -> [rows,n] (replicate each row's scalar). Dedicated broadcast
    op (relax.broadcast_to), not an outer-product matmul."""
    return bb.emit(relax.op.broadcast_to(x, relax.ShapeExpr([rows, n])))


def rms_norm(bb, x, w, seq, d, eps=0.0):
    """RMSNorm(x[seq,d], w[1,d]) -> [seq,d].

    ms = mean(x^2, axis=-1) + eps; y = x / sqrt(ms) * w
    reduce via ones-matmul, broadcast via ones-matmul, 1/rms via ones-divide.
    eps (e.g. 1e-5) added as a constant tensor (immediate can't encode it).
    """
    sq = bb.emit(relax.op.multiply(x, x))                       # [seq,d]
    # scale by 1/d BEFORE reducing so the running sum stays near mean(x^2): sum(x^2)
    # over large d (e.g. 3072) otherwise overflows FP16 (max 65504) -> inf -> 0 output.
    sq = bb.emit(relax.op.multiply(sq, _c(np.full((seq, d), 1.0 / d))))
    mean = reduce_sum_lastdim(bb, sq, seq, d)                   # = mean(x^2), overflow-safe
    if eps:
        mean = bb.emit(relax.op.add(mean, _c(np.full((seq, 1), eps))))       # + eps
    rms = bb.emit(relax.op.sqrt(mean))                         # [seq,1]
    inv = bb.emit(relax.op.divide(_c(np.ones((seq, 1))), rms))  # 1/rms  [seq,1]
    scale = broadcast_col(bb, inv, seq, d)                      # [seq,d]
    xn = bb.emit(relax.op.multiply(x, scale))                  # x/rms
    wb = bb.emit(relax.op.broadcast_to(w, relax.ShapeExpr([seq, d])))  # w[1,d] -> [seq,d]
    return bb.emit(relax.op.multiply(xn, wb))                  # * weight


def _llama3_scale_freqs(freqs, factor=32.0, low=1.0, high=4.0, old_ctx=8192):
    """Llama-3 RoPE frequency scaling (smooth low/high-frequency interpolation)."""
    out = np.empty_like(freqs)
    low_wl = old_ctx / low                                     # wavelength thresholds
    high_wl = old_ctx / high
    for i, f in enumerate(freqs):
        wl = 2.0 * np.pi / f
        if wl > low_wl:
            out[i] = f / factor                               # low freq: scale down
        elif wl < high_wl:
            out[i] = f                                        # high freq: unchanged
        else:
            s = (old_ctx / wl - low) / (high - low)           # smooth interpolation
            out[i] = (1 - s) * f / factor + s * f
    return out


def rope_tables(seq, hd, base=10000.0, llama3_scaling=False):
    """Host-precomputed cos/sin tables [seq,hd] (NPU can't do sin/cos) and the
    rotate_half permutation matrix [hd,hd] (so rotate_half = q @ Rot, no slice/concat).
    base=500000 + llama3_scaling=True for Llama 3.2."""
    half = hd // 2
    freqs = base ** (-2.0 * np.arange(half) / hd)              # [half]
    if llama3_scaling:
        freqs = _llama3_scale_freqs(freqs)
    cos = np.zeros((seq, hd)); sin = np.zeros((seq, hd))
    for p in range(seq):
        ang = p * freqs                                       # [half]
        cos[p, :half] = cos[p, half:] = np.cos(ang)
        sin[p, :half] = sin[p, half:] = np.sin(ang)
    rot = np.zeros((hd, hd))
    for j in range(half):
        rot[j + half, j] = -1.0          # rh[:, j<half]  = -q[:, j+half]
    for j in range(half, hd):
        rot[j - half, j] = 1.0           # rh[:, j>=half] =  q[:, j-half]
    return cos, sin, rot


def rope(bb, q, cos_c, sin_c, rot_c):
    """RoPE: q_embed = q*cos + rotate_half(q)*sin. q is [seq,hd].
    cos_c/sin_c [seq,hd], rot_c [hd,hd] are shared relax constants."""
    rh = bb.emit(relax.op.matmul(q, rot_c))                    # rotate_half via perm matrix
    a = bb.emit(relax.op.multiply(q, cos_c))
    b = bb.emit(relax.op.multiply(rh, sin_c))
    return bb.emit(relax.op.add(a, b))


def softmax_lastdim(bb, s, rows, cols, stable=True):
    """softmax over last dim of s[rows,cols]. 0710: stable via max-subtraction —
    reduce-max is still not a native op, so codegen folds it with vector-max
    (emit_row_max, 0x12) over strided column loads. exp -> rowsum -> broadcast ->
    divide. stable=False keeps the legacy no-max path (report.md §6.1)."""
    if stable:
        m = bb.emit(relax.op.max(s, axis=[-1], keepdims=True))          # [rows,1] row max
        mb = bb.emit(relax.op.broadcast_to(m, relax.ShapeExpr([rows, cols])))
        s = bb.emit(relax.op.subtract(s, mb))                          # s - rowmax
    e = bb.emit(relax.op.exp(s))                                # [rows,cols]
    rowsum = bb.emit(relax.op.sum(e, axis=[-1], keepdims=True))  # [rows,1]
    denom = bb.emit(relax.op.broadcast_to(rowsum, relax.ShapeExpr([rows, cols])))  # [rows,cols]
    return bb.emit(relax.op.divide(e, denom))


def causal_mask(seq):
    """[seq,seq] additive mask: 0 on/below diagonal, large-negative above."""
    m = np.zeros((seq, seq), dtype="float32")
    for i in range(seq):
        for j in range(seq):
            if j > i:
                m[i, j] = -30000.0
    return m


def attention_singlehead_causal(bb, q, k, v, seq, hd):
    """Single-head causal attention. q,k,v are [seq,hd].
    scores = (q @ k^T) / sqrt(hd) + causal_mask ; softmax ; @ v.
    k^T via element-copy transpose; scale via constant; softmax without max-sub."""
    kt = bb.emit(relax.op.permute_dims(k, axes=[1, 0]))         # [hd,seq]
    s = bb.emit(relax.op.matmul(q, kt))                        # [seq,seq]
    inv = 1.0 / float(np.sqrt(hd))
    s = bb.emit(relax.op.multiply(s, _c(np.full((seq, seq), inv))))
    s = bb.emit(relax.op.add(s, _c(causal_mask(seq))))
    p = softmax_lastdim(bb, s, seq, seq)                        # [seq,seq]
    return bb.emit(relax.op.matmul(p, v))                      # [seq,hd]


def silu(bb, z, rows, cols):
    """SiLU(z) = z * sigmoid(z).  z is [rows,cols].
    0710: the standard HW activation IS SiLU, exposed on any matrix op's
    activation bit — codegen lowers relax.nn.silu to a single m_add(+0, act=SiLU)
    per chunk, replacing the old 5-op exp/add/div/mul decomposition."""
    return bb.emit(relax.op.nn.silu(z))


def swiglu(bb, x, Wg, Wu, Wd, seq, d, f):
    """SwiGLU FFN: down( silu(x@Wg) * (x@Wu) ).  x[seq,d], Wg/Wu[d,f], Wd[f,d]."""
    gate = bb.emit(relax.op.matmul(x, Wg))                     # [seq,f]
    up = bb.emit(relax.op.matmul(x, Wu))                       # [seq,f]
    s = silu(bb, gate, seq, f)                                 # [seq,f]
    h = bb.emit(relax.op.multiply(s, up))                      # [seq,f]
    return bb.emit(relax.op.matmul(h, Wd))                     # [seq,d]
