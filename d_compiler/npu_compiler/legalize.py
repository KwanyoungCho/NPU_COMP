"""Relax-level legalization helpers — decompose ops the NPU ISA lacks into
NPU-supported primitives (matmul + elementwise + sqrt/exp), per report.md §5.

These are *builders* (emit decomposed ops into a relax.BlockBuilder), reusable
when constructing the model graph. The key tricks:
  - reduce-sum over last dim  ->  matmul with a ones[K,1] vector
  - broadcast / outer product ->  matmul with a ones row/col
  - 1/x                       ->  divide(ones, x)
No 64x64 tiling here (B0 logical). softmax max-subtraction is intentionally omitted.
"""
import numpy as np
from tvm import relax


def _c(arr):
    """fp16 relax constant from a numpy-able array."""
    return relax.const(np.asarray(arr, dtype="float16"))


def reduce_sum_lastdim(bb, x, rows, k):
    """x[rows,k] -> [rows,1], sum over last dim, via matmul with ones[k,1]."""
    return bb.emit(relax.op.matmul(x, _c(np.ones((k, 1)))))


def broadcast_col(bb, x, rows, n):
    """x[rows,1] -> [rows,n] by outer product with ones[1,n]."""
    return bb.emit(relax.op.matmul(x, _c(np.ones((1, n)))))


def rms_norm(bb, x, w, seq, d, eps=0.0):
    """RMSNorm(x[seq,d], w[1,d]) -> [seq,d], eps=0 (matches proxy).

    ms = mean(x^2, axis=-1); y = x / sqrt(ms) * w
    reduce via ones-matmul, broadcast via ones-matmul, 1/rms via ones-divide.
    """
    assert eps == 0.0, "eps!=0 needs a constant-add (TODO when real model)"
    sq = bb.emit(relax.op.multiply(x, x))                       # [seq,d]
    ssum = reduce_sum_lastdim(bb, sq, seq, d)                   # [seq,1]
    mean = bb.emit(relax.op.multiply(ssum, _c(np.full((seq, 1), 1.0 / d))))  # /d
    rms = bb.emit(relax.op.sqrt(mean))                         # [seq,1]
    inv = bb.emit(relax.op.divide(_c(np.ones((seq, 1))), rms))  # 1/rms  [seq,1]
    scale = broadcast_col(bb, inv, seq, d)                      # [seq,d]
    xn = bb.emit(relax.op.multiply(x, scale))                  # x/rms
    wb = bb.emit(relax.op.matmul(_c(np.ones((seq, 1))), w))     # broadcast w[1,d] -> [seq,d]
    return bb.emit(relax.op.multiply(xn, wb))                  # * weight


def rope_tables(seq, hd, base=10000.0):
    """Host-precomputed cos/sin tables [seq,hd] (NPU can't do sin/cos) and the
    rotate_half permutation matrix [hd,hd] (so rotate_half = q @ Rot, no slice/concat)."""
    half = hd // 2
    cos = np.zeros((seq, hd)); sin = np.zeros((seq, hd))
    for p in range(seq):
        for i in range(half):
            th = p * (base ** (-2.0 * i / hd))
            cos[p, i] = cos[p, i + half] = np.cos(th)
            sin[p, i] = sin[p, i + half] = np.sin(th)
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


def softmax_lastdim(bb, s, rows, cols):
    """softmax over last dim of s[rows,cols]. NO max-subtraction (ISA has no
    reduce-max) -> safe only when scores are small (report.md §6.1).
    exp -> rowsum(ones-matmul) -> broadcast -> divide."""
    e = bb.emit(relax.op.exp(s))                                # [rows,cols]
    rowsum = bb.emit(relax.op.matmul(e, _c(np.ones((cols, 1)))))  # [rows,1]
    denom = bb.emit(relax.op.matmul(rowsum, _c(np.ones((1, cols)))))  # [rows,cols]
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
    """SiLU(z) = z * sigmoid(z) = z / (1 + exp(-z)).  z is [rows,cols].
    HW activation (x^2*sigmoid) is unusable, so build from exp/add/div/mul.
    negate via zeros-subtract; +1 via ones-add; reciprocal via ones-divide."""
    zeros = _c(np.zeros((rows, cols)))
    ones = _c(np.ones((rows, cols)))
    neg = bb.emit(relax.op.subtract(zeros, z))                 # -z
    den = bb.emit(relax.op.exp(neg))                           # exp(-z)
    den = bb.emit(relax.op.add(den, ones))                     # 1+exp(-z)
    sig = bb.emit(relax.op.divide(ones, den))                  # sigmoid(z)
    return bb.emit(relax.op.multiply(z, sig))                  # z*sigmoid(z)


def swiglu(bb, x, Wg, Wu, Wd, seq, d, f):
    """SwiGLU FFN: down( silu(x@Wg) * (x@Wu) ).  x[seq,d], Wg/Wu[d,f], Wd[f,d]."""
    gate = bb.emit(relax.op.matmul(x, Wg))                     # [seq,f]
    up = bb.emit(relax.op.matmul(x, Wu))                       # [seq,f]
    s = silu(bb, gate, seq, f)                                 # [seq,f]
    h = bb.emit(relax.op.multiply(s, up))                      # [seq,f]
    return bb.emit(relax.op.matmul(h, Wd))                     # [seq,d]
