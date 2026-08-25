"""N3 gate: v09 compute units.

Two proof strategies:
  * FP16 mode must be bit-exact with the frozen 0818 source model.  A tiny
    mechanical converter (exactly the spec section-4 migration rule: multiply
    ver.08 element addresses by 4, everything else verbatim) turns a ver.08
    program into a v09 program; both runs must produce identical bytes.
  * v09-only behavior (signed immediates, seeded reduce-max, standard GELU,
    dtype feeders, dequant-in-matmul incl. group scale changes over a MAC
    chain, VQUANT/VDEQUANT) is checked against numpy references that
    replicate the simulator arithmetic order.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))

from npu_compiler.isa_0818 import (
    ACT_SILU, IMM, MAIN, PARTIAL, SCALAR, VECTOR, Asm,
    enc_addr_hi, enc_addr_lo, enc_broadcast, enc_vlen)
from npu_compiler import isa_v09 as V
from npu_compiler.source_runtime_0818 import run as run_0818
from npu_compiler.v09_runtime import V09Error, run as run_v09

# ------------------------------------------------------------- helpers


def fp16_image(values):
    """FP16 array -> 32-bit cell image (little-endian pairs)."""
    flat = np.asarray(values, dtype="<f2").reshape(-1)
    assert flat.size % 2 == 0, "keep test images cell-aligned"
    return flat.view("<u4").copy()


def image_fp16(image):
    return np.ascontiguousarray(np.asarray(image, dtype="<u4")).view("<f2")


def convert_0818_program(words, n_elements):
    """Mechanical ver.08 -> v09 program conversion (spec section 4).

    Stages the whole FP16 G-buffer image into SRAM nibble 0 (GLOAD), rewrites
    every two-half address by *4 (element -> nibble), keeps shapes/vlen/ops
    verbatim, and replaces the trailing 0xF0 with GSTORE + HALT.
    """
    assert n_elements % 2 == 0
    cells = n_elements // 2
    out = list(V.enc_gload(0, cells, 0, rows=1, cols=cells))
    i = 0
    while i < len(words):
        word = words[i]
        op = word & 0xFF
        mode = (word >> 30) & 3
        if op == 0x80:
            nxt = words[i + 1]
            assert (nxt & 0xFF) == 0x80 and ((nxt >> 29) & 1) == 1
            operand, partial = (word >> 30) & 3, (word >> 28) & 1
            address = ((word >> 8) & 0xFFFF) | (((nxt >> 8) & 0xFFFF) << 16)
            out.append(enc_addr_lo(operand, address * 4, partial))
            out.append(enc_addr_hi(operand, address * 4, partial))
            i += 2
            continue
        if op == 0x15 and mode == SCALAR:
            nxt = words[i + 1]
            assert (nxt & 0xFF) == 0x15
            address = ((word >> 8) & 0xFFFF) | (((nxt >> 8) & 0xFFFF) << 16)
            out.append(enc_broadcast(SCALAR, address * 4, False))
            out.append(enc_broadcast(SCALAR, (address * 4) >> 16, True))
            i += 2
            continue
        if op == 0xF0:
            assert i == len(words) - 1, "0xF0 must terminate the test program"
            out += V.enc_gstore(0, cells, 0, rows=1, cols=cells)
            out.append(V.enc_halt())
            i += 1
            continue
        out.append(word)
        i += 1
    return out


def assert_parity(asm, gbuf_fp16):
    """Run a ver.08 program on the 0818 oracle and its conversion on v09."""
    values = np.asarray(gbuf_fp16, dtype=np.float16).reshape(-1)
    oracle = run_0818(asm, values, output_dtype=np.float16)[:values.size]
    program = convert_0818_program(asm.words, values.size)
    images, _ = run_v09(program, fp16_image(values))
    ours = image_fp16(images[-1])
    np.testing.assert_array_equal(oracle.view(np.uint16), ours.view(np.uint16))


def f32(value):
    return np.float32(value)


def seq_matmul_f32(a, b):
    """FP32 matmul with the simulator's flat k accumulation order."""
    a = np.asarray(a); b = np.asarray(b)
    out = np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)
    for m in range(a.shape[0]):
        for n in range(b.shape[1]):
            acc = f32(0)
            for k in range(a.shape[1]):
                acc = f32(acc + f32(f32(a[m, k]) * f32(b[k, n])))
            out[m, n] = acc
    return out


def region(asm, operand, addr, rows, cols, stride=None):
    asm.matrix_region(operand, addr, rows, stride if stride is not None else cols,
                      addr, rows, cols)


# ------------------------------------------- FP16 parity vs the 0818 oracle


def test_fp16_vector_battery_matches_0818():
    rng = np.random.default_rng(11)
    n = 64
    gbuf = np.zeros(1024, dtype=np.float16)
    gbuf[:n] = rng.standard_normal(n).astype(np.float16)
    gbuf[n:2 * n] = (rng.random(n).astype(np.float16) + np.float16(0.5))
    asm = Asm()
    asm.vlen(n)
    asm.addr(0, 0); asm.load(0, 0)
    asm.addr(1, n); asm.load(0, 1)
    asm.v_add(VECTOR); asm.addr(2, 2 * n); asm.save(0)
    asm.v_mul(IMM, 3); asm.addr(2, 3 * n); asm.save(0)
    asm.v_div(VECTOR); asm.addr(2, 4 * n); asm.save(0)
    asm.v_muladd(VECTOR); asm.addr(2, 5 * n); asm.save(0)
    asm.v_max(VECTOR); asm.addr(2, 6 * n); asm.save(0)
    asm.v_shift(2, IMM); asm.addr(2, 7 * n); asm.save(0)
    asm.addr(0, n); asm.load(0, 0)
    asm.v_exp(); asm.addr(2, 8 * n); asm.save(0)
    asm.v_sqrt(); asm.addr(2, 9 * n); asm.save(0)
    asm.v_cos(); asm.addr(2, 10 * n); asm.save(0)
    asm.v_sin(); asm.addr(2, 11 * n); asm.save(0)
    asm.v_sign_inv(); asm.addr(2, 12 * n); asm.save(0)
    asm.v_broadcast_addr(5); asm.addr(2, 13 * n); asm.save(0)
    asm.finish()
    assert_parity(asm, gbuf)


def test_fp16_matmul_battery_matches_0818():
    rng = np.random.default_rng(23)
    m, k, n = 4, 6, 4
    gbuf = np.zeros(512, dtype=np.float16)
    gbuf[:m * k] = rng.standard_normal(m * k).astype(np.float16)
    gbuf[64:64 + k * n] = rng.standard_normal(k * n).astype(np.float16)
    asm = Asm()
    region(asm, 0, 0, m, k)
    region(asm, 1, 64, k, n)
    region(asm, 2, 128, m, n)
    asm.load(1, 0); asm.load(1, 1)
    asm.m_mul(VECTOR)
    asm.save(1)
    # MAC chain: split k into two halves accumulated in the array.
    half = k // 2
    region(asm, 0, 0, m, half, stride=k)
    region(asm, 1, 64, half, n, stride=n)
    asm.load(1, 0); asm.load(1, 1)
    asm.m_mul(VECTOR)
    region(asm, 0, half, m, half, stride=k)
    region(asm, 1, 64 + half * n, half, n, stride=n)
    asm.load(1, 0); asm.load(1, 1)
    asm.m_mul(VECTOR, mac=True)
    region(asm, 2, 160, m, n)
    asm.save(1)
    # fused SiLU and vendor-legacy GELU keep their 0818 arithmetic.
    region(asm, 0, 0, m, k)
    region(asm, 1, 64, k, n)
    asm.load(1, 0); asm.load(1, 1)
    asm.m_mul(VECTOR, activation=ACT_SILU)
    region(asm, 2, 192, m, n)
    asm.save(1)
    asm.finish()
    assert_parity(asm, gbuf)


# ------------------------------------------- v09 fixes vs exact references


def run_simple(program, gbuf_fp16):
    images, counters = run_v09(program, fp16_image(gbuf_fp16))
    return image_fp16(images[-1]), counters


def staged(n_elements):
    cells = n_elements // 2
    return list(V.enc_gload(0, cells, 0, rows=1, cols=cells)), cells


def finish(program, cells):
    program += V.enc_gstore(0, cells, 0, rows=1, cols=cells)
    program.append(V.enc_halt())
    return program


def vec_addr(operand, nibble):
    return [enc_addr_lo(operand, nibble), enc_addr_hi(operand, nibble)]


def test_signed_immediates():
    gbuf = np.zeros(64, dtype=np.float16)
    gbuf[:8] = np.arange(8, dtype=np.float16)
    program, cells = staged(gbuf.size)
    program.append(enc_vlen(8))
    program += vec_addr(0, 0)
    program.append(V.enc_load(0, 0))
    from npu_compiler.isa_0818 import enc_add
    program.append(enc_add(IMM, -3 & 0xFFFF))
    program += vec_addr(2, 8 * 4)
    program.append(V.enc_save(0))
    program.append(enc_broadcast(IMM, -5 & 0xFFFF))
    program += vec_addr(2, 16 * 4)
    program.append(V.enc_save(0))
    out, _ = run_simple(finish(program, cells), gbuf)
    np.testing.assert_array_equal(
        out[8:16], (gbuf[:8].astype(np.float32) + f32(-3)).astype(np.float16))
    np.testing.assert_array_equal(out[16:24], np.full(8, -5, np.float16))


def test_reduce_seeded_max_and_long_flat_sum():
    """512-lane single-instruction reduces (the 256-lane datapath strip-mines
    internally; no architectural carry state)."""
    from npu_compiler.isa_0818 import enc_reduce_max, enc_reduce_sum
    rng = np.random.default_rng(7)
    values = (-rng.random(512).astype(np.float16) - np.float16(1))  # all < 0
    gbuf = np.zeros(1024, dtype=np.float16)
    gbuf[:512] = values
    program, cells = staged(gbuf.size)
    program.append(enc_vlen(512))
    program += vec_addr(0, 0)
    program.append(V.enc_load(0, 0))
    program.append(enc_reduce_max())
    program += vec_addr(2, 600 * 4)
    program.append(V.enc_save(0))
    program += vec_addr(0, 0)
    program.append(V.enc_load(0, 0))
    program.append(enc_reduce_sum())
    program += vec_addr(2, 602 * 4)
    program.append(V.enc_save(0))
    out, _ = run_simple(finish(program, cells), gbuf)
    assert out[600] == values.max()          # ver.08 zero-seed would give 0
    acc = f32(0)
    for value in values:
        acc = f32(acc + f32(value))
    assert out[602] == np.float16(acc)


def test_standard_gelu_activation_mode():
    x = np.asarray([-2.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, -1.5],
                   dtype=np.float16)
    gbuf = np.zeros(128, dtype=np.float16)
    gbuf[:8] = x
    program, cells = staged(gbuf.size)
    _matrix_setup(program, 0, 1, 8, V.DT_FP16, 128, 8, V.DT_FP16, 256)
    program.append(V.enc_load(1, 0))
    from npu_compiler.isa_0818 import enc_m_add
    program.append(enc_m_add(IMM, 0, activation=V.ACT_GELU_STD))
    program.append(V.enc_save(1))
    out, _ = run_simple(finish(program, cells), gbuf)
    xf = x.astype(np.float32)
    inner = f32(0.7978845608028654) * (xf + f32(0.044715) * xf * xf * xf)
    expected = (f32(0.5) * xf *
                (f32(1.0) + np.float32(np.tanh(np.float64(inner))))
                ).astype(np.float16)
    np.testing.assert_array_equal(out[64:72], expected)


def _dequant_reference(acc_f32_or_int, w_scale, a_scale=None):
    out = np.asarray(acc_f32_or_int, dtype=np.float32)
    out = out * np.asarray(w_scale, np.float32)[None, :]
    if a_scale is not None:
        out = out * np.asarray(a_scale, np.float32)[:, None]
    return out.astype(np.float16)


def _matrix_setup(program, a_nib, m, k, a_dtype, w_nib, n, w_dtype, d_nib):
    for operand, addr, rows, cols, dtype in (
            (0, a_nib, m, k, a_dtype), (1, w_nib, k, n, w_dtype),
            (2, d_nib, m, n, V.DT_FP16)):
        program += [enc_addr_lo(operand, addr, MAIN),
                    enc_addr_hi(operand, addr, MAIN),
                    V.enc_mrows(operand, rows, MAIN, dtype),
                    V.enc_mcols(operand, cols, MAIN, dtype),
                    enc_addr_lo(operand, addr, PARTIAL),
                    enc_addr_hi(operand, addr, PARTIAL),
                    V.enc_mrows(operand, rows, PARTIAL, dtype),
                    V.enc_mcols(operand, cols, PARTIAL, dtype)]


def test_w8a16_feeder_and_drain_dequant():
    rng = np.random.default_rng(5)
    m, k, n = 2, 4, 4
    a = rng.standard_normal((m, k)).astype(np.float16)
    w_int = rng.integers(-128, 128, size=(k, n)).astype(np.int8)
    w_scale = np.asarray([0.01, 0.02, 0.5, 1.5], dtype=np.float32)
    # global image: A (4 cells) | W packed (4 cells) | scales (4 cells) | out
    image = np.concatenate([
        fp16_image(a), w_int.reshape(-1).view(np.uint8).view("<u4"),
        w_scale.view("<u4"), np.zeros(4, np.uint32)])
    program = list(V.enc_gload(0, 4, 0, rows=1, cols=4))         # A -> nib 0
    program += V.enc_gload(4, 4, 64, rows=1, cols=4)             # W -> nib 64
    program += V.enc_gload(8, 4, 128, rows=1, cols=4)            # sw -> nib 128
    _matrix_setup(program, 0, m, k, V.DT_FP16, 64, n, V.DT_INT8, 256)
    program += V.enc_wscale(128)
    program += [V.enc_load(1, 0), V.enc_load(1, 1)]
    from npu_compiler.isa_0818 import enc_m_mul
    program.append(enc_m_mul(VECTOR))
    program.append(V.enc_save(1))
    program += V.enc_gstore(12, 4, 256, rows=1, cols=4)
    program.append(V.enc_halt())
    images, _ = run_v09(program, image)
    out = np.ascontiguousarray(images[-1][12:16]).view("<f2").reshape(m, n)
    acc = seq_matmul_f32(a.astype(np.float32), w_int.astype(np.float32))
    np.testing.assert_array_equal(out, _dequant_reference(acc, w_scale))


def test_w8a8_integer_path_and_double_scale_drain():
    rng = np.random.default_rng(3)
    m, k, n = 2, 4, 4
    qa = rng.integers(-127, 128, size=(m, k)).astype(np.int8)
    qw = rng.integers(-127, 128, size=(k, n)).astype(np.int8)
    a_scale = np.asarray([0.03, 0.07], dtype=np.float32)
    w_scale = np.asarray([0.01, 0.02, 0.5, 1.5], dtype=np.float32)
    image = np.concatenate([
        qa.reshape(-1).view(np.uint8).view("<u4"),                # 2 cells
        qw.reshape(-1).view(np.uint8).view("<u4"),                # 4 cells
        a_scale.view("<u4"), w_scale.view("<u4"),                 # 2 + 4 cells
        np.zeros(4, np.uint32)])
    program = list(V.enc_gload(0, 2, 0, rows=1, cols=2))          # qa -> nib 0
    program += V.enc_gload(2, 4, 64, rows=1, cols=4)              # qw -> nib 64
    program += V.enc_gload(6, 2, 128, rows=1, cols=2)             # sa -> 128
    program += V.enc_gload(8, 4, 192, rows=1, cols=4)             # sw -> 192
    _matrix_setup(program, 0, m, k, V.DT_INT8, 64, n, V.DT_INT8, 512)
    program += V.enc_ascale(128) + V.enc_wscale(192)
    program += [V.enc_load(1, 0), V.enc_load(1, 1)]
    from npu_compiler.isa_0818 import enc_m_mul
    program.append(enc_m_mul(VECTOR))
    program.append(V.enc_save(1))
    program += V.enc_gstore(12, 4, 512, rows=1, cols=4)
    program.append(V.enc_halt())
    images, _ = run_v09(program, image)
    out = np.ascontiguousarray(images[-1][12:16]).view("<f2").reshape(m, n)
    acc = qa.astype(np.int64) @ qw.astype(np.int64)               # exact ints
    np.testing.assert_array_equal(
        out, _dequant_reference(acc.astype(np.float32), w_scale, a_scale))


def test_groupwise_scale_change_over_mac_chain():
    """Group quantization rides the existing MAC bit: re-point 0x8B between
    groups and keep accumulating -- no extra flags or accumulator."""
    rng = np.random.default_rng(13)
    m, k, n = 2, 4, 4
    groups = 2
    kg = k // groups
    a = rng.standard_normal((m, k)).astype(np.float16)
    w_int = rng.integers(-8, 8, size=(k, n)).astype(np.int8)
    w_scale = np.asarray([[0.5, 0.25, 1.0, 2.0],
                          [0.125, 4.0, 0.75, 1.5]], dtype=np.float32)
    image = np.concatenate([
        fp16_image(a), w_int.reshape(-1).view(np.uint8).view("<u4"),
        w_scale.reshape(-1).view("<u4"), np.zeros(4, np.uint32)])
    program = list(V.enc_gload(0, 4, 0, rows=1, cols=4))
    program += V.enc_gload(4, 4, 64, rows=1, cols=4)
    program += V.enc_gload(8, 8, 128, rows=1, cols=8)
    from npu_compiler.isa_0818 import enc_m_mul
    for g in range(groups):
        a_nib = g * kg * 4          # column offset inside A (stride k)
        w_nib = 64 + g * kg * n * 2
        for operand, addr, rows, cols, stride, dtype in (
                (0, a_nib, m, kg, k, V.DT_FP16),
                (1, w_nib, kg, n, n, V.DT_INT8),
                (2, 512, m, n, n, V.DT_FP16)):
            program += [enc_addr_lo(operand, addr, MAIN),
                        enc_addr_hi(operand, addr, MAIN),
                        V.enc_mrows(operand, rows, MAIN, dtype),
                        V.enc_mcols(operand, stride, MAIN, dtype),
                        enc_addr_lo(operand, addr, PARTIAL),
                        enc_addr_hi(operand, addr, PARTIAL),
                        V.enc_mrows(operand, rows, PARTIAL, dtype),
                        V.enc_mcols(operand, cols, PARTIAL, dtype)]
        program += V.enc_wscale(128 + g * n * 8)
        program += [V.enc_load(1, 0), V.enc_load(1, 1)]
        program.append(enc_m_mul(VECTOR, mac=g > 0))
    program.append(V.enc_save(1))
    program += V.enc_gstore(16, 4, 512, rows=1, cols=4)
    program.append(V.enc_halt())
    images, _ = run_v09(program, image)
    out = np.ascontiguousarray(images[-1][16:20]).view("<f2").reshape(m, n)
    total = np.zeros((m, n), np.float32)
    for g in range(groups):
        acc = seq_matmul_f32(a[:, g * kg:(g + 1) * kg].astype(np.float32),
                             w_int[g * kg:(g + 1) * kg].astype(np.float32))
        total = total + acc * w_scale[g][None, :]
    np.testing.assert_array_equal(out, total.astype(np.float16))


def test_vquant_vdequant_and_fp32_scale_production():
    x = np.asarray([0.5, -1.25, 3.0, -6.5, 2.25, 0.125, -0.75, 5.5],
                   dtype=np.float16)
    n = x.size
    gbuf = np.zeros(1024, dtype=np.float16)   # staged SRAM spans 4096 nibbles
    gbuf[:n] = x
    program, cells = staged(gbuf.size)
    program.append(enc_vlen(n))
    # absmax -> scale = absmax/127 stored FP32 at nibble 256
    program += vec_addr(0, 0)
    program.append(V.enc_load(0, 0))
    from npu_compiler.isa_0818 import enc_sign_inv, enc_minmax, enc_div
    program.append(enc_sign_inv())
    program += vec_addr(2, 64 * 4)
    program.append(V.enc_save(0))
    program += vec_addr(1, 64 * 4)
    program.append(V.enc_load(0, 1))
    program += vec_addr(0, 0)
    program.append(V.enc_load(0, 0))
    program.append(enc_minmax(True, VECTOR))
    program += vec_addr(2, 80 * 4)
    program.append(V.enc_save(0))
    program += vec_addr(0, 80 * 4)
    program.append(V.enc_load(0, 0))
    from npu_compiler.isa_0818 import enc_reduce_max
    program.append(enc_reduce_max())
    program += vec_addr(2, 96 * 4)
    program.append(V.enc_save(0))
    program.append(enc_vlen(1))
    program += vec_addr(0, 96 * 4)
    program.append(V.enc_load(0, 0))
    program.append(enc_div(IMM, 127))
    program += vec_addr(2, 1024)     # FP32 scale at nibble 1024
    program += [V.enc_mrows(2, 1, PARTIAL, V.DT_FP32),
                V.enc_mcols(2, 1, PARTIAL, V.DT_FP32)]
    program.append(V.enc_save(0))
    # VQUANT x -> INT8 at nibble 2048, then VDEQUANT back to FP16 at 3072
    program.append(enc_vlen(n))
    program += V.enc_ascale(1024)
    program += vec_addr(0, 0)
    program += [V.enc_mrows(0, 1, PARTIAL, V.DT_FP16),
                V.enc_mcols(0, n, PARTIAL, V.DT_FP16)]
    program += vec_addr(2, 2048)
    program += [V.enc_mrows(2, 1, PARTIAL, V.DT_INT8),
                V.enc_mcols(2, n, PARTIAL, V.DT_INT8)]
    program.append(V.enc_vquant())
    program += [enc_addr_lo(0, 2048), enc_addr_hi(0, 2048),
                V.enc_mrows(0, 1, PARTIAL, V.DT_INT8),
                V.enc_mcols(0, n, PARTIAL, V.DT_INT8)]
    program.append(V.enc_vdequant())
    program += vec_addr(2, 3072)
    program += [V.enc_mrows(2, 1, PARTIAL, V.DT_FP16),
                V.enc_mcols(2, n, PARTIAL, V.DT_FP16)]
    program.append(V.enc_save(0))
    out, counters = run_simple(finish(program, cells), gbuf)
    xf = x.astype(np.float32)
    absmax = np.float16(np.abs(x).max())          # exact fp16 values
    scale = f32(absmax) / f32(127)
    q = np.clip(np.rint(xf / scale), -127, 127)
    np.testing.assert_array_equal(out[768:768 + n],
                                  (q * scale).astype(np.float16))
    assert counters["vquant"] == 1 and counters["vdequant"] == 1


def test_vquant_saturation_and_int4():
    x = np.asarray([100.0, -100.0, 3.0, -3.0, 0.4, -0.4, 7.5, -7.5],
                   dtype=np.float16)
    scale_cells = np.asarray([1.0], dtype=np.float32).view("<u4")
    image = np.concatenate([fp16_image(x), scale_cells,
                            np.zeros(7, np.uint32)])       # out at cells 5..9
    program = list(V.enc_gload(0, 4, 0, rows=1, cols=4))   # x -> nib 0
    program += V.enc_gload(4, 1, 2048, rows=1, cols=1)     # scale=1.0 @ 2048
    program.append(enc_vlen(8))
    program += V.enc_ascale(2048)
    program += vec_addr(0, 0)
    program += vec_addr(2, 4096)
    program += [V.enc_mrows(2, 1, PARTIAL, V.DT_INT4),
                V.enc_mcols(2, 8, PARTIAL, V.DT_INT4)]
    program.append(V.enc_vquant())
    program += [enc_addr_lo(0, 4096), enc_addr_hi(0, 4096),
                V.enc_mrows(0, 1, PARTIAL, V.DT_INT4),
                V.enc_mcols(0, 8, PARTIAL, V.DT_INT4)]
    program.append(V.enc_vdequant())
    program += vec_addr(2, 6144)
    program += [V.enc_mrows(2, 1, PARTIAL, V.DT_FP16),
                V.enc_mcols(2, 8, PARTIAL, V.DT_FP16)]
    program.append(V.enc_save(0))
    program += V.enc_gstore(5, 4, 6144, rows=1, cols=4)
    program.append(V.enc_halt())
    images, _ = run_v09(program, image)
    out = np.ascontiguousarray(images[-1][5:9]).view("<f2")
    expected = np.clip(np.rint(x.astype(np.float32)), -7, 7).astype(np.float16)
    np.testing.assert_array_equal(out, expected)


def test_illegal_dtype_paths_are_errors():
    gbuf = np.zeros(64, dtype=np.float16)
    image = fp16_image(gbuf)

    def base():
        return list(V.enc_gload(0, 32, 0, rows=1, cols=32))

    def expect(program, needle):
        try:
            run_v09(program + [V.enc_halt()], image)
        except V09Error as error:
            assert needle in str(error), str(error)
            return
        raise AssertionError(f"expected error {needle!r}")

    p = base() + [enc_vlen(8)] + vec_addr(0, 0) + vec_addr(2, 512)
    p.append(V.enc_vquant())      # DST descriptor left at default FP16
    expect(p, "vquant destination dtype")
    p = base() + [enc_vlen(8)] + vec_addr(0, 0)
    p += [V.enc_mrows(0, 1, PARTIAL, V.DT_INT8),
          V.enc_mcols(0, 8, PARTIAL, V.DT_INT8), V.enc_load(0, 0)]
    expect(p, "vector load requires FP16")
    from npu_compiler.isa_0818 import enc_m_mul
    p = base()
    _matrix_setup(p, 0, 2, 2, V.DT_INT8, 64, 2, V.DT_FP16, 512)
    p += [V.enc_load(1, 0), V.enc_load(1, 1), enc_m_mul(VECTOR)]
    expect(p, "INT activation requires INT weight")
    p = base()
    _matrix_setup(p, 0, 2, 2, V.DT_INT4, 64, 2, V.DT_INT4, 512)
    p += [V.enc_load(1, 0), V.enc_load(1, 1), enc_m_mul(VECTOR)]
    expect(p, "cannot be INT4")


if __name__ == "__main__":
    test_fp16_vector_battery_matches_0818()
    test_fp16_matmul_battery_matches_0818()
    test_signed_immediates()
    test_reduce_seeded_max_and_long_flat_sum()
    test_standard_gelu_activation_mode()
    test_w8a16_feeder_and_drain_dequant()
    test_w8a8_integer_path_and_double_scale_drain()
    test_groupwise_scale_change_over_mac_chain()
    test_vquant_vdequant_and_fp32_scale_production()
    test_vquant_saturation_and_int4()
    test_illegal_dtype_paths_are_errors()
    print("ALL V09 C-MODEL N3 COMPUTE TESTS PASSED")
