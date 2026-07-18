"""A4 Stage 5a: tile-blocked layout convention (memplan.pack_tiled/unpack_tiled).

Locks the physical byte-order for tile-blocked activations: [R,N] logical <->
[ceil(R/64),ceil(N/64),64,64] physical, zero-padded. pack then unpack must be
the identity (dropping padding), for both 64-multiple and ragged shapes."""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE)))
from npu_compiler import memplan


def test_roundtrip():
    for (R, N) in [(64, 64), (128, 192), (8, 16), (128, 3072), (65, 130), (1, 128), (200, 64)]:
        a = (np.random.RandomState(R * 1000 + N).randn(R, N)).astype("float16")
        flat = memplan.pack_tiled(a)
        assert flat.size == memplan.tiled_numel((R, N)), f"size {flat.size} vs {memplan.tiled_numel((R,N))}"
        back = memplan.unpack_tiled(flat, R, N)
        assert back.shape == (R, N) and np.array_equal(back, a), f"roundtrip mismatch {R}x{N}"
    return "pack/unpack identity for 64-multiple + ragged shapes"


def test_tile_contiguous():
    """Each 64x64 tile is stored contiguously (the property matmul TILE-mode relies on):
    tile (ti,tj) occupies flat[(ti*Nt+tj)*4096 : +4096] and equals a[ti*64:.., tj*64:..]."""
    R, N, T = 128, 192, 64
    a = np.arange(R * N).reshape(R, N).astype("float16")
    flat = memplan.pack_tiled(a)
    Nt = N // T
    for ti in range(R // T):
        for tj in range(Nt):
            tile = flat[(ti * Nt + tj) * T * T:(ti * Nt + tj + 1) * T * T].reshape(T, T)
            assert np.array_equal(tile, a[ti * T:(ti + 1) * T, tj * T:(tj + 1) * T]), f"tile {ti},{tj}"
    return "each 64x64 tile is contiguous in tile-blocked storage"


def _f16(x):
    return np.asarray(x, dtype=np.float16).astype(np.float32)


def test_tile_mode_matmul():
    """A4 5b: matmul with tile-blocked A and/or C == row-major, BYTE-EXACT (same MAC
    order; layout only changes addressing, and TILE skips the A-gather / C-scatter)."""
    from npu_compiler import isa, memplan as MP, tir_backend
    import npu_compiler.runtime as rt

    def run(a_data, b_data, M, K, N, a_tiled, c_tiled, b_packed):
        asm = isa.Asm(); mp = MP.MemPlan()
        a_off = mp.scratch_alloc(a_data.size); b_off = mp.scratch_alloc(b_data.size)
        cn = MP.tiled_numel((M, N)) if c_tiled else M * N
        c_off = mp.scratch_alloc(cn)
        tir_backend.emit_matmul_into(asm, mp, c_off, a_off, b_off, M, K, N,
                                     b_pack_nt=(N // 64 if b_packed else None),
                                     a_tiled=a_tiled, c_tiled=c_tiled)
        asm.halt()
        g = np.zeros(mp.top, np.float32)
        g[a_off:a_off + a_data.size] = a_data.reshape(-1)
        g[b_off:b_off + b_data.size] = b_data.reshape(-1)
        full = rt.run(asm, g, gn=mp.top)
        if c_tiled:
            return MP.unpack_tiled(full[c_off:c_off + cn], M, N)
        return full[c_off:c_off + M * N].reshape(M, N)

    out = {}
    for (M, K, N) in [(64, 64, 64), (128, 192, 128), (64, 128, 256), (128, 3072, 128)]:
        rng = np.random.default_rng(M + K + N)
        A = _f16(rng.standard_normal((M, K)) * 0.3)
        B = _f16(rng.standard_normal((K, N)) * 0.1)
        Crow = run(A, B, M, K, N, False, False, False)                 # A,B,C row-major
        Ctile = run(MP.pack_tiled(A), MP.pack_tiled(B), M, K, N,       # A tile, B packed, C tile
                    a_tiled=True, c_tiled=True, b_packed=True)
        assert np.array_equal(Crow, Ctile), f"{M}x{K}x{N}: TILE != ROW"
        # sanity vs float64
        rel = float(np.max(np.abs(Crow - A @ B))) / (float(np.max(np.abs(A @ B))) + 1e-9)
        out[f"{M}x{K}x{N}"] = round(rel, 5)
    return f"TILE-mode A/C == row-major byte-exact; rel-vs-f64 {out}"


if __name__ == "__main__":
    print("[PASS] roundtrip:", test_roundtrip())
    print("[PASS] tile-contiguous:", test_tile_contiguous())
    print("[PASS] tile-mode matmul:", test_tile_mode_matmul())
    print("ALL LAYOUT (A4 5a/5b) TESTS PASSED")
