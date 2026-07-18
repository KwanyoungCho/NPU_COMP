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


if __name__ == "__main__":
    print("[PASS] roundtrip:", test_roundtrip())
    print("[PASS] tile-contiguous:", test_tile_contiguous())
    print("ALL LAYOUT (A4 5a) TESTS PASSED")
