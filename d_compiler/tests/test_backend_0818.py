"""End-to-end tests for the row-major ver.08 vendor backend."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))

from tvm import relax

from npu_compiler import driver
from npu_compiler.backend_0818 import CodegenError


def _sinfo(shape):
    return relax.TensorStructInfo(list(shape), "float16")


def _unary_module(shape, operation):
    bb = relax.BlockBuilder()
    x = relax.Var("x", _sinfo(shape))
    with bb.function("main", [x]):
        with bb.dataflow():
            y = bb.emit(operation(x))
            output = bb.emit_output(y)
        bb.emit_func_output(output)
    return bb.finalize()


def _binary_module(shape, operation):
    bb = relax.BlockBuilder()
    lhs = relax.Var("lhs", _sinfo(shape))
    rhs = relax.Var("rhs", _sinfo(shape))
    with bb.function("main", [lhs, rhs]):
        with bb.dataflow():
            y = bb.emit(operation(lhs, rhs))
            output = bb.emit_output(y)
        bb.emit_func_output(output)
    return bb.finalize()


def _matmul_module(rows, inner, cols):
    bb = relax.BlockBuilder()
    lhs = relax.Var("lhs", _sinfo([rows, inner]))
    rhs = relax.Var("rhs", _sinfo([inner, cols]))
    with bb.function("main", [lhs, rhs]):
        with bb.dataflow():
            y = bb.emit(relax.op.matmul(lhs, rhs))
            output = bb.emit_output(y)
        bb.emit_func_output(output)
    return bb.finalize()


def test_row_major_direct_subtile_matmul():
    rows, inner, cols = 17, 65, 33  # forces two K tiles and ver.08 MAC
    rng = np.random.default_rng(818)
    lhs = np.asarray(rng.normal(0, 0.2, (rows, inner)), dtype=np.float16)
    rhs = np.asarray(rng.normal(0, 0.2, (inner, cols)), dtype=np.float16)
    asm, mp = driver.compile_module(_matmul_module(rows, inner, cols), backend="0818")
    assert set(mp.layout.values()) == {"row"}
    assert any((word & 0xFF) == 0x42 and ((word >> 27) & 1) for word in asm.words)
    got = driver.run_compiled(asm, mp, {"lhs": lhs, "rhs": rhs})
    expected = np.asarray(lhs.astype(np.float32) @ rhs.astype(np.float32),
                          dtype=np.float16).astype(np.float32)
    assert np.array_equal(got, expected)


def test_negative_row_max_avoids_native_vendor_bug():
    mod = _unary_module([4, 7], lambda x: relax.op.max(x, axis=[1], keepdims=True))
    values = -np.arange(1, 29, dtype=np.float16).reshape(4, 7)
    asm, mp = driver.compile_module(mod, backend="0818")
    assert not any((word & 0xFF) == 0x19 for word in asm.words)
    got = driver.run_compiled(asm, mp, {"x": values})
    assert np.array_equal(got, values.max(axis=1, keepdims=True).astype(np.float32))


def test_native_row_sum_sets_scalar_save_length():
    mod = _unary_module([4, 7], lambda x: relax.op.sum(x, axis=[1], keepdims=True))
    values = np.arange(1, 29, dtype=np.float16).reshape(4, 7)
    asm, mp = driver.compile_module(mod, backend="0818")
    reduce_index = next(index for index, word in enumerate(asm.words) if (word & 0xFF) == 0x14)
    assert asm.words[reduce_index + 1] == (1 << 8) | 0x82  # vector save consumes current vlen
    got = driver.run_compiled(asm, mp, {"x": values})
    expected = np.asarray(values.astype(np.float32).sum(axis=1, keepdims=True),
                          dtype=np.float16).astype(np.float32)
    assert np.array_equal(got, expected)


def test_full_capacity_transpose_has_no_tile_blocked_scratch():
    mod = _unary_module([64, 64], lambda x: relax.op.permute_dims(x, axes=[1, 0]))
    values = np.arange(4096, dtype=np.float16).reshape(64, 64)
    asm, mp = driver.compile_module(mod, backend="0818")
    assert mp.top == 8192  # input + row-major output; no gather/transpose tile buffer
    got = driver.run_compiled(asm, mp, {"x": values})
    assert np.array_equal(got, values.T.astype(np.float32))


def test_native_scalar_broadcast():
    mod = _unary_module([4, 1], lambda x: relax.op.broadcast_to(x, [4, 7]))
    values = np.arange(4, dtype=np.float16).reshape(4, 1)
    asm, mp = driver.compile_module(mod, backend="0818")
    assert any((word & 0xFF) == 0x15 for word in asm.words)
    got = driver.run_compiled(asm, mp, {"x": values})
    expected = np.broadcast_to(values, (4, 7)).astype(np.float32)
    assert np.array_equal(got, expected)


def test_vendor_capacity_is_checked_at_compile_time():
    mod = _binary_module([3000], relax.op.add)  # two inputs + output = 9000 FP16
    try:
        driver.compile_module(mod, backend="0818")
    except CodegenError as error:
        assert "capacity is 8192" in str(error)
    else:
        raise AssertionError("expected the real vendor G-buffer limit to be rejected")


def test_source_backend_extends_same_row_major_plan():
    mod = _binary_module([3000], relax.op.add)  # 9000 entries exceeds vendor storage
    lhs = np.arange(3000, dtype=np.float16)
    rhs = np.full(3000, 2, dtype=np.float16)
    asm, mp = driver.compile_module(mod, backend="source-0818")
    assert mp.top == 9000
    assert asm.execution_target == "source-0818"
    got = driver.run_compiled(asm, mp, {"lhs": lhs, "rhs": rhs})
    assert np.array_equal(got, np.asarray(lhs + rhs, dtype=np.float16).astype(np.float32))


if __name__ == "__main__":
    test_row_major_direct_subtile_matmul()
    test_negative_row_max_avoids_native_vendor_bug()
    test_native_row_sum_sets_scalar_save_length()
    test_full_capacity_transpose_has_no_tile_blocked_scratch()
    test_native_scalar_broadcast()
    test_vendor_capacity_is_checked_at_compile_time()
    test_source_backend_extends_same_row_major_plan()
    print("ALL 0818 BACKEND TESTS PASSED")
