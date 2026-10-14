"""Current scheduled TIR: native/Python word parity and independent NumPy values."""
import unittest

import numpy as np
from tvm import relax

from test_architecture_contracts import graph, lower
from npu_compiler import npu_link


class TirCodegenV09(unittest.TestCase):
    def check_operation(self, values, operation, expected, exact=False):
        mod = lower(graph([v.shape for v in values], operation))
        asm, plan = npu_link.compile_program(mod, native_mode="compare")
        inputs = dict(zip((p.name_hint for p in plan.func.params), values))
        got, _ = npu_link.run_program(asm, plan, plan.func, inputs)
        expected = np.asarray(expected, dtype="float16")
        if exact:
            np.testing.assert_array_equal(got.view("uint16"), expected.view("uint16"))
        else:
            np.testing.assert_allclose(got, expected, rtol=.002, atol=2e-5)

    def test_matmul_single_and_multiple_tiles(self):
        rng = np.random.default_rng(926)
        for m, k, n in ((64, 64, 64), (128, 128, 128)):
            with self.subTest(shape=(m, k, n)):
                x = rng.normal(0, .2, (m, k)).astype("float16")
                w = rng.normal(0, .2, (k, n)).astype("float16")
                self.check_operation([x, w], relax.op.matmul,
                                     x.astype("float32") @ w.astype("float32"))

    def test_binary_operations(self):
        rng = np.random.default_rng(9)
        x, y = [rng.uniform(.5, 1.5, (4, 64)).astype("float16") for _ in range(2)]
        for name in ("add", "subtract", "multiply", "divide"):
            with self.subTest(op=name):
                expected = getattr(np, name)(x.astype("float32"), y.astype("float32"))
                self.check_operation([x, y], getattr(relax.op, name), expected, exact=True)

    def test_unary_operations(self):
        x = np.random.default_rng(10).uniform(.5, 1.5, (4, 64)).astype("float16")
        for name in ("sqrt", "exp", "negative"):
            with self.subTest(op=name):
                self.check_operation([x], getattr(relax.op, name),
                                     getattr(np, name)(x.astype("float32")))

    def test_reductions(self):
        x = np.random.default_rng(12).normal(0, .2, (4, 64)).astype("float16")
        for name in ("sum", "max"):
            with self.subTest(op=name):
                op = lambda a: getattr(relax.op, name)(a, axis=[1], keepdims=True)
                expected = getattr(np, name)(x.astype("float32"), axis=1, keepdims=True)
                self.check_operation([x], op, expected, exact=True)

    def test_movement(self):
        x = np.random.default_rng(13).normal(size=(4, 64)).astype("float16")
        cases = [
            (x[:, :1].copy(), lambda a: relax.op.broadcast_to(a, (4, 64)),
             np.broadcast_to(x[:, :1], (4, 64))),
            (x, relax.op.permute_dims, x.T),
            (x, lambda a: relax.op.strided_slice(a, axes=[1], begin=[16], end=[48]),
             x[:, 16:48]),
        ]
        for i, (value, op, expected) in enumerate(cases):
            with self.subTest(case=i):
                self.check_operation([value], op, expected, exact=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
