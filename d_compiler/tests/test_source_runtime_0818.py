"""Extended-capacity tests for the vendor-compatible source C-model runtime."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))

from npu_compiler.isa_0818 import DST, IMM, SRC1, Asm
from npu_compiler.source_runtime_0818 import run


def test_dynamic_gbuffer_above_vendor_limit():
    values = np.zeros(10001, dtype=np.float16)
    values[9000] = np.float16(3.25)
    program = Asm()
    program.vlen(1).addr(SRC1, 9000).load(0, SRC1)
    program.v_add(IMM, 2).addr(DST, 10000).save(0).finish()
    output = run(program, values)
    assert output.size == 10001
    assert output[9000] == np.float16(3.25)
    assert output[10000] == np.float16(5.25)


def test_broadcast_uses_full_32bit_scalar_address():
    values = np.zeros(70001, dtype=np.float16)
    values[70000] = np.float16(6.5)
    program = Asm().vlen(3).v_broadcast_addr(70000)
    program.addr(DST, 66000).save(0).finish()
    output = run(program, values)
    assert np.array_equal(output[66000:66003], np.full(3, 6.5, dtype=np.float16))


if __name__ == "__main__":
    test_dynamic_gbuffer_above_vendor_limit()
    test_broadcast_uses_full_32bit_scalar_address()
    print("ALL EXTENDED 0818 SOURCE-RUNTIME TESTS PASSED")
