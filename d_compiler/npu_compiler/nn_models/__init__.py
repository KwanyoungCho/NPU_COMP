"""Standard TVM frontend model definitions (relax.frontend.nn).

These replace the hand-written Relax graph builders as the reference way to
describe a model: an ``nn.Module`` exported with ``export_tvm()``.  The same
definition builds for llvm (CPU validation) and for the NPU target.
"""
