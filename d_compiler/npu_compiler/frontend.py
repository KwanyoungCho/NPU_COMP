"""PyTorch import frontend: real model -> Relax IRModule our backend can consume.

Uses TVM's torch.export frontend, then normalizes:
  - FoldConstant: precompute the Linear weight transpose (permute_dims(const)->const)
  - (tuple output, bias broadcast, fp16 G-buffer) are handled in memplan/driver.

Export with CONCRETE example inputs => static shapes (prefill at a fixed length).
Start small (Linear) and grow op coverage as we add layers.
"""
import torch
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program


def import_torch(model, example_inputs):
    """torch nn.Module + example inputs (tuple) -> normalized Relax IRModule."""
    from . import import_legalize
    model = model.eval()
    exported = torch.export.export(model, example_inputs)
    mod = from_exported_program(exported)
    mod = relax.transform.FoldConstant()(mod)     # fold weight transpose etc.
    mod = import_legalize.legalize(mod)           # high-level ops -> our primitives
    return mod
