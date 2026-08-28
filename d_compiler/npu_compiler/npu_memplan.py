"""Static address assignment derived from TVM's standard memory planning.

The NPU has no runtime allocator: every tensor needs a compile-time address in
one flat buffer.  Rather than hand-rolling liveness analysis, we run the stock
sequence (``ToNonDataflow`` -> ``RemovePurityChecking`` -> ``CallTIRRewrite``
-> ``StaticPlanBlockMemory``) and read its result: the pass groups tensors into
reusable storage pools and gives each tensor a (storage, offset) pair.  All we
add is assigning each pool a base address in the flat buffer.

  params/constants   bump-allocated first (live for the whole program)
  storage pools      bump-allocated after, each sized by the pass
  tensor address     pool_base + offset
"""
from __future__ import annotations

import tvm
from tvm import relax

DTYPE_BYTES = {"float16": 2, "float32": 4, "int8": 1, "int32": 4, "uint8": 1}


class MemPlanError(RuntimeError):
    pass


def plan_pipeline():
    """The standard passes that turn a lowered module into planned allocations."""
    return tvm.transform.Sequential([
        relax.transform.ToNonDataflow(),
        relax.transform.RemovePurityChecking(),
        relax.transform.CallTIRRewrite(),
        relax.transform.StaticPlanBlockMemory(),
    ], name="npu_memory_planning")


def _int(value):
    if isinstance(value, relax.PrimValue):
        value = value.value
    return int(value)


def _shape_of(expr):
    return [int(d) for d in expr.struct_info.shape]


def _param_tensors(param):
    """(name, shape, dtype) per tensor; packed params arrive as one tuple."""
    sinfo = param.struct_info
    if isinstance(sinfo, relax.TupleStructInfo):
        return [(f"{param.name_hint}.{i}", [int(d) for d in field.shape], field.dtype)
                for i, field in enumerate(sinfo.fields)]
    return [(param.name_hint, [int(d) for d in sinfo.shape], sinfo.dtype)]


def _numel(shape):
    total = 1
    for dim in shape:
        total *= dim
    return total


class StaticPlan:
    """Flat-buffer placement for one planned function."""

    def __init__(self, unit_bytes=2):
        self.unit_bytes = unit_bytes
        self.address = {}       # var name -> address in `unit_bytes` units
        self.nbytes = {}        # var name -> byte size
        self.pool_base = {}     # storage var name -> address
        self.pool_bytes = {}    # storage var name -> byte size
        self.top = 0            # total size in `unit_bytes` units

    def _alloc(self, nbytes):
        if nbytes % self.unit_bytes:
            nbytes += self.unit_bytes - nbytes % self.unit_bytes
        base = self.top
        self.top += nbytes // self.unit_bytes
        return base

    def summary(self):
        return {
            "total_units": self.top,
            "total_bytes": self.top * self.unit_bytes,
            "pools": len(self.pool_base),
            "tensors": len(self.address),
        }


def assign_addresses(mod, func_name="prefill", unit_bytes=2):
    """Run the standard planning passes and place every tensor in a flat buffer.

    Returns ``(planned_module, StaticPlan)``.  Parameters keep their argument
    order, so the host fills the buffer the same way it does today.
    """
    planned = plan_pipeline()(mod)
    func = planned[func_name]
    plan = StaticPlan(unit_bytes)

    for param in func.params:
        for name, shape, dtype in _param_tensors(param):
            nbytes = _numel(shape) * DTYPE_BYTES[dtype]
            plan.address[name] = plan._alloc(nbytes)
            plan.nbytes[name] = nbytes

    body = func.body
    blocks = body.blocks if isinstance(body, relax.SeqExpr) else []
    for block in blocks:
        for binding in block.bindings:
            value = binding.value
            if isinstance(value, relax.Var):
                # an alias binding (the rewrite names a call's result); it
                # shares the allocation it was bound from
                if value.name_hint in plan.address:
                    plan.address[binding.var.name_hint] = plan.address[value.name_hint]
                    plan.nbytes[binding.var.name_hint] = plan.nbytes[value.name_hint]
                continue
            if not isinstance(value, relax.Call):
                continue
            name = value.op.name if hasattr(value.op, "name") else str(value.op)
            var = binding.var.name_hint
            if name == "relax.memory.alloc_storage":
                size = _int(value.args[0].values[0])
                dtype = value.args[3].value
                nbytes = size * DTYPE_BYTES[dtype]
                plan.pool_bytes[var] = nbytes
                plan.pool_base[var] = plan._alloc(nbytes)
            elif name == "relax.memory.alloc_tensor":
                storage = value.args[0].name_hint
                if storage not in plan.pool_base:
                    raise MemPlanError(f"{var}: unknown storage {storage}")
                offset = _int(value.args[1])
                dtype = value.args[3].value
                plan.address[var] = plan.pool_base[storage] + \
                    offset * DTYPE_BYTES[dtype] // unit_bytes
                plan.nbytes[var] = _numel(_shape_of(binding.var)) * DTYPE_BYTES[dtype]
            elif name == "relax.builtin.alloc_tensor":
                # the pass leaves the function result unplanned; give it its own room
                shape = _shape_of(binding.var)
                dtype = binding.var.struct_info.dtype
                nbytes = _numel(shape) * DTYPE_BYTES[dtype]
                plan.address[var] = plan._alloc(nbytes)
                plan.nbytes[var] = nbytes
    return planned, plan


def unplanned_footprint(mod, func_name="prefill", unit_bytes=2):
    """Bytes the same module would need with no reuse (bump allocation)."""
    planned = plan_pipeline()(mod)
    func = planned[func_name]
    total = 0
    for param in func.params:
        for _, shape, dtype in _param_tensors(param):
            total += _numel(shape) * DTYPE_BYTES[dtype]
    body = func.body
    for block in (body.blocks if isinstance(body, relax.SeqExpr) else []):
        for binding in block.bindings:
            value = binding.value
            if not isinstance(value, relax.Call):
                continue
            name = value.op.name if hasattr(value.op, "name") else str(value.op)
            if name in ("relax.memory.alloc_tensor", "relax.builtin.alloc_tensor"):
                dtype = binding.var.struct_info.dtype
                total += _numel(_shape_of(binding.var)) * DTYPE_BYTES[dtype]
    return total
