"""Relax execution-plan compiler for the fixed-memory 0818 vendor target.

Unlike the legacy backend, this compiler never assigns a full graph to one
global-buffer address space.  Relax values live in host memory and every
arithmetic binding dispatches a bounded kernel to :class:`VendorSession`.
Slice and concat remain host layout operations; they perform no model math.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tvm import relax

from .v3_runtime import VendorSession


class V3CompileError(RuntimeError):
    pass


SUPPORTED_CALLS = {
    "relax.matmul",
    "relax.add",
    "relax.subtract",
    "relax.multiply",
    "relax.divide",
    "relax.sqrt",
    "relax.exp",
    "relax.negative",
    "relax.cos",
    "relax.sin",
    "relax.nn.silu",
    "relax.sum",
    "relax.max",
    "relax.broadcast_to",
    "relax.permute_dims",
    "relax.strided_slice",
    "relax.concat",
}


def _op_name(call):
    return call.op.name if hasattr(call.op, "name") else str(call.op)


def _int(value):
    if isinstance(value, relax.PrimValue):
        value = value.value
    return int(value)


@dataclass(frozen=True)
class PlannedBinding:
    output: int
    name: str
    args: tuple
    attrs: tuple


class RelaxVendorPlan:
    """Static Relax binding plan whose compute target is the vendor executable."""

    def __init__(self, func):
        if not isinstance(func, relax.Function):
            raise V3CompileError(f"expected Relax Function, got {type(func)}")
        if not isinstance(func.body, relax.SeqExpr):
            raise V3CompileError("V3 requires a normalized Relax SeqExpr")

        self.param_names = tuple(param.name_hint for param in func.params)
        self._slots = {param: index for index, param in enumerate(func.params)}
        self._slot_count = len(func.params)
        self.constants = []
        self.bindings = []
        self.host_layout_bindings = 0

        for block in func.body.blocks:
            for binding in block.bindings:
                output = self._slot_count
                self._slot_count += 1
                self._slots[binding.var] = output
                value = binding.value
                if isinstance(value, relax.Call):
                    name = _op_name(value)
                    if name not in SUPPORTED_CALLS:
                        raise V3CompileError(f"unsupported Relax op for vendor V3: {name}")
                    args = tuple(self._ref(arg) for arg in value.args)
                    attrs = self._attrs(name, value)
                    if name in ("relax.strided_slice", "relax.concat"):
                        self.host_layout_bindings += 1
                    self.bindings.append(PlannedBinding(output, name, args, attrs))
                elif isinstance(value, relax.Var):
                    self.bindings.append(
                        PlannedBinding(output, "alias", (self._ref(value),), ()))
                elif isinstance(value, relax.Tuple):
                    self.bindings.append(
                        PlannedBinding(output, "tuple", (self._ref(value),), ()))
                else:
                    raise V3CompileError(f"unsupported Relax binding value {type(value)}")
        self.output = self._ref(func.body.body)

    def _ref(self, expr):
        if isinstance(expr, relax.Var):
            if expr not in self._slots:
                raise V3CompileError(f"value used before definition: {expr}")
            return ("slot", self._slots[expr])
        if isinstance(expr, relax.Constant):
            index = len(self.constants)
            self.constants.append(np.asarray(expr.data.numpy(), dtype=np.float16))
            return ("const", index)
        if isinstance(expr, relax.Tuple):
            return ("tuple", tuple(self._ref(field) for field in expr.fields))
        if isinstance(expr, relax.ShapeExpr):
            return ("shape", tuple(_int(value) for value in expr.values))
        if isinstance(expr, relax.PrimValue):
            return ("prim", _int(expr))
        # TVM sometimes exposes the strided-slice integer tuple as an Array-like
        # object rather than a Relax Tuple.
        if hasattr(expr, "__iter__") and not isinstance(expr, (str, bytes)):
            return ("values", tuple(_int(value) for value in expr))
        raise V3CompileError(f"unsupported Relax operand {type(expr)}")

    @staticmethod
    def _attrs(name, call):
        if name in ("relax.sum", "relax.max"):
            return (
                ("axis", tuple(int(value) for value in call.attrs.axis)),
                ("keepdims", bool(call.attrs.keepdims)),
            )
        if name == "relax.permute_dims":
            axes = call.attrs.axes
            return (("axes", None if axes is None else tuple(int(value) for value in axes)),)
        if name == "relax.concat":
            return (("axis", int(call.attrs.axis)),)
        return ()

    def _value(self, ref, slots):
        kind, payload = ref
        if kind == "slot":
            return slots[payload]
        if kind == "const":
            return self.constants[payload]
        if kind == "tuple":
            return tuple(self._value(field, slots) for field in payload)
        if kind in ("shape", "values"):
            return payload
        if kind == "prim":
            return payload
        raise AssertionError(kind)

    @staticmethod
    def _layout_slice(source, axes, begin, end, strides=None):
        slices = [slice(None)] * source.ndim
        if strides is None:
            strides = (1,) * len(axes)
        for axis, first, last, step in zip(axes, begin, end, strides):
            axis %= source.ndim
            slices[axis] = slice(first, last, step)
        return np.ascontiguousarray(source[tuple(slices)], dtype=np.float16)

    @staticmethod
    def _dispatch(binding, args, vendor):
        name = binding.name
        attrs = dict(binding.attrs)
        if name == "alias" or name == "tuple":
            return args[0]
        if name == "relax.matmul":
            return vendor.gemm(args[0], args[1])
        binary = {
            "relax.add": "add",
            "relax.subtract": "subtract",
            "relax.multiply": "multiply",
            "relax.divide": "divide",
        }
        if name in binary:
            return vendor.binary(binary[name], args[0], args[1])
        unary = {
            "relax.sqrt": "sqrt",
            "relax.exp": "exp",
            "relax.negative": "negative",
            "relax.cos": "cos",
            "relax.sin": "sin",
            "relax.nn.silu": "silu",
        }
        if name in unary:
            return vendor.unary(unary[name], args[0])
        if name in ("relax.sum", "relax.max"):
            source = args[0]
            axes = tuple(axis % source.ndim for axis in attrs["axis"])
            if axes != (source.ndim - 1,) or not attrs["keepdims"]:
                raise V3CompileError(
                    f"V3 reduction requires last-axis keepdims, got axis={axes}")
            return (vendor.reduce_sum_last(source) if name == "relax.sum"
                    else vendor.reduce_max_last(source))
        if name == "relax.broadcast_to":
            return vendor.broadcast_to(args[0], args[1])
        if name == "relax.permute_dims":
            axes = attrs["axes"]
            if axes is None:
                axes = tuple(reversed(range(args[0].ndim)))
            if args[0].ndim != 2 or axes != (1, 0):
                raise V3CompileError(f"V3 transpose requires rank-2 axes (1,0), got {axes}")
            return vendor.transpose2d(args[0])
        if name == "relax.strided_slice":
            strides = args[4] if len(args) > 4 else None
            return RelaxVendorPlan._layout_slice(args[0], args[1], args[2], args[3], strides)
        if name == "relax.concat":
            return np.ascontiguousarray(
                np.concatenate(args[0], axis=attrs["axis"]), dtype=np.float16)
        raise AssertionError(name)

    def run(self, inputs, vendor=None):
        """Run the compiled plan; inputs are a name dictionary or positional list."""
        if isinstance(inputs, dict):
            missing = [name for name in self.param_names if name not in inputs]
            if missing:
                raise KeyError(f"missing V3 inputs: {', '.join(missing)}")
            values = [inputs[name] for name in self.param_names]
        else:
            values = list(inputs)
            if len(values) != len(self.param_names):
                raise ValueError(f"expected {len(self.param_names)} inputs, got {len(values)}")
        slots = [None] * self._slot_count
        for index, value in enumerate(values):
            slots[index] = np.asarray(value, dtype=np.float16)

        owns_vendor = vendor is None
        if owns_vendor:
            vendor = VendorSession()
        try:
            for binding in self.bindings:
                args = tuple(self._value(ref, slots) for ref in binding.args)
                slots[binding.output] = self._dispatch(binding, args, vendor)
            return self._value(self.output, slots)
        finally:
            if owns_vendor:
                vendor.close()

    def summary(self):
        counts = {}
        for binding in self.bindings:
            counts[binding.name] = counts.get(binding.name, 0) + 1
        return {
            "parameters": len(self.param_names),
            "bindings": len(self.bindings),
            "constants": len(self.constants),
            "host_layout_bindings": self.host_layout_bindings,
            "ops": counts,
        }


def compile_module(mod, func_name="main"):
    """Compile a Relax function into a fixed-buffer vendor execution plan."""
    return RelaxVendorPlan(mod[func_name])
