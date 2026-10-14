"""Static entry validation and byte-checked tensor tree ABI."""
import numpy as np
from tvm import relax, tir
from .npu_errors import LinkError


def bindings(func):
    body = func.body
    if isinstance(body, relax.SeqExpr):
        return [binding for block in body.blocks for binding in block.bindings]
    return []


def tensor_leaves(info, name):
    if isinstance(info, relax.TupleStructInfo):
        for index, field in enumerate(info.fields):
            yield from tensor_leaves(field, f"{name}.{index}")
    elif isinstance(info, relax.TensorStructInfo) and info.shape is not None:
        try:
            shape = tuple(int(dim) for dim in info.shape)
            dtype = np.dtype(str(info.dtype))
        except (TypeError, ValueError) as error:
            raise LinkError(f"{name}: ABI requires a static shape and supported dtype") from error
        if any(dim <= 0 for dim in shape):
            raise LinkError(f"{name}: ABI dimensions must be positive, got {shape}")
        yield name, shape, dtype
    else:
        raise LinkError(f"{name}: ABI supports static tensor/tuple values only")


def constant_key(expr):
    array = expr.data.numpy()
    return str(array.dtype), array.shape, array.tobytes()


def output_tree(func, plan):
    def named(name, info):
        if isinstance(info, relax.TupleStructInfo):
            members = plan.tuples.get(name, [f"{name}.{i}" for i in range(len(info.fields))])
            if len(members) != len(info.fields):
                raise LinkError(f"{name}: incomplete tuple placement")
            return [named(member, field) for member, field in zip(members, info.fields)]
        _, shape, dtype = next(tensor_leaves(info, name))
        if name not in plan.address:
            raise LinkError(f"output {name}: no static placement")
        return leaf(plan.address[name], shape, dtype, name)

    def leaf(address, shape, dtype, name):
        size = int(np.prod(shape)) * dtype.itemsize
        offset = address * plan.unit_bytes
        if offset < 0 or offset + size > plan.top * plan.unit_bytes:
            raise LinkError(f"output {name}: byte interval exceeds image")
        return dict(name=name, shape=list(shape), dtype=str(dtype), offset_bytes=offset, size_bytes=size)

    def expr(value):
        if isinstance(value, relax.Var):
            return named(value.name_hint, value.struct_info)
        if isinstance(value, relax.Tuple):
            return [expr(field) for field in value.fields]
        if isinstance(value, relax.TupleGetItem):
            return expr(value.tuple_value)[value.index]
        if isinstance(value, relax.Constant):
            a = value.data.numpy()
            return leaf(plan.graph_constants[constant_key(value)], a.shape, a.dtype, "constant")
        raise LinkError(f"unsupported entry return expression: {type(value).__name__}")

    return expr(func.body.body if isinstance(func.body, relax.SeqExpr) else func.body)


def read_outputs(image, tree):
    raw = np.ascontiguousarray(image).view(np.uint8).reshape(-1)
    def read(node):
        if isinstance(node, list):
            return tuple(read(item) for item in node)
        start, size = node["offset_bytes"], node["size_bytes"]
        if start < 0 or start + size > raw.size:
            raise LinkError(f"output {node['name']}: runtime image is too short")
        return raw[start:start + size].view(np.dtype(node["dtype"])).reshape(node["shape"]).copy()
    return read(tree)


def validate_entry(mod, name, plan):
    """Every statement is accounted for; outputs must have a real writer.

    Initialization is tracked by byte interval so storage/reshape aliases
    cannot turn an unwritten allocation into a valid output.
    """
    func = mod[name]
    initialized = []
    def mark(address, size):
        initialized.append((address * plan.unit_bytes, address * plan.unit_bytes + size))
    def available(start, size):
        end = start + size
        for lo, hi in sorted(initialized):
            if lo > start:
                break
            if hi > start:
                start = hi
            if start >= end:
                return True
        return size == 0
    for param in func.params:
        for leaf, _, _ in tensor_leaves(param.struct_info, param.name_hint):
            mark(plan.address[leaf], plan.nbytes[leaf])
    for address, array in plan.const_data:
        mark(address, array.nbytes)
    administrative = {
        "relax.memory.alloc_storage", "relax.memory.alloc_tensor",
        "relax.builtin.alloc_tensor", "relax.reshape", "relax.memory.view",
        "relax.memory.ensure_zero_offset",
        "relax.memory.kill_tensor", "relax.memory.kill_storage",
    }
    for binding in bindings(func):
        value = binding.value
        where = f"{name}/{binding.var.name_hint}"
        if not isinstance(binding, relax.VarBinding):
            raise LinkError(f"{where}: unsupported binding {type(binding).__name__}")
        if isinstance(value, (relax.Var, relax.Tuple, relax.TupleGetItem, relax.Constant)):
            continue
        if not isinstance(value, relax.Call):
            raise LinkError(f"{where}: unsupported expression {type(value).__name__}")
        if not isinstance(value.op, relax.GlobalVar):
            op = getattr(value.op, "name", str(value.op))
            if op not in administrative:
                raise LinkError(f"{where}: unsupported residual call {op}; no device implementation")
            continue
        prim = mod[value.op]
        if not isinstance(prim, tir.PrimFunc):
            raise LinkError(f"{where}: call {value.op.name_hint} is not a device PrimFunc")
        args = []
        for arg in value.args:
            args.extend(arg.fields if isinstance(arg, relax.Tuple) else [arg])
        if len(args) != len(prim.params):
            raise LinkError(f"{where}: PrimFunc argument count mismatch")
        reads, writes = set(), set()
        def visit(node):
            if isinstance(node, tir.BufferLoad):
                reads.add(node.buffer.data)
            elif isinstance(node, tir.BufferStore):
                writes.add(node.buffer.data)
        tir.stmt_functor.post_order_visit(prim.body, visit)
        pending = []
        for param, arg in zip(prim.params, args):
            if param not in prim.buffer_map:
                raise LinkError(f"{where}: scalar PrimFunc arguments are unsupported")
            buf = prim.buffer_map[param]
            if isinstance(arg, relax.Constant):
                address = plan.graph_constants[constant_key(arg)]
                size = arg.data.numpy().nbytes
            elif isinstance(arg, relax.Var) and arg.name_hint in plan.address:
                address, size = plan.address[arg.name_hint], plan.nbytes[arg.name_hint]
            else:
                raise LinkError(f"{where}: unplaced kernel argument")
            expected = int(np.prod([int(d) for d in buf.shape])) * np.dtype(str(buf.dtype)).itemsize
            if expected != size or str(arg.struct_info.dtype) != str(buf.dtype):
                raise LinkError(f"{where}: kernel buffer shape/dtype disagrees with argument")
            # A reduction reads its own initialized output. Input-only buffers
            # must already be defined; writing blocks are checked by codegen.
            if buf.data in reads and buf.data not in writes and not available(address * plan.unit_bytes, size):
                raise LinkError(f"{where}: reads an undefined tensor")
            if buf.data in writes:
                pending.append((address, size))
        for address, size in pending:
            mark(address, size)
    tree = output_tree(func, plan)
    def check(node):
        if isinstance(node, list):
            for field in node:
                check(field)
        elif not available(node["offset_bytes"], node["size_bytes"]):
            raise LinkError(f"{name}: output {node['name']} has no producer")
    check(tree)
    plan.outputs = tree
