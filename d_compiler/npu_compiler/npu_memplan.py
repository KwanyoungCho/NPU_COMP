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
from .npu_errors import LinkError
from .npu_abi import tensor_leaves, constant_key

DTYPE_BYTES = {"float16": 2, "float32": 4, "int8": 1, "int32": 4, "uint8": 1}


class MemPlanError(LinkError):
    pass


def plan_pipeline(views=True):
    """The standard passes that turn a lowered module into planned allocations."""
    from .npu_views import OutputViews
    return tvm.transform.Sequential(([
        relax.transform.RewriteDataflowReshape(),
        OutputViews(),
    ] if views else []) + [
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
    return [(name, shape, str(dtype)) for name, shape, dtype
            in tensor_leaves(param.struct_info, param.name_hint)]


def _numel(shape):
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


class StaticPlan:
    """Flat-buffer placement for one planned function."""

    def __init__(self, unit_bytes=2):
        if unit_bytes not in (1, 2, 4):
            raise MemPlanError("address unit must divide a 32-bit DMA cell")
        self.unit_bytes = unit_bytes
        self.address = {}       # var name -> address in `unit_bytes` units
        self.nbytes = {}        # var name -> byte size
        self.pool_base = {}     # storage var name -> address
        self.pool_bytes = {}    # storage var name -> byte size
        self.tuples = {}        # tuple var name -> member var names
        self.const_data = []    # (address, numpy array) the host must write
        self.top = 0            # total size in `unit_bytes` units
        self.snapshots = []     # (kernel index, var name, struct info) captured
        self.native = {}        # kernel name -> did the native walker cover it
        self.native_diagnostics = {}  # coverage/fallback explanation per kernel
        self.sram_peak = {}     # kernel name -> SRAM nibbles it needs
        self.kernel_span = []   # (kernel name, first word, last word+1)
        self.kernel_records = []  # one record per call, including direct emitters
        self.schedules = {}     # kernel name -> candidate selection diagnostics
        self.sram_layouts = {}  # per-buffer placement and lifetime conflicts
        self.numeric_policy = None      # selected by the linker/target
        self.frontend_precision = None  # declared by the importer, when known
        self.codegen_contract = None    # shared lowering contract hash
        self.fusion = None              # region/program cost selection
        self.residency = None           # versioned DMA forwarding decisions
        self.views = []                 # standard TVM reshape aliases
        self.compilation = None

    def _alloc(self, nbytes):
        # round every allocation up to a whole 32-bit cell so DMA, which moves
        # cells, never has to touch a neighbour's storage
        cell = 4
        if nbytes % cell:
            nbytes += cell - nbytes % cell
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


def _unique_names(func):
    """Give every binding variable a distinct name.

    The plan (and the linker) key tensors by ``name_hint``, which the
    hand-written frontends kept unique by construction.  Traced modules reuse
    hints like ``lv`` across bindings, and a duplicate would silently alias
    two different tensors to the last one's address.
    """
    from tvm.relax.expr_functor import PyExprMutator, mutator

    @mutator
    class Rename(PyExprMutator):
        def __init__(self):
            super().__init__()
            self.seen = set()

        def _fresh(self, var):
            name = var.name_hint
            while name in self.seen:
                name = name + "_"
            self.seen.add(name)
            if name == var.name_hint:
                return var
            maker = (relax.DataflowVar
                     if isinstance(var, relax.DataflowVar) else relax.Var)
            return maker(name, var.struct_info)

        def visit_var_def_(self, var):
            return self._fresh(var)

        def visit_dataflow_var_def_(self, var):
            return self._fresh(var)

    # params are visited as definitions first, so they claim their own names;
    # pre-seeding them would make the renamer collide with itself and append
    # an underscore to every parameter
    return Rename().visit_expr(func)


def assign_addresses(mod, func_name="prefill", unit_bytes=2, views=True):
    """Run the standard planning passes and place every tensor in a flat buffer.

    Returns ``(planned_module, StaticPlan)``.  Parameters keep their argument
    order, so the host fills the buffer the same way it does today.
    """
    planned = plan_pipeline(views=views)(mod)
    planned[func_name] = _unique_names(planned[func_name])
    func = planned[func_name]
    plan = StaticPlan(unit_bytes)

    for param in func.params:
        def register_tuple(name, info):
            if isinstance(info, relax.TupleStructInfo):
                plan.tuples[name] = [f"{name}.{i}" for i in range(len(info.fields))]
                for member, field in zip(plan.tuples[name], info.fields):
                    register_tuple(member, field)
        register_tuple(param.name_hint, param.struct_info)
        for name, shape, dtype in _param_tensors(param):
            if dtype not in DTYPE_BYTES:
                raise MemPlanError(f"{name}: unsupported device ABI dtype {dtype}")
            nbytes = _numel(shape) * DTYPE_BYTES[dtype]
            plan.address[name] = plan._alloc(nbytes)
            plan.nbytes[name] = nbytes

    # graph constants (e.g. an index vector) need a home the kernels can read
    constants = {}
    body = func.body
    blocks = body.blocks if isinstance(body, relax.SeqExpr) else []
    def collect_constant(expr):
        if isinstance(expr, relax.Constant):
            array = expr.data.numpy()
            key = constant_key(expr)
            if key not in constants:
                address = plan._alloc(array.nbytes)
                constants[key] = address
                plan.const_data.append((address, array))
    relax.analysis.post_order_visit(body, collect_constant)
    for block in blocks:
        for binding in blocks and block.bindings or []:
            value = binding.value
            if isinstance(value, relax.Constant):
                plan.address[binding.var.name_hint] = constants[constant_key(value)]
                plan.nbytes[binding.var.name_hint] = value.data.numpy().nbytes
                continue
            if not isinstance(value, relax.Call):
                continue
            for arg in value.args:
                if isinstance(arg, relax.Constant):
                    array = arg.data.numpy()
                    # TVM re-wraps objects per access, so key by content
                    key = (str(array.dtype), array.shape, array.tobytes())
                    if key in constants:
                        continue
                    nbytes = array.size * array.dtype.itemsize
                    address = plan._alloc(nbytes)
                    constants[key] = address
                    plan.const_data.append((address, array))
    plan.graph_constants = constants
    for block in blocks:
        for binding in block.bindings:
            value = binding.value
            if isinstance(value, relax.Tuple):
                # a multi-output kernel (split) binds its results as a tuple;
                # remember the members so TupleGetItem can resolve them
                def member(expr, name):
                    if isinstance(expr, relax.Var):
                        return expr.name_hint
                    if isinstance(expr, relax.Constant):
                        plan.address[name] = constants[constant_key(expr)]
                        plan.nbytes[name] = expr.data.numpy().nbytes
                    elif isinstance(expr, relax.Tuple):
                        plan.tuples[name] = [member(field, f"{name}.{i}")
                                             for i, field in enumerate(expr.fields)]
                    else:
                        raise MemPlanError(f"{name}: unsupported tuple member")
                    return name
                member(value, binding.var.name_hint)
                continue
            if isinstance(value, relax.TupleGetItem):
                members = plan.tuples.get(value.tuple_value.name_hint)
                if members and value.index < len(members):
                    source = members[value.index]
                    if source in plan.tuples:
                        plan.tuples[binding.var.name_hint] = plan.tuples[source]
                    else:
                        plan.address[binding.var.name_hint] = plan.address[source]
                        plan.nbytes[binding.var.name_hint] = plan.nbytes[source]
                continue
            if isinstance(value, relax.Var):
                # an alias binding (the rewrite names a call's result); it
                # shares the allocation it was bound from
                if value.name_hint in plan.address:
                    plan.address[binding.var.name_hint] = plan.address[value.name_hint]
                    plan.nbytes[binding.var.name_hint] = plan.nbytes[value.name_hint]
                elif value.name_hint in plan.tuples:
                    plan.tuples[binding.var.name_hint] = plan.tuples[value.name_hint]
                continue
            if not isinstance(value, relax.Call):
                continue
            name = value.op.name if hasattr(value.op, "name") else str(value.op)
            var = binding.var.name_hint
            if name == "relax.reshape":
                source = value.args[0]
                if not isinstance(source, relax.Var) or source.name_hint not in plan.address:
                    raise MemPlanError(f"{var}: unplaced reshape source")
                dtype = binding.var.struct_info.dtype
                nbytes = _numel(_shape_of(binding.var)) * DTYPE_BYTES[dtype]
                if dtype != source.struct_info.dtype or nbytes != plan.nbytes[source.name_hint]:
                    raise MemPlanError(f"{var}: reshape must preserve dtype and byte size")
                plan.address[var] = plan.address[source.name_hint]
                plan.nbytes[var] = nbytes
                plan.views.append(dict(name=var, source=source.name_hint,
                                       address_bytes=plan.address[var] * unit_bytes,
                                       size_bytes=nbytes))
            elif name in ("relax.memory.view", "relax.memory.ensure_zero_offset"):
                source = value.args[0]
                if not isinstance(source, relax.Var) or source.name_hint not in plan.address:
                    raise MemPlanError(f"{var}: unplaced view source")
                offset = _int(value.args[3]) if name == "relax.memory.view" else 0
                dtype = binding.var.struct_info.dtype
                nbytes = _numel(_shape_of(binding.var)) * DTYPE_BYTES[dtype]
                if offset < 0 or offset % unit_bytes or offset + nbytes > plan.nbytes[source.name_hint]:
                    raise MemPlanError(f"{var}: view byte interval/alignment exceeds source")
                plan.address[var] = plan.address[source.name_hint] + offset // unit_bytes
                plan.nbytes[var] = nbytes
                plan.views.append(dict(name=var, source=source.name_hint, offset_bytes=offset,
                                       address_bytes=plan.address[var] * unit_bytes, size_bytes=nbytes))
            elif name == "relax.memory.alloc_storage":
                # StaticPlanBlockMemory emits token->bytes here; dtype is
                # an allocation hint, not the unit of the shape expression.
                nbytes = _int(value.args[0].values[0])
                plan.pool_bytes[var] = nbytes
                plan.pool_base[var] = plan._alloc(nbytes)
            elif name == "relax.memory.alloc_tensor":
                storage = value.args[0].name_hint
                if storage not in plan.pool_base:
                    raise MemPlanError(f"{var}: unknown storage {storage}")
                offset = _int(value.args[1])
                dtype = value.args[3].value
                nbytes = _numel(_shape_of(binding.var)) * DTYPE_BYTES[dtype]
                if offset < 0 or offset % unit_bytes:
                    raise MemPlanError(f"{var}: byte offset {offset} cannot be "
                                       f"represented in {unit_bytes}-byte units")
                if offset + nbytes > plan.pool_bytes[storage]:
                    raise MemPlanError(f"{var}: tensor exceeds storage {storage}")
                plan.address[var] = plan.pool_base[storage] + offset // unit_bytes
                plan.nbytes[var] = nbytes
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


def compact_unused_storage(words, plan):
    """Relocate whole unused TVM storage pools after SRAM/codegen decisions.

    Keep every pool touched by DMA or the external ABI, including later
    versions sharing a slot. A live pool's internal layout never changes.
    This final address fixup adds neither allocation search nor DMA packets.
    """
    from .isa_v09 import OP_GLOAD, OP_GSTORE, decode_dma
    from .v09_stats import packets
    unit = plan.unit_bytes
    before = plan.top*unit
    detail = dict(policy='remove wholly unused TVM storage pools',before_bytes=before,
                  after_bytes=before,saved_bytes=0,removed_pools=[],removed_tensors={})
    plan.dram_compaction = detail
    if plan.snapshots:
        detail['reason'] = 'snapshot memory contract'; return
    required = []
    def require(address,size):
        required.append((address,address+size))
    for param in plan.func.params:
        for name,_,_ in tensor_leaves(param.struct_info,param.name_hint):
            require(plan.address[name]*unit,plan.nbytes[name])
    def output_leaves(tree):
        if isinstance(tree,list):
            return [v for child in tree for v in output_leaves(child)]
        return [tree]
    outputs = output_leaves(plan.outputs)
    for item in outputs: require(item['offset_bytes'],item['size_bytes'])
    for address,array in plan.const_data: require(address*unit,array.nbytes)
    require(plan.constant_base*unit,len(plan.constant_values)*2)
    require(plan.one_fp32_base*unit,4)
    dma = []
    for index,op,packet in packets(words):
        if op not in (OP_GLOAD,OP_GSTORE): continue
        d = decode_dma(packet)
        start = d['g_addr']*4
        # Conservatively protect the whole strided span, including gaps.
        require(start,((d['rows']-1)*d['g_stride']+d['cols'])*4)
        dma.append((index,start))
    removed = []
    for name,address in plan.pool_base.items():
        lo = address*unit
        hi = lo+(plan.pool_bytes[name]+3)//4*4
        if not any(lo < end and start < hi for start,end in required):
            removed.append((lo,hi,name))
    removed.sort()
    if not removed:
        return
    def relocate(address):
        if any(lo <= address < hi for lo,hi,_ in removed): return None
        return address-sum(hi-lo for lo,hi,_ in removed if hi <= address)
    def units(address):
        result = relocate(address*unit)
        if result is None: raise MemPlanError('live address inside removed storage')
        return result//unit
    for index,address in dma:
        words[index+1] = relocate(address)//4
    old_addresses = dict(plan.address)
    for name,address in old_addresses.items():
        moved = relocate(address*unit)
        if moved is None:
            detail['removed_tensors'][name] = dict(original_address_bytes=address*unit,
                                                   size_bytes=plan.nbytes[name])
            del plan.address[name]
        else: plan.address[name] = moved//unit
    for lo,hi,name in removed:
        detail['removed_pools'].append(dict(name=name,original_address_bytes=lo,size_bytes=hi-lo))
        del plan.pool_base[name]; del plan.pool_bytes[name]
    plan.pool_base = {name:units(a) for name,a in plan.pool_base.items()}
    plan.const_data = [(units(a),v) for a,v in plan.const_data]
    plan.graph_constants = {k:units(a) for k,a in plan.graph_constants.items()}
    plan.constant_base = units(plan.constant_base)
    plan.one_fp32_base = units(plan.one_fp32_base)
    plan.top = units(plan.top)
    for item in outputs: item['offset_bytes'] = relocate(item['offset_bytes'])
    for view in plan.views:
        view['original_address_bytes'] = view['address_bytes']
        view['address_bytes'] = relocate(view['address_bytes'])
    scoped = {c['index']:c['parameter_sram_nibbles'] for c in plan.call_schedules}
    for record in plan.kernel_records:
        for i,arg in enumerate(record['arguments']):
            old = arg['address_bytes']
            arg['original_address_bytes'] = old
            if i in scoped.get(record['index'],{}):
                arg.update(address_bytes=None,storage='sram',
                           sram_address_nibbles=scoped[record['index']][i])
            else:
                arg['address_bytes'] = relocate(old)
                arg['storage'] = 'dram' if arg['address_bytes'] is not None else 'unused'
    for value in getattr(plan,'joint_memory',{}).get('retained_values',[]):
        value['original_address_bytes'] = value['address_bytes']
        value['address_bytes'] = relocate(value['address_bytes'])
    for event in (plan.residency or {}).get('events',[]):
        event['original_global_address_bytes'] = event['global_address_bytes']
        event['global_address_bytes'] = relocate(event['global_address_bytes'])
    detail.update(after_bytes=plan.top*unit,saved_bytes=before-plan.top*unit)
