"""Call-site schedule selection with explicit cross-kernel SRAM lifetimes.

Whole intermediate tensors may stay in reserved SRAM until their last reader.
Every beam state carries those reservations into schedule selection. Inputs
are bound to retained regions; output copies write into them instead of DRAM.
The all-DRAM path remains available. Final externally observable values and
ambiguous aliases keep their original storage contract.
"""
from dataclasses import dataclass, replace
from math import prod

import tvm
from tvm import relax, tir

from .npu_abi import bindings, constant_key, tensor_leaves
from .npu_errors import LinkError
from .npu_sram import align
from .npu_compilation import current, phase
from .npu_schedule import _fit_kernel, _place_sram, _sram_buffers
from .tir_codegen_v09 import Walker, SramEmitter, V09TirError
from .v09_asm import V09Asm
from .v09_stats import analyze
from .peephole import eliminate_dead_stores
from . import native


def overlap(a, size, b, other):
    return a < b + other and b < a + size


@dataclass
class Call:
    gv: object
    prim: object
    addresses: tuple
    sizes: tuple
    reads: tuple
    writes: tuple


@dataclass(frozen=True)
class Value:
    key: tuple
    address: int                 # global bytes; identifies a version, not just a slot
    size: int
    last: int
    readers: tuple              # (call index, argument index)


def discover(module, plan):
    """Use actual byte intervals so reshape/subviews and pool reuse are explicit."""
    calls = []
    for binding in bindings(plan.func):
        call = binding.value
        if not (isinstance(call, relax.Call) and isinstance(call.op, relax.GlobalVar)
                and isinstance(module[call.op], tir.PrimFunc)):
            continue
        prim = module[call.op]
        args = [field for arg in call.args for field in
                (arg.fields if isinstance(arg, relax.Tuple) else [arg])]
        addresses = tuple(plan.graph_constants[constant_key(arg)] if isinstance(arg,relax.Constant)
                          else plan.address[arg.name_hint] for arg in args)
        buffers = [prim.buffer_map[p] for p in prim.params]
        sizes = tuple(prod(int(d) for d in b.shape) * {'float16':2,'float32':4,'int8':1,
                      'uint8':1,'int32':4}[str(b.dtype)] for b in buffers)
        read, write = set(), set()
        def visit(node):
            if isinstance(node,tir.BufferLoad): read.add(node.buffer.data)
            if isinstance(node,tir.BufferStore): write.add(node.buffer.data)
        tir.stmt_functor.post_order_visit(prim.body,visit)
        calls.append(Call(call.op,prim,addresses,sizes,
                          tuple(i for i,b in enumerate(buffers) if b.data in read),
                          tuple(i for i,b in enumerate(buffers) if b.data in write)))
    protected = []
    def outputs(tree):
        if isinstance(tree,list):
            for child in tree: outputs(child)
        else: protected.append((tree['offset_bytes'],tree['size_bytes']))
    outputs(plan.outputs)
    for param in plan.func.params:
        for name,_,_ in tensor_leaves(param.struct_info,param.name_hint):
            protected.append((plan.address[name]*2,plan.nbytes[name]))
    for address,array in plan.const_data:
        protected.append((address*2,array.nbytes))
    values, reasons = [], []
    for index,call in enumerate(calls):
        for output in call.writes:
            buf=call.prim.buffer_map[call.prim.params[output]]
            address,size=call.addresses[output]*2,call.sizes[output]
            if (str(buf.dtype)!='float16' or output in call.reads
                    or buf.strides or int(buf.elem_offset)!=0):
                continue
            if any(overlap(address,size,a,s) for a,s in protected):
                continue
            if any(i!=output and overlap(address,size,call.addresses[i]*2,call.sizes[i])
                   for i in call.reads+call.writes):
                continue
            readers, reason = [], None
            for next_index,following in enumerate(calls[index+1:],index+1):
                for arg in following.reads:
                    start,span=following.addresses[arg]*2,following.sizes[arg]
                    if not overlap(address,size,start,span): continue
                    if arg in following.writes:
                        # Redirecting a read/write parameter would also move
                        # its next version's store to SRAM. That version has
                        # no reservation contract, so later DRAM reads would
                        # see stale data. Keep this chain on its original ABI.
                        reason='in-place consumer update'; break
                    incoming=following.prim.buffer_map[following.prim.params[arg]]
                    dtype=str(incoming.dtype)
                    if (start<address or start+span>address+size or dtype!='float16'
                            or incoming.strides or int(incoming.elem_offset)!=0):
                        reason='partial overlap or dtype reinterpretation'; break
                    readers.append((next_index,arg))
                if reason: break
                writes=[(following.addresses[i]*2,following.sizes[i]) for i in following.writes
                        if overlap(address,size,following.addresses[i]*2,following.sizes[i])]
                if writes:
                    # A partial update could leave old bytes observable later.
                    if any(a>address or a+s<address+size for a,s in writes):
                        reason='partial overwrite'; break
                    break  # complete new version replaces this allocation
            if reason:
                reasons.append(dict(producer=index,argument=output,reason=reason)); continue
            if readers:
                values.append(Value((index,output),address,size,max(i for i,_ in readers),tuple(readers)))
    return calls, values, reasons


def parameter_scopes(prim, indices):
    """Expose the retained input/output contract before Schedule runs."""
    if not indices: return prim
    replacements, variables = {}, {}
    for index in indices:
        old=prim.buffer_map[prim.params[index]]
        data=tir.Var(old.data.name,tvm.ir.PointerType(tvm.ir.PrimType(old.dtype),'global.sram'))
        variables[old.data]=data
        replacements[old]=tir.decl_buffer(old.shape,old.dtype,old.name,data=data,
            strides=old.strides,elem_offset=old.elem_offset,scope='global.sram',
            data_alignment=old.data_alignment,offset_factor=old.offset_factor)
    def buf(b): return replacements.get(b,b)
    def region(r): return tir.BufferRegion(buf(r.buffer),r.region)
    def rewrite(node):
        if isinstance(node,tir.BufferLoad) and node.buffer in replacements:
            return tir.BufferLoad(buf(node.buffer),node.indices)
        if isinstance(node,tir.BufferStore) and node.buffer in replacements:
            return tir.BufferStore(buf(node.buffer),node.value,node.indices)
        if isinstance(node,tir.Block):
            return tir.Block(node.iter_vars,[region(r) for r in node.reads],
                [region(r) for r in node.writes],node.name_hint,node.body,node.init,
                [buf(b) for b in node.alloc_buffers],
                [tir.MatchBufferRegion(buf(m.buffer),region(m.source)) for m in node.match_buffers],node.annotations)
    body=tir.stmt_functor.ir_transform(prim.body,None,rewrite)
    body=tir.stmt_functor.substitute(body,variables)
    return tir.PrimFunc(prim.params,body,prim.ret_type,
                       {p:buf(b) for p,b in prim.buffer_map.items()},prim.attrs)


@dataclass
class Choice:
    words: list
    scheduled: object
    layout: object
    diagnostics: dict
    scratch: tuple
    sram: dict
    top: int
    workspace_start: int
    parameter_sram: dict
    native: bool
    native_diagnostics: dict
    statistics: dict
    workspace_limit: int
    retained_bytes: int


def emit(prim, addresses, sram, scratch, constants, one, mode):
    with phase('bound_kernel_codegen'):
        return _emit(prim, addresses, sram, scratch, constants, one, mode)


def _emit(prim, addresses, sram, scratch, constants, one, mode):
    diagnostics={}
    words=native.codegen_kernel(prim,addresses,sram,scratch,constants,one,
                                diagnostics=diagnostics) if mode else None
    if words is not None and mode=='use': return list(map(int,words)),True,diagnostics
    asm=V09Asm(); walker=Walker(asm,{},SramEmitter(asm))
    walker.constants=constants; walker.one_fp32=one; walker.scratch_slots=scratch
    for p,a in zip(prim.params,addresses):
        b=prim.buffer_map[p]; walker.bases[b.data]=a; walker.dtypes[b.data]=str(b.dtype)
    for b,a in sram.items(): walker.declare_sram(b,a)
    with phase('bound_kernel_python', fallback_reason=diagnostics.get('reason')):
        walker.run(prim,{})
    if words is not None and list(map(int,words))!=asm.words:
        raise LinkError('resident kernel: native/Python instruction mismatch')
    return asm.words,words is not None,diagnostics


def reserve(live,size,capacity):
    """Allocate high addresses; all kernel workspace stays below live values."""
    size=align(size*2)
    ceiling=capacity//8*8
    for address,span in sorted(live.values(),reverse=True):
        if ceiling-address-span>=size: return ceiling-size,size
        ceiling=address
    if ceiling>=size: return ceiling-size,size
    return None


def probe(prim, addresses, sizes, sram, scratch, constants, one, mode, reuse=True):
    """Reuse costs when only disjoint DRAM parameter addresses have changed.

    Physical SRAM addresses stay in the key: descriptor half-word equality
    affects peephole costs. Global aliases and unknown bindings take the full
    path. The selected candidate is always emitted and checked at real addresses.
    """
    def create():
        words, was_native, detail = emit(prim,addresses,sram,scratch,constants,one,mode)
        with phase('bound_kernel_peephole'):
            cleaned = eliminate_dead_stores(words)
        with phase('bound_kernel_statistics'):
            statistics = analyze(cleaned,detail=False)
        return words, was_native, detail, statistics
    params = [prim.buffer_map[p] for p in prim.params]
    global_ranges = [(address*2,size) for b,address,size in zip(params,addresses,sizes)
                     if b.scope() != 'global.sram']
    if any(overlap(a,s,b,t) for i,(a,s) in enumerate(global_ranges)
           for b,t in global_ranges[i+1:]):
        return create()
    ordered = [b for b in params if b in sram] + _sram_buffers(prim)
    if len(ordered) != len(sram) or any(b not in sram for b in ordered):
        return create()
    settings = (tuple(sram[b] for b in ordered),tuple(scratch),
                tuple(sorted(constants.items())),one,mode,
                tuple(a%4 for a,_ in global_ranges))
    return current().probe(prim,settings,create,enabled=reuse)


def plan_program(module, plan, profile, scratch_base, slot_count, constants, one, mode,
                 *, _reuse_costs=True):
    calls,values,excluded=discover(module,plan)
    diagnostics=dict(policy='bounded call-site schedule and retained-SRAM search',
                     beam_width=profile.memory_beam_width,excluded=excluded,
                     candidates=len(values),trials=0,rejections=[],calls=[])
    if not values: return {},diagnostics
    if profile.memory_beam_width<1: raise LinkError('memory_beam_width must be positive')
    by_key={v.key:v for v in values}
    readers={(i,p):v.key for v in values for i,p in v.readers}
    # state = (retained regions, selected prefix, DMA, words, peak)
    states=[({},[],0,0,0)]
    cache={}
    for index,call in enumerate(calls):
        outputs=[v for v in values if v.key[0]==index]
        extensions=[]
        for live,prefix,dma,word_count,peak in states:
            # Keep/spill is a decision for each produced value, not a named
            # mapping template. Bound unusually wide multi-output branches.
            reservations=[dict(live)]
            for value in outputs:
                expanded=[]
                for reservation in reservations:
                    expanded.append(reservation)
                    slot=reserve(reservation,value.size,profile.sram_nibbles)
                    if slot is not None:
                        kept=dict(reservation);kept[value.key]=slot;expanded.append(kept)
                reservations=sorted(expanded,key=lambda r:(-sum(by_key[k].size*len(by_key[k].readers)
                    for k in r if k[0]==index),len(r)))[:max(2,profile.memory_beam_width*2)]
                if live not in reservations: reservations.append(dict(live))
            for reservation in reservations:
                bindings_by_param={}
                for p in call.reads:
                    key=readers.get((index,p))
                    if key in live:
                        bindings_by_param[p]=live[key][0]+(call.addresses[p]*2-by_key[key].address)*2
                for value in outputs:
                    if value.key in reservation:
                        bindings_by_param[value.key[1]]=reservation[value.key][0]
                capacity=min((a for a,_ in reservation.values()),default=profile.sram_nibbles)
                signature=(index,tuple(sorted(bindings_by_param.items())),capacity)
                diagnostics['trials']+=1
                if signature not in cache:
                    try:
                        prim=parameter_scopes(call.prim,tuple(bindings_by_param))
                        budget=replace(profile,sram_bytes=capacity//2)
                        def create():
                            working=tvm.IRModule({call.gv:prim},attrs=module.attrs)
                            detail={}
                            recipe=_fit_kernel(working,call.gv,prim,budget,scratch_base,slot_count,detail)
                            return recipe,detail
                        recipe,detail=current().recipe(prim,budget,scratch_base,slot_count,create)
                        scheduled,elems,layout=recipe
                        scratch=tuple(scratch_base+i*elems*4 for i in range(slot_count))
                        start=scratch_base+slot_count*elems*4
                        sram,local_top=_place_sram(layout,start,capacity)
                        for p,address in bindings_by_param.items():
                            if address < local_top or address+call.sizes[p]*2 > profile.sram_nibbles:
                                raise LinkError('retained parameter overlaps workspace or exceeds SRAM')
                            sram[scheduled.buffer_map[scheduled.params[p]]]=address
                        words,was_native,native_detail,stats=probe(scheduled,call.addresses,
                            call.sizes,sram,scratch,constants,one,mode,reuse=_reuse_costs)
                        top=max([local_top]+[a+s for a,s in reservation.values()])
                        cache[signature]=Choice(words,scheduled,layout,detail,scratch,sram,top,start,
                            dict(bindings_by_param),was_native,native_detail,stats,capacity,
                            sum(s for _,s in reservation.values())//2)
                    except (LinkError,V09TirError) as error:
                        diagnostics['rejections'].append(dict(call=index,reason=str(error),
                                                             parameter_sram=dict(bindings_by_param)))
                        cache[signature]=None
                choice=cache[signature]
                if choice is None: continue
                retained={k:v for k,v in reservation.items() if by_key[k].last>index}
                extensions.append((retained,prefix+[choice],dma+choice.statistics['dma_bytes'],
                                   word_count+choice.statistics['words'],max(peak,choice.top)))
        if not extensions:
            raise LinkError(f'no legal SRAM/schedule plan at call {index}')
        # Only merge states with the same remaining values and physical layout.
        best={}
        for state in extensions:
            key=tuple(sorted(state[0].items()))
            if key not in best or state[2:]<best[key][2:]: best[key]=state
        states=sorted(best.values(),key=lambda s:s[2:])[:profile.memory_beam_width]
        # Preserve the nonretaining continuation for capacity fallback.
        if () in best and all(s is not best[()] for s in states): states.append(best[()])
    winner=min(states,key=lambda s:s[2:])
    choices=dict(enumerate(winner[1]))
    # Cost reuse cannot silently change the selected program's contract.
    # Validate deferred candidates before relocating their SRAM arena.
    for index,choice in list(choices.items()):
        if choice.words is not None: continue
        try:
            words,was_native,detail=emit(choice.scheduled,calls[index].addresses,choice.sram,
                choice.scratch,constants,one,mode)
            stats=analyze(eliminate_dead_stores(words),detail=False)
            if stats != choice.statistics:
                raise LinkError('reused candidate cost differs at actual addresses')
        except (LinkError,V09TirError) as error:
            result,diagnostics=plan_program(module,plan,profile,scratch_base,slot_count,
                constants,one,mode,_reuse_costs=False)
            diagnostics['cost_reuse_fallback']=str(error)
            return result,diagnostics
        choices[index]=replace(choice,words=words,native=was_native,native_diagnostics=detail)
    # Search reserves from the top to keep addresses stable while exploring.
    # Pack that arena down after selection, preserving all relative addresses
    # and keeping it above every workspace used while a value is retained.
    occupied=[c for c in choices.values() if c.retained_bytes]
    shift=0
    if occupied:
        lowest=min(c.workspace_limit for c in occupied)
        workspace=max(align(c.workspace_start)+c.layout.peak_nibbles for c in occupied)
        shift=max(0,(lowest-workspace)//8*8)
    if shift:
        for index,choice in list(choices.items()):
            if not choice.retained_bytes: continue
            parameters={p:a-shift for p,a in choice.parameter_sram.items()}
            sram=dict(choice.sram)
            for p,a in parameters.items():
                sram[choice.scheduled.buffer_map[choice.scheduled.params[p]]]=a
            words,was_native,native_detail=emit(choice.scheduled,calls[index].addresses,sram,
                choice.scratch,constants,one,mode) if parameters else (
                    choice.words,choice.native,choice.native_diagnostics)
            choices[index]=replace(choice,words=words,sram=sram,parameter_sram=parameters,
                top=max(align(choice.workspace_start)+choice.layout.peak_nibbles,choice.top-shift),
                workspace_limit=choice.workspace_limit-shift,native=was_native,
                native_diagnostics=native_detail,statistics=analyze(eliminate_dead_stores(words),detail=False))
    for index,choice in choices.items():
        diagnostics['calls'].append(dict(index=index,name=calls[index].gv.name_hint,
            parameter_sram_nibbles=choice.parameter_sram,
            workspace_limit_nibbles=choice.workspace_limit,
            retained_bytes=choice.retained_bytes,
            workspace_peak_bytes=(align(choice.workspace_start)+choice.layout.peak_nibbles)//2,
            sram_peak_bytes=choice.top//2,selected=choice.diagnostics.get('selected'),
            statistics=choice.statistics))
    diagnostics.update(retained_values=[dict(producer=v.key[0],argument=v.key[1],
        address_bytes=v.address,size_bytes=v.size,last_use=v.last,readers=list(v.readers))
        for v in values if v.key[1] in choices[v.key[0]].parameter_sram],
        dma_bytes=sum(c.statistics['dma_bytes'] for c in choices.values()),
        words=sum(c.statistics['words'] for c in choices.values()),
        sram_peak_bytes=max(c.top for c in choices.values())//2,
        arena_relocation_bytes=shift//2)
    return choices,diagnostics
