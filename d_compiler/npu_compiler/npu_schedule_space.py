"""Parameterized matmul space: blocking, loop order and operand placement.

The 64x64 instruction and an uninterrupted K accumulator chain are constraints,
not mapping templates. Each operand independently chooses an enclosing loop.
Estimates order trials; scheduled TIR and actual emission decide feasibility.
"""
from itertools import product
from math import prod


def factors(extent, tile):
    count = extent // tile
    small, large = [], []
    i = 1
    while i * i <= count:
        if count % i == 0:
            small.append(i * tile)
            if i * i != count:
                large.append(count // i * tile)
        i += 1
    return small + large[::-1]


def loop_spec(extents, config, tile=64):
    m, n, k = ((v + tile - 1) // tile * tile for v in extents)
    mp, np, kp = (config[x] for x in ('m_panel', 'n_panel', 'k_chunk'))
    sizes = dict(m_panel=m//mp, n_panel=n//np, m_tile=mp//tile,
                 n_tile=np//tile, k_group=k//kp, k_tile=kp//tile)
    outer = [axis + '_panel' for axis in config['outer_order']]
    inner = [axis + '_tile' for axis in config['inner_order']]
    return [(name, sizes[name]) for name in outer + inner + ['k_group', 'k_tile']]


def _iteration_signature(loops):
    # Splitting adjacent loops of the same axis does not change the order of
    # 64-wide tiles. Cache placement prefixes below still distinguish a copy
    # at a panel/group boundary from a copy per individual tile.
    result = []
    for name, size in loops:
        if size == 1:
            continue
        axis = name.split('_')[0]
        if result and result[-1][0] == axis:
            result[-1] = (axis, result[-1][1] * size)
        else:
            result.append((axis, size))
    return tuple(result)


def prioritize(pool, available_bytes, *, prove_operand_bound=False):
    """Prune only proven simultaneous operand storage; otherwise rank hints.

    The caller certifies ordinary, aligned rank-2 FP16 parameter operands.
    Both caches are live at a matrix operation, so their summed storage is a
    lower bound. Output/scratch/padding estimates are never a rejection proof.
    Unknown candidate metadata (including custom schedules) remains eligible.
    """
    eligible = [c for c in pool if not (prove_operand_bound and
                sum(c.get('operand_cache_bytes', ())) > available_bytes)]
    ordered = sorted(eligible, key=lambda c: c.get('footprint_hint', 0) > available_bytes)
    return ordered, dict(capacity_pruned=len(pool)-len(eligible),
        eligible_candidates=len(eligible),
        capacity_deprioritized=sum(c.get('footprint_hint', 0) > available_bytes for c in eligible))


def candidates(extents, tile=64, resident=()):
    """Enumerate equivalent-normalized schedules, with explicit search axes.

    A resident operand needs no cache stage: its placement choices collapse.
    Tile factors divide the padded axes; ragged data retains the padding proof.
    """
    m, n, k = ((v + tile - 1) // tile * tile for v in extents)
    rows, cols, depth = extents
    seen, result = set(), []
    for mp, np, kp, outer, inner in product(
            factors(m,tile), factors(n,tile), factors(k,tile), ('mn','nm'), ('mn','nm')):
        config = dict(m_panel=mp, n_panel=np, k_chunk=kp,
                      outer_order=outer, inner_order=inner)
        loops = loop_spec(extents, config, tile)
        # Unit loops are still legal cache locations, but equivalent copies
        # need not be generated once for every spelling of a unit loop.
        locations = []
        signatures = set()
        for index, (name, _) in enumerate(loops):
            signature = _iteration_signature(loops[:index+1])
            if signature not in signatures:
                signatures.add(signature)
                locations.append(index)
        choices = [locations if operand not in resident else [locations[-1]] for operand in (0,1)]
        for a_at, b_at in product(*choices):
            semantic = (_iteration_signature(loops),
                        tuple(_iteration_signature(loops[:at+1])
                              if operand not in resident else ()
                              for operand,at in enumerate((a_at,b_at))))
            if semantic in seen:
                continue
            seen.add(semantic)
            footprints, traffic = [], 0
            for operand, at in enumerate((a_at,b_at)):
                axes = ('m','k') if operand == 0 else ('n','k')
                footprint = tile*tile*prod(size for key,size in loops[at+1:] if key[0] in axes)
                repeats = prod(size for key,size in loops[:at+1] if key[0] not in axes)
                footprints.append(0 if operand in resident else footprint)
                traffic += 0 if operand in resident else 2*depth*(rows if operand==0 else cols)*repeats
            spec = dict(config, a_at=loops[a_at][0], b_at=loops[b_at][0])
            result.append(dict(mapping=spec, tile_pad=(m,n,k)!=extents,
                               estimated_operand_load_bytes=traffic,
                               operand_cache_bytes=tuple(2*v for v in footprints),
                               footprint_hint=2*(sum(footprints)+tile*tile)))
    return sorted(result, key=lambda c:(c['estimated_operand_load_bytes'], c['footprint_hint'],
                                        tuple(c['mapping'].values())))
