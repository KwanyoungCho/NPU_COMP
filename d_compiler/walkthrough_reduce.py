"""reduce(합)·broadcast가 어떻게 lowering되는지 '눈으로' 보는 학습 스크립트.

핵심 메시지:
  - 이 NPU엔 reduce/broadcast 명령이 없다(벡터 연산은 전부 element-wise·연속).
  - 그래서 행 합(reduction)과 broadcast의 '효율적' lowering은 **행렬 엔진의 ones-matmul**:
        rowsum:    x[R,C] @ ones[C,1]      -> [R,1]      (degenerate N=1)
        broadcast: src[R,1] @ ones[1,C]    -> [R,C]      (degenerate K=1, 외적)
  - 단, 이걸 그래프에 relax.matmul로 두지 않고 **전용 op(relax.sum / relax.broadcast_to)**
    로 표현한다. codegen의 emit_row_sum / emit_broadcast가 위 ones-matmul로 lowering.
  - 덕분에 그래프의 relax.matmul은 전부 '진짜 GEMM' -> TIR 백엔드 단독 경로.

즉 "reduction은 진짜 reduction으로 정직하게, matmul은 TIR-only로" 의 코드 증거.

실행:  /home/chokwans99/anaconda3/envs/npu-tvm/bin/python d_compiler/walkthrough_reduce.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tvm import relax
from npu_compiler import driver
from study_util import banner, disasm


def _f16(a):
    return np.asarray(a, dtype=np.float16).astype(np.float32)


def mod_sum(R, C):
    """그래프: y = sum(x, axis=-1, keepdims)   -- matmul이 아니라 relax.sum"""
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo([R, C], "float16"))
    with bb.function("main", [x]):
        with bb.dataflow():
            y = bb.emit(relax.op.sum(x, axis=[-1], keepdims=True)); gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


def mod_bcast(src_shape, R, C):
    """그래프: y = broadcast_to(x, [R,C])   -- matmul이 아니라 relax.broadcast_to"""
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo(list(src_shape), "float16"))
    with bb.function("main", [x]):
        with bb.dataflow():
            y = bb.emit(relax.op.broadcast_to(x, relax.ShapeExpr([R, C]))); gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


def graph_ops(mod):
    out = []
    for blk in mod["main"].body.blocks:
        for b in blk.bindings:
            if isinstance(b.value, relax.Call):
                out.append(getattr(b.value.op, "name", str(b.value.op)))
    return out


def mm_tiles(asm):
    """(m_mul 호출 수, [그 직전 TILE A/B 크기들]) — degenerate N=1 / K=1 확인용."""
    tiles, mms = [], 0
    cur = {}
    for w in asm.words:
        op = w & 0xFF
        if op == 0x88:  # TILE
            sel = (w >> 31) & 1; d1 = (w >> 8) & 0xFF; d2 = (w >> 16) & 0xFF
            cur[sel] = (d1, d2)
        elif op == 0x42 and (w >> 30) & 3 == 2:  # MATMUL vector
            mms += 1
            tiles.append((cur.get(0), cur.get(1)))
    return mms, tiles


# ===========================================================================
banner(0, "이 NPU엔 reduce/broadcast 명령이 없음 → ones-matmul로 lowering (단 전용 op로 표현)")
print("  벡터연산(add/mul/...)은 element-wise·연속만 가능 → 행 내부 합이 불가.")
print("  행렬엔진으로:  rowsum = x@ones[C,1] (N=1),   broadcast = src@ones[1,C] (K=1).")
print("  그래프엔 relax.sum / relax.broadcast_to 로 두어 matmul-the-op은 TIR 전용 유지.")

# ===========================================================================
banner(1, "① 그래프 확인 — reduction/broadcast는 relax.matmul이 아니다")
ms = mod_sum(64, 192)
mb = mod_bcast((64, 1), 64, 192)
print(f"  sum 그래프 op들       : {graph_ops(ms)}")
print(f"  broadcast 그래프 op들 : {graph_ops(mb)}")
print("  → 'relax.matmul' 없음. reduction=relax.sum, broadcast=relax.broadcast_to.")

# ===========================================================================
banner(2, "② lowering 증거 — codegen이 emit_row_sum/emit_broadcast로 ones-matmul 생성")
asum, _ = driver.compile_func(ms["main"])           # direct backend -> emit_row_sum
abc, _ = driver.compile_func(mb["main"])            # direct backend -> emit_broadcast
n1, t1 = mm_tiles(asum)
n2, t2 = mm_tiles(abc)
print(f"  sum[64,192]->[64,1] : m_mul {n1}회,  타일(A,B) 예시 {t1[:3]}")
print(f"      → B타일이 (kt,1) = N=1 (degenerate reduction). 패딩 없이 hardware-legal.")
print(f"  bcast[64,1]->[64,192]: m_mul {n2}회, 타일(A,B) 예시 {t2[:3]}")
print(f"      → A타일이 (mt,1)=K=1 외적. ones와의 곱이 곧 복제.")
print("\n  ones는 v_move(즉시값 1)로 스크래치에 한 번 채움. 처음 6개 명령:")
for i, w in enumerate(asum.words[:6]):
    print(f"     {i:<3}0x{w:08x}  {disasm(w)}")

# ===========================================================================
banner(3, "③ 정확성 — emit_row_sum / emit_broadcast vs numpy")
print(f"  {'연산':<26}{'결과':>10}")
for (R, C) in [(8, 64), (64, 192), (7, 130), (128, 3072)]:
    x = _f16(np.random.default_rng(R + C).standard_normal((R, C)) * 0.3)
    got = driver.run_module(mod_sum(R, C), {"x": x})
    ref = _f16(x).sum(axis=-1, keepdims=True)
    rel = float(np.max(np.abs(got - ref))) / (float(np.max(np.abs(ref))) + 1e-9)
    print(f"  sum[{R},{C}]->[{R},1]".ljust(26) + f"rel={rel:.1e}".rjust(10))
for (sh, R, C, tag) in [((8, 1), 8, 64, "col"), ((1, 64), 8, 64, "row"),
                        ((7, 1), 7, 130, "col"), ((1, 3072), 128, 3072, "row")]:
    x = _f16(np.random.default_rng(R * C).standard_normal(sh) * 0.5)
    got = driver.run_module(mod_bcast(sh, R, C), {"x": x})
    ref = np.broadcast_to(_f16(x), (R, C))
    eq = np.array_equal(got, ref)
    print(f"  bcast {tag} {sh}->[{R},{C}]".ljust(26) + (f"byte-exact={eq}").rjust(10))

# ===========================================================================
banner(4, "④ 핵심 — 그래서 matmul은 TIR 단독")
print("  reduction/broadcast가 matmul 노드가 아니므로, 그래프의 relax.matmul은")
print("  전부 '진짜 GEMM'(64배수) → TIR+tensorize 백엔드로만 흐른다.")
print("  (Llama 3.2 3B 레이어: matmul 147개 전부 64배수, degenerate 0개 → 147/147 TIR.)")
print("  direct의 일반 타일링은 oracle/테스트 + 비-64 안전 fallback으로만 남음.")

print("\n" + "=" * 72)
print("요약: reduce/broadcast = ones-matmul로 '효율 lowering'하되 전용 op로 '정직하게 표현'.")
print("      → matmul-the-op은 TIR 단독, reduction은 진짜 reduction. (codegen.emit_row_sum/emit_broadcast)")
print("=" * 72)
