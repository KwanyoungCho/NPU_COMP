# Llama3 레이어 코드젠 (NPU c-model)

재구현 시뮬레이터(`../_poc/mysim`)를 타깃으로 Llama3 transformer 레이어를 NPU 명령어로
생성·검증하는 코드젠. **타일링 없음 / one-shot** 가정, 축소 차원에서 float 참조 대비 검증.

## 파일
| 파일 | 내용 | 검증 결과 |
|------|------|-----------|
| `npu.py` | 어셈블러(명령 인코딩) + G-buffer 메모리맵 + matmul 매크로 + mysim 실행 | — |
| `test_matmul.py` | one-shot matmul (8×64 @ 64×64) | byte-exact (mismatch 0) |
| `test_rmsnorm.py` | RMSNorm (합=ones-matmul, broadcast=ones-matmul) | PASS (maxerr 2e-3) |
| `test_attention.py` | single-head causal attention | byte-exact (mismatch 0) |
| `test_swiglu.py` | SwiGLU FFN (SiLU=z/(1+exp(-z))) | PASS (maxerr 0.016) |
| `llama_layer.py` | **전체 레이어** (RMSNorm→GQA+RoPE+causal→res→RMSNorm→SwiGLU→res) | **PASS (rel 0.12%)** |

## 실행
```bash
# (최초 1회) 시뮬레이터 빌드
g++ -O2 -std=c++17 ../_poc/mysim.cpp -o mysim
python3 llama_layer.py      # 전체 레이어 생성 + 검증
python3 test_matmul.py      # 개별 커널 검증
```

## 설정 (reduced, one-shot)
`SEQ=8, D=64, H=4 (q-heads), KV=2 (GQA), HD=16, F=128` — 모든 matmul 차원 ≤255라
한 명령에 인코딩 가능. 풀 Llama3(4096 등)는 동일 구조에 **타일링**(64×64 / 256 lane)만 추가하면 됨.

## 핵심 매핑 기법 (ISA에 전용 op가 없어 우회)
- **reduction(합)**: `v⊙v @ ones` → 합. RMSNorm 제곱합, softmax 분모.
- **broadcast(스칼라→벡터)**: `ones[n×1] @ scalar[1×1]` → n개 복제. per-token 정규화/softmax 나눗셈.
- **전치(Q·Kᵀ)**: 데이터 미리 전치 배치 — `Kᵀ = Wkᵀ @ Xnᵀ`로 런타임 전치 없이 직접 생성
  (Xnᵀ는 `transpose()` 매크로로 elem-copy; per-head 분리 matmul로 strided 접근 회피).
- **SiLU**: HW activation은 `x²·sigmoid(x)`라 SiLU 아님 → `z/(1+exp(-z))`를 exp/div로 조합. negate는 zeros 벡터와 sub.
- **멀티헤드 출력**: head별 `O_h@Wo_h`를 누산(add) → concat 불필요(strided 회피).
- **causal mask**: `-30000` 마스크 행렬을 score에 add.

## 알려진 제약 / TODO
- **softmax FP16 안정성**: max-subtraction 없음(ISA에 max-reduction op 없음). score가 ~11 초과 시
  `exp`가 FP16 한계(65504) 초과 → inf/nan. 현재는 작은 가중치(σ~0.2, 실제 Llama 수준)로 회피.
  완전 안정화엔 max-reduce 프리미티브 필요(또는 mysim에 reduce 명령 추가).
- **eps**: immediate가 정수라 RMSNorm eps≈1e-5를 인코딩 불가 → 현재 eps=0. 필요시 데이터 상수로 추가.
- **타일링 없음**: 차원 ≤255 가정. 풀 차원은 64×64 matmul 타일링 + K-누산, 256-lane 벡터 타일링 추가 필요.
- **전치 elem-copy**: `transpose()`가 vlen=1 복사 unroll이라 명령어 많음(축소 차원이라 OK, 풀 차원선 블록 전치 권장).
