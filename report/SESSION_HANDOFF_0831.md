# NPU Compiler 작업 인계 문서 — 2026-08-31

`report/SESSION_HANDOFF_0828.md`의 후속. 그때 남겼던 "남은 일"을 전부 진행한 결과다.
branch는 `cmodel-v09`.

---

## 1. 이번에 끝난 것

### A1. 서술자 dead-store 제거 (백로그 최우선 항목)

직선 스트림이라 모든 서술자 레지스터 값이 한 번의 순회로 결정된다. 따라서 "이미 그
값인 레지스터에 다시 쓰는 명령"은 **증명 가능하게** 죽은 명령이다.
`npu_compiler/peephole.py`가 C-model이 실제로 들고 있는 상태만 모델링한다 —
주소 half(0x80), vector length(0x82), rows/cols와 dtype(0x88/0x89),
scale 주소 half(0x8A/0x8B). 그 외에는 아무것도 이 상태를 안 건드린다.
0x15는 주소 half를 바꾸지만 동시에 실행하므로 후보가 아니다.
DMA payload word는 opcode로 오독하지 않고 건너뛴다.

측정 (모두 **비최적화 스트림과 bit-exact** 확인):

| 커널 | 전 | 후 |
|---|---|---|
| `matmul [64,128]x[128,192]` | 206 | 110 (−46.6%) |
| `matmul` padded `[7,128]x[128,96]` | 3,353 | 2,413 (−28.0%) |
| `softmax [8,64,64]` | 29,709 | 20,008 (−32.7%) |
| `rms_norm [4,2560]` | 315 | 242 (−23.2%) |
| `silu [6,9728]` | 343 | 273 (−20.4%) |
| **1층 검증 프로그램** | 29,946 | **19,321 (−35.5%)** |

### S6. target 등록 + build 진입점

`npu_target()`는 진짜 `tvm.target.Target`(`ext_dev -keys=npu -model=v09`)다.
`ext_dev`는 TVM 표준 속성만 받으므로 머신 상수는 `NpuProfile`(SRAM 8 MiB / tile 64 /
lanes 256 / scratch slot 6)에 모았고, **링커와 스케줄이 그걸 읽는다**(전에는 모듈
상수였다). 그래프 절반은 `relax.get_pipeline("npu")`로 등록했다.
`npu_target.build(mod, target)` → `NpuExecutable`(word 스트림 + 메모리 계획,
`.run()` / `.save()`).

**`relax.build`는 일부러 종점이 아니다.** 그 끝은 할당기와 호출 기제를 가진 런타임의
`runtime.Module`인데 이 기계엔 둘 다 없다 — 직선 스트림 하나, 평면 이미지 하나,
주소는 전부 컴파일타임 확정. 미완성이 아니라 구조상 그렇다는 걸 docstring에 적었다.

### S7. Gemma 4 E2B 프론트엔드 (세 번째 family)

Llama/Qwen3와 달리 델타가 크다: 층마다 sliding/full 두 종류(head_dim 256 vs 512,
theta 다름, full은 proportional RoPE), 뒤쪽 20개 층이 앞선 owner 층의 **K/V 공유**,
**per-layer embedding** 주입(gate → projection), norm 5종, **weight 없는 V-norm**,
**scale 1.0** 어텐션, tanh-GELU MLP, 층별 출력 스칼라.

검증은 최종 logits가 아니라 **HF의 층별 hidden state**와 대조했다
(`build/gemma4_layer_reference_hello.npz`):

| 깊이 | cosine | 무엇을 덮는가 |
|---|---|---|
| 1층 | 0.999983 | 층 본체 전부 |
| 16층 | 0.999746 | full-attention 층(4·9·14) + 첫 공유 층(15) |
| 34층 | 0.999868 | 공유 층 20개 전부 |

> 함정: 그 파일의 `hidden_NN`은 **층 NN에 들어가는** 상태다. 하나 어긋나게 비교해서
> 처음엔 cosine 0.018이 나왔다. 테스트 코드에 그 사실을 적어놨다.

codegen에 `tanh`가 없어서 gelu_tanh가 lower되지 않았다. `1 - 2/(exp(2x)+1)`로 전개했다
(`(e^2x-1)/(e^2x+1)`가 아니라): 지수가 FP16을 넘치면 몫이 0이 되어 결과가 정확히 1이
되는 반면, 차분 형태는 무한대를 무한대로 나눈다. 큰 음수도 따로 처리할 필요가 없다.
±40 범위에서 max|diff| 0.0007, 전부 유한.

러너는 family에 무관해졌다 — 각 프론트엔드가 `model_config` / `runtime_inputs` /
`load_params`를 내놓고, `run_nn_npu.py --model {llama,qwen3,gemma}`가 family별
golden token과 참조 logits를 안다.

### S8. 양자화를 Relax pass로 — **그래프 절반만**

`QuantizeWeightsW8A16`이 파라미터 weight의 matmul을 `scale → quantize → qmatmul`로
재작성한다. **`LiftTransformParams` 앞에** 두는 게 요점이다: scale과 packed weight는
파라미터만의 함수라 표준 pass가 알아서 호스트 쪽으로 hoist한다 → 토큰당 추가 비용 0.
출력 채널별 granularity인 이유는 scale이 **합산 축에서 상수여야** 하기 때문이다
(그래야 내적에서 빠져나와 부분합이 누산기에 들어갈 때 곱해질 수 있다).

llvm 측정: float32 기준 **cosine 0.999998**(비양자화 0.999999), argmax 일치,
lifted param 11→16(int8·float32 포함), numpy mirror와 max|diff| 0.0049.

---

## 2. 남은 일 — S8의 NPU codegen 절반

그래프 pass는 동작하지만 **NPU에서는 아직 못 돈다.** 막는 지점은 정확히 셋이고
백로그 §F에 적어놨다:

1. **스케줄** — `npu_qmatmul`은 리덕션 뒤에 dequant(`* scale[n]`) 블록이 하나 더
   붙는다. `schedule_matmul_sram`이 이를 못 받아
   `BlockNode write buffers do not match`로 실패한다.
   unpad와 같은 방식으로 `reverse_compute_at` 하면 될 가능성이 높다
2. **혼재 폭 SRAM** — `_flat`과 staging이 **원소당 4 nibble**을 가정한다.
   INT8은 2, FP32(scale)는 8이 필요하다. 버퍼별 폭을 dtype에서 받아 주소 계산,
   `_sram_layout` 크기, DMA의 원소↔셀 환산에 반영해야 한다
   (`npu_memplan`도 원소당 2바이트를 가정한다)
3. **서술자** — 가중치 피연산자에 `dtype=INT8`, 열 타일마다 `wscale(...)`.
   oracle(`backend_v09`)에 검증된 구현이 있으니 옮기면 된다

> 대안 하나: scale을 FP32 대신 **FP16**으로 두면 dequant가 평범한 broadcast 곱이 되어
> (3)이 거의 사라진다. 정밀도 손실은 INT8 반올림 오차에 묻힌다. 다만 (2)는
> 어느 경로로 가든 남는다.

그 밖에 백로그 A2(커널 경계를 넘는 DMA 병합), B1(weight 재적재), C2(상수 take → slice)
등은 그대로다.

---

## 3. 하지 말아야 할 것 (측정으로 확인한 것들)

- **융합 켜지 말 것**: 링크·실행은 되지만 결과가 틀리고(cosine 0.1749) word도 안 준다
- **llvm을 정확도 최종 기준으로 쓰지 말 것**: TVM의 float16 matmul은 float16으로
  누적한다. 우리는 FP32 누적이라 **우리가 더 정확하다**. 어긋나면 float32 numpy로 판정
- **타일 규모 테스트만 믿지 말 것**: 지난번 Qwen3 버그(고정 8192 슬롯)는 모든 테스트
  shape가 8192보다 작아서 원리적으로 못 잡혔다. 실차원 테스트가 그래서 있다
