# TVM API를 동작 기준으로 읽는 사전

[목차](../README.md). 기준은 설치된 TVM 0.19.0이다. API 이름을 외우기보다
“어떤 표현을 받아 무엇으로 바꾸며, 아직 무엇을 결정하지 않는가”를 구분한다.
이 문서는 현재 compiler가 쓰는 기능을 설명하며 최신 TVM 전체 기능 목록은 아니다.

## IR을 만드는 API

| API/표현 | 입력 → 결과 | 작은 예 |
|---|---|---|
| `relax.frontend.nn.Module.export_tvm` | 모델과 입력 spec → Relax 모듈, parameter 명세 | Llama의 정적 S/C 함수 작성 |
| `relax.BlockBuilder` | tensor 호출과 함수 범위 → type/shape가 연결된 graph | matmul 뒤 add 결과를 변수로 binding |
| `TensorStructInfo` | shape/dtype → 값의 구조 정보 | 실제 x 배열 없이 [64,128] FP16임을 앎 |
| `te.placeholder` | shape/dtype → 수식의 입력 배열 | A[M,K] |
| `te.compute` | output shape와 원소 수식 → tensor 계산 | B[i,j]=A[i,j]+1 |
| `te.reduce_axis` | reduction 범위 → 합산 축 | k=0..K-1 |
| `BlockBuilder.call_te` | TE 함수와 Relax 인자 → TIR PrimFunc와 call_tir | custom matmul legalizer |
| `relax.call_tir` | PrimFunc 이름/인자/output 정보 → graph 호출 | 호출 순서를 Relax에 유지 |
| `tvm.script` | IR의 읽기 쉬운 Python 형식 | `.script()` dump는 배열 출력값이 아니라 계산 프로그램 |

PrimFunc의 block은 읽기·쓰기 영역과 iteration 관계를 가진 계산 단위다. Block 이름이
같다는 이유로 같은 연산이라 판단하지 않는다. 현재 kernel 분석도 접근/수식/계약을 본다.

## Graph pass

| API | 하는 일 | 현재 사용에서의 경계 |
|---|---|---|
| `tvm.transform.Sequential` | 여러 pass를 순서대로 호출 | 최적화 후보를 자동 탐색하는 기능 자체는 아님 |
| `CanonicalizeBindings` | binding/alias 표현을 정돈 | 물리 메모리 주소를 정하지 않음 |
| `EliminateCommonSubexpr` | 같은 순수 계산을 공유 | 다른 weight의 같은 shape까지 동일시하지 않음 |
| `RewriteDataflowReshape` | 지원 reshape를 view로 다루기 쉬운 표현으로 정리 | transpose를 항상 복사 없이 처리하지 않음 |
| `LiftTransformParams` | parameter-only 계산을 별도 함수로 추출 | 추출된 함수를 실제 실행하는 것은 host 준비 단계 |
| `LegalizeOps` | Relax op를 TE/TIR 구현 호출로 낮춤 | 전체 Relax 함수가 사라지지 않음 |
| `FoldConstant` | constant만으로 계산 가능한 결과를 미리 평가 | runtime 입력의 계산은 남음. NPU는 부분합 순서 wrapper 사용 |
| `AnnotateTIROpPattern` | TIR 계산 성격을 fusion용 attribute로 분류 | NPU SRAM 적합성이나 속도 판정 아님 |
| `FuseOps` | 의존 그래프와 op pattern으로 Relax 영역 구성 | 실제 TIR 결합은 별도 |
| `FuseOpsByPattern` | 명시된 graph pattern과 지원 조건으로 그룹 구성 | pattern이 있다는 이유로 항상 최종 선택되는 것은 아님 |
| `FuseTIR` | 묶인 TIR 호출들을 한 PrimFunc로 결합 | SRAM 배치와 vector/matrix 명령 선택은 후속 |
| `DeadCodeElimination` | 도달/사용되지 않는 함수와 값 정리 | 읽는 원소 수를 줄이는 scheduling과 다름 |

`CanonicalizeAttention`, `FoldRepeatedGemv`, 양자화 rewrite, `NpuOutputViews`는
우리 모듈이 제공하는 변환이다. 표준 pass 안의 확장 지점 또는 TVM IR 변환 API로
구현했지만 TVM 기본 제공 pass인 것처럼 부르지 않는다.

## Schedule API: 선언한 계산을 실행 순서로 배치하기

`tir.Schedule`은 TIR를 받아 block/loop에 대한 변환을 적용한다. `get_block`과
`get_loops`는 변환할 대상을 참조하는 handle을 돌려준다. 변환 후에는 실제 loop
구조가 달라지므로 이전 handle이나 인덱스를 무조건 그대로 쓰지 않는다.

```python
sch = tir.Schedule(module)
c = sch.get_block("C")
i, j = sch.get_loops(c)
io, ii = sch.split(i, factors=[None, 2])
sch.compute_at(sch.get_block("B"), io)
```

이는 C를 2행씩 계산하면서 필요한 B도 그 안에서 만드는 변환이다. `.trace`에는
적용한 API, `.mod.script()`에는 바뀐 loop가 나온다. 결과 TIR에 `compute_at`이라는
runtime 명령이 남지 않는다. 실제 [trace](../examples/schedule_api/02_compute_at.trace.txt)를 참조한다.

| API | 구체적 역할 | 적용할 때 확인하는 것 |
|---|---|---|
| `split` | loop를 여러 loop로 나눔 | 나머지와 predicate, factor |
| `fuse` | 연속 loop를 한 loop로 평탄화 | 독립 원소인지, index/stride가 유지되는지 |
| `reorder` | loop 순서를 바꿈 | 의존성, reduction, 단일 accumulator |
| `cache_read` | 별도 scope의 읽기 cache와 복사 stage 생성 | 어느 소비자가 공유할지 |
| `cache_write` | 계산의 쓰기를 cache로 옮기고 복사 stage 생성 | 외부 출력은 여전히 필요함 |
| `compute_at` | 생산 block을 소비 loop 위치 안으로 이동 | 모든 관련 소비가 허용 범위에 있는지 |
| `reverse_compute_at` | 소비 block을 생산 loop 위치 안으로 이동 | 생산된 영역으로 필요한 소비를 수행할 수 있는지 |
| `set_scope` | 기존 중간 buffer의 storage scope 변경 | 새 copy와는 다름 |
| `pad_einsum` | einsum 축을 tile 배수로 확장하는 계산과 경계 구성 | 원래 결과와 zero padding의 의미 |
| `decompose_reduction` | reduction 초기화와 update 분리 | 어느 loop에서 accumulator를 새로 시작하는지 |
| `TensorIntrin.register` | 수식 descriptor와 장치 구현 대응 등록 | shape/dtype/scope/수식 계약 |
| `tensorize` | 일치하는 loop 영역을 intrinsic으로 치환 | 틀린 수식을 이름만으로 치환하지 않음 |
| `rfactor` | reduction을 부분합과 결합으로 분해 | 수치 순서. 현재 CPU reference의 K=64 grouping |

Compute_at이 주소 배치를 자동으로 최적화하는 것은 아니다. Buffer가 어디서 얼마만큼
쓰이는지 드러낸 뒤 compaction·USMP·NPU 보관 계획이 그 정보를 사용한다.

## TIR 후처리와 메모리 pass

| API | 변화 | 현재 목적 |
|---|---|---|
| `LowerInitBlock` | init block을 낮은 수준 조건/저장으로 표현 | codegen이 초기화와 reduction을 처리 |
| `PlanAndUpdateBufferAllocationLocation` | 사용 위치에 맞춰 allocation 위치 정리 | 좁아진 cache 영역이 드러나게 함 |
| `ConvertBlocksToOpaque` | block iter binding을 낮추고 index에 반영 | buffer compaction/codegen이 읽기 쉬운 형태 |
| `CompactBufferAllocation` | 실제 사용 범위에 맞춰 allocation 축소 | 전체 tensor 크기의 cache를 작은 stripe로 줄임 |
| `LowerMatchBuffer` | 부분 buffer view의 alias/index를 낮춤 | 분석용 USMP TIR 또는 지원 경로 |
| `ToNonDataflow` | DataflowVar를 일반 실행 형태로 변환 | 메모리 planning 입력 준비 |
| `RemovePurityChecking` | 실행용 표현에 남은 purity 제약 정리 | 후속 allocation/call rewrite 허용 |
| `CallTIRRewrite` | 결과 allocation과 TIR 호출을 명시 | output buffer를 가진 호출 형태 |
| `StaticPlanBlockMemory` | 수명에 따른 storage pool 재사용 | DRAM flat 주소의 기본 pool 구조 |
| USMP `extract_buffer_info` | allocation과 접근에서 충돌 정보 추출 | 커널 내부 SRAM 수명 분석 |
| USMP greedy 배치 | 충돌하지 않는 buffer의 offset 공유 | 가능한 경우 기본 bump보다 SRAM 절약 |

실행용 TIR와 분석용 복사본을 구분한다. USMP 분석을 위해 낮춘 IR를 반드시 그대로
실행 codegen에 쓰는 것은 아니다. Buffer identity로 분석 결과를 연결한다.

## 검증과 재사용 API

`tvm.ir.structural_hash/equal`은 IR 구조가 같은지 판정하는 도구다. Hash만 같다고
동일성을 확정하지 않고 equality도 검사한다. Cache key에는 실행 설정과 메모리 조건도
포함된다. `relax.build(...,"llvm")`와 `VirtualMachine`은 host transform과 CPU reference에
사용한다. NPU의 최종 program을 LLVM VM이 대신 실행하는 구조는 아니다.

## 선택형 MetaSchedule API

| API | 제공하는 기능 | 현재 NPU에서의 사용 범위 |
|---|---|---|
| `Workload` | 비교할 TIR module을 하나의 작업으로 표현 | 실행 계약이 같은 커널의 trace 조회 단위 |
| `TuningRecord` | 작업, Schedule trace, target와 측정 정보를 연결 | 이미 선택한 trace를 저장; `run_secs=None` |
| `JSONDatabase` | Workload와 TuningRecord를 파일에 보존 | build 사이 재사용. NPU adapter가 최신 개선 기록을 찾아 검증 |
| `tir.Schedule.trace` | 적용한 Schedule 명령과 선택값 기록 | 새 TIR에 재적용한 뒤 lowering·SRAM·codegen 검증 |
| `TuneContext` | 작업, target, 후보 생성기와 탐색 전략을 묶음 | 선택형 NPU matmul 탐색의 실행 문맥 |
| `ScheduleFn` | 함수 호출로 후보 Schedule 생성 | 기존 NPU schedule 규칙을 그대로 호출 |
| `sample_categorical` | 범주형 후보 중 하나를 확률적으로 선택하고 기록 | 정리된 후보 목록의 설정 번호를 같은 확률로 추첨 |
| `ReplayFunc` | 후보 생성 함수를 다시 실행하여 다른 선택 시도 | 같은 NPU 후보 공간의 검사 순서를 바꾸는 비교 경로 |
| `relax_integration.extract_tasks` | 같은 TIR 작업을 합치고 호출 횟수 집계 | 작업 목록·반복 횟수 진단; TaskScheduler로 예산을 분배하지는 않음 |

DB의 `reuse`/`tune`은 NPU adapter가 정한 사용 모드다. `reuse`는 기록을 검증해 적용하고,
`tune`은 그 기록을 기준으로 추가 탐색한다. DMA bytes를 초 단위 실행시간처럼 저장하지
않으므로 TVM 기본 latency 순위 조회로 NPU 승자를 고르지는 않는다. 기본 실행은
`space`이며 이 두 기능은 선택형이다. 전체 256-token prefill 비교에서는 이전 DB로
탐색을 건너뛰지 않도록 DB를 연결하지 않는다.

## 쓰지 않았다고 바로 문제가 되는 것은 아닌 API

DLight는 schedule 규칙을 적용하는 체계이고 MetaSchedule은 탐색·측정 framework다.
기본 경로는 표준 Schedule API 위의 NPU 후보 생성·비용 검사를 사용하며 DLight는
호출하지 않는다. 09-29에는 MetaSchedule의 `JSONDatabase`/`TuningRecord`를 이용한
선택형 trace 재사용과 `TuneContext`/`ScheduleFn`/`ReplayFunc`를 이용한 선택형 탐색을
추가했다. `sample_categorical`로 NPU 설정을 뽑고 실제 SRAM·명령을 공통 평가기로
검사한다. 이는 EvolutionarySearch나 학습형 CostModel을 적용한 구성은 아니다.
API 역할과 실제 결과는 [6장](../current/06_matmul_schedule.md)과
[11장 E8](../current/11_experiments.md#e8-metaschedule로-탐색하면-llama-커널이-더-좋아지는가)에 있다.
이 API들이 자동으로 하드웨어 계약·SRAM 보관·좋은 비용 모델을 해결하지는 않는다.
CombineParallelMatmul 같은 추가 graph pass도 지원되는 형태와 concat/split 비용을
측정한 뒤 판단해야 한다. 현재 호출 여부와 일반적인 TVM 기능의 존재를 구분한다.

이 설명은 저장소 구현과 설치된 TVM source의 `python/tvm/relax/transform`,
`src/relax/transform/fuse_ops.cc`, `python/tvm/tir/schedule` 및 USMP 구현을 대조했다.
