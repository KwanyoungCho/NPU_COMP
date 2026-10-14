# NPU compiler 발표자 노트 (상세판)

슬라이드별 설명 순서, 세부 역할, 해석 시 주의점과 근거 경로. 본문은 약 90분 분량이며 파트를 선택해 줄일 수 있다.

## 01. 표지

현재 메모리 계층(DRAM/SRAM/연산장치)을 사용하는 v09 FP16 backend를 설명합니다. 모델의 계산이 어떻게 TVM Relax → TIR → scheduled TIR → ISA로 내려가고, runtime이 그 프로그램으로 실제 token을 생성하는지 Llama 3.2 3B와 작은 예제 두 개로 끝까지 따라갑니다. 과거 측정과 09-29 탐색 개선 결과를 구분합니다. 별도로 실행 중인 E10은 28-layer·256-token prefill이며 완료 전 비용이나 출력 통과를 주장하지 않습니다.

## 02. 발표 구성: 계산 → 배치 → 실행, 그리고 검증

_개요_

각 파트는 "입력 → 해결할 문제 → 실제 동작 → 결과" 순서로, 같은 예제를 단계마다 다시 봅니다.

발표는 10개 파트입니다. 앞부분(1~4)은 계산의 의미를 정하는 단계, 중간(5~8)은 배치와 명령 생성, 뒷부분(9~10)은 runtime과 실험·한계입니다. 세 예제를 반복해서 사용하므로 청중은 한 예제가 단계마다 어떻게 변하는지 추적할 수 있습니다.

근거: docs/README.md

## 03. 대상 장치: 세 층의 메모리와 고정된 수치 계약

_1 · 전체 구조_

데이터가 실제로 어떻게 움직이는가: DRAM의 weight를 panel 단위로 SRAM에 올리고(GLOAD), SRAM의 64×64 tile을 연산장치에 넣고(LOAD), 결과를 SRAM에 내린 뒤(SAVE) DRAM으로 보냅니다(GSTORE).

DRAM(global)은 DMA만 접근하며 32-bit cell 단위 주소를 씁니다. SRAM은 8 MiB, nibble 주소입니다. 연산장치는 64×64 matrix unit(FP32 accumulator 1개)과 256-lane vector unit입니다. GLOAD/GSTORE는 DRAM↔SRAM, LOAD/SAVE는 SRAM↔연산장치 이동이라 서로 다른 경계입니다. 그림의 SRAM 구성(literal·scratch, 입력 panel, 출력 tile, 커널 사이 보관값)은 8장의 실제 배치 순서입니다. 논리 VLEN 65535와 물리 256 lanes의 관계는 RTL 미확인 항목으로 문서가 명시합니다.

근거: docs/current/01_overview.md, reference/execution_contract.md, npu_compiler/device.py

## 04. Llama 3.2 3B: 발표 내내 사용할 shape와 크기

_1 · 전체 구조_

Decode는 S=1, prefill은 prompt 길이 S. Weight는 checkpoint의 [N,K] 배치이며 계산에는 [K,N]이 필요합니다.

Llama 한 layer의 weight는 FP16으로 192 MiB, 28 layers면 5.25 GiB, LM head가 751.5 MiB입니다. 전체 검증 기록의 DRAM image 약 6.43 GB와 일치하는 규모입니다. 8 MiB SRAM에는 어떤 projection weight도 통째로 들어가지 않으므로 panel 단위 streaming이 필수라는 점을 여기서 미리 보여줍니다.

근거: docs/examples/llama_ir/summary.json, docs/current/01_overview.md

## 05. Compile, load(prepare), run은 서로 다른 시점의 작업

_1 · 전체 구조_

Compile은 shape/dtype만 보고 프로그램을 확정하고, load는 weight를 변환·배치하며, run은 입력만 갱신해 같은 명령을 재실행합니다.

npu_target.build는 relax.build와 달리 NpuExecutable(words + plan + lowered module)을 돌려줍니다. 장치에 runtime allocator가 없어 모든 주소가 compile 시점에 정해지기 때문입니다. Weight가 함수 parameter로 남아 있으면 값만 바꿔 같은 프로그램을 재사용할 수 있지만, constant folding된 weight는 그렇지 않습니다. 정적 shape가 달라지면 다른 프로그램이 필요합니다.

근거: docs/current/01_overview.md, npu_compiler/npu_target.py, npu_generate.py

## 06. 같은 계산을 네 수준에서 읽는다: Relax → TIR → scheduled TIR → ISA

_1 · 전체 구조_

작은 예제 P = X[64,128] @ B[128,64]의 matmul이 각 수준에서 어떻게 보이는지. 아래 코드는 모두 실제 dump에서 발췌했습니다.

Relax에는 호출 관계(무엇을 계산하는가)만 있습니다. LegalizeOps 뒤 TIR에는 어떤 원소를 읽어 더하는지, FP32로 누산해 FP16으로 cast하는 수치 의미가 있습니다. Schedule은 같은 계산을 64×64 조각과 SRAM 복사 위치로 재배치하고, codegen은 descriptor 설정·LOAD·M_MUL·SAVE·DMA 명령으로 펼칩니다. LegalizeOps 뒤에도 Relax 함수가 남아 호출 순서를 유지한다는 점이 중요합니다.

근거: docs/examples/pipeline/01~07, two_matmuls_recorded/program.json

## 07. 전체 pipeline: 한 번 지나가는 직선이 아니라 비용으로 되돌아오는 탐색

_1 · 전체 구조_

실제 호출 구조. 후보 graph마다 schedule·메모리·codegen까지 내려가 (DMA bytes, words, SRAM peak)를 얻고 그 비용으로 선택합니다.

npu_target.build → npu_build.lower_and_select가 진입점입니다. graph_prefix는 한 번만 적용되고, 그 뒤 attention(off/score/full) × GQA·producer 변환 × generic fusion(off/vectors/matmul) 조합으로 후보 source를 만듭니다. 각 후보는 npu_fusion.select로 영역별 검증을 거친 뒤 npu_link.compile_program으로 entry 전체를 컴파일하고 v09_stats.cost로 비교합니다. 기본값은 최초 합법 module 이후 120초이며 None으로 해제할 수 있습니다. Runtime은 선택된 NpuExecutable과 host transform을 사용합니다.

근거: npu_compiler/npu_target.py, npu_build.py, npu_link.py, tvm_pipeline.py

## 08. TVM 표준 기능 위에 NPU backend의 결정을 얹는다

_1 · 전체 구조_

TVM이 제공하는 것(표현·변환·분석)과 backend가 정의하는 것(장치 계약·후보·비용·명령)을 구분합니다. DLight는 미사용이며 MetaSchedule trace 재사용·ReplayFunc 탐색은 선택형입니다.

TVM은 Relax graph pass, TE/TIR 표현, tir.Schedule loop 변환, CompactBufferAllocation, StaticPlanBlockMemory, USMP 충돌 분석을 제공합니다. NPU backend는 LegalizeOps의 custom map, TensorIntrin 등록, 후보 생성과 비용 검사, 커널 사이 SRAM 보관, ISA 인코딩과 runtime image를 담당합니다. DLight는 규칙 모음일 뿐 SRAM 수명 관리의 다른 이름이 아니므로, 표준 Schedule API 위에 직접 탐색을 두었습니다. MetaSchedule은 09-29부터 선택형 두 기능(JSONDatabase trace 재사용, ReplayFunc 추첨 탐색)에만 사용하며 기본 경로는 여전히 space 탐색입니다.

근거: docs/current/01_overview.md, reference/tvm_api.md

## 09. 코드 지도: 어느 현상이 어느 단계의 결정인지 먼저 정한다

_1 · 전체 구조_

d_compiler/npu_compiler (약 12.7k lines, Python + C++). 파일 순서로 읽기보다 관심 있는 결정을 만드는 단계부터 봅니다.

이 표는 이후 슬라이드의 "구현 연결"에 대응합니다. 예를 들어 첫 tile만 맞고 뒤 tile이 틀리면 ISA 주소 단위(isa_v09, tir_codegen) 문제이고, SRAM 보관을 켤 때만 틀리면 npu_memory_search의 값 버전·alias 판단을 먼저 봅니다.

근거: docs/current/12_operations_limits.md, wc -l npu_compiler/*.py npu_codegen/*.cc

## 10. Llama는 relax.frontend.nn으로 정의하고 정적 명세로 export한다

_2 · 모델 → Relax_

Compiler가 보는 것은 Python의 for-loop가 아니라 export된 계산 그래프입니다. Embedding lookup은 host에 남기고 input_embeds를 device 입력으로 받습니다.

nn_models/llama.py는 Attention(q/k/v/o Linear, bias 없음), MLP(down(silu(gate)·up)), DecoderLayer(pre-norm, residual 2개)를 relax.frontend.nn Module로 씁니다. RoPE는 host가 준 cos/sin과 rotate_half(split → negative → concat)로 표현하고, GQA는 op.repeat로 K/V를 3배 반복합니다. get_default_spec(seq, capacity)가 정적 shape를 고정하고 export_tvm이 packed_params tuple과 num_input attribute를 붙입니다. build_generate는 세 entry를 export한 뒤 decode 함수에 npu.cache_append 계약을 선언합니다.

근거: npu_compiler/nn_models/llama.py, docs/current/02_model_relax.md

## 11. Llama 한 layer의 계산 그래프 (prefill, S=7): tensor의 모양으로 보기

_2 · 모델 → Relax_

각 glyph는 tensor 하나이고 화살표 위 글자가 연산입니다. 겹친 glyph는 head 축(24 또는 8), 회색 삼각형은 causal mask. 두 residual과 GQA repeat, RoPE의 위치를 봅니다. 이 그래프가 28번 반복됩니다.

Attention 앞 RMSNorm 결과가 Q/K/V projection에 들어갑니다. Q는 [7,24,128], K/V는 [7,8,128]입니다. RoPE는 Q와 K에만 적용됩니다. K/V를 repeat(3, axis=1)로 24 head로 늘린 뒤 head를 앞으로 보내 [24,7,128] batch matmul을 합니다. scores [24,7,7]에 1/√128 scale과 causal mask를 더하고 softmax, PV, O projection 뒤 첫 residual을 더합니다. MLP는 gate/up 두 갈래([7,8192] 각각, weight 48 MiB씩)를 SiLU·곱으로 합쳐 down projection 후 두 번째 residual을 더합니다. glyph의 칸 수는 축약이며 실제 tile 수(예: [7,3072]는 폭 48 tiles)의 비율이 아닙니다.

근거: docs/examples/llama_ir/01_authored_relax.py, docs/current/02_model_relax.md

## 12. 실제 authored Relax (decode, 1 layer, C=9) 발췌

_2 · 모델 → Relax_

실제 3B 폭, layer 수만 28→1로 줄인 export 결과. 변수 이름은 자동 생성이지만 계산은 앞 슬라이드의 그래프와 1:1 대응합니다.

decode 함수의 signature에 6개 실행 입력과 packed_params가 있고 npu.cache_append attribute가 붙어 있습니다. permute_dims(weight)가 layer마다 나타나는데 이것은 3장의 LiftTransformParams가 host로 옮깁니다. cache concat 뒤 repeat로 24 head가 되고, scores [24,1,10]에 scale 0.0884(=1/√128)와 mask [1,1,10]을 더합니다. 반환 tuple은 logits와 현재 token의 K/V이며 runtime의 append DMA가 cache slot에 옮깁니다.

근거: docs/examples/llama_ir/01_authored_relax.py (IR export만, checkpoint/실행 없음)

## 13. 작은 예제 ①: SiLU(X @ Wᵀ + bias) — 끝까지 따라갈 4개 연산

_2 · 모델 → Relax_

X[64,128], checkpoint weight[64,128], bias[64]. 64×64 tile로 보면 X는 1×2, Wᵀ는 2×1, 출력은 1 tile입니다. bias는 broadcast와 fusion을 설명하기 위해 넣었고 실제 Llama projection에는 없습니다.

이 예제의 authored Relax는 함수 하나에 permute_dims, matmul, add, silu 네 연산이 있습니다. num_input=1이므로 x만 실행 입력이고 weight/bias는 parameter입니다. tile 관점에서 X의 두 tile(K=0..63, 64..127)과 Wᵀ의 두 tile이 같은 출력 tile P에 누적되고, bias는 P의 모든 행에 같은 64개 값을 더하며, SiLU는 원소별입니다. 같은 예제의 host transform, semantic TIR, scheduled TIR, ISA packet, 실제 실행 출력이 docs/examples/pipeline에 모두 남아 있어 이후 슬라이드가 이 파일들을 인용합니다.

근거: docs/examples/pipeline/01_authored_relax.py, docs/current/02_model_relax.md

## 14. Relax를 읽는 데 필요한 최소 용어

_2 · 모델 → Relax_

Dataflow block은 순수 데이터 의존성을 표현할 뿐 하드웨어 동시 실행을 약속하지 않습니다. 실행 순서와 storage 재사용은 후속 단계가 정합니다.

IRModule 하나에 prefill, host transform 함수, 그리고 나중의 TIR PrimFunc들이 함께 있습니다. TensorStructInfo 덕분에 실제 배열 없이도 shape/dtype 추론이 가능합니다. GlobalVar는 모듈 안의 다른 함수를 참조하며 call_tir가 PrimFunc를 호출할 때 사용됩니다. BlockBuilder는 작은 예제나 legalizer가 graph를 만들 때 쓰는 도구입니다.

근거: docs/current/02_model_relax.md, reference/tvm_api.md

## 15. Graph prefix: 후속 lowering이 이해하기 쉬운 graph로 만드는 5개 pass

_3 · Graph 정리와 host_

한 번만 적용됩니다(prepared=True). 1번만 NPU 전용이고 나머지는 TVM 표준 pass입니다.

graph_prefix는 CanonicalizeAttention, CanonicalizeBindings, EliminateCommonSubexpr, RewriteDataflowReshape, LiftTransformParams 순서의 Sequential입니다. 표준 pass를 호출해도 임의의 layout 변환이 모두 zero-copy가 되거나 모든 연산이 장치에 맞게 되는 것은 아니며, 정확한 지원 조건은 이후 단계가 확인합니다. GQA repeat → broadcast matmul(FoldRepeatedGemv), concat producer fusion은 항상 적용되는 prefix가 아니라 비용 비교 대상인 선택적 후보입니다.

근거: npu_compiler/tvm_pipeline.py graph_prefix(), docs/current/03_graph_host.md

## 16. LiftTransformParams: weight transpose는 언제, 어디서 실행되는가

_3 · Graph 정리와 host_

작은 예제의 실제 IR. device 함수에서 permute_dims가 사라지고 반환 tuple의 순서가 (bias, lv)로 바뀝니다. 변환된 Wᵀ는 DRAM image의 정해진 주소에 한 번 놓입니다.

Compile은 host 함수를 만드는 시점이고, 실제 weight 배열을 transpose하는 것은 runtime 준비 단계에서 host 함수를 LLVM VM으로 실행할 때입니다. 그림 아래의 DRAM image는 bundle/program.json의 실제 주소입니다: x 0, bias 16,384, lv(Wᵀ) 16,512, 출력 gv 32,896, literal pool 41,088 (총 41,100 B). Runtime은 원래 순서를 추측하지 않고 변환된 signature(x, bias, lv)에 맞춰 이름·shape·dtype을 결합합니다. Llama LM head weight만 해도 751.5 MiB라 이 변환이 호출마다 들어가면 매우 큰 이동입니다.

근거: docs/examples/pipeline/02_graph_prefix.py, 05_host_transform.py, bundle/program.json (memory.addresses), docs/current/03_graph_host.md

## 17. CanonicalizeAttention: attention을 세부 연산으로 드러내는 이유

_3 · Graph 정리와 host_

고수준 relax.nn.attention을 그대로 LegalizeOps에 넘기면 표준 attention legalizer가 자체 TE 계산을 만들어, 우리가 등록한 matmul·softmax의 NPU 정의를 거치지 않습니다.

CanonicalizeAttention은 relax.nn.attention / attention_bias만 대상으로 하며 정적 FP16 rank-4 BSNH 입력을 요구합니다. Q를 [b·h,s,d], K를 미리 transpose한 [b·h,d,t], V를 [b·h,t,dv]로 바꾸고 scale을 FP16 상수로 곱하거나 √d로 나눕니다. causal(TopLeft/BottomRight)과 window는 tril/triu로 max 전과 exp 후에 적용해 masked 원소를 정확히 0으로 만듭니다(그림의 회색 삼각형). 결과는 BSNH로 되돌립니다. 설치된 TVM 0.19의 DecomposeOpsForInference는 이 attention 예제를 그대로 두었다는 IR 비교 기록이 있어, 표준 pass 이름이 비슷하다고 같은 변환이라 판단하지 않습니다.

근거: npu_compiler/npu_attention.py, docs/examples/attention_canonicalization/comparison.json

## 18. Runtime의 host transform 실행과 parameter cache

_3 · Graph 정리와 host_

IR로 표현된 변환을 그대로 실행해야 linked program의 image가 기대하는 값과 정확히 같습니다. NumPy로 재구현하면 같은 규칙이 두 곳에 생깁니다.

_run_transform은 entry_transform_params만 도달 가능한 module을 relax.build(…,"llvm")로 빌드해 VirtualMachine으로 실행합니다. _transform은 결과를 디스크 cache에 저장하는데, 키는 host IR JSON과 모든 weight의 shape·dtype·바이트 내용을 넣은 SHA-256입니다. flock, atomic rename, 배열별 sha256 검증, memmap 로드를 사용합니다. _bind_inputs는 func.params의 앞부분(runtime 입력)을 이름으로 검사하고 나머지를 변환 결과와 zip합니다. build_image는 plan.address×unit_bytes 위치에 각 leaf를 쓰고 const_data, 상수 pool, FP32 1.0을 채웁니다.

근거: npu_compiler/npu_generate.py (_run_transform, _transform, _bind_inputs), npu_link.build_image

## 19. LegalizeOps 뒤에도 Relax는 남는다: call_tir + PrimFunc

_4 · LegalizeOps와 수치_

작은 예제의 실제 unfused IR. Relax 함수는 호출 순서를, 각 PrimFunc는 원소 계산을 정의합니다. op_pattern attribute가 다음 단계 FuseOps의 입력입니다.

LegalizeOps는 고수준 Relax 연산을 원소 단위 계산으로 정의하지만 SRAM 주소나 64×64 실행 순서는 정하지 않습니다. call_tir는 TIR 함수를 호출하고 결과 shape/dtype을 알려주는 표현이지 그래프 연결을 지우는 명령이 아니므로 그 뒤에 FuseOps를 수행할 수 있습니다. TIR은 TVM Tensor IR이며 dump는 TVM Script 문법입니다. 다른 compiler가 TIR이라는 약자를 써도 같은 IR이 아닙니다.

근거: docs/examples/pipeline/03_unfused_semantic_tir.py, docs/current/04_semantic_tir.md

## 20. TIR 읽기: block, 축의 종류, reads/writes, TE와 call_te

_4 · LegalizeOps와 수치_

TE의 placeholder/compute/reduce_axis는 "이 원소의 값은 어떤 수식인가"를 선언하고, bb.call_te가 그 정의에서 PrimFunc와 call_tir를 만듭니다.

i,j는 결과 위치를 고르는 spatial 축, k는 같은 결과에 여러 값을 더하는 reduction 축입니다. T.init은 reduction 초기값, T.reads/T.writes는 block의 접근 영역으로 schedule이 이동·분할의 합법성을 분석할 때 씁니다. Buffer는 shape/dtype/scope를 가진 배열 표현입니다. Block 이름이 같다고 같은 연산이 아니며, 현재 kernel 분석(npu_kernel.matmul_blocks)도 이름이 아니라 접근·수식·계약을 보고 matmul을 인식합니다.

근거: docs/current/04_semantic_tir.md, reference/tvm_api.md, npu_compiler/npu_legalize.py

## 21. NPU legalize map: 기본 lowering과 다른 dtype·저장 경계가 필요한 연산만 바꾼다

_4 · LegalizeOps와 수치_

TVM 기본 RMSNorm은 FP32 중간값을 유지하지만 우리 vector unit은 FP16을 저장하고 내부 reduction만 FP32입니다. 그래서 per-op recipe만 교체합니다.

legalize_map은 relax.matmul, relax.split, relax.sum, relax.mean, relax.nn.rms_norm, relax.nn.softmax 여섯 개를 등록합니다. 각 legalizer는 지원 조건을 검사하고 밖이면 ValueError로 거절합니다. 예를 들어 softmax는 마지막 축, 정적 길이 1..65504, FP16만 허용합니다. RMSNorm은 마지막 축·1차원 weight·epsilon 범위(2^-48 ≤ ε ≤ 65504²)를 검사합니다. split은 axis-0 정적 split을 memory.view + ensure_zero_offset으로 바꿔 decode의 layer별 cache 분리가 복사 없이 되게 합니다.

근거: npu_compiler/npu_legalize.py legalize_map(), npu_views.legalize_split

## 22. FP16 저장 / FP32 누산: K 전체 동안 accumulator를 끊지 않는다

_4 · LegalizeOps와 수치_

K=128 출력 tile 하나의 128개 곱은 64개씩 두 그룹의 FP32 부분합이 되어 같은 accumulator에 더해지고, 마지막에 한 번만 FP16으로 반올림됩니다. 중간에 FP16으로 저장했다 다시 더하면 반올림 위치가 달라집니다.

입력과 출력이 FP16이어도 많은 곱을 FP16으로 계속 더하면 오차가 커집니다. 현재 정책은 FP16 입력을 곱해 FP32로 누산하고 완료된 결과만 FP16으로 저장하는 것입니다. 장치는 K=64 단위 부분합을 만들므로 CPU reference도 split(k,[None,64])+rfactor로 같은 grouping을 표현하고, FoldConstants도 같은 기준을 씁니다. 단순 flat-K CPU loop와 모든 입력에서 bit-exact하다고 약속하지 않습니다(덧셈 결합 순서). E1에서는 같은 grouping의 LLVM 출력과 최대 절대차 0이었습니다.

근거: docs/current/04_semantic_tir.md, npu_numeric.tile_order_reference, docs/examples/pipeline/08_llvm_reference.py

## 23. Softmax와 RMSNorm의 실제 정의: reduction과 FP16 범위

_4 · LegalizeOps와 수치_

"FP32 reduction"은 softmax 전체가 FP32라는 뜻이 아닙니다. 입력·exp·저장된 합·출력은 FP16이고 합산 내부만 FP32입니다.

최댓값을 빼면 실수 수학에서 결과는 같고 exp의 큰 양수 입력을 피합니다. 유한 logits에서 exp(x−max)≤1이므로 마지막 축 길이가 65504 이하면 FP16 합이 범위 안에 있습니다. RMSNorm은 x=1000이면 x²=10⁶으로 FP16 범위를 넘고 합산만 FP32로 바꿔도 이미 저장한 제곱값의 overflow는 복구할 수 없어서, 입력을 absmax로 먼저 줄입니다. |u|≤1이고 Σz²도 크게 자라지 않지만 FP16 중간 반올림 때문에 별도의 target 근사이며 HF FP32와 동일하지 않습니다.

근거: npu_compiler/npu_legalize.py (_softmax_te, _rms_norm_te), docs/current/07_operators.md

## 24. Fusion: 커널 수가 줄어드는 것만으로 이득을 판단하지 않는다

_5 · Fusion 후보 선택_

작은 예제에서 분리(off)와 결합(auto)의 실제 데이터 이동. 두 구성 모두 중간 tile P·A가 SRAM에 머물러 DMA는 같고, 커널 경계의 descriptor·LOAD/SAVE 반복과 짧은 vector 행이 words와 SRAM 작업 공간을 바꿉니다.

off는 3커널(matmul 43 words, add 391, silu 2,373)이고 auto는 1커널 1,572 words입니다. 양쪽 모두 GLOAD(X, Wᵀ, bias, 상수) 4개와 GSTORE(Y) 1개만 DRAM을 지나므로 DMA는 41,100 B로 같습니다. 그림처럼 중간 tile P, A, sig가 SRAM에서 다음 계산으로 바로 이어지는 것은 커널을 나눠도 가능합니다(8장). Fused 커널은 축 결합으로 vector 행이 128 폭이 되어 명령 수가 절반이 되고, USMP가 buffer offset을 공유해 SRAM peak가 50,700 → 42,508 B로 줄었습니다. 이득의 형태는 예제마다 다르므로 실제 비용으로 확인합니다.

근거: docs/current/05_fusion.md, docs/examples/pipeline/stage_summary.json (E1)

## 25. TVM은 무엇을 보고 묶는가: op_pattern, FuseOps, FuseTIR

_5 · Fusion 후보 선택_

AnnotateTIROpPattern이 TIR의 연산 성격을 분류하고, FuseOps가 의존 그래프·post-dominator·경로 pattern으로 영역을 만들며, FuseTIR가 TIR 계산을 한 PrimFunc로 결합합니다.

FuseOps는 pattern 숫자가 같다는 이유로 묶지 않습니다. OutEWiseFusable(matmul) 출력 뒤에 elemwise/broadcast 경로가 이어지면 붙이고, injective와 tuple 연결은 후속 차례에서 처리하며, CommReduce는 다른 복합 연산에 임의로 붙이지 않고, Opaque는 경계입니다. 가지가 나뉘면 중간의 모든 관련 경로가 조건을 만족해야 합니다. 그림의 작은 예제: transpose(2)는 host 함수라 device 그래프에 없고, matmul(4) → add(0) → silu(0)이 한 영역이 되어 FuseTIR가 fused_matmul_add_silu PrimFunc를 만듭니다. NPU는 지원하지 않는 matrix 복합 영역과 W8A8을 opaque(8)로 표시해 경계를 보호합니다.

근거: docs/current/05_fusion.md, TVM 0.19 src/relax/transform/fuse_ops.cc, docs/examples/pipeline/03_unfused_semantic_tir.py

## 26. FuseOpsByPattern: attention 전체 또는 score chain을 composite로

_5 · Fusion 후보 선택_

패턴은 pattern 언어(is_op/wildcard)로 쓰고, supported 검사 함수가 정적 rank-3 FP16 64배수 조건을 확인합니다. 조건 밖이면 더 작은 score fusion으로 물러납니다.

전체 attention 패턴은 matmul → (multiply) → add → softmax → matmul입니다. supported 검사는 q/k/v가 FP16 rank-3 정적 shape이고 softmax axis가 마지막이며 h>0, k[0]==v[0]==h, k[1]==d, v[1]==k[2], 그리고 m, d, t, dv가 모두 64의 배수인지 확인합니다. decode는 m=1이라 조건을 만족하지 않습니다. score chain 패턴(npu.masked_softmax)은 softmax(add(multiply(scores, scale), mask))와 scale이 없는 변형(Gemma)입니다. 코드 주석의 seq 2048 수치는 설계 근거로 남긴 과거 관측이며 이번 실험표가 아닙니다.

근거: npu_compiler/tvm_pipeline.py (_attention_patterns, _masked_softmax_patterns, _matrix_producer_patterns)

## 27. npu_fusion.select: 영역 하나를 실제 codegen 비용으로 검사한다

_5 · Fusion 후보 선택_

TVM이 만든 FuseOps 영역마다 unfused(기준)와 fused를 schedule·메모리·명령 생성까지 내려 비교하고, 진 영역은 inline_functions로 다시 펼칩니다.

먼저 W8A8 PrimFunc와 matmul이 2개 이상인 PrimFunc(또는 matmul_epilogues=False일 때 모든 matmul PrimFunc)에 op_pattern 8(kOpaque)을 붙여 경계로 둡니다. FuseOps로 영역을 만들고 FuseTIR로 결합한 뒤, entry에서 도달 가능한 Primitive 함수마다 unfused 기준을 probe하고 fused wrapper를 probe합니다. unfused probe의 실패는 unsupported 입력이나 compiler 오류이므로 숨기지 않습니다. fused가 사전식 비용에서 작을 때만 채택하고, 나머지는 inline해 다시 FuseTIR·DCE합니다. region_cost cache로 같은 영역의 중복 계산을 줄입니다.

근거: npu_compiler/npu_fusion.py select(), _probe(), docs/examples/pipeline/bundle/program.json fusion.regions

## 28. npu_build.lower_and_select: 같은 IR인 source를 먼저 제거하고, 전체 entry의 비용으로 module 하나를 고른다

_5 · Fusion 후보 선택_

최대 27개 조합 중 structural_equal인 source는 generic fusion 검사 전에 버립니다 (E10 prefill: source 9 → 3, 후보 9). 남은 후보는 entry마다 컴파일한 (DMA, words, SRAM)을 사전식으로 비교하며, 첫 합법 module 뒤 추가 탐색은 기본 120초, None이면 제한 없음입니다.

options()는 attention(False/score/full) × (gqa, producers) 세 구성마다 graph_pipeline으로 source를 만들고, 이미 만든 source와 structural_equal이면 generic 3종을 검사하지 않고 건너뜁니다. E10의 256-token prefill_cache에서는 GQA fold와 producer 변환이 IR을 바꾸지 않아 9개 source 중 3개만 남고, generic off/vectors/matmul을 곱한 9개 구성이 검사 대상입니다(회색 18칸). decode를 함께 요청하는 경우에는 결과가 다를 수 있습니다. generic이 켜지면 npu_fusion.select로 후보를 만들고 이미 본 후보와 같으면 duplicate로 기록합니다. compile_all은 entry마다 EntryCache.compile을 호출하며, 완료된 entry 하나가 incumbent보다 나쁘면 나머지 entry를 건너뛰고 noncompetitive로 기록합니다. 채택 조건은 모든 entry가 비악화이고 하나 이상이 개선인 경우이며, 첫 합법 결과 뒤 graph_search_seconds 예산을 시작합니다(기본 120초, None이면 시간 제한 없음, phase 경계에서만 검사). schedule_database가 설정된 경우 후보 module마다 relax_integration.extract_tasks로 작업 종류·호출 수 목록을 진단용으로 남깁니다. 오른쪽 격자의 ✓/⏱은 과거 7-token 기록입니다: 후보 0만 legal·selected, 후보 1에서 예산 소진.

근거: npu_compiler/npu_build.py lower_and_select(), npu_compilation.optimization_budget, evidence/current/llama_selection.json (7-token 기록), llama_prefill256_full_search_progress.json (E10)

## 29. 과거 7-token Llama 선택 기록: 120초 안에서 unfused 기준이 선택되었다

_5 · Fusion 후보 선택_

compile 189.7 s. 후보 0(모두 off)이 legal·selected, 후보 1(generic vectors)에서 추가 탐색 예산 120 s가 소진. "모든 fusion이 더 나쁘다"는 증거가 아닙니다.

phase별 self seconds를 보면 native codegen 41.5 s, schedule probe codegen 29.3 s, residency 22.4 s, entry compile 자체 20.8 s가 큽니다. cache 통계는 schedule recipe hit 272 / miss 383, trial hit 184 / miss 248, cost probe hit 294 / miss 270, region miss 40입니다. native가 거절한 schedule은 "FP32 matrix chain interrupted" 5건입니다. 후보 1은 vectors generic fusion을 검사하던 중 예산이 끝났으므로 이후 후보(attention score/full 등)는 검사되지 않았습니다.

근거: docs/evidence/current/llama_selection.json (과거 7-token 보존 기록; 새 256-token E10과 구분)

## 30. 64×64 장치로 K=128을 계산하기: 출력 tile 하나의 K 누산은 끊지 않는다

_6 · Matmul scheduling_

작은 예제 ② 두 번째 matmul H[64,128] @ W2[128,64]의 실제 명령 20 words. X₀·B₀로 accumulator를 시작하고(MAC=0), X₁·B₁을 같은 accumulator에 더한 뒤(MAC=1), 한 번만 FP16으로 SAVE합니다.

위 그림은 tile 단위 데이터 흐름이고 아래 strip은 77-word 프로그램의 word 52~71을 그대로 옮긴 것입니다. 각 operand는 descriptor(ADDR MAIN/PARTIAL, COLS)를 설정한 뒤 LOAD로 matrix unit에 들어갑니다. 두 번째 K tile에서는 SRC1 MAIN 주소가 +256 nibbles(= 64열), SRC2가 +16384 nibbles(= 64행) 이동하고 COLS는 peephole이 생략했습니다. accumulator가 1개이고 partial spill이 금지이므로 출력 tile 하나의 K chain을 끝내고 다음 tile로 갑니다(K 최내측). 이것은 모든 하드웨어의 일반 법칙이 아니라 이 장치의 계약입니다.

근거: docs/examples/two_matmuls_recorded/program.json (words 52–71), scheduled_0001.py, docs/current/06_matmul_schedule.md

## 31. SRAM에 올리는 panel은 matrix tile보다 크다: 실제 폭 gate projection

_6 · Matmul scheduling_

X[64,3072] @ W[3072,8192]. W는 48 MiB = 48×128 tiles라 SRAM에 들어가지 않으므로 X(1×48 tiles, 384 KiB)를 상주시키고 W의 열 panel j(48×1 tiles, 384 KiB)를 128번 교체합니다.

그림의 각 칸은 64×64 tile입니다. X panel 하나를 한 번 올리면 모든 출력 tile이 재사용하고, W의 열 panel j가 올라오면 그 안의 K 방향 48개 tile로 M_MUL 48개를 연속 실행해 출력 tile (0, j)를 완성합니다. 출력 열 tile은 128개이므로 총 6,144 M_MUL이며 W 전체 48 MiB가 한꺼번에 SRAM에 올라가는 일은 없습니다. 같은 X/W 크기라도 W를 몇 번 다시 읽는지에 따라 DMA는 달라지고, SRAM을 가득 채우는 것이 목표가 아니라 앞 커널 출력 보관 공간과 임시값 공간이 함께 필요합니다. down projection은 반대로 W panel(8192×256, 4 MiB)을 유지하고 H는 앞 커널의 SRAM 보관값을 씁니다.

근거: docs/current/06_matmul_schedule.md, evidence/current/refinement.json (M=64 MLP 정적 compile 기록)

## 32. 탐색 공간: m/n panel, K chunk, loop 순서, 그리고 두 operand의 cache 위치

_6 · Matmul scheduling_

npu_schedule_space.candidates: 64×64 명령과 끊기지 않는 K chain은 제약이지 mapping 템플릿이 아닙니다. 출력 tile을 도는 순서와 각 operand의 cache 위치가 재읽기 횟수를 정합니다.

factors(extent, 64)는 padding한 축 길이의 64배수 약수입니다. 예를 들어 N=256이면 64/128/256, N=8192면 8개, K=3072면 10개입니다. product(m_panel, n_panel, k_chunk, outer_order, inner_order)에서 loop_spec은 항상 [outer panel 2개, inner tile 2개, k_group, k_tile] 구조를 만들어 K가 최내측입니다. 왼쪽 아래 격자는 출력 [128,256] = 2×4 tiles를 m_panel=n_panel=64로 도는 순서입니다: mn은 X 행 panel을 고정한 채 W 열 panel을 바꾸고(X 재사용), nm은 그 반대입니다. a_at/b_at는 cache stage를 넣을 loop이며, unit loop와 같은 축의 연속 loop는 실제 순회와 복사 주기가 같으면 signature로 합칩니다(다음 슬라이드). 추정 DMA는 FP16 operand bytes × 반복 횟수이며 검사 순서만 정하고 선택 기준은 실제 emission입니다. 오른쪽 아래 표의 "개선 전" 후보 수는 09-28 compile 기록이고 "현재" 수는 candidates()/prioritize()로 계산한 값입니다. 새 compile 기록이 아닙니다.

근거: npu_compiler/npu_schedule_space.py (factors, loop_spec, candidates), evidence/current/refinement.json (09-28 기록)

## 33. 후보 정규화와 용량 필터: 같은 tile 순회·복사 주기는 한 후보로, 증명 가능한 SRAM 초과만 미리 제외

_6 · Matmul scheduling_

npu_schedule_space.candidates / prioritize (09-29 개선). 512-token down [512,8192]×[8192,3072]: 22,236 → 5,034 후보, 그중 1,596은 두 입력 cache의 합만으로 8 MiB를 넘어 제외, 3,438을 검사 목록에 남김 (E9 기록).

① loop_spec은 같은 축을 panel과 tile 두 loop로 나누므로 n_panel=64/128/256은 모두 N tile을 1,2,3,4 순서로 도는 같은 순회입니다. _iteration_signature는 크기 1인 loop를 빼고 같은 축의 이웃 loop를 곱해 (m2, n4, k2) 하나로 만듭니다. cache 위치는 그 loop까지의 prefix signature로 구분하므로, b_at=n_panel이라도 n_panel=64는 tile마다, 128은 2 tile마다, 256은 한 번 복사하는 서로 다른 후보로 남습니다. toy [128,256,128]에서 loop 위치 조합 1,728개가 후보 73개가 됩니다. ② prioritize는 두 operand cache가 matrix 명령 시점에 동시에 살아 있다는 사실만 사용합니다. 두 operand가 서로 다른 함수 parameter이고 FP16, scope global/global.sram, shape가 정확히 (M,K)/(K,N), M·N·K가 모두 64배수일 때만(ordinary) 두 cache bytes 합이 (SRAM − scratch prefix)를 넘는 후보를 제외합니다. padding이나 computed producer, W8A8처럼 증명이 성립하지 않으면 footprint 추정으로 순서만 뒤로 보내고 제거하지 않습니다. 출력·scratch·epilogue 추정은 제외 근거로 쓰지 않습니다. 표의 512-down 수치는 E9 기록이고 나머지 "계산" 수치는 현재 코드의 candidates()/prioritize()를 scratch 96 KiB 가정으로 실행한 값입니다.

근거: npu_compiler/npu_schedule_space.py (_iteration_signature, candidates, prioritize), npu_schedule.py (_fit_matmul의 ordinary 판정), docs/current/06_matmul_schedule.md, 11_experiments.md E8/E9

## 34. 후보 검사 흐름: DB 재생(선택) → incumbent → 정규화된 후보를 실제 schedule·SRAM 배치·명령 생성으로 검사

_6 · Matmul scheduling_

_fit_matmul. 실패·용량 초과도 시도로 세고, words 한도는 기준 후보 크기에 맞춰 늘어나며, 합법 후보 2개를 보기 전에는 words로 끝내지 않습니다. 검사 순서만 space(추정 DMA 순)와 meta_schedule(ReplayFunc 추첨)이 다릅니다.

순서: schedule_database가 있으면 workload(계산 구조 + 실행 조건)로 trace를 조회해 새 Schedule에 재적용하고 공통 lowering·SRAM 배치·용량·codegen을 다시 검사합니다. reuse 모드는 여기서 끝나고(database_hit, 새 후보 0회), tune 모드는 그 결과를 incumbent로 둡니다. DB가 없거나 실패하면 legacy 규칙(A/B/B_stream/None, legal 목표 2, 시도 64)으로 incumbent를 잡습니다. 확장 탐색은 candidates()로 정규화된 pool을 만들고 prioritize로 증명된 용량 초과를 뺀 뒤 space는 추정 DMA 순으로, meta_schedule은 ReplayFunc가 sample_categorical로 뽑은 순서로 검사합니다. 각 후보는 _measure_trial(trial cache)에서 tir.Schedule 변환과 TIR 후처리(LowerInitBlock → PlanAndUpdateBufferAllocationLocation → ConvertBlocksToOpaque → CompactBufferAllocation), lower_stores, forward_output_tiles, pointwise coalescing을 거친 뒤 plan_sram으로 USMP 배치를 얻습니다. SRAM top이 용량을 넘으면 capacity_rejected, MAC chain이 끊기면 codegen_rejected입니다. words 한도는 max(schedule_probe_word_budget 100,000, 기준 words × legal 목표 8)이며 기준은 incumbent words, 없으면 첫 합법 후보 words입니다. 종료 사유와 generated/eligible/examined/legal 수, 유효 words 한도가 진단에 남습니다. 승자가 DB 기록보다 좋으면(또는 기록이 없으면) trace를 commit합니다.

근거: npu_compiler/npu_schedule.py (_fit_matmul, _measure_trial, _probe_accumulator_chain), npu_schedule_space.prioritize, device.NpuProfile

## 35. 표준 Schedule API가 실제로 바꾸는 것과 그 자체로 하지 않는 일

_6 · Matmul scheduling_

tir.Schedule은 TIR를 받아 block/loop 변환을 적용합니다. 변환 뒤 loop 구조가 달라지므로 이전 handle을 그대로 쓰지 않습니다.

schedule_matmul_sram이 적용하는 순서는 계약 검사 → 다른 producer(pad, dequant)의 cache_read/set_scope → 필요 시 pad_einsum → split(i,[None,m_panel])→split(·,[None,64]), j·k 동일 → reorder → cache_write + reverse_compute_at(출력 write-back) → fused epilogue의 reverse_compute_at/set_scope → compute_inline → operand cache_read + compute_at(a_at/b_at) → decompose_reduction(k_group) → tensorize 두 번(gemm, fill)입니다. compute_at이 주소 배치를 최적화하는 것은 아니며, buffer가 어디서 얼마나 쓰이는지 드러낸 뒤 compaction·USMP·보관 계획이 그 정보를 사용합니다.

근거: docs/current/06_matmul_schedule.md, reference/tvm_api.md, npu_compiler/npu_intrin.py schedule_matmul_sram

## 36. compute_at을 작은 예제로 보기: 필요한 생산을 사용 위치로 옮긴다

_6 · Matmul scheduling_

B = A + 1, C = B × 2 (4×4 FP32 CPU 교육 예제, NPU tile이 2라는 뜻은 아님). C의 행 loop를 2행씩 나눈 뒤 B를 그 안으로 옮깁니다.

변환 전에는 B 전체를 만든 뒤 C 전체를 만들므로 B 4×4가 오래 살아 있습니다. split으로 i를 (2,2)로 나누고 B를 i_0 안에 compute_at하면 각 행 묶음에서 필요한 B 2행만 만들고 바로 C가 사용합니다. reverse_compute_at은 반대로 B를 만드는 loop 기준으로 소비자 C를 끌어옵니다. Matmul 뒤 add/SiLU나 SRAM 출력 복사 stage를 생산 tile 근처에 두려면 이 구분이 필요합니다.

근거: docs/examples/schedule_api/01_before.py, 02_compute_at.py, 02_compute_at.trace.txt, 03_reverse_compute_at.py

## 37. cache_read → compute_at → CompactBufferAllocation: 복사 buffer가 실제 접근 범위로 줄어든다

_6 · Matmul scheduling_

cache_read만 추가하면 입력 전체를 복사하는 buffer가 생깁니다. 복사 stage를 행 묶음 안에 넣으면 필요한 영역이 좁아지고, compaction이 allocation을 그 범위로 축소합니다.

아래 격자는 SRAM cache buffer의 실제 크기와 한 반복에서 채워지는 영역입니다. ①에서는 4×4 전체를 한 번에 복사하고, ②에서는 i_0마다 2행만 복사하지만 buffer 선언은 여전히 4×4이며, ③ CompactBufferAllocation이 접근 범위(2×4)로 선언을 줄입니다. 이 메커니즘이 실제 matmul schedule에서 panel 크기를 결정합니다. a_at을 m_panel loop에 두면 X cache buffer가 [m_panel, K]로, k_group loop에 두면 [m_panel, k_chunk]로 compaction됩니다. USMP는 compaction된 크기와 수명을 보고 SRAM offset을 배치합니다.

근거: docs/examples/schedule_api/04_cache_read.py, 05_cache_read_compute_at.py, 08_cache_read_compacted.py

## 38. Tensorize: 64×64 계산 조각을 등록한 TensorIntrin으로 치환한다

_6 · Matmul scheduling_

수식·dtype·scope가 일치하면 npu_gemm_acc_sram / npu_fill_zero_sram으로 바뀝니다. 이는 최종 opcode가 아니라 codegen이 descriptor·LOAD·M_MUL로 펼칠 call_extern입니다.

description은 A,B (64,64) FP16, C (64,64) FP32, 모두 scope global.sram, strides [s,1] symbolic, offset_factor 1로 등록됩니다. symbolic stride 덕분에 64×64 view가 더 넓은 panel 안에 있을 수 있습니다. implementation은 call_extern("npu_gemm_acc", C.ptr, sc, A.ptr, sa, B.ptr, sb)입니다. fill intrinsic은 C=0을 call_extern("npu_fill_zero")로 바꾸며, codegen에서 이것은 명령을 내지 않고 새 누산 시작 상태로만 기록되어 첫 M_MUL이 MAC=0이 됩니다. register_all은 import 시 실행되고 구조 동일성 검사로 중복 등록을 막습니다.

근거: npu_compiler/npu_intrin.py (_sram_gemm_desc/_impl, register_all), docs/examples/pipeline/07_scheduled_fused_matmul_add_silu.py

## 39. 실제 선택 기록: 작은 예제의 4개 후보와 실제 폭 MLP의 incumbent

_6 · Matmul scheduling_

작은 예제는 X/W 16 KiB씩을 K loop 바깥에서 한 번 올리고 2개의 M_MUL이 같은 SRAM 배열의 다른 부분을 읽습니다. 실제 폭 MLP는 정적 compile 기록입니다(executed=false). 후보 수는 개선 전 탐색 공간의 기록이며 현재 정규화 공간에서는 203 / 17개입니다.

작은 예제에서 space 후보 4개는 모두 추정 32,768 B이고, legacy incumbent ("B_stream",64)는 1,574 words, 선택된 a_at=b_at=m_panel 후보는 1,572 words로 DMA가 같고 words에서 이겼습니다. 선택 후보의 SRAM peak(85,016 nibbles)가 incumbent(68,632)보다 크지만 사전식 비교에서 SRAM은 세 번째 기준입니다. 실제 폭 MLP(M=64)는 5커널, 140,594 words, DMA 152,174,604 B, SRAM peak 7,446,540 B, compile 21.96 s로 기록되어 있으며 전체 모델 출력 검증이 아닙니다. 표의 후보 수 1,433/299는 09-28 기록의 정규화 전 공간이며, 현재 코드의 candidates()는 같은 shape에서 203개(용량 증명 제외 후 162)와 H resident 17개(제외 후 12)를 만듭니다. 기록의 선택 자체는 그대로입니다.

근거: docs/examples/pipeline/bundle/program.json schedules, evidence/current/refinement.json, mlp64.json

## 40. 긴 prefill(S=3072)에서 X와 W가 모두 큰 경우: 재읽기 횟수가 비용을 정한다

_6 · Matmul scheduling_

설명용 계산이며 새 compile 측정이 아닙니다. X 18 MiB, gate W 48 MiB. X는 64행(384 KiB), W는 256열(1.5 MiB)씩 올린다고 가정합니다.

X panel을 바깥에 유지하면 48개 X panel마다 W 전체를 열 panel로 순회하므로 X 18 MiB + W 48 MiB × 48회입니다. W panel을 바깥에 두면 32개 W panel마다 X 전체를 행 panel로 순회하므로 W 48 MiB + X 18 MiB × 32회입니다. 원래 tensor 크기는 같지만 재읽기 횟수가 다릅니다. 실제 탐색은 더 큰 panel과 다른 cache 위치도 검사하고 앞 커널의 SRAM 입력을 쓸 수 있으면 그 부분의 DRAM 읽기도 제외합니다. 그래프 전체의 tile streaming은 지원하지 않으므로 이 산술을 장문 전체 모델의 최적 실행 결과로 해석하지 않습니다.

근거: docs/current/06_matmul_schedule.md (설명용 산술)

## 41. 누산 체인 검사: codegen이 MAC chain을 보장하고, lower_stores가 FP32 buffer 사용을 제한한다

_6 · Matmul scheduling_

"SRAM에 공간이 충분하다"만으로 schedule이 실행 가능한 것은 아닙니다. 출력 tile의 K 누적을 끊는 후보는 세 단계 중 하나에서 거절됩니다.

emit_gemm은 C가 zeroed 집합에 있으면 이전 tile을 flush하고 pending=(C, stride)로 두어 MAC=0을 냅니다. 같은 C가 이어지면 MAC=1이고, 다른 C가 오면 "FP32 matrix chain interrupted"를 raise합니다. flush(=FP16 SAVE)는 vector op, reduction, SRAM→SRAM 이동, write-back DMA 전에, 그리고 kernel 끝에 실행되며 입력 GLOAD staging은 flush하지 않습니다. lower_stores는 FP32 buffer의 store를 copy/zero/acc+cast(fp16)만 허용하고 load는 FP16 cast 아래에서만 허용해 FP16 물리 slot으로 retype합니다. USMP는 matrix 목적지 buffer를 pin해 SAVE가 지연되는 동안 다른 buffer와 겹치지 않게 합니다.

근거: npu_compiler/tir_codegen_v09.py (emit_gemm, flush), npu_accumulator.py (lower_stores, forward_output_tiles), npu_sram.py (pinned)

## 42. MetaSchedule 연결(선택형): ReplayFunc 추첨 순서와 JSONDatabase trace 재생

_6 · Matmul scheduling_

기본은 space(추정 DMA 순). schedule_search="meta_schedule"는 같은 후보 pool에서 sample_categorical로 번호를 뽑고, compilation(schedule_database=…)는 실행 조건이 같은 커널의 trace를 저장·재검증합니다. 학습형 cost model·latency 측정은 없습니다.

ReplayFunc 경로: TuneContext(mod=canonical PrimFunc, target=npu, space_generator=ScheduleFn(make_schedule, sch_rules=[], postprocs=[], mutator_probs={}), search_strategy=ReplayFunc(), rand_state=schedule_seed, num_threads=1)를 만들고 pre_tuning(max_trials=schedule_max_attempts, num_trials_per_iter=1)합니다. generate_measure_candidates마다 make_schedule이 sample_categorical(range(len(pool)), 균등)로 번호를 뽑아 그 mapping으로 schedule_matmul_sram을 실행합니다. 같은 번호가 다시 나오면 schedule을 다시 만들지 않고 "duplicate sample"로 시도만 소비합니다. postprocs를 비워 TVM 내부의 무제한 재시도를 막고, 거절은 공통 NPU 검사 경로에서 합니다. notify_runner_results에는 RunnerResult(None, None)을 주어 latency를 넣지 않고, 승자는 기존 (dma, words, sram) 평가기가 고릅니다. DB 경로: SelectedScheduleDatabase는 compiler source sha·TVM 버전/commit·contract·native library sha의 fingerprint로 디렉터리를 나누고 JSONDatabase(module_equality="structural")를 엽니다. workload는 global_symbol·op_pattern을 뺀 PrimFunc에 npu.selected_schedule_contract(profile에서 탐색 설정 joint_memory/beam/graph_search_seconds/candidates/trials/attempts/words/search/seed를 뺀 것 + scratch_base + slot_count + native_probe)를 붙인 module이며, 조회는 structural hash와 equal입니다. 기록은 TuningRecord(trace, workload, run_secs=None, target)이고 같은 workload의 최신 개선 기록을 사용합니다. 재사용 시 trace.apply_to_schedule → 공통 lowering → plan_sram → 용량 → probe codegen을 다시 거치며 실패하면 database_rejection을 남기고 원래 탐색으로 돌아갑니다. 작은 예제 3개에서 세 경로의 명령이 같고 C-model 출력이 bit 단위로 같았습니다. Attention 전체·일반 vector·W8A8은 DB 대상이 아닙니다.

근거: npu_compiler/npu_meta_schedule.py (replay_candidates, SelectedScheduleDatabase), npu_schedule.py _fit_matmul, evidence/current/meta_schedule_reuse.json, docs/current/06_matmul_schedule.md

## 43. 원소별 연산의 codegen: SiLU 하나가 ISA 하나가 아니다

_7 · Vector · 정규화 · attention_

VLEN=128인 한 행이 SRAM scratch slot을 거치며 SIGN_INV → EXP → BROADCAST(1.0) → ADD → DIV로 sigmoid가 되고, 마지막 곱은 별도 block입니다. 각 단계마다 descriptor·LOAD·SAVE가 붙습니다.

Walker는 innermost 축을 vector 행으로 보고 표현식 tree를 6개 scratch slot에 직렬화합니다(slot 폭 = 커널의 가장 넓은 행). 연속 load는 주소만, stride-0 load나 literal은 BROADCAST, binary op는 VLEN·ADDR·LOAD·ADDR·LOAD·OP·ADDR·SAVE 순서입니다. 그림의 각 행은 SRAM에 실제로 저장되는 FP16 128-원소 행이며 오른쪽 word가 그 단계의 실제 opcode입니다. E1 fused 프로그램의 opcode 통계가 이를 보여줍니다: SIGN_INV 32, EXP 32, ADD 96, DIV 32, MUL 32, BROADCAST 64, 주소 설정 0x80이 606 words.

근거: npu_compiler/tir_codegen_v09.py (_emit_pointwise, _materialize, _sigmoid), docs/examples/pipeline/09_packets.json words 440-468

## 44. 물리 256 lanes와 논리 VLEN, 그리고 generic vector scheduling

_7 · Vector · 정규화 · attention_

v09 규약은 물리 폭 256과 논리 길이 ≤65535를 구분합니다. Generic vector 커널은 tensor 전체를 SRAM에 놓아 보고 안 들어가면 행 묶음을 반으로 줄이며, 연속 원소별 축은 결합해 긴 행으로 만듭니다.

VLEN=1000이면 장치 내부에서 256+256+256+232로 처리한다는 전제이며 RTL 검증은 별도입니다. 오른쪽 위 그림: 64×64 tensor와 임시값 전체가 SRAM에 안 들어가면 32행, 16행으로 stripe를 반씩 줄이고 compute_at과 buffer compaction으로 각 묶음의 작업 공간을 줄입니다. 오른쪽 아래: 연속된 원소별 축을 표준 fuse/split으로 합쳐 64행×VLEN 64를 32행×VLEN 128로 만들면 명령 수가 절반이 됩니다(작은 예제의 npu.pointwise_vector_width=128). broadcast, 비연속 stride, 겹치는 입출력, 임시 공간 한도를 검사하고 행별 reduction은 축을 합치지 않습니다. 현재 비용 모델에는 lane 활용률이나 연산별 cycle이 없습니다.

근거: docs/current/07_operators.md, reference/execution_contract.md, npu_schedule.py (generic stripe halving, _coalesce_pointwise)

## 45. RoPE와 GQA: host가 cos/sin을 만들고 NPU는 곱셈·부호 반전·재배치만 한다

_7 · Vector · 정규화 · attention_

Head dimension 128을 앞/뒤 절반 (a | b)로 나누고 rotate_half = [−b | a]를 만든 뒤 x·cos + rotate_half(x)·sin을 계산합니다. Q/K에만 적용하고 V에는 적용하지 않습니다.

그림의 각 행은 한 token·한 head의 128 원소 벡터입니다(8칸으로 축약). IR에서는 split(64) → negative → concat → multiply(cos) / multiply(sin) → add로 표현되며 모두 vector 명령입니다. rope_freqs는 base^(−2i/hd)를 만들고 linear는 factor로 나누며 llama3는 wavelength에 따라 세 구간을 적용합니다. 결과를 [freqs, freqs]로 이어 rotate_half와 맞추고, rope_inputs가 float64로 각도를 계산해 FP16 cos/sin [S,1,hd]로 내립니다. validate_rope_scaling은 default/linear/llama3만 받습니다. GQA는 op.repeat(k, 3, axis=1)이며 persistent cache는 KV head 8개로 저장됩니다. decode에서는 FoldRepeatedGemv가 repeat를 broadcast matmul로 바꾸는 후보를 만들 수 있습니다(m=1 전용).

근거: npu_compiler/nn_models/llama.py (_rotate_half, rope_freqs, llama3_scale_freqs, validate_rope_scaling), npu_graph_opt.FoldRepeatedGemv

## 46. Attention을 연결해서 읽기, 그리고 full-fusion의 query 분할

_7 · Vector · 정규화 · attention_

Prefill head 하나: Q[S,128]×Kᵀ[128,S] → scores[S,S] → /√128 + mask → 행별 softmax → P×V[S,128]. 전체를 묶으면 query 행 block만 계산하고 그 block의 score 행 전체를 SRAM에서 소비합니다.

그림의 격자 한 칸은 64×64 tile입니다(S=512로 축약). 진한 행은 현재 query block이고 회색 삼각형은 causal mask로 기여가 0인 영역입니다. schedule_attention_blocked는 query 행을 block으로 split하고 reorder(row_o, head, row_i, …)로 row block을 head loop 바깥에 둡니다. 다른 모든 block은 head loop에 compute_at되고 scores [1,block,seq], max, exp, sum, 정규화 결과, PV accumulator는 set_scope로 SRAM에 둡니다. Q/K/V cache_read는 head에, mask(head 축 1 broadcast)는 row_o에 둡니다. _fit_attention은 가장 큰 block부터 시작해 실패하면 반으로 줄이고 floor min(rows,64)에서도 실패하면 LinkError입니다. 한 block이 score 행 전체를 갖고 있어 softmax는 정확하며 online rescaling(FlashAttention)이 없습니다.

근거: npu_compiler/npu_intrin.py schedule_attention_blocked, npu_schedule._fit_attention, docs/current/07_operators.md

## 47. 세 가지 메모리 계획이 서로 다른 범위를 본다

_8 · 메모리 계획_

① DRAM은 Relax tensor 수명으로 pool을 나누고 flat 주소를 주며, ② 커널 안에서는 USMP가 buffer offset을 겹치고, ③ 커널 사이에서는 보관값을 SRAM 위쪽에 예약합니다. 커널을 합치지 않아도 SRAM만으로 연결할 수 있습니다.

위 그림은 두 메모리의 실제 구성입니다. DRAM image에는 parameter leaf(변환된 순서), graph 상수, StaticPlanBlockMemory가 만든 storage pool(수명이 겹치지 않는 tensor가 공유), 출력 tree, literal pool이 있습니다. SRAM에는 literal·scratch, 커널의 USMP pool(buffer가 offset을 공유), 그리고 커널 사이 보관값이 위에서 아래로 예약됩니다. 세 계획은 서로 자동으로 해결되지 않습니다. USMP를 호출했다고 persistent KV cache나 전체 그래프의 보관 정책이 생기지 않습니다. compile_program 안의 실제 순서는 assign_addresses → validate_entry → scalar pool → plan_program(joint SRAM search) → 호출별 schedule/USMP/codegen → residency → peephole → baseline 비교 → compact_unused_storage → cache append 재구성입니다.

근거: docs/current/08_memory.md, npu_compiler/npu_link.py _compile_program (단계 순서)

## 48. 기본 DRAM 배치는 표준 TVM 결과에서 시작한다

_8 · 메모리 계획_

ToNonDataflow → RemovePurityChecking → CallTIRRewrite → StaticPlanBlockMemory. CallTIRRewrite가 allocation과 TIR 호출을 명시하고, StaticPlanBlockMemory가 동시에 살아 있지 않은 tensor의 pool 공유를 정합니다.

plan_pipeline은 RewriteDataflowReshape → OutputViews → ToNonDataflow → RemovePurityChecking → CallTIRRewrite → StaticPlanBlockMemory입니다. assign_addresses는 parameter(인자 순서, tuple leaf p.0.1)를 먼저, graph 상수(내용 키로 dedupe)를 다음, 그리고 binding을 프로그램 순서로 걸으며 alloc_storage→pool, alloc_tensor→pool_base+offset, builtin.alloc_tensor(함수 결과)→자체 slot, reshape/view→alias로 처리합니다. 아래 strip은 작은 예제 bundle의 실제 결과입니다: plan.address는 unit(2 B) 단위로 x 0, bias 8192, lv 8256, gv 16448이며 bytes로는 0 / 16,384 / 16,512 / 32,896입니다. 모든 allocation은 4-byte DMA cell로 반올림되고 literal pool은 linker가 마지막에 붙입니다(41,088).

근거: npu_compiler/npu_memplan.py (plan_pipeline, assign_addresses), docs/examples/pipeline/06_initial_dram_plan.py, bundle/program.json memory

## 49. 수명 분석: 실제 폭 MLP의 G/S/U/H는 언제까지 살아 있는가

_8 · 메모리 계획_

gate → SiLU → up → multiply → down (최신 실제 폭 MLP 측정의 순서). M=64에서 G/S/U/H는 각각 [64,8192] FP16 = 1 MiB입니다.

G는 SiLU가 끝나면 더 이상 필요하지 않고, S는 multiply까지, U도 multiply까지, H는 down까지 필요합니다. 표는 값의 관계를 설명하며 실제 peak에는 weight 조각, scratch, 정렬, 주소 사이 빈 공간도 들어갑니다. 살아 있는 tensor 크기의 합과 최고 주소는 같지 않을 수 있습니다. 이 순서는 최신 측정의 순서이며 과거 문서의 up→gate 순서를 현재 결과에 적용하지 않습니다.

근거: docs/current/08_memory.md, evidence/current/mlp64.json

## 50. USMP: 커널 내부 buffer의 충돌 분석과 greedy 배치

_8 · 메모리 계획_

작은 예제 fused 커널의 실제 결과. 8개 buffer(총 163,840 nibbles)가 충돌하지 않는 것끼리 offset을 공유해 pool 81,920 nibbles에 들어갑니다 (baseline bump 150,808 → peak 85,016 nibbles, 32,896 B 절약).

분석용 복사본에 LowerMatchBuffer/LowerOpaqueBlock을 적용하고 실행용 TIR은 바꾸지 않습니다. 각 Allocate에 WorkspacePoolInfo("npu_sram")를 붙여 extract_buffer_info로 충돌 정보를 얻고, offset을 buffer.data Var identity로 다시 연결합니다. gemm/fill 목적지(matmul_global.sram)와 알 수 없는 extern call의 pointer는 pinned로 모든 buffer와 충돌 처리되어 offset 0을 혼자 씁니다. 차트에서 같은 offset을 공유하는 쌍(T_add와 lv panel, compute_intermediate와 T_multiply, sigmoid 결과·bias·x panel)은 수명이 겹치지 않는 buffer들입니다. greedy_by_size와 greedy_by_conflicts를 돌려 더 작은 peak를 택하되 bump보다 좋을 때만 씁니다. 크기는 nibble(fp16=4)로 8-nibble(32-bit cell) 정렬입니다.

근거: npu_compiler/npu_sram.py plan_sram, docs/examples/pipeline/bundle/program.json sram_layouts

## 51. 커널 사이 SRAM 보관 탐색: 값의 버전을 찾고, 보관 공간을 예약한 뒤, 호출별로 다시 schedule한다

_8 · 메모리 계획_

npu_memory_search.plan_program. 보관값은 SRAM 위에서 아래로 예약되고, 커널 workspace는 그 아래 천장(cap)까지만 쓰도록 재schedule됩니다. beam width 4, all-DRAM 진행도 항상 비교. tensor 전체 보관만 있습니다.

discover는 PrimFunc 호출마다 인자의 byte 범위와 읽기/쓰기를 얻고, 출력 tree·parameter leaf·graph 상수 범위를 보호합니다. 후보는 FP16이고, 같은 호출에서 읽지 않고(RMW 아님), compact하며, 보호 범위·같은 호출의 다른 인자와 겹치지 않는 쓰기입니다. 이후 호출을 앞으로 훑어 부분 겹침·dtype 재해석, in-place 갱신, 부분 덮어쓰기는 거절하고 전체 덮어쓰기는 버전 종료로 봅니다. 오른쪽 그림처럼 reserve는 SRAM 위에서 아래로 first-fit이며 workspace는 항상 그 아래입니다. parameter_scopes가 bound parameter를 global.sram으로 바꿔 schedule이 cache stage를 생략하고 출력 tile을 직접 쓰게 합니다. 프로그램 수준에서는 all-DRAM baseline과 (dma, words, peak)를 비교해 엄격히 작을 때만 채택합니다.

근거: npu_compiler/npu_memory_search.py (discover, reserve, parameter_scopes, plan_program)

## 52. 작은 예제 ②: 두 matmul이 SRAM 44 KiB에서 H를 공유하는 실제 주소

_8 · 메모리 계획_

X[64,64] @ W1[64,128] → H[64,128]; H @ W2[128,64] → Y[64,64]. 단위 bytes. H는 DRAM에 저장되지 않고 두 번째 matmul이 이 위치를 직접 읽습니다.

SRAM 낮은 주소에는 literal pool과 scratch(6 slots × 128 × 4 nibbles)가 있어 pool base가 nibble 3096 = byte 1548입니다. 첫 커널은 X(8192 B)를 1548에, W1(16384 B)을 9740에, H(16384 B)를 26124에 둡니다. 두 번째 커널은 Y 작업 공간이 X 자리를, W2가 W1 자리를 재사용하고 H는 유지됩니다. 전체 peak는 42,508 B이며 용량 45,056 B(44 KiB) 안입니다. 탐색 기록: H는 처음 nibble 57344에 예약되었다가 arena를 2,548 B 내려 nibble 52248(byte 26124)로 이동했고, 두 커널 모두 bump 배치(USMP가 더 줄이지 못함)였으며 시도 4회였습니다.

근거: docs/examples/two_matmuls_recorded/program.json, scheduled_0000.py, scheduled_0001.py, docs/current/08_memory.md

## 53. 임시 출력 tile을 없애고 H에 바로 쓴다: view와 row stride

_8 · 메모리 계획_

H는 행마다 128 FP16 = row stride 256 B. 두 번째 열 tile의 시작은 H + 128 B이지 H + 8192 B가 아닙니다. 두 tile은 각 행의 왼쪽/오른쪽 절반이므로 stride를 유지해야 합니다.

이전에는 별도 64×64 출력 buffer에 SAVE하고 H로 복사했습니다. 현재는 출력 parameter가 SRAM에 보관될 때 forward_output_tiles가 fill+gemm intrinsic의 view를 출력에 직접 겨냥하고 identity copy를 제거합니다. scheduled_0000.py에서 compute가 scope global.sram의 match_buffer이고 C = match_buffer(compute[0:64, i1*64:…], strides=("C_s0",1))로 나타나며 attr npu.forwarded_output_tiles=1이 붙습니다. FP32 누산이 끝난 최종 저장 위치를 바꾸는 것이지 K 중간 partial sum을 H에 내리는 것이 아닙니다. E3에서 words 85→77, bit-exact, DMA/SRAM peak는 그대로였습니다.

근거: npu_compiler/npu_accumulator.forward_output_tiles, docs/examples/two_matmuls_recorded/scheduled_0000.py, evidence/current/refinement.json (E3)

## 54. 보관 후 남은 DRAM 예약 제거와 주소 재배치, 그리고 후단 residency forwarding

_8 · 메모리 계획_

SRAM-only 전달이 확정되면 아무 곳에서도 쓰지 않는 pool 전체를 제거하고 나머지 주소를 당깁니다. 두 matmul 예제에서 H의 DRAM pool 16,384 B가 사라지고 출력 gv가 57,344 → 40,960으로 이동합니다.

compact_unused_storage는 parameter leaf, 출력, const_data, literal pool, FP32 1.0, 그리고 모든 GLOAD/GSTORE의 strided span(빈틈 포함)을 반드시 남기고, cell 반올림 범위가 이들과 겹치지 않는 pool만 제거합니다. relocate는 제거된 pool 크기만큼 주소를 빼며 DMA g_addr word를 제자리에서 고치고 plan.address, pool_base, const_data, constant_base, one_fp32_base, top, 출력 offset, views, kernel_records(storage: sram/dram/unused), joint_memory, residency 기록을 갱신합니다. 오른쪽 아래 strip은 두 matmul 예제의 실제 주소입니다. npu_residency.forward는 codegen 뒤 명령 stream 위에서 GSTORE된 SRAM 사본을 추적해 커널 진입 시 GLOAD를 제거하거나 SRAM 복사로 바꾸는 write-through forwarding이며 생산자의 GSTORE는 남깁니다.

근거: npu_compiler/npu_memplan.compact_unused_storage, npu_residency.forward, docs/examples/two_matmuls_recorded/program.json (memory, dram_compaction)

## 55. Residual과 KV cache는 왜 다르게 취급되는가, 그리고 현재 메모리 계획의 경계

_8 · 메모리 계획_

Residual도 중간값이면 마지막 add까지 보관 후보입니다. KV cache는 다음 프로그램 호출에도 필요하고 ABI 입력/출력으로 관찰되므로 DRAM 상태를 유지합니다.

64×3072 FP16 residual이 384 KiB라는 사실만으로 반드시 보관됐다고 말할 수 없습니다. 사이의 모든 작업 공간과 비용을 봐야 합니다. KV cache는 kv라는 이름을 찾아 예외 처리하는 것이 아니라 외부 가시성 계약(출력 tree, parameter leaf)으로 보호됩니다. 현재는 빈 구간 재사용과 보관 영역 이동이 있지만 모든 조각화를 해결하는 전역 allocator는 아니며, 남은 낭비가 실제 프로그램 비용에 큰지 확인한 뒤 추가 개선을 검토합니다.

근거: docs/current/08_memory.md, npu_memory_search.discover (보호 범위)

## 56. Codegen: 주소가 연결된 scheduled TIR를 정적으로 전개된 명령 stream으로

_9 · ISA · linker · runtime_

Walker는 TIR을 직접 해석합니다. 모든 For는 완전히 unroll되고 loop/branch 명령은 없습니다. 아래는 두 matmul 예제의 두 번째 커널(K loop 2회)이 29 words의 직선 stream이 되는 실제 결과입니다.

Linker가 Walker(asm, bases, SramEmitter)를 만들고 parameter를 signature 순서로 2-byte 단위 base에 바인딩하며 SRAM buffer를 nibble로 선언합니다. visit는 For를 unroll하고 IfThenElse를 compile 시점에 결정하며 LetStmt는 거절합니다. _match_nest는 loop nest 전체를 잡아 출력 행마다 vector 명령 하나를 냅니다. 그림의 strip: GLOAD W2(4 words) → K tile 0의 descriptor·LOAD·M_MUL(MAC=0) → K tile 1(MAC=1) → DST descriptor·SAVE → GSTORE Y(4 words) → HALT. 네 부류의 emission이 있습니다: tensorized matmul, elementwise(6 scratch slot 직렬화), reduction(REDUCE_SUM/MAX + BROADCAST), 복사 stage(scope가 다르면 DMA, 같으면 LOAD/COPY/SAVE 또는 BROADCAST/strided LOAD). Python walker와 C++ native walker는 같은 계약을 따르며 커널 단위로 섞일 수 있습니다. 정적 전개이므로 Llama prefill은 8.58M words가 됩니다.

근거: npu_compiler/tir_codegen_v09.py (Walker.visit, _match_nest, _emit_pointwise, _emit_reduction, _emit_movement, emit_gemm), docs/examples/two_matmuls_recorded/program.json

## 57. 네 가지 이동 명령, 세 가지 주소 단위, descriptor 상태 모델

_9 · ISA · linker · runtime_

Descriptor는 다음 LOAD/연산/SAVE가 쓸 주소·shape·dtype "상태"입니다. 명령마다 배열 정보를 반복하는 대신 register를 바꾸고 뒤 명령이 참조합니다. 아래는 77-word 예제 두 번째 matmul 직전의 실제 register 값입니다.

H가 SRAM에 보관되면 GSTORE/GLOAD는 사라질 수 있지만 H를 matrix unit에 넣는 LOAD까지 없어지지는 않습니다. 한 숫자를 다른 단위로 읽으면 2배/4배 어긋난 주소가 됩니다. 0x80은 SRC1/SRC2/DST × MAIN/PARTIAL 주소의 16-bit 절반을, 0x88/0x89는 rows/cols와 dtype을, 0x82는 VLEN을 씁니다. MAIN은 전체 stride, PARTIAL은 이번 접근 영역의 시작과 크기입니다. 표의 값: SRC1 MAIN 0xCC18 = nibble 52,248 = byte 26,124(H), cols 128(행 stride 256 B); SRC2 MAIN 0x4C18 = 19,480(W2), cols 64; DST 0x0C18 = 3,096 = byte 1,548(Y). 두 번째 K tile에서 SRC1은 +0x100(64열), SRC2는 +0x4000(64행)만 바뀌고 COLS는 peephole이 생략합니다. region()은 operand 하나의 64×64 view를 위해 8 words를 냅니다.

근거: docs/current/09_isa.md, reference/execution_contract.md, npu_compiler/isa_compute.py, isa_v09.py, two_matmuls_recorded/program.json

## 58. 실제 명령 해독 ①: X를 SRAM으로 가져오는 4-word GLOAD packet

_9 · ISA · linker · runtime_

77-word 예제의 word offset 8: 000c18a0 00000000 00000800 00010800. DRAM의 X[64,64] FP16 8,192 B가 SRAM byte 1548에 도착합니다.

w0의 상위 24 bit는 SRAM nibble 주소(8-nibble 정렬), 하위 8 bit는 opcode입니다. 0xC18 = 3096 nibbles = byte 1548, 앞 슬라이드의 X 주소와 일치합니다. w1은 global cell 주소 0(= byte 0, X가 image 맨 앞), w2는 row stride 0x800 = 2048 cells, w3는 rows 1(상위 16 bit) / cols 0x800 = 2048 cells = 8192 bytes입니다. 1D DMA는 rows=1, cols=cells, stride=cols로 인코딩되고 0xFFFF cells/rows를 넘으면 분할됩니다. DMA는 dtype 없는 cell 이동이며 4-byte cell 경계의 부분 데이터는 이웃 값을 손상하지 않도록 별도 처리(bounce)합니다.

근거: docs/examples/two_matmuls_recorded/program_memory.bin, program.json, npu_compiler/isa_v09.py

## 59. 실제 명령 해독 ②: M_MUL의 MAC bit와 77-word 프로그램의 흐름

_9 · ISA · linker · runtime_

80000042와 88000042의 차이는 bit 27(MAC)뿐입니다. 둘 다 mode VECTOR, activation OFF. 아래 strip은 77 words 전체를 opcode 종류별로 칠한 것입니다.

Word 0-3은 상수 pool GLOAD, 4-47은 커널 0(첫 matmul: W1 GLOAD, X GLOAD, 두 출력 tile 각각 MAC=0 → H 직접 SAVE), 48-75는 커널 1(W2 GLOAD가 W1 자리에, K tile 두 개 MAC=0/1, Y SAVE, Y GSTORE), 76은 HALT입니다. strip에서 DMA packet(주황)이 4 words씩 차지하고 나머지는 descriptor(파랑)·LOAD/SAVE(청록)·M_MUL(보라) 1 word씩임을 볼 수 있습니다. Peephole 효과: 두 번째 tile(word 40)에서는 SRC1 descriptor가 통째로 사라지고 SRC2 low half만 바뀌며, 커널 1에는 0x88 word가 없습니다(rows=64가 커널 0에서 이어짐). opcode 통계는 0x42×4, 0x80×26, 0x88×6, 0x89×9, 0x90×8, 0x98×3, 0xA0×4, 0xA8×1, 0xFF×1입니다.

근거: docs/examples/two_matmuls_recorded/program.json, program_memory.bin, docs/current/09_isa.md, npu_compiler/isa_compute.py

## 60. Linker: 커널 stream을 호출 순서로 연결하고, forwarding·peephole·주소 압축으로 마무리한다

_9 · ISA · linker · runtime_

npu_link._compile_program. 커널은 독립적으로 emit되고(각자 필요한 descriptor를 다시 냄) 경계의 중복 설정은 뒤의 peephole이 제거합니다. 아래 strip은 77-word 예제의 실제 구성입니다.

순서: 계약 검사(tile 64, cell 32, numeric policy) → npu_memplan.assign_addresses → scalar pool(호출되는 PrimFunc의 FP16 literal + {1.0, 2.0}, 짝수 개로 padding, DRAM constant_base와 FP32 1.0, 첫 커널 전에 SRAM nibble 0으로 GLOAD, scratch_base 계산) → binding 순서대로 커널(joint search가 미리 emit한 words / W8A8 reference emitter / GlobalVar별 cache된 recipe를 _place_sram 후 native 또는 Python walker) → HALT → residency forwarding → peephole → joint vs all-DRAM baseline 비교 → compact_unused_storage → cache append. kernel_records의 span [4,48), [48,76)이 strip의 경계입니다. Peephole은 descriptor register 상태만 정확히 추적하며 LOAD/SAVE/연산/DMA/BROADCAST는 건드리지 않습니다. Forwarding의 효과 분석은 각 명령이 어떤 주소를 쓰는지(WriteEffects) 알아야 하며, 알 수 없는 opcode는 cache 전체를 비웁니다.

근거: npu_compiler/npu_link.py, peephole.py, npu_residency.py, v09_effects.py, two_matmuls_recorded/program.json kernel_records

## 61. Python codegen과 native C++ codegen의 일치, 그리고 통계의 정의

_9 · ISA · linker · runtime_

하나의 contract JSON이 양쪽 op table을 만들고 hash로 묶입니다. native의 unsupported는 Python fallback, invalid_schedule은 fallback 없는 LinkError, 내부 오류는 propagate합니다.

v09_contract.py는 JSON 바이트의 sha256을 CONTRACT_HASH로, npu_codegen/generate_contract.py는 같은 hash와 kMaxVlen, op X-macro를 v09_contract.h로 씁니다. build_identity.py는 .cc/.h를 정렬 순서로 hash해 NPU_BUILD_SOURCE_HASH로 export합니다. native.load()는 .so가 없으면 Python을 쓰고 hash 불일치면 RuntimeError("rebuild")를 냅니다. 모드는 use(기본, NPU_NATIVE=0로 비활성), compare(둘 다 실행해 첫 다른 word 보고), python입니다. v09_stats.analyze는 DMA를 4-word packet으로 읽어 words와 instructions를 구분하고, dma bytes를 Σrows·cols·4로, sram_peak를 커널 top의 최댓값/2로, image를 top·unit_bytes로 계산합니다. check_counters가 C-model counters와 대조합니다.

근거: npu_compiler/v09_contract.json, native.py, npu_codegen/build_identity.py, v09_stats.py, tests/test_native_codegen.py

## 62. 결과물: NpuExecutable과 export_bundle

_9 · ISA · linker · runtime_

Binary만으로는 inference 결과를 얻을 수 없습니다. 같은 plan에 맞는 입력과 weight image, 그리고 출력 tree의 shape/dtype/offset이 필요합니다.

NpuExecutable에는 words, plan, 낮춰진 module, 입력 signature(param_names)가 있습니다. export_bundle은 program_memory.bin(little-endian uint32), program.json(주소·통계·선택·출력 ABI·provenance), lowered.py(선택된 semantic IRModule), scheduled_*.py(joint-memory 경로의 호출별 TIR), values를 주면 global_memory.bin(초기 DRAM image)을 씁니다. 일반 recipe 경로의 단일 fused 커널은 자동 dump 대상이 아니므로 작은 예제 runner가 compile context의 recipe를 꺼내 07_scheduled_fused_matmul_add_silu.py로 남겼습니다. 전체 모델은 매우 큰 image와 pickle을 문서에 보관하지 않고 측정·hash를 남겼습니다.

근거: npu_compiler/npu_target.py NpuExecutable, docs/current/09_isa.md, docs/examples/pipeline/bundle/

## 63. Runtime 전체 흐름: generate()는 프로그램 2개를 한 번 링크하고 입력만 바꿔 반복한다

_9 · ISA · linker · runtime_

배포된 모델 = prompt 길이 S의 prefill_cache 1개 + 고정 capacity C의 decode 1개. 장치에 동적 shape가 없으므로 decode는 항상 C+1 슬롯을 attention하고 mask가 현재 길이를 전달합니다.

generate(family, config, assets, token_ids, steps, capacity)는 capacity(기본 S+steps−1)마다 family.build_generate로 세 entry를 export하고 _lower(lower_and_select, entries=("prefill_cache","decode"))로 링크합니다. weight는 한 번만 읽고 bucket 사이 parameter ABI가 같아야 합니다. prefill은 run_tuple_program(host transform → build_image → v09_runtime.run → read_outputs)이고 첫 token은 logits[-1]의 FP32 argmax입니다. host가 K [L,kv,hd,C], V [L,kv,C,hd] 배열을 만들어 prefill K/V S행을 복사하고, decode loop는 DecodeSession으로 매 step embedding·cos/sin·mask만 patch합니다. 매 step 새로 링크하지 않습니다.

근거: npu_compiler/npu_generate.py generate(), run_tuple_program, DecodeSession; d_compiler/run_nn_generate.py

## 64. Prefill 프로그램과 decode 프로그램: host와 NPU가 나누어 맡는 일

_9 · ISA · linker · runtime_

Prefill은 prompt S개 token을 한 번에, decode는 새 token 하나와 과거 cache를 입력으로 다음 logits를 만듭니다.

host: prompt token → embedding 행(assets.embedding은 전체 table을 올리지 않고 행만 읽음), cos/sin, causal mask [1,S,S] (−1e4, 과거 per-head mask는 S=2048에서 201 MB 전송·192 MiB image였음). NPU: 28 layers → 마지막 hidden의 norm/LM head → logits [1,V] + K/V. host: argmax → 첫 생성 token. 반복: 그 token의 embedding, 현재 position의 cos/sin, valid-length mask [1,1,C+1] (−65504) → NPU decode + cache append → argmax. Persistent cache shape는 K [28,8,128,C], V [28,8,C,128]입니다.

근거: docs/current/10_runtime_llama.md, npu_compiler/nn_models/llama.py (runtime_inputs, causal_mask), npu_generate.length_mask

## 65. 고정 shape와 유효 길이는 다르다: capacity C=9, 유효 길이 L=7

_9 · ISA · linker · runtime_

Decode는 기존 C개 슬롯과 현재 K/V 하나를 논리적으로 연결해 C+1 길이로 attention합니다. 현재 token의 임시 위치는 C, persistent cache에 기록할 위치는 첫 빈 슬롯 L입니다.

예를 들어 C=9, L=7이면 attention 입력은 [과거 0..6 | 빈 7,8 | 현재 token 9]이고 mask는 [0…0 | −65504 | 0]입니다. 의미상 제외할 슬롯도 shape와 명령에는 남아 있습니다. mask는 기여를 막을 뿐 계산의 동적 생략이 아닙니다. AppendPlan.position은 mask에서 유효 prefix 길이를 복원하며 하나의 유효 prefix + 현재 token 형태가 아니면 거절하고, cache가 가득 차면(L ≥ C) 오류입니다. DecodeSession은 position이 순차적이어야 함을 검사합니다.

근거: docs/current/10_runtime_llama.md, npu_generate.length_mask, npu_cache.AppendPlan.position

## 66. KV append는 누가 실행하는가: npu.cache_append 계약 → AppendPlan → DMA tail

_9 · ISA · linker · runtime_

K cache는 [128, C]로 transposed 저장이라 새 K(128×1)는 열 하나 = 행마다 2 B를 stride C·2 B로 쓰고, V cache는 [C, 128]이라 새 V는 연속 256 B 행 하나입니다. host는 position으로 주소를 정하고 전송은 NPU 명령이 합니다.

declare는 decode 함수에 {"version":1, "mask":"mask", "entries":[{"parameter":"k_cache","output":1,"axis":3}, {"parameter":"v_cache","output":2,"axis":2}]}를 붙입니다. 그림은 layer l, head h 하나의 cache입니다(28 × 8 쌍이 있음). K는 hd × C로 저장되어 slot L에 쓰는 것이 열 하나를 stride C·2 B로 쓰는 2D DMA이고, V는 C × hd라 행 하나를 연속으로 쓰는 1D DMA입니다. 행 bytes가 4의 배수가 아니면(홀수 C 등) cell RMW 경로를 씁니다. AppendPlan은 출력 tuple이 FP16이고 각 출력 shape가 cache shape에서 time 축만 1인지, source/destination이 alias하지 않는지, capacity가 일치하는지, mask가 FP16 [1,1,C+1]인지 검사합니다. parity 0/1 두 template를 미리 만들고 words(position)이 fixup으로 주소를 재배치합니다. 기록: 새 K/V 논리 크기 114,688 B, append DMA 458,752 B, decode base 7,310,801 words, 실행 stream 7,324,697 words.

근거: npu_compiler/npu_cache.py (declare, AppendPlan, _template, words), docs/current/10_runtime_llama.md

## 67. DecodeSession과 PersistentTransport: 무엇이 재사용되고 무엇이 바뀌는가

_9 · ISA · linker · runtime_

이전 실행의 최종 global image 파일을 다음 입력으로 이어 쓰고, 바뀐 host 입력과 program suffix만 patch합니다. 아래 두 strip은 step마다 실제로 바뀌는 바이트입니다.

DecodeSession은 transformed weights, compile_program(residency=True), AppendPlan, build_image, PersistentTransport를 만듭니다. run은 mask에서 position을 얻어 순차성을 검사하고, cache 입력이 초기화 뒤 다시 오면 거절하며, runtime 입력의 shape/dtype을 검사해 plan.address×2 위치의 patch 목록을 만듭니다(6,676 B). tail을 HALT 앞에 넣고 transport.run(words, patches)를 부릅니다. PersistentTransport는 program_memory.bin에서 바뀐 첫 word부터만 다시 쓰고(55,568 B), global_memory.bin에 patch를 적용한 뒤 subprocess를 실행하고 saved_global_memory.bin을 rename으로 다음 입력으로 만듭니다. C-model은 여전히 호출마다 전체 image(≈6.43 GB)를 읽고 씁니다.

근거: npu_compiler/npu_generate.DecodeSession, v09_runtime.PersistentTransport, docs/current/10_runtime_llama.md

## 68. 실제 검증된 3-token 생성: "Hello, NPU compiler!" → " I'm trying"

_9 · ISA · linker · runtime_

FP16, 28 layers, prompt 7 tokens, capacity 9, greedy 3 tokens. 보존된 기록이며 이번 문서 작업에서 전체 모델을 다시 실행하지 않았습니다.

tokenizer 결과는 [128000, 9906, 11, 452, 6459, 19979, 0]의 7 tokens입니다. prefill이 358(" I")을 예측하고, decode position 7에서 2846("'m"), position 8에서 4560(" trying")을 예측합니다. 마지막 4560은 아직 입력으로 처리하지 않았으므로 이 시점 cache에는 그 token의 K/V가 없습니다. HF와 argmax가 모두 일치하고 logits cosine은 0.99999 이상, 최대 절대차는 0.024~0.039입니다. 판정 기준은 유한 logits, token 일치, cosine ≥ 0.999이며 최대 절대차는 기록만 했습니다. 실행 948.99 s는 완료된 첫 entry를 재사용한 검증이므로 cold compile 시간이 아닙니다.

근거: docs/evidence/current/llama_validation.json, completion.json (compiler source sha256 44694fc6…)

## 69. Bucket, layer 반복, 그리고 C-model 실행 규약

_9 · ISA · linker · runtime_

여러 capacity를 미리 컴파일하고 유효 길이가 커질 때 큰 bucket으로 바꿀 수 있습니다. 전환 시 유효 prefix를 한 번 복사합니다. 28층은 최종 stream에 모두 전개되며 schedule 판단만 재사용합니다.

cache_buckets는 증가하는 capacity 목록이고 마지막이 capacity와 같아야 합니다. bucket(position)은 position보다 큰 첫 bucket이며 실제 필요한 bucket만 컴파일합니다(used). 그림: C=16 프로그램으로 position 15까지 진행하다가 유효 길이가 16이 되면 DecodeSession을 닫고 유효 cache prefix 16 slot을 C=32 배열로 복사해 새 session을 만듭니다. transitions에 valid_cache_bytes/input_cache_bytes를 기록합니다. 작은 context의 낭비(빈 slot 계산)를 줄이는 대신 프로그램 수, 초기 image, 전환 복사가 늘어납니다. C-model 파일 규약: program_memory.bin과 global_memory.bin을 읽고 SNAPSHOT(0xF0)마다 그리고 HALT(0xFF)에서 saved_global_memory.bin에 image를 덧붙이며 perf_counters.txt를 씁니다.

근거: npu_generate.generate (cache_buckets, transitions), v09_runtime.run, _poc/mysim_v09.cpp 파일 규약

## 70. Compile 시간은 어떻게 제어되는가: build 안의 cache, 예산, phase 계측

_10 · 실험 · 한계 · 정리_

compilation() context는 build 하나 동안 살아 있으며 nested 호출은 바깥 context를 재사용합니다. 모든 cache는 structural_hash로 찾고 structural_equal로 확인합니다.

recipe cache는 PrimFunc 구조(global_symbol/op_pattern 제외)와 profile(joint_memory/beam/graph_search_seconds 제외, sram_bytes 포함)·scratch·slot을 키로 schedule+SRAM layout을 재사용합니다. 28 layers에서 약 32개 커널이 1,206회 호출되어 호출별 scheduling이 link 시간의 64%였던 것이 도입 이유입니다. trial cache는 mapping별 측정(실패도 _RejectedTrial로)을, probe cache는 비용만(승자는 실제 주소로 재emit·검증), region cache는 fusion 영역 비용을, EntryCache(256 MiB words / 32 entries LRU)는 graph 후보 사이에 linked entry 전체를 재사용합니다. 예산은 첫 합법 module 뒤 arm되고 phase 진입마다 검사되며, 초과 시 OptimizationBudgetExceeded가 budget_exhausted로 기록됩니다. progress 이벤트는 phase_begin/end, candidate_begin/end(status), entry_compiled/reused 등입니다.

근거: npu_compiler/npu_compilation.py (recipe, trial, probe, region_cost, EntryCache, optimization_budget, phase, describe)

## 71. 실험 근거의 세 수준과 서로 다른 지표

_10 · 실험 · 한계 · 정리_

모든 숫자를 하나의 성능 점수로 합치지 않습니다. 컴파일 시간, 장치가 요청한 전송량, 명령 크기, 저장 공간, 수치 결과를 따로 측정합니다.

새 실행 검증은 이번 문서 작업에서 현재 소스로 작은 입력을 실제 실행한 것(E1, E2)이고, 보존된 실행 검증은 기존 로그·JSON·source hash를 그대로 보존한 것(E3, E6, E7)이며, 정적 분석/기존 단일 측정은 명령·배치 분석이나 한 번의 compile wall time(E4, E5)입니다. 작은 예제와 기존 최종 검증의 compiler source SHA-256은 44694fc6…로 같습니다. 중간 개선 단계 기록은 source hash가 다르며 현재 소스의 새 benchmark로 바꾸지 않았습니다.

근거: docs/current/11_experiments.md, docs/evidence/README.md, examples/reproduction.json

## 72. E1. Fusion은 DMA가 같아도 명령과 작업 공간을 줄일 수 있다

_10 · 실험 · 한계 · 정리_

SiLU(X[64,128] @ W[64,128].T + bias[64]), seed 914, FP16. 현재 public build로 fusion off/auto를 각각 컴파일해 같은 값으로 C-model을 실행했습니다. 두 구성 모두 SRAM 전달 기능을 사용합니다.

off는 3커널(matmul 43 words, add 391 words, silu 2,373 words), auto는 1커널 1,572 words입니다. DMA는 41,100 B로 같고(입력·weight·bias GLOAD 4개 + 출력 GSTORE 1개), words 2,812→1,577, SRAM peak 50,700→42,508 B입니다. Fusion 전후 FP16 비트가 같고 동일 수치 grouping의 LLVM 출력과 최대 절대차는 0입니다. C-model DMA counters(dma_cells_loaded 8227, stored 2048)가 정적 해독과 일치했습니다. 이 사례가 모든 fusion의 이득을 대표하지는 않습니다.

근거: docs/examples/pipeline/stage_summary.json (새 실행 검증)

## 73. E2. SRAM을 늘리면 언제 중간 DRAM 왕복이 사라지는가

_10 · 실험 · 한계 · 정리_

X[64,64] @ W1[64,128] @ W2[128,64], seed 928, fusion off. 보관 탐색 옵션과 SRAM 용량만 바꾸어 5개 구성을 실제 실행했습니다. 모든 구성이 같은 FP16 출력이며 DMA counters가 일치했습니다.

32 KiB에서는 보관 옵션을 켜도 첫 matmul(X 8 KiB + W1 16 KiB + H 16 KiB = 40 KiB + pool)이 들어가지 않아 선택 결과가 같습니다. 44 KiB에서 보관을 켜면 H 16 KiB의 저장·재읽기 합계 32,768 B가 사라져 DMA 81,932→49,164 B, words 87→77, image 65,548→49,164 B가 되고 SRAM peak는 34,316→42,508 B로 늘어납니다. 64 KiB로 늘려도 이 예제에는 추가 이득이 없습니다. 보관 탐색 off에도 기존 후단 residency/peephole은 유지되므로 모든 메모리 최적화를 켜고 끈 실험이 아닙니다.

근거: docs/examples/memory/measurements.json (새 실행 검증)

## 74. E3 · E4. 직접 SRAM 출력과 미사용 DRAM 제거: 작은 예제와 실제 폭 MLP

_10 · 실험 · 한계 · 정리_

E3는 44 KiB 두 matmul에서 이미 SRAM 보관을 사용하는 상태로 pointwise 결합·출력 직접 저장·미사용 DRAM 제거를 비활성/활성한 보존 비교. E4는 M=64 실제 폭 MLP의 정적 compile 기록(executed=false).

E3: words 85→77, DMA 49,164 그대로, SRAM peak 42,508 그대로, image 65,548→49,164, bit-exact. E2의 보관 off/on(87→77)과 다른 비교이므로 87→77과 85→77을 섞지 않습니다. E4: 선택 커널 5개(gate→SiLU→up→multiply→down) 그대로, words 143,404→140,594, DMA 152,174,604 그대로, SRAM peak 7,446,540 그대로, image 154,927,116→151,781,388. 서로 다른 중간값이 같은 기본 DRAM pool을 재사용하므로 제거량은 4 MiB가 아니라 3 MiB입니다. 과거 fused epilogue 명령이 매우 길었던 문제를 줄였어도 최종 선택이 fusion으로 바뀌지는 않았습니다.

근거: docs/evidence/current/refinement.json, mlp64.json, examples/two_matmuls_recorded/program.json

## 75. E5. Compile 속도 개선은 코드 품질을 바꾸지 않았다 (binary hash 동일)

_10 · 실험 · 한계 · 정리_

기존에 기록한 단일 wall-clock 관측값. 일부 측정 중 다른 검증도 실행됐습니다. 반복 평균이나 하드웨어 inference latency가 아닙니다.

M=7 실제 폭 MLP: 중복 실패 검사·시도 제한 개선으로 84.163 s → 22.073 s, TIR trial 생성 347→112회, 143,883 words와 binary hash 동일. M=64: 무제한 시도 → 64회 제한에서 21.791 s → 21.958 s, 두 경우 모두 47회 trial이므로 제한이 실제로 작동하지 않았고 140,594 words hash 동일. 모든 입력에서 일정 배율로 빨라진다는 결론은 낼 수 없습니다. 주요 개선은 invalid accumulator schedule을 native/Python에서 두 번 검사하지 않기, 동일 entry/region 비용 재사용, 실패도 포함하는 탐색 한도입니다. 전체 검증의 948.993 s는 cold compile 시간이 아닙니다(fresh_full_build_seconds=null).

근거: docs/evidence/current/compile_comparison.json

## 76. E6. 전체 Llama 3.2 3B 출력 검증과 프로그램 규모

_10 · 실험 · 한계 · 정리_

FP16, 28 layers, prompt 7 tokens, capacity 9, greedy 3 tokens. HF와 argmax가 모두 일치하고 logits cosine ≥ 0.99999입니다.

기본 entry 통계: prefill_cache 8,582,352 words / DMA 6,457,527,556 B / SRAM peak 8,374,292 B / image 6,429,587,576 B, decode base 7,310,801 words / 6,438,980,676 B / 7,342,100 B / 6,427,349,032 B. Decode 실제 실행에는 append tail이 추가되므로 base 통계와 tail 포함 counters를 혼합하지 않습니다. DMA의 대부분은 layer마다 weight 192 MiB × 28 + LM head 751.5 MiB ≈ 6 GB를 panel로 읽는 것입니다. graph 탐색은 시간 한도에 도달해 unfused 기준을 선택했으며 "모든 후보 중 전역 최적"이라는 의미는 없습니다.

근거: docs/evidence/current/llama_validation.json, llama_selection.json, completion.json

## 77. E7. 기능 검증 gate: 17개 suite가 확인하는 축

_10 · 실험 · 한계 · 정리_

기존 최종 gate는 17개 검사 묶음이 실패/skip 없이 완료되었습니다. 묶음 안에 여러 assertion이 있으므로 "17개 shape만 테스트했다"는 뜻이 아닙니다. 이번 문서 작업에서는 재실행하지 않았습니다.

run_gate.sh는 실제 subprocess exit code, timeout, skip을 validation.json으로 기록하며 [PASS] 문자열만 찾아 성공으로 판정하지 않습니다. 가장 긴 suite는 nn_decode(207 s)이고 pipeline_extensions 68.6 s, compiler_contracts 44.6 s, priority_contracts 37.3 s, npu_target 34.8 s입니다. 선택 검사는 --tests로 일부 suite만, --require-no-skips로 skip 금지, --output으로 로그 위치를 지정합니다.

근거: docs/evidence/current/regression_validation.json, gate_logs/, d_compiler/run_gate.sh

## 78. E9. 탐색을 보완해도 512-token down의 최종 명령은 같았다

_10 · 실험 · 탐색 개선_

실제 Llama layer-0 weight + 합성 activation. 단일 projection의 비교이며 전체 모델 generation 검증이 아닙니다.

후보 정규화, 직접 FP16 입력에 한정한 SRAM 초과 사전 제외, 기준 words를 반영한 예산을 적용한 뒤 두 방식 모두 추가 합법 후보4개를 검사했습니다. Space10.586초, ReplayFunc33.264초이며 최종 binary는 같습니다. 이 후보 범위에서 기존 schedule보다 나은 정적 비용을 얻지 못했습니다. C-model은 최대 절대차0.000030517578125와 DMA counters 일치를 확인했습니다. DB reuse는 약4초이며 탐색 전략과 예산을 바꾸어도 호환되는 trace를 재사용합니다. 관련 회귀 검사34개가 통과했습니다. 전체 탐색공간의 최적성 또는 실제 장치 latency를 주장하지 않습니다.

근거: docs/current/11_experiments.md E9; evidence/current/llama_schedule_refinement.json

## 79. E10 중단: 큰 cache fusion의 명령 생성 병목을 수정했다

_10 · 실험 · 전체 Llama 재검증_

28 layers 전체 · prefill_cache 하나 · graph 시간 제한 None · matmul 합법 목표16 / 최대 시도128 · SRAM 8 MiB · 전체 재검증 미실행

Attention off/score/full 3종과 generic off/vectors/matmul 3종입니다. 이 prefill에서는 GQA·producer 설정이 동일 IR이어서 source 중복을 제거한 뒤9개가 남습니다. 표는 일반 fusion 영역 함수 종류이며 layer별 호출 수나 최종 채택 수가 아닙니다. 영역 검사 후 풀리는 fusion과 추가 중복도 가능합니다. 분리된 커널도 SRAM 보관과 직접 출력을 사용하므로 공정하게 비교합니다. Graph 시간제한은 없지만 matmul 시도128·합법목표16·words예산10억, 메모리 beam4이므로 모든 schedule/보관 조합의 완전탐색은 아닙니다. 같은 계산·메모리 조건은 재사용합니다. HF last logits와28-layer K/V reference는 준비됐고 compile은 과도한 메모리로 중단했고 C-model 비교는 미실행입니다. 생산자 반복 배치 거절·직사각형 행 분할·길이1 DMA 처리를 수정하여 독립 검사4개와 회귀54개가 통과했습니다. 비용이나 fusion 채택, NPU token일치 결과는 아직 확정하지 않습니다.

근거: docs/current/11_experiments.md E10; evidence/current/llama_prefill256_full_search_progress.json

## 80. 현재 지원 경계: 무엇을 받아들이고 무엇을 거절하는가

_10 · 실험 · 한계 · 정리_

ABI verifier는 정의·호출·배치의 일관성을 검사하지만 임의 TIR의 모든 index 접근이 안전함을 증명하는 정리 증명기는 아닙니다. 지원 subset 밖의 계산을 조용히 통과시키지 않습니다.

정적 shape만 지원하며 FP16 matmul은 rank≥2, batch broadcast, unit/ragged 축을 처리하고 rank-1은 명시적 reshape가 필요합니다. control flow는 지원된 정적 호출을 직선 stream으로 전개하며 임의 dynamic control flow나 packed call은 거절합니다. softmax는 마지막 축 1..65504, RMSNorm은 마지막 축·FP16·정적 width·지원 epsilon, attention full-fusion은 정적 rank-3 FP16 64배수, RoPE는 default/linear/llama3, cache는 FP16 정적 capacity와 additive mask, runtime은 C-model adapter입니다. 보장하지 않는 것: 실장치 latency, 모든 동적 shape, 모든 TVM op, 모든 NaN/Inf edge semantics, 모든 HF 모델, 자동 mixed precision, quantized KV, key online softmax, 임의 inter-kernel tile streaming.

근거: docs/current/12_operations_limits.md, reference/execution_contract.md

## 81. 옵션과 API: 무엇을 켜고 끄며, 오해하기 쉬운 점은 무엇인가

_10 · 실험 · 한계 · 정리_

npu_target.build(module, func_name, fusion, residency, profile) 와 NpuProfile 필드. 각 옵션은 한 단계만 바꾸며 다른 최적화를 함께 끄지 않습니다.

fusion auto는 합법 기준과 여러 graph 후보를 비교하지만 전역 최적을 보장하지 않습니다. fusion off는 attention/generic fusion만 끄고 메모리 최적화는 유지합니다. fusion on은 pattern/generic fusion을 요청하며 지원되지 않는 복합 kernel은 실패할 수 있습니다. joint_memory는 중간 FP16 tensor 전체 보관과 schedule을 함께 선택하고, reuse_sram은 커널 내부 수명 재사용(USMP)입니다. graph_search_seconds는 전체 compile timeout이 아니고, schedule_max_attempts는 64개의 성공 후보를 보장하지 않습니다. cache_buckets는 미리 만든 capacity별 decode 선택이지 매 step 임의 shape의 동적 실행이 아닙니다. schedule_search/schedule_seed/schedule_probe_word_budget과 compilation(schedule_database=…, schedule_database_mode=…)는 09-29에 추가·정리된 항목이며 기본 동작(space, DB 없음)을 바꾸지 않습니다. DB는 중첩 context에서 지정할 수 없고 버전 fingerprint별 하위 디렉터리를 씁니다.

근거: docs/current/12_operations_limits.md, npu_compiler/npu_target.py, device.py

## 82. 오류가 나면 어느 단계부터 보는가

_10 · 실험 · 한계 · 정리_

관찰한 증상을 단계의 결정에 대응시키면 확인할 파일과 예제가 정해집니다. Native/Python 일치는 계약 준수의 근거이지 모델 수식 오류까지 배제하지 않습니다.

LLVM부터 HF와 크게 다르면 모델 수식, weight 순서, mask/RoPE, 정밀도 정책을 봅니다. 작은 K는 맞고 큰 K가 틀리면 FP32 accumulator 연결과 부분합 순서, 첫 tile만 맞으면 MAIN stride·PARTIAL 주소·단위, SRAM 보관을 켤 때만 틀리면 값 버전·마지막 소비·alias·덮어쓰기, 주소 압축 후 틀리면 DMA와 output/constant/append 주소 일관성, prefill은 맞고 decode가 틀리면 현재 token·position·mask·append 슬롯, words가 너무 많으면 짧은 vector·주소 설정 반복·정적 전개, compile이 오래 걸리면 cache hit/miss·거절 이유·phase 시간·종료 한도를 확인합니다.

근거: docs/current/12_operations_limits.md

## 83. 남은 이슈: 구현 착수 목록이 아니라 검토 목록

_10 · 실험 · 한계 · 정리_

효과가 작거나 custom 복잡도가 큰 변경은 원하지 않습니다. 기존 구현을 유지한 상태에서 필요성과 규모를 먼저 평가합니다. 이번 문서 작업에서 최적화를 추가 구현하지 않았습니다.

각 항목은 현재 한계와 재개 시 먼저 확인할 것을 짝지어 둔 것입니다. 특히 vector 256/긴 VLEN은 RTL 계약 확인이 먼저이고, attention online softmax와 메모리 퇴거/부분 tile은 큰 변경이라 설계부터 분리합니다. 성능 모델은 RTL/장치 timing을 확보한 뒤 DMA→words→SRAM 비교와의 상관을 확인해야 합니다.

근거: docs/current/12_operations_limits.md

## 84. 정리: 작은 예제 하나가 지나온 길 — 계산에서 실행까지

_정리_

SiLU(X[64,128]·Wᵀ+bias)가 각 단계에서 무엇으로 바뀌었는지. 같은 흐름이 실제 Llama 28 layers에서는 8.58M words의 prefill과 7.31M words의 decode가 됩니다.

① authored Relax 4연산 → ② graph prefix로 transpose가 host로 (prefill(x, bias, lv) + prefill_transform_params) → ③ LegalizeOps로 call_tir 3개 + PrimFunc(FP32 누산 정의) → ④ fusion 후보 비교로 fused_matmul_add_silu 채택(words 2,812→1,577) → ⑤ schedule: a_at=b_at=m_panel, X/W 16 KiB 1회 복사, M_MUL 2개(MAC 0/1), SiLU vector 전개 폭 128 → ⑥ USMP greedy_by_conflicts peak 85,016 nibbles, 보관할 중간값 없음 → ⑦ codegen/link: 1,577 words, DMA 41,100 B, 5 packets → ⑧ runtime: image 41,100 B, C-model 실행, 출력 offset 32,896에서 [64,64] FP16 읽기, LLVM reference와 최대 절대차 0.

근거: docs/examples/pipeline/*, two_matmuls_recorded/*, evidence/current/*

## 85. 기존 main(ver.08 / source-0818)과 최신 v09는 실행 경계가 다르다

_부록 A1 · 기존 main backend_

main commit 1143abbc…로 고정해 설명합니다. 둘의 words나 실행 시간을 숫자만 비교해 성능 순위를 만들 수 없습니다: 컴파일 단위, 메모리 경계, 실행기, codegen 세대가 다릅니다.

main은 Relax graph를 수작업 연산별 emitter로 연결하고 flat G-buffer와 host의 실행 경계를 사용하며, 같은 layer 프로그램을 host가 weight를 바꾸어 반복 실행하고, context별 정적 decode 프로그램과 host가 반환된 K/V로 배열을 확장하는 방식입니다. LM head는 K×64 packed panel로, 종료는 0xF0 snapshot입니다. 당시 결과: Llama 3.2 3B에서 같은 token [358, 2846, 4560], 최종 decode cosine 약 0.9999881, context8 decode 프로그램 453,349 words(layer 프로그램 통계). Gemma 4 E2B, Qwen3-4B도 main backend에서 HF 일치 기록이 있으나 v09 generation CLI에서 재검증된 것은 아닙니다.

근거: docs/main/README.md, pipeline.md, results.md

## 86. 양자화 W8A16 / W8A8: 저장 dtype과 연산 dtype을 구분한다

_부록 A2 · 양자화_

Pass와 커널은 구현되어 있지만 최근 전체 28-layer 연속 생성 검증은 FP16입니다. 양자화 generation까지 완성·검증됐다고 이해하지 않도록 본문과 분리합니다.

W[K,N]의 열 j마다 sW[j]=max(|W[:,j]|,tiny)/127, Wq=clip(round(W/sW),−127,127) INT8, half-to-even, zero-point 없는 대칭 양자화, weight per-output-channel, W8A8 activation per-row입니다. 양자화 rewrite는 LiftTransformParams 전에 적용되어 scale/quantize가 host로 옮겨집니다. W8A16은 DRAM INT8 W + FP16 scale → SRAM에서 FP16 복원 → 기존 FP16 GEMM chain(mixed matrix mode 미사용)입니다. W8A8은 실행 중 행별 absmax·FP32 scale·VQUANT로 INT8 X를 만들고 64개 K마다 INT32 부분합에 sW·sX를 적용해 FP32 accumulator에 더한 뒤 FP16 저장입니다. 검증: W8A16 matmul, W8A8 [64,64]@[64,64], [7,128]@[128,192] bit equality, 작은 모델 cosine ≈ 0.999998. run_nn_generate/npu_generate.generate에는 양자화 선택이 연결되어 있지 않습니다.

근거: docs/current/13_quantization.md, npu_compiler/npu_quantize.py, npu_w8a8_schedule.py, evidence/current/quantization.log

## 87. TVM API를 동작 기준으로 읽기: 무엇을 받아 무엇으로 바꾸며, 아직 무엇을 정하지 않는가

_부록 A3 · TVM API 사전_

설치된 TVM 0.19.0 기준. 이름을 외우기보다 각 API가 결정하는 것과 결정하지 않는 것을 구분합니다.

IR을 만드는 API(export_tvm, BlockBuilder, TensorStructInfo, te.placeholder/compute/reduce_axis, call_te, call_tir, tvm.script), graph pass(Sequential, CanonicalizeBindings, EliminateCommonSubexpr, RewriteDataflowReshape, LiftTransformParams, LegalizeOps, FoldConstant, AnnotateTIROpPattern, FuseOps, FuseOpsByPattern, FuseTIR, DeadCodeElimination), TIR 후처리(LowerInitBlock, PlanAndUpdateBufferAllocationLocation, ConvertBlocksToOpaque, CompactBufferAllocation, LowerMatchBuffer), 메모리(ToNonDataflow, RemovePurityChecking, CallTIRRewrite, StaticPlanBlockMemory, USMP), 검증(structural_hash/equal, relax.build llvm + VirtualMachine)로 나뉩니다. CanonicalizeAttention, FoldRepeatedGemv, 양자화 rewrite, OutputViews는 우리 모듈의 변환이며 TVM 기본 pass가 아닙니다. 선택형 MetaSchedule API(Workload/TuningRecord/JSONDatabase, Schedule.trace, TuneContext/ScheduleFn/ReplayFunc, sample_categorical, extract_tasks)는 6장 절과 같이 기존 NPU 비용 기준을 유지한 채 저장·재생과 검사 순서 변경에만 사용합니다.

근거: docs/reference/tvm_api.md

## 88. 재현 환경과 명령

_부록 A4 · 재현_

작은 예제부터 시작하고 소스·환경·설정을 기록합니다. 기존 측정값이 현재도 같을 것이라고 가정해 재사용하지 않습니다.

TVM 0.19.0 (commit c4dc0c29…), LLVM 18.1.8, Python 3.11.15, conda 환경 npu-tvm이 기준입니다. native build 명령은 library가 없거나 소스와 맞지 않을 때 필요합니다. reproduce.py는 작은 compile 7개 구성과 C-model 실행을 하며 전체 Llama나 checkpoint 다운로드는 하지 않습니다. export_llama_ir.py는 큰 weight를 읽거나 codegen하지 않고 IR만 만듭니다. 재현 결과는 보존된 문서 예제를 덮어쓰지 않도록 별도 디렉터리에 둡니다. 예제 출력이 달라졌을 때 문서 숫자만 먼저 바꾸지 말고 source/profile/numeric policy와 선택 후보를 확인합니다.

근거: docs/current/12_operations_limits.md, docs/examples/README.md, d_compiler/toolchain.lock.json

## 89. 마무리

질문을 받습니다. 세부 IR·API·실험 근거는 docs/README.md의 13장과 examples/, evidence/에서 확인할 수 있습니다.
