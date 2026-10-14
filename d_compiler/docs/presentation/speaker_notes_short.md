# NPU compiler 발표자 노트 (요약판)

13장 요약판의 슬라이드별 설명과 검증 범위.

## 01. 표지

13장 요약판입니다. 각 장은 그림 하나로 그 단계의 실제 동작을 보여주고, IR은 2~4줄 요약만 둡니다. 예제는 SiLU(X[64,128]·Wᵀ+bias), 두 matmul 44 KiB, Llama 3.2 3B(S=7, C=9)이며 과거 측정과 현재 탐색 개선을 구분하고 E10은 메모리 증가로 중단됐으며 전체 재검증은 아직 수행하지 않았습니다.

## 02. 한 장으로 보는 전체 pipeline: 같은 계산이 단계마다 다른 모습이 된다

_전체 pipeline_

작은 예제 SiLU(X[64,128]·Wᵀ+bias)의 실제 산출물로 본 5단계. 후보 graph마다 ③④⑤까지 내려가 (DMA bytes, words, SRAM peak)를 얻고 그 비용으로 선택합니다.

① 모델을 relax.frontend.nn으로 export하면 연산 호출과 shape만 있는 Relax 그래프가 됩니다. ② graph prefix가 weight만 쓰는 transpose를 host 함수로 옮기고, LegalizeOps가 각 연산의 원소 계산(FP32 누산, FP16 cast)을 TIR PrimFunc로 정의합니다. ③ FuseOps 영역을 실제 명령 비용으로 검사하고, matmul은 panel·순서·cache 위치를 탐색해 64×64 intrinsic으로 tensorize합니다. ④ DRAM은 TVM StaticPlanBlockMemory의 pool을 flat 주소로, SRAM은 USMP가 충돌하지 않는 buffer의 offset을 공유하고 커널 사이 보관값을 위쪽에 예약합니다. ⑤ walker가 TIR을 완전히 unroll해 descriptor·LOAD/SAVE·M_MUL·DMA packet의 직선 stream을 만들고, runtime은 DRAM image와 함께 C-model에 넘겨 출력 offset에서 결과를 읽습니다. 후보 비교는 ⑤의 통계로 ③으로 되돌아갑니다.

근거: docs/examples/pipeline (01~09), bundle/program.json; npu_target.build → npu_build.lower_and_select → npu_link.compile_program

## 03. 장치: DRAM ↔ SRAM ↔ 연산장치, 데이터는 panel과 64×64 tile로 움직인다

_① 장치_

weight는 DRAM에서 panel 단위로 SRAM에 올라오고(GLOAD), 64×64 tile이 연산장치로 들어가(LOAD) FP32 accumulator 한 개에서 K를 누산한 뒤 FP16으로 SRAM에 내려오며(SAVE), 결과가 DRAM으로 나갑니다(GSTORE).

DRAM(global)은 DMA만 접근하며 32-bit cell 단위 주소를 씁니다. SRAM은 8 MiB, nibble 주소입니다. 연산장치는 64×64 matrix unit(FP32 accumulator 1개)과 256-lane vector unit입니다. GLOAD/GSTORE는 DRAM↔SRAM, LOAD/SAVE는 SRAM↔연산장치 이동이라 서로 다른 경계입니다. SRAM 구성(literal·scratch, 입력 panel, 출력 tile·임시, 커널 사이 보관값)은 메모리 계획의 실제 배치 순서입니다. 한 숫자를 다른 단위로 읽으면 2배/4배 어긋난 주소가 됩니다. C-model 통과는 기능 검증이며 RTL 검증을 대신하지 않습니다.

근거: docs/current/01_overview.md, reference/execution_contract.md, npu_compiler/device.py

## 04. 모델을 Relax로 쓰고, weight만 쓰는 계산은 host로 옮긴다

_② 계산의 의미_

Llama 한 layer를 tensor 모양으로 본 그래프(왼쪽). 오른쪽: LiftTransformParams가 W transpose를 host 함수로 분리해 Wᵀ가 DRAM image의 정해진 주소에 한 번 놓이고, device 함수는 token마다 입력만 바꿉니다.

relax.frontend.nn으로 정의한 Llama는 export_tvm으로 정적 shape의 Relax IRModule이 됩니다(prefill S=7, decode C=9). Attention은 Q/K/V projection, RoPE(Q,K), GQA repeat(8→24 head), QKᵀ·scale·mask·softmax·PV, O projection과 residual이고 MLP는 gate/up([7,8192], weight 48 MiB씩)·SiLU·곱·down과 residual입니다. graph prefix(CanonicalizeAttention, CanonicalizeBindings, EliminateCommonSubexpr, RewriteDataflowReshape, LiftTransformParams)가 parameter만 쓰는 permute_dims를 host 함수로 옮기고, runtime 준비 단계에서 LLVM VM이 이를 1회 실행해 DRAM image에 넣습니다. 반환 순서가 (bias, lv)로 바뀌므로 runtime은 signature로 binding합니다.

근거: nn_models/llama.py, docs/examples/llama_ir/01_authored_relax.py, pipeline/02_graph_prefix.py, bundle/program.json

## 05. 수치 계약: FP16 저장, FP32 누산 — K 전체 동안 accumulator를 끊지 않는다

_② 계산의 의미_

LegalizeOps가 각 Relax 연산의 원소 계산을 TIR로 정의합니다(Relax 호출 순서는 남음). matmul은 128개 곱을 64개씩 두 FP32 부분합으로 같은 accumulator에 더하고 마지막에 한 번만 FP16으로 반올림합니다.

NPU legalize map은 matmul(FP16×FP16→FP32 sum→FP16), softmax(max·exp·FP32 sum·FP16 sum·divide, 마지막 축 1..65504), rms_norm(absmax로 먼저 scale해 x² overflow 방지), sum/mean, split(view)만 바꾸고 나머지는 TVM 기본 legalization입니다. 장치는 K=64 단위 부분합을 만들므로 CPU reference도 split+rfactor로 같은 grouping을 쓰며, 중간에 FP16으로 저장했다 다시 더하는 schedule은 "FP32 matrix chain interrupted"로 거절됩니다. E1에서 같은 grouping의 LLVM 출력과 최대 절대차 0. 정밀도 정책은 고정이며 오차 예산 자동 탐색은 없습니다.

근거: npu_compiler/npu_legalize.py, npu_numeric.py, docs/examples/pipeline/03_unfused_semantic_tir.py, 08_llvm_reference.py

## 06. Fusion은 커널 수가 아니라 실제 프로그램 비용으로 고른다

_③ 배치_

작은 예제의 분리(off) vs 결합(auto): 두 경우 모두 중간 tile P·A가 SRAM에 머물러 DMA는 같고(41,100 B), 커널 경계의 descriptor·LOAD/SAVE 반복과 짧은 vector 행이 words와 SRAM 작업 공간을 바꿉니다.

TVM FuseOps가 op_pattern(matmul 4 → add 0 → silu 0)으로 영역을 만들면 npu_fusion.select가 영역마다 unfused 기준과 fused 후보를 실제 schedule·메모리·codegen까지 내려 (DMA, words, SRAM) 사전식으로 비교하고 진 영역은 inline합니다. 그 위에서 lower_and_select가 attention(off/score/full) × GQA·producer × generic(off/vectors/matmul) 최대 27개 graph 후보를 entry 전체 비용으로 비교합니다. 채택 조건은 모든 entry 비악화 + 하나 이상 개선이며 기본 추가 탐색은 120초이고 None으로 해제합니다. 과거 7-token 전체 Llama 기록에서는 후보 0(모두 off)만 legal·selected, 후보 1 검사 중 예산 소진, 25개 미검사였습니다. 이는 "모든 fusion이 더 나쁘다"의 증거가 아닙니다.

근거: npu_compiler/npu_fusion.py, npu_build.py, docs/examples/pipeline/stage_summary.json (E1), evidence/current/llama_selection.json

## 07. Matmul: 64×64 tile은 고정, 자유도는 panel(무엇을 SRAM에 얼마나 오래)과 순서

_③ 배치_

실제 폭 gate projection X[64,3072]@W[3072,8192]: W는 48 MiB라 X(1×48 tiles, 384 KiB)를 상주시키고 W 열 panel j(48×1 tiles)를 128번 교체합니다. 아래는 두 matmul 예제 두 번째 커널의 실제 20 words.

탐색 축은 m/n panel, K chunk, outer/inner 순서, X/W cache 위치(a_at/b_at)이며 K는 항상 최내측입니다(accumulator 1개). 후보는 추정 DMA 순으로 검사하되 실제 tir.Schedule 변환 → CompactBufferAllocation → USMP 배치 → 실제 명령 생성으로 합법성과 (DMA, words, SRAM)을 확인하며 기본 시도64·legal8 목표입니다. 동일 후보를 정규화하고 증명 가능한 SRAM 초과만 미리 제외합니다. Words 예산은 max(설정값, 기준 words×legal 목표)이며 최소 두 합법 후보 전에는 words 때문에 종료하지 않습니다. 시도 한도는 별도입니다. TVM trace DB 재사용과 ReplayFunc 탐색은 선택형입니다. E10은 시도128·legal16 목표로 확장합니다. gate/up은 ("A",64) input-stationary(X 유지, W panel 384 KiB 교체, 6,144 M_MUL), down은 ("B",256) weight-stationary(W 4 MiB 유지, H는 SRAM 보관값)가 선택됐습니다. 아래 word strip은 X₀·B₀(MAC=0) → X₁·B₁(MAC=1) → SAVE의 실제 명령이며 두 번째 tile에서는 주소 descriptor만 바뀝니다(COLS는 peephole이 생략).

근거: npu_schedule_space.py, npu_schedule.py, npu_intrin.py, evidence/current/refinement.json, two_matmuls_recorded/program.json (words 52–71)

## 08. Matmul 이외: SiLU 하나가 vector 명령 5개, attention은 query block으로 score를 SRAM에서 소비

_③ 배치_

원소별 연산은 VLEN 길이의 행이 SRAM scratch slot을 거치며 단계마다 descriptor·LOAD·SAVE가 붙습니다(왼쪽, 실제 opcode). Full-attention fusion은 query 행 block마다 K/V 전체와 score 행 전체를 SRAM에 두고 정확한 softmax를 합니다(오른쪽).

Walker는 표현식을 6개 scratch slot에 직렬화합니다. sigmoid는 SIGN_INV → EXP → BROADCAST(1.0) → ADD → DIV이고 SiLU의 마지막 곱은 별도 block입니다. 중간값은 FP16 SRAM 행이라 SiLU 하나가 ISA 하나가 아니며, generic vector 커널은 전체가 안 들어가면 행 묶음을 반으로 줄이고 연속 축을 결합해 행을 길게 만듭니다(64×64 → 32×128). Attention full fusion은 정적 rank-3 FP16 64배수 조건에서만 가능하고 row block을 head loop 바깥에 두어 block마다 모든 head를 처리하며, 실패하면 block을 반으로 줄입니다. key까지 나누는 online softmax(FlashAttention)는 없습니다. decode(m=1)는 조건 밖이라 score chain fusion 또는 분리 matmul을 씁니다.

근거: tir_codegen_v09.py (_materialize, _sigmoid), docs/examples/pipeline/09_packets.json, npu_intrin.schedule_attention_blocked, npu_schedule._fit_attention

## 09. 메모리 계획: DRAM pool, 커널 안의 offset 공유, 커널 사이 SRAM 보관

_④ 메모리_

① TVM StaticPlanBlockMemory가 수명이 겹치지 않는 tensor를 같은 pool에 두고 backend가 flat 주소를 줍니다. ② USMP가 커널 buffer의 충돌을 분석해 offset을 공유합니다. ③ 보관 탐색이 중간값 H를 SRAM 위쪽에 예약하고 남은 용량으로 커널을 다시 schedule합니다.

왼쪽 아래는 작은 예제 fused 커널의 실제 USMP 결과입니다: 8개 buffer(총 163,840 nibbles)가 충돌하지 않는 것끼리 offset을 공유해 81,920 nibbles에 들어가며(bump 150,808 → peak 85,016), gemm 목적지는 pinned로 offset 0을 혼자 씁니다. 오른쪽은 두 matmul 예제(SRAM 44 KiB)의 실제 주소입니다: 첫 커널이 H를 26,124에 직접 SAVE하고 두 번째 커널이 그 위치를 직접 LOAD하므로 H의 GSTORE/GLOAD 32,768 B가 사라져 DMA 81,932 → 49,164 B, 미사용 DRAM pool 제거로 image 65,548 → 49,164 B가 됩니다. 32 KiB에서는 첫 matmul이 안 들어가 보관이 적용되지 않습니다. 보관은 tensor 전체만, beam 4, all-DRAM 진행도 항상 비교하며 이득이 실제 비용으로 확인될 때만 채택합니다. KV cache와 출력·parameter는 ABI 계약으로 보호됩니다.

근거: npu_memplan.py, npu_sram.py, npu_memory_search.py, bundle/program.json sram_layouts, two_matmuls_recorded/program.json, examples/memory/measurements.json (E2)

## 10. ISA: 정적으로 전개된 명령 stream — descriptor 설정, LOAD/SAVE, M_MUL, DMA packet

_⑤ 실행_

두 matmul 예제의 77 words 전체(위)와 실제 word 해독(아래). walker가 TIR loop를 완전히 unroll하므로 loop·branch 명령이 없고, linker가 커널 stream을 이어 붙인 뒤 peephole·forwarding·DRAM 압축으로 마무리합니다.

word 0-3 상수 pool GLOAD, 4-47 커널 0(W1·X GLOAD, 두 출력 tile 각각 MAC=0 → H 직접 SAVE), 48-75 커널 1(W2 GLOAD가 W1 자리, K tile 두 개 MAC=0/1, Y SAVE, GSTORE), 76 HALT. DMA packet은 4 words(w0 SRAM nibble 주소+opcode, w1 global cell, w2 stride, w3 rows/cols)이고 나머지는 1 word입니다. Descriptor(0x80 주소, 0x88/0x89 rows/cols·dtype, 0x82 VLEN)는 상태 register라 뒤 명령이 참조하며, peephole은 값이 같을 때만 중복 설정을 제거합니다(커널 1에 0x88 없음). M_MUL word의 bit 27이 MAC이며 80000042(MAC=0)와 88000042(MAC=1)만 다릅니다. Python walker와 native C++ walker는 같은 contract hash를 공유하고 words·DMA bytes·SRAM peak 통계와 C-model counters를 대조합니다. 결과물 bundle: program_memory.bin, program.json(주소·통계·ABI·provenance), lowered.py, scheduled_*.py, global_memory.bin.

근거: two_matmuls_recorded/program_memory.bin, program.json; tir_codegen_v09.py, v09_walker.cc, npu_link.py, peephole.py, isa_v09.py

## 11. Runtime: prefill 1개 + decode 1개 프로그램, cache는 DRAM에 남고 새 K/V는 DMA tail이 슬롯에 쓴다

_⑤ 실행_

Decode는 capacity C개 슬롯 + 현재 token으로 항상 C+1 길이를 attention하고 mask가 유효 길이를 전달합니다. host는 embedding·cos/sin·mask만 patch(6,676 B)하고, program 끝의 append tail이 새 K/V를 슬롯 L에 옮깁니다.

generate()는 prefill_cache(S=7)와 decode(C=9) 두 프로그램을 한 번 링크하고 weight를 한 번 읽습니다. prefill 뒤 host가 argmax로 첫 token을 고르고 K/V S행을 capacity 배열 앞에 복사합니다. 각 decode step에서 host는 token embedding, position의 cos/sin, length_mask(유효 슬롯 0, 나머지 −65504, 마지막 현재 token 0)를 image의 정해진 주소에 patch합니다. K cache는 [128,C]로 transposed 저장이라 새 K(128×1)는 열 하나(행마다 2 B, stride 18 B)이고 V cache는 [C,128]이라 새 V는 연속 256 B 행입니다. AppendPlan이 position으로 DMA 주소를 fixup한 tail을 HALT 앞에 넣고, PersistentTransport가 program suffix와 입력만 patch한 뒤 C-model을 실행하고 saved image를 다음 입력으로 rename합니다. 기록: "Hello, NPU compiler!" → 358, 2846, 4560 = " I'm trying", HF argmax 일치, cosine ≥ 0.99999.

근거: npu_generate.py (generate, DecodeSession, length_mask), npu_cache.AppendPlan, v09_runtime.PersistentTransport, evidence/current/llama_validation.json

## 12. 검증된 것과 아직 없는 것

_검증 · 경계_

작은 예제는 실제 C-model 실행에서 fused/unfused bit-exact, LLVM reference와 최대 절대차 0. 전체 Llama 3.2 3B는 HF와 같은 greedy token 3개, logits cosine ≥ 0.99999. 아래 수치는 모두 보존 기록이며 ISA words나 DMA를 실장치 latency로 바꾸어 말하지 않습니다.

E1(fusion off/auto)과 E2(SRAM 32/44/64 KiB)는 이번 문서 작업에서 현재 소스로 실제 실행한 것이고, E3(직접 저장 85→77 words), E6(전체 Llama), E7(17개 회귀 suite)은 보존된 실행 기록, E4(실제 폭 MLP)와 E5(compile 84→22 s, binary 동일)는 정적 분석·단일 관측입니다. 지원 경계: 정적 shape, 지원된 정적 호출의 직선 전개, softmax 마지막 축 1..65504, RMSNorm FP16, full-attention은 rank-3 64배수, RoPE default/linear/llama3, FP16 정적 capacity cache, C-model runtime. 보장하지 않는 것: 실장치 latency, 동적 shape, 모든 TVM op/HF 모델, 자동 mixed precision, quantized KV, key online softmax, inter-kernel tile streaming. 양자화(W8A16/W8A8)는 kernel bit equality까지만이고 generation에는 미연결입니다.

근거: docs/current/11_experiments.md (E1~E7), 12_operations_limits.md, evidence/current/*

## 13. 09-29 탐색 개선: 중복 후보 정규화, 증명된 용량 초과 제외, 적응형 words 예산 — MetaSchedule은 선택형

_탐색 개선 · 전체 모델 실험_

같은 tile 순회·복사 주기의 표기는 한 후보로 합치고(512-token down 22,236 → 5,034), 두 입력 cache 합이 8 MiB를 넘는 1,596개를 증명으로 제외한 뒤 3,438개를 검사 목록에 둡니다. 기본 탐색은 space, ReplayFunc 추첨과 trace DB는 옵션입니다.

① loop_spec은 같은 축을 panel·tile 두 loop로 나누므로 n_panel=64/128/256은 모두 N tile을 같은 순서로 도는 표기입니다. 크기 1 loop를 빼고 같은 축의 이웃 loop를 곱한 signature로 합치되, cache 복사 위치(prefix)가 다르면 별도 후보로 둡니다. toy [128,256,128]은 loop 위치 조합 1,728 → 후보 73입니다. ② 두 operand cache는 matrix 명령 시점에 동시에 살아 있으므로 그 합이 하한입니다. 정렬 rank-2 FP16 직접 입력 2개(64배수, 함수 parameter)일 때만 (SRAM − scratch prefix)를 넘는 후보를 제외하고, padding·computed producer는 순서만 뒤로 보냅니다. ③ words 한도는 max(100,000, 기준 words × legal 목표 8)이며 legal 2개 전에는 words로 끝내지 않고, 시도 64회(실패 포함)는 별도입니다. ④ space는 추정 DMA 순, meta_schedule은 TVM ReplayFunc가 sample_categorical로 뽑은 순서로 같은 pool을 검사하며 둘 다 실제 SRAM 배치·명령 생성으로 (DMA, words, SRAM)을 얻습니다. JSONDatabase에는 run_secs=None인 선택된 trace만 저장하고 reuse는 재생·재검증, tune은 incumbent로 두고 개선될 때만 기록합니다. 실제 Llama 4개 커널에서 두 방식의 최종 binary는 같았고(E8), 512-token down은 개선 후 두 방식 모두 4회 검사·합법 4개로 이전과 같은 명령이었습니다(E9). ⑤ E10은 28-layer·256-token prefill 하나를 graph 시간 제한 없이 컴파일하다 메모리 증가로 중단한 실험이며, GQA·producer 변환이 같은 IR이라 source 3개 × generic 3 = 9 구성입니다. 비용·fusion 채택·출력 일치는 완료 전까지 주장하지 않습니다.

근거: npu_schedule_space.py, npu_schedule.py, npu_meta_schedule.py; evidence/current/llama_schedule_refinement.json (E9), llama_prefill256_full_search_progress.json (E10)
