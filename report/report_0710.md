# NPU 0710 ISA 반영 보고서 — 우리 소스 반영(Phase 1 완료) & 컴파일러 재타겟 계획

> 작성일: 2026-07-10
> 대상: 2026-07-10 NPU 업데이트(`0710_npu_update/`, 박형철 ver.07)
> 선행 문서: `report/update_0710.md`(업데이트 분석), `d_compiler/RETARGET_0710.md`(실행 계획)
> **이 문서는 처음 보는 사람도 이해하도록 배경부터 서술한다.**

---

## 0. 한 문장 요약

NPU 하드웨어가 대규모 업데이트(strided load/save, matmul MAC/activation, native
reduce/broadcast 등)됐고, 그 새 명령어들을 **우리 시뮬레이터·인코더(`isa.py`,
`mysim.cpp`)에 완벽 반영해 벤더 시뮬레이터와 전체 예제 byte-exact 일치까지 검증**했다
(Phase 1 완료). 이어 **컴파일러 재타겟(Phase 2)** 에서 reduce-sum·transpose·row-broadcast를
네이티브 명령으로, softmax를 (reduce-max 우회 fold-max 기반) **stable softmax**로 바꿨고,
전체 테스트·byte-exact 회귀를 통과했다. (§7.5)

---

## 1. 배경 — 무엇이, 왜 바뀌었나

우리는 이 NPU(c-model)를 타깃으로 LLM(Llama)을 컴파일하는 컴파일러를 만들어 왔다.
초기 NPU는 명령이 제한적이라, 우리가 **소프트웨어로 우회**해야 했던 오버헤드가 많았다:
- 넓은 행렬의 타일을 읽을 때 연속 메모리만 되어 **per-row 복사(gather)**,
- 결과를 넓은 행렬에 쓸 때 **per-row 복사(scatter)**,
- matmul 부분합 누산을 위한 **save·load·add**,
- reduce/broadcast를 **ones-matmul**로 흉내, 등.

이번 2026-07-10 업데이트는 **우리 연구실(성균관대)과 서울과기대가 요청한 HW 기능들을
벤더가 반영**한 것이다. 즉 위 우회들의 상당수가 **하드웨어 명령**으로 직접 지원된다.
(자세한 신구 비교는 `report/update_0710.md`.)

### 1.1 새로 생긴 핵심 명령 (opcode)
| 명령 | opcode | 우리 우회를 대체하는 것 |
|---|---|---|
| **strided load / save** | 0x90/0x98 의 비트[29] | gather / scatter (+ K^T transpose) |
| **matmul MAC** (C += A·B) | 0x42 의 비트[28] | K-accumulate(save+load+add) |
| **matmul activation** (SiLU) | 0x42 의 비트[29] | 별도 활성화 elementwise |
| **reduce-sum** | 0x14 | ones-matmul 합산 |
| **broadcast** | 0x15 | ones-matmul 브로드캐스트 |
| **exp / cos / sin** | 0x0F / 0x18 | softmax·RoPE 우회 |
| **sign-inversion / copy** | 0x16 / 0x17 | RoPE rotate_half |

---

## 2. 우리가 다루는 두 개의 "시뮬레이터"

혼동을 막기 위해 정리한다.
- **`_poc/mysim.cpp`** — **우리가 소스를 가진** c-model(157줄). 우리가 빌드해서 테스트에
  쓴다. `--gout`으로 결과를 파일로 뽑는 편의 기능이 있어 컴파일러 테스트에 유리.
- **`0710_npu_update/a_npu/a.out`** — **벤더가 새로 준 c-model 바이너리**(소스 비공개).
  새 명령을 모두 실행하는 **권위 있는 정답(오라클)**. 단 stdout으로만 출력하고,
  최신 libstdc++(GLIBCXX_3.4.32)가 필요.

**전략**: 우리 `mysim.cpp`를 새 ISA로 확장하고, **벤더 `a.out`을 정답지로 삼아
byte-exact 대조**한다. 그러면 우리 테스트 루프(빠른 `--gout`)를 유지하면서 정확성을
벤더 기준으로 보장할 수 있다.

> **다행인 점**: 두 시뮬레이터의 입력 포맷이 **이미 동일**하다 — `program_memory.bin`
> (32-bit little-endian + 개행), `G_buffer_data.bin`(FP16 little-endian). 그래서 파일
> 포맷 변환 없이 곧바로 대조 가능했다.

---

## 3. 실행 계획 (전체 로드맵)

| Phase | 내용 | 상태 |
|---|---|---|
| **0** | 새 a.out 실행 환경·포맷 호환 확인 | ✅ 완료 |
| **1a** | `isa.py`(인코더)에 새 명령 추가 | ✅ 완료·검증 |
| **1b** | `mysim.cpp`(실행기)에 새 명령 의미 구현 | ✅ 완료 |
| **1c** | 우리 mysim == 벤더 a.out **byte-exact** 검증 | ✅ **완료(63/63)** |
| **2** | **컴파일러 재타겟**(새 명령을 emit) | ▶ 다음 |
| **3** | M1~M6 재검증 + 재측정 + 리포트 갱신 | 대기 |

이하 §4~§6은 **이미 완료한 Phase 0·1**의 상세이고, §7이 **다음(Phase 2·3) 계획**이다.

---

## 4. Phase 1a — 인코더(`isa.py`) 확장 & 검증

### 4.1 무엇을 했나
`isa.py`는 파이썬으로 32-bit 명령어를 만드는 **인코더**다(실행기 아님). 조사해 보니
우리 `isa.py`는 **이미 이 명령 포맷**(0x80/0x82/0x88/0x90/0x98, 0x42, vector ops)을
쓰고 있었다 — 즉 **처음부터 다시 짤 필요 없이 "확장"** 이면 됐다. 추가한 것:
- `load`/`save`에 **strided** 파라미터(비트[29] + `ncols`[23:16] + `start`[15:8]),
- `matmul`에 **MAC** 비트[28],
- 신규 인코더 `reduce_sum(0x14)`·`broadcast(0x15)`·`sign_inv(0x16)`·`copy(0x17)`·`cos/sin(0x18)`.
- (하위호환: 기존 호출은 기본 인자로 그대로 동작)

### 4.2 검증 — 벤더 예제와 인코딩 byte-exact
벤더가 제공한 예제 프로그램들(`b_program/inst_*/program_memory.bin`)의 모든 32-bit
워드에 대해 **`reencode(w) == w`**(우리가 필드를 분해→재조립해도 동일)임을 확인.
신규 명령 포함 전부 통과 → **우리 인코더의 비트 레이아웃이 벤더와 정확히 일치**.

---

## 5. Phase 1b/1c — 실행기(`mysim.cpp`) 확장 & byte-exact 검증 ★

### 5.1 새 명령의 "실제 의미"를 어떻게 알아냈나
벤더 `a.out`을 각 예제에 대해 돌려 **입력(PE_in)·출력(PE_out) 트레이스**를 관찰하고,
그 의미를 역공학했다. (테스트용 G-buffer는 `G[i]=i`라 값·주소를 눈으로 추적하기 쉬움.)

| 명령 | 관찰된 의미 |
|---|---|
| **reduce-sum** | 벡터 `[5,6,7,8,9,10,11,12]` → **`[68]`**(합 하나) |
| **broadcast** | 스칼라/즉값을 vlen 길이로 채움 → `[5,5,5,…]` |
| **sign-inversion** | `[5,6,…]` → `[-5,-6,…]` |
| **copy** | 항등 복사 |
| **cos/sin** | `pout[i]=cos/sin(pin[i])` |
| **strided load** | 소스 `R0×R1`에서 `[start, start+ncols)` 열을 **열-major(=전치)로** 로드. 예: 2×3 `[[5,6,7],[8,9,10]]`의 열0,1 → `[5,8,6,9]` |
| **strided save** | pout(열-major)를 dest의 해당 열들에 기록 (전치 저장의 역) |
| **matmul MAC** | 결과를 덮어쓰지 않고 **누산**(`C += A·B`) |

가장 중요한 발견: **strided load는 "열-블록을 전치해서 읽는다"** — 그래서 `Q@K^T`가
`load Q + strided-load K + matmul`로 되고, 별도 transpose가 사라진다(벤더도 "성균관대
요청 stride load로 transpose 구현"이라 명시).

### 5.2 `mysim.cpp`에 반영
위 의미대로 `_poc/mysim.cpp`에 구현:
- `0x90/0x98`에 strided 분기(열-major 로드/세이브),
- `0x42`에 MAC 비트(`pout[i*cB+j] += a`),
- 신규 `0x14/0x15/0x16/0x17/0x18`,
- elementwise 길이를 **로드된 타일 크기(pin1.size())** 기준으로(strided 대응),
- **활성화 함수를 SiLU(`x·sigmoid(x)`)로 교체** — 벤더가 기존 `x²·sigmoid(x)`가
  부적합하다며 표준 SiLU로 바꿨기 때문(이걸 안 고쳐 `_w_act` 예제 5개가 처음엔 실패).

### 5.3 검증 — 전체 예제 byte-exact
`b_program`의 **모든 63개 예제**에 대해, 각 예제의 program을 우리 mysim과 벤더 a.out에
각각 돌려 **PE_in/PE_out 데이터흐름을 대조**:

```
PASS=63 FAIL=0   → ★ 전체 b_program byte-exact 일치 ✓
```

재현: `bash d_compiler/validate_isa_0710.sh` (벤더 a.out은 `LD_LIBRARY_PATH`에
conda npu-tvm의 libstdc++ 필요).

→ **"우리 소스에 완벽 반영"이라는 목표를 정량적으로 달성**(벤더 c-model과 동작 동일).

---

## 6. 현재까지 변경/추가된 파일

| 파일 | 변경 |
|---|---|
| `npu_compiler/isa.py` | 새 명령 인코더(strided load/save, MAC, reduce-sum·broadcast·sign-inv·copy·cos/sin) + decode |
| `_poc/mysim.cpp` | 새 명령 실행 의미 + strided/​MAC + **SiLU 활성화** |
| `d_compiler/validate_isa_0710.sh` | 벤더 a.out 대비 전체 byte-exact 검증 스크립트(신규) |
| `d_compiler/RETARGET_0710.md` | 전체 실행 계획(신규) |
| `report/update_0710.md` | 업데이트 신구 비교 분석(신규) |
| `report/report_0710.md` | 이 문서(신규) |

---

## 7. 다음 단계 — Phase 2: 컴파일러 재타겟 (계획)

이제 **컴파일러(`codegen.py`/`tir_backend.py`/`memplan.py`/`model.py`)가 새 명령을
emit**하도록 바꾼다. 매핑:

| 기존(우리 SW 우회) | → 새 명령 |
|---|---|
| gather (per-row 복사) | **strided load** (No.cols/start로 타일 직접) |
| scatter (per-row 복사) | **strided save** |
| K^T transpose | **strided load**(전치 로드) — 전치 캐시/permute 제거 |
| K-accumulate (save+load+add) | **matmul MAC 비트** |
| 활성화 elementwise | **matmul activation 비트** |
| ones-matmul reduce/broadcast | **reduce-sum / broadcast** |
| RoPE rotate_half | **sign-inversion + copy** (+ native cos/sin) |
| softmax (max 생략) | **exp** + **fold-max(min/max)** 기반 stable softmax |

**재평가 대상(측정 후 결정)**: 우리 기존 최적화 — **가중치 패킹(−91.8%)·O-proj
융합·활성화 gather 재사용·전치 KV캐시** — 는 strided load/save·MAC가 흡수하므로
**상당수 불필요**해질 수 있다. Phase 3에서 재측정해 제거/유지를 정한다.

### Phase 3 (재검증·재측정)
- `tests/test_decode.py` M1~M6 재수행(새 ISA/mysim에서 정합).
- `analyze_kernels`/`analyze_pack` 재측정 → 새 오버헤드 프로파일(gather/scatter/accum/
  transpose 소멸 반영; 기존 "gather 93%"는 구 ISA 기준이었음).
- 리포트 갱신.

---

## 7.5 Phase 2 — 진행 상황 (컴파일러 재타겟, 완료분)

계획대로 `codegen.py`/`legalize.py`를 새 명령으로 재타겟했다. **완료·검증된 항목**:

**(a) reduce-sum 네이티브화** — `emit_row_sum`을 ones-matmul → **네이티브 reduce-sum(0x14)**
로 교체. 행마다 `vlen(C); load(행); v_reduce_sum; save`. RMSNorm의 `mean(x²)`와
softmax의 rowsum이 이 경로를 탄다.

**(b) transpose 네이티브화** — `emit_transpose`를 per-element 복사(O(R·C)) →
**strided load(0x90 bit[29])** 로 교체. `[rt,C]` 블록을 열-major로 읽으면 전치 블록
`[ct,rt]`가 **명령 1개**로 나온다(`v_copy`로 pin1→pout 이동, `rt==R`이면 dst가 연속이라
save 1회). K^T가 여기서 흡수된다.

**(c) stable softmax + reduce-max 우회** — `softmax_lastdim(stable=True)`로 **max 차감**
추가. 하드웨어에 **reduce-max가 여전히 없어**, `emit_row_max`를 **vector-max(0x12)
fold**로 구현: 각 열 j를 strided load로 (전체 R행 동시에) 읽어 running acc에 `v_max`.
→ O(C) 벡터-max(누산은 R lane 병렬)로, O(R·C) 아님. `s−rowmax`로 exp 오버플로 제거.

**(d) row-broadcast 네이티브화** — `[1,C]→[R,C]`는 **native copy(0x17)** 로 소스 행을
full-address load 후 복제.

### ★ 중요한 함정 — 네이티브 broadcast의 16-bit 주소 한계

**col-broadcast(`[R,1]→[R,C]`)에 native broadcast(0x15, SCALAR)를 처음 썼다가
MEDIUM 레이어에서 NaN**이 났다. 원인: **SCALAR 피연산자 주소는 명령의 16-bit
즉치(immediate) 필드**로 인코딩된다(`cst=(instr>>8)&0xFFFF`). load/save는 set-addr(0x80,
lo+hi)로 **full 32-bit** 주소를 쓰지만, **scalar-즉치 연산은 하위 64K만** 가리킨다.
MEDIUM에서 `off[rowmax]=143872 > 65535`라 주소가 잘려 **엉뚱한 값을 broadcast** →
max 차감이 무력화 → exp 오버플로 → NaN.

→ **결론**: native broadcast(0x15)는 **즉치 상수 채우기**엔 유용하나, 큰 버퍼(>64K)에
있는 **텐서 값의 per-row broadcast엔 부적합**. col-broadcast는 full-address를 쓰는
**ones-matmul 외적(col[R,1]@ones[1,C], degenerate K=1)** 으로 유지했다. reduce-sum/
reduce-max/transpose는 전부 load/save(full-address)만 쓰므로 이 한계와 무관.

### 검증
- **전체 테스트 통과**: `test_isa/matmul/rmsnorm/swiglu/elementwise/tiling/runtime/layer/
  tir_backend/import/attention/real_layer/decode(M1~M6)` — MEDIUM/REDUCED 정합, 3B 컴파일 성공.
- **byte-exact 회귀 유지**: `validate_isa_0710.sh` = **PASS=63 FAIL=0**.
- NaN 원인(위 함정)은 **디버깅으로 최초 NaN 바인딩(broadcast 출력)까지 추적**해 확정.

### 남은 Phase 2 항목 (미착수)
- **gather/scatter → strided load/save**, **K-accumulate → matmul MAC 비트**,
  **활성화 → matmul activation 비트**(SiLU), **RoPE rotate_half → sign-inv+copy(+cos/sin)**.
  이들은 TIR matmul 백엔드(`tir_backend.py`) 변경이 필요해 별도로 진행한다.
- 기존 최적화(가중치 패킹·O-proj 융합·전치 캐시) **재측정 후 제거/유지 결정**.

---

## 8. 결론

- **Phase 1(우리 소스 반영) 완료·검증**: `isa.py`·`mysim.cpp`가 새 ISA를 지원하고,
  벤더 c-model과 **전체 63개 예제 byte-exact 일치**.
- **Phase 2 일부 완료·검증**: reduce-sum·transpose·row-broadcast 네이티브화 + **stable
  softmax(reduce-max는 vector-max fold로 우회)**. 전체 테스트·byte-exact 회귀 통과.
- **얻은 교훈**: native broadcast(0x15)의 SCALAR 소스는 **16-bit 즉치 주소**라 >64K
  버퍼엔 못 쓴다 → col-broadcast는 full-address ones-matmul 유지. (load/save만 full 32-bit)
- 남은 Phase 2: gather/scatter→strided, K-accum→MAC, 활성화→matmul act, RoPE→sign-inv
  (모두 TIR 백엔드 변경). 남는 순수 SW 과제는 **명령 수 최소화(HW 루프 부재)**.
