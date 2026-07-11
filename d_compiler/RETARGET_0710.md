# NPU 0710 ISA 반영 — 재타겟 실행 계획

> 목표: 2026-07-10 NPU 업데이트(strided load/save, matmul MAC/activation, native
> reduce/broadcast/exp/cos-sin/sign-inv/copy)를 **우리 소스에 완벽 반영**한 뒤,
> **컴파일러를 새 ISA로 재타겟**한다. 배경: `report/update_0710.md`.

## 조사 결과 (착수 전 확정 사실)
- 우리 `isa.py`는 **이미 이 instruction format**(0x80/0x82/0x88/0x90/0x98, 0x42, vector ops)을 쓰고 b_program과 byte-exact. → **확장만** 필요.
- 우리 `_poc/mysim.cpp`(157줄, 우리 소스, `--gout` 지원)는 새 명령(0x14~0x18, strided[29], MAC[28])을 **아직 실행 안 함** → 확장 필요.
- 벤더 `a_npu/a.out`(신규, 권위 오라클)은 **conda libstdc++로 실행**(`LD_LIBRARY_PATH=.../npu-tvm/lib`), 입력 `program_memory.bin`(32-bit LE + `\n`)·`G_buffer_data.bin`(fp16 LE) **포맷이 우리와 동일**, 출력은 **stdout 트레이스**(--gout 없음).
- **전략**: 우리 `mysim.cpp`+`isa.py`를 새 ISA로 확장하고, **벤더 a.out을 오라클로 byte-exact 검증**. 우리 `--gout` 테스트 루프 유지.

## 확정 인코딩 (b_program 소스 검증)
| 명령 | 인코딩 |
|---|---|
| reduce-sum | `0x14` (로드된 벡터를 합산) |
| broadcast imm | `(0<<30)+(imm<<8)+0x15` |
| broadcast scalar | `(1<<30)+(addr<<8)+0x15` |
| sign-inversion | `0x16` |
| copy | `0x17` |
| cos / sin | `(0<<27)+0x18` / `(1<<27)+0x18` (operator=bit[27]) |
| matmul MAC | `0x42` + `(1<<28)` (activation=`(1<<29)`) |
| strided load | `0x90` + `(1<<29)` + `(ncols<<16)` + `(start<<8)` |
| strided save | `0x98` + `(1<<29)` + `(ncols<<16)` + `(start<<8)` |

---

## Phase 0 — 환경/오라클 (거의 완료)
- [x] 새 a.out 실행 확인(conda libstdc++). 포맷 호환 확인.
- [ ] `run_oracle.sh` wrapper: `LD_LIBRARY_PATH=<conda> a.out` in cwd + stdout 파싱. (검증용)

## Phase 1 — ISA/시뮬레이터를 우리 소스에 반영
### 1a. `isa.py` 확장 (인코더)
- `enc_load`/`enc_save`에 `strided`,`ncols`,`start` 파라미터 추가(비트29·23:16·15:8).
- `enc_m_mul`(및 _enc_matrix)에 `mac` 비트[28] 추가.
- 신규: `enc_reduce_sum()`, `enc_broadcast(mode,imm)`, `enc_sign_inv()`, `enc_copy()`, `enc_cossin(is_sin)`.
- `decode()`(round-trip 검증)에 위 opcode 추가.
### 1b. `_poc/mysim.cpp` 확장 (실행 의미)
- strided load/save: `ncols`/`start`로 넓은 행렬에서 열-범위 타일 로드/세이브(행 stride=행렬 폭).
- matmul MAC[28]: `C += A·B`(PE-out 덮어쓰기 대신 누산), accumulate buffer 초기화 규약.
- reduce-sum(0x14), broadcast(0x15), sign-inv(0x16), copy(0x17), cos/sin(0x18) 구현.
### 1c. 검증: 우리 mysim == 벤더 a.out (byte-exact)
- 각 `b_program/inst_*`의 program_memory.bin을 우리 mysim으로 실행 → 벤더 a.out stdout과 대조.
- 신규 명령 중심으로 자동 대조 스크립트(`tests/test_isa_0710.py`).

## Phase 2 — 컴파일러 재타겟 (instruction 지원과 함께)
### 2a. `tir_backend.py` (matmul 경로)
- **gather → strided load** (No.cols/start로 타일 직접 로드; per-row 복사 제거).
- **scatter → strided save**.
- **K-accumulate → matmul MAC 비트** (save 부분합 + v_add 제거).
- **K^T transpose → strided load**(K를 전치로 로드) → 전치 캐시/permute 제거.
- **activation → matmul activation 비트**.
### 2b. `codegen.py` (elementwise/reduce/broadcast/RoPE/softmax)
- `relax.sum` → **native reduce-sum**(ones-matmul 제거), `relax.broadcast_to` → **native broadcast**.
- RoPE: **sign-inversion+copy**(rotate_half), **cos/sin** native(또는 상수 유지).
- softmax: **exp** native + **fold-max(min/max)** 기반 stable softmax(reduce-max 우회, report update_0710 §5).
### 2c. `memplan.py`
- **가중치 패킹 재검토**: strided load가 gather를 흡수 → 패킹 이득 소멸 가능 → `pack_params`/`pack` 기본 off 검토·측정 후 결정.
### 2d. `model.py`
- 대부분 relax 레벨 그대로(백엔드가 흡수). RoPE·softmax 빌더만 새 primitive 반영 검토.
- **O-proj 융합·활성화 gather 재사용**: strided/​MAC로 대체되므로 유지 여부 재평가.

## Phase 3 — 재검증·재측정
- [ ] `tests/test_decode.py` M1~M6 재수행(새 ISA/mysim, byte-exact/토큰 일치).
- [ ] `test_real_layer`·`test_tir_backend` 회귀.
- [ ] `analyze_kernels`/`analyze_pack` 재측정 → **새 오버헤드 프로파일**(gather/scatter/accum/transpose 소멸 반영).
- [ ] report 갱신(update_0710 후속 또는 report 신규): "새 ISA에서의 명령 프로파일 재정의".

## 리스크/결정 포인트
1. **패킹·O-proj 융합·활성화 재사용**: strided load/save+MAC로 상당수 불필요 → 제거 vs 유지(측정 후 결정).
2. **stable softmax**: fold-max 명령 수 증가 감수(정확성 우선).
3. **벤더 a.out stdout 파싱** vs **우리 mysim --gout 유지**: 후자 채택(빠름), a.out은 오라클.
4. **strided load 의미 정합**: 벤더 a.out과 byte-exact 대조로 확정(Phase 1c).
