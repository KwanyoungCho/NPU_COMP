# NPU 업데이트 분석 보고서 (2026-07-10) — 기존 ISA 대비 상세 비교

> 대상: `0710_npu_update/` (박형철, NPU simulator 진행 보고 / instruction format ver.07)
> 구성 파일: `20260708_01_박형철_구현내용.pdf`, `20260708_02_NPU_design_instruction_format_ver_07.pdf`, `p006b.tar`
> 관련 문서: `report/report_0616.md`(오버헤드 분석), `report/report_0703.md`(decode 생성·입력 가중치 패킹)

---

## 0. 요약

- 이번 업데이트는 **성균관대(본 연구실)와 서울과기대가 요청한 HW 기능들을 NPU 설계자가 대거 반영**한 것이다.
- 우리가 **소프트웨어로 우회했던 오버헤드**(gather / scatter / K^T transpose / K-accumulate / reduce / broadcast / activation)의 **대부분이 하드웨어 명령으로 직접 지원**된다.
- 결과적으로 **우리 SW 최적화 여러 개(가중치 패킹·O-proj 융합·활성화 재사용·전치 캐시·ones-matmul reduce/broadcast)의 역할이 축소·소멸**하고, 컴파일러를 **새 ISA로 재타겟**해야 한다.
- **아직 없는 것**: reduce-max(→ stable softmax 약점), HW loop/auto-increment(→ 여전히 unroll), 버퍼 파일-쓰기.

---

## 1. 업데이트 개요 — 벤더가 컨소시엄 요청을 반영

진행 보고 PDF는 각 기능을 **"구현 여부(O/X)"** 로 표기하며, 서울과기대/성균관대의 요청 해석을 인용한다. 즉 이 업데이트는 **우리가 report_0616 §6.5 / report_0703 §7.4에서 "새 ISA로 필요하다"고 짚은 항목들**(strided load/save, m_mul accumulate, reduce/broadcast, activation, transpose)을 실제로 HW에 넣은 것이다.

특히 문서에 명시:
- transpose: **"성균관대에서 요청하신 stride load 기능으로 구현됨"**
- matmul-accumulate: **"matmul에 누산 모드(더하기) 비트 추가"**
- strided load-store: **"현재 load는 연속(contiguous)만 지원 → 타일 추출에 per-row 로드 다수 또는 호스트 사전 재배치 필요"** (문제 인식) → 해결

---

## 2. 파일 구성 (`p006b.tar`)

| 경로 | 내용 |
|---|---|
| `a_npu/a.out` | **새 시뮬레이터 실행 바이너리**(ELF, 소스 비공개 — 기존 mysim처럼 "주어진 실행기") |
| `a_npu/{G_buffer_data.bin, program_memory.bin}` | 초기 G-buffer / 프로그램 메모리 |
| `b_program/inst_XXXX_*/` | **명령어별 example**(각 디렉토리에 `a.out`, `program_memory.bin`, `w_support_program_mem_gen.cpp`=인코딩 생성기) |
| `c_hex_data/` | G-buffer 바이너리 생성기(`w_support_binary_data_gen.cpp`) |

**인코딩 스펙은 PDF, 실제 비트 인코딩은 `w_support_program_mem_gen.cpp`, 동작은 `a.out`** 에 있다.

---

## 3. 새 ISA 전체 (32-bit, little-endian, opcode = bits[7:0])

### 3.1 Vector(SIMD) 명령
| 명령 | opcode | 명령 | opcode |
|---|---|---|---|
| add | 0x01 | compare | 0x11 |
| subtract | 0x02 | **min/max** | 0x12 |
| bitwise logical | 0x08 | type-convert | 0x13 |
| shift | 0x09 | ★ **reduce-sum** | 0x14 |
| multiply | 0x0A | ★ **broadcast** | 0x15 |
| divide | 0x0B | ★ **sign-inversion** | 0x16 |
| **multiply-add(FMA)** | 0x0C | ★ **copy** | 0x17 |
| move | 0x0D | ★ **cos/sin** | 0x18 |
| **square-root** | 0x0E | | |
| **exponential** | 0x0F | | |

### 3.2 Matrix 명령
| 명령 | opcode | 옵션 |
|---|---|---|
| matrix add | 0x40 | activation 비트[29] |
| matrix subtract | 0x41 | activation 비트[29] |
| **matrix multiply** | 0x42 | **activation 비트[29] + MAC 비트[28]** |
| matrix move | 0x43 | |

각 명령: {matrix,matrix}, {matrix,scalar}, {matrix,immediate} 지원.

### 3.3 Memory 명령
| 명령 | opcode | 비고 |
|---|---|---|
| set {vector,matrix} start addr | 0x80 | [31:30] 1st-load/2nd-load/save, [29] low/high |
| set vector length | 0x82 | length[23:8] |
| **set matrix tile** | 0x88 | R0(row)[15:8], R1(col)[23:16], [31] 1st/2nd |
| **load** | 0x90 | [31] vec/mat, [30] 1st/2nd, **[29] strided**, No.cols[23:16], start-col[15:8] |
| **save** | 0x98 | [31] vec/mat, **[29] strided**, No.cols[23:16], start-col[15:8] |

★ = 이번 추가(하늘색). 프로그램 구조: `set addr → set tile → load → matmul(a/b) → set save addr → save`.

---

## 4. 기존 vs 신규 상세 비교 (우리 pain point별) ★핵심

> "기존"은 report_0616/0703에서 우리가 쓰던 방식, "신규"는 이번 HW 명령.

### 4.1 gather (입력 strided 접근)
- **기존**: NPU가 연속 load만 지원 → 넓은 행렬의 64×64 타일을 **per-row 복사(gather)** 로 모음. 우리 대응 = **가중치 패킹(−91.8%, report_0703 §7.3)**, 활성화 gather 재사용.
- **신규**: **Load 비트[29] strided** — `No.columns`(추출 열 수)·`start column`으로 **넓은 행렬에서 타일을 직접 strided load**. gather가 **load 1개로 흡수**.

### 4.2 scatter (출력 strided 쓰기)
- **기존**: 넓은 출력 타일을 **per-row 복사(scatter)**. 우리 대응 = **O-proj head 융합(scatter 24→1)**.
- **신규**: **Save 비트[29] strided** — 결과 타일을 넓은 행렬의 strided 위치에 직접 save. scatter가 **save 1개로 흡수**.

### 4.3 K^T transpose
- **기존**: decode 최대 비용(**50.9%**). 우리 대응 = 원소별 전치 / **전치 KV 캐시**.
- **신규**: **strided load로 K를 전치 형태로 로드** → `Q@K^T = load Q + strided-load K + matmul`. **transpose 제거**(문서 명시).

### 4.4 K-accumulate (Tiled GEMM 부분합)
- **기존**: matmul이 PE-out을 **덮어씀** → 타일마다 `save·load·matrix_add` 오버헤드(우리 "accum" role). 우리 제안 = "m_mul accumulate".
- **신규**: **matmul MAC 비트[28] (C += A·B)** + **accumulate buffer 초기화 지원** → 누산이 **matmul 안에서** 처리. accum 오버헤드 소멸.

### 4.5 activation (SiLU)
- **기존**: SwiGLU의 SiLU를 별도 elementwise(exp·mul·div 조합)로. 기존 활성화가 `x²·sigmoid` 형태로 부적합했음.
- **신규**: **matmul activation 비트[29] ON + 표준 SiLU/sigmoid** → 활성화가 **matmul에 융합**.

### 4.6 reduce-sum / broadcast
- **기존**: 둘 다 **ones-matmul**로 우회(RMSNorm 제곱합·평균, softmax 지수합·나눗셈).
- **신규**: **Reduce-sum(0x14)**, **Broadcast(0x15)** 네이티브 → ones-matmul 불필요, 명령 감소·정확도↑.

### 4.7 RoPE
- **기존**: rotate_half = slice+neg+concat 또는 rot 행렬 matmul, cos/sin은 호스트 사전계산.
- **신규**: **sign-inversion(0x16)**(−x) + **copy(0x17)** 로 rotate_half, **cos/sin(0x18)** 온-디바이스.

### 4.8 exponential
- **기존**: softmax exp 우회.
- **신규**: **Vector exp(0x0F)** 네이티브.

### 4.9 c-model 자체 개선
| 항목 | 기존 | 신규 |
|---|---|---|
| 버퍼 용량 | 16KB(FP16 **8192개**)·program 32K **고정** → 초과분 무시/침범 | ✅ **동적 할당/확장** (큰 모델 수용) |
| PC 무한증가 | 출력 끝없이 발생 | ✅ **halt/종료 조건 추가** |
| 버퍼 파일-쓰기 | 없음(우리는 DEVNULL+`--gout` 우회) | ❌ 미추가 → **load 명령으로 버퍼 확인** 권장 |

---

## 5. 아직 없는 것 (남은 SW 과제)

| 미구현 | 영향 | 벤더 제안/우회 |
|---|---|---|
| **Reduce-max** ❌ | **stable softmax의 근본 블로커**(우리 `ws=0.05` 제약의 원인). 진행보고에 "대체 방법 없음"으로 표기 | 서울과기대 해석 기반: **Vector min/max(0x12)로 row/col별 max를 구해 벡터로 조합**. 전용 명령은 없음 → 번거로움 |
| **Loop + 주소 auto-increment** ❌ | GEMM 타일 반복이 HW 루프로 안 됨 → **컴파일러가 여전히 전부 unroll**(타일 수 = 명령 수) | load/save 주소를 바꿔 반복(=현재 우리 방식). |
| **버퍼 파일-쓰기** ❌ | 결과를 파일로 못 뽑음 | load로 확인 |

---

## 6. 우리 컴파일러·최적화에 미치는 영향 ★

### 6.1 기존 SW 최적화들의 운명
| 우리 최적화 (기존) | 새 ISA에서의 상태 |
|---|---|
| **가중치 패킹 (−91.8%/토큰)** | 존재 이유가 "gather 명령 제거"였음 → **strided load가 gather를 흡수**하므로 **이득 대폭 축소/불필요**(재검토) |
| **O-proj head 융합 (scatter 24→1)** | scatter가 **strided save**로 싸짐 + **matmul MAC**로 head 바로 누산 → **이득 축소** |
| **활성화 gather 재사용** | strided load가 싸지면 재-load 부담↓ → **이득 축소**(중복 load 회피 가치는 일부 잔존) |
| **전치 KV 캐시** | **strided load가 전치 흡수** → **전치 저장 불필요** |
| **ones-matmul reduce/broadcast** | **네이티브 reduce-sum/broadcast로 대체** |
| **m_mul accumulate (제안이었음)** | **MAC 비트로 실제 구현됨** → K-accum 소멸 |

### 6.2 오버헤드 프로파일 변화(예상)
- 우리 기존 분석(report_0703 §7): gather ~93%, useful 4~7% — **이는 구 ISA 기준**.
- 새 ISA에선 gather/scatter/accum/transpose가 명령에서 사라져 **useful matmul 비중이 대폭 상승**할 것으로 예상. 즉 **우리가 측정한 "병목 대부분이 이미 HW로 해결"** 된 상태.

### 6.3 남는 SW 가치
1. **명령 수 최소화** — HW 루프가 없으므로 **타일 수 = 명령 수**. 타일링·재사용·operand 스케줄링은 여전히 SW 몫.
2. **stable softmax 우회** — reduce-max 부재 → min/max 기반 구현이 남은 거의 유일한 SW 정확성 과제.
3. **메모리 계획** — 버퍼가 커졌지만 배치·재사용(§ report 논의한 buffer liveness)은 여전히 유효.

---

## 7. 인코딩 상세 (소스로 검증)

### 7.1 Instruction format
- 32-bit, little-endian, **opcode = bits[7:0]**.
- 주소지정 모드 [31:30]: `00`=immediate(Imm[15:0] in [23:8]), `01`=scalar(buffer addr in [23:8]), `10`=vector_2(vector-vector).

### 7.2 matmul의 activation/MAC 비트 (`inst_1023b`, line 58)
```
(2 << 30) + (1 << 29) + 0x42
= [31:30]=10 (vector_2), [29]=1 activation ON, [28]=0 MAC OFF, opcode 0x42 (matmul)
→ MAC 켜려면 + (1 << 28)
```

### 7.3 strided load / save (`inst_1034`)
```
load  (line 56): (1<<31)+(0<<30)+(1<<29)+(2<<16)+(0<<8)+0x90
                 = matrix, 1st, STRIDED, No.columns=2, start_col=0
save  (line 61): (1<<31)+(1<<29)+(2<<16)+(1<<8)+0x98
                 = matrix, STRIDED, No.columns=2, start_col=1
```

### 7.4 대표 프로그램 구조 (matmul 예)
```
set 1st addr (low/high)   0x80
set 2nd addr (low/high)   0x80
set tile 1st (R0×R1)      0x88
set tile 2nd (R0×R1)      0x88
load 1st matrix           0x90   (strided면 [29]=1 + No.cols/start)
load 2nd matrix           0x90
matrix multiply           0x42   (+activation[29] / +MAC[28])
set save addr             0x80
save                      0x98   (strided면 [29]=1 + No.cols/start)
```

---

## 8. 다음 단계

1. **새 ISA로 백엔드 재타겟** — strided load/save, matmul MAC/activation, native reduce-sum/broadcast/exp/sin/cos를 emit. 기존 gather/scatter/ones-matmul/전치/누산 우회 코드는 제거.
2. **재측정** — 새 ISA에서 3B 커널 명령 프로파일 재산출(대부분의 오버헤드가 이미 사라졌을 것으로 예상 → 우리 −91.8% 등 수치의 의미 재정의).
3. **stable softmax** — min/max 기반 reduce-max 우회 구현(남은 핵심 SW 정확성 과제).
4. **새 시뮬레이터 연동** — codegen↔`a.out`을 새 명령/프로그램 포맷·새 sim 바이너리에 맞게 갱신, 소차원 재검증(M1~M6 재수행).

---

## 부록. 이번 추가/수정 명령 (b_program 기준, 하늘색 강조분)
`reduce_sum(0x14)`, `broadcast_immediate/scalar(0x15)`, `sign_inversion(0x16)`, `copy(0x17)`, `cos/sin(0x18)`, `matrix_add_matrix (+ _w_act)`, `matrix_mul (+ _w_act 변종)`, `matrix_stride_load_store(load/save 0x90/0x98의 strided)`, 표준 활성화(SiLU/sigmoid), c-model(버퍼 확장·halt).
