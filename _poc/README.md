# NPU c-model 재구현 (mysim)

소스가 없는 NPU c-model 바이너리(`a_npu/a.out`)를 역추출하여 **동작이 byte-exact로 동일한**
C++ 시뮬레이터를 재구현한 것. `b_program/inst_*` 예제 **55개 전부 출력이 1바이트도 안 틀림**
(`PASS=55 FAIL=0`).

추가로 원본의 한계 3가지(고정 캡 / write-back 없음 / 자체 종료 없음)를 해소한 개선 옵션 포함.

---

## 파일

| 파일 | 역할 |
|------|------|
| `mysim.cpp` | 재구현 시뮬레이터 소스 (전체 ISA + 개선 옵션) |
| `verify_all.sh` | 컴파일 후 55개 예제를 원본 golden과 byte-exact diff |
| `capture_goldens.sh` | 원본 `a_npu/a.out`으로 golden trace 55개 재생성 |
| `goldens/` | 원본 기준 trace 55개 (검증 기준값) |

## 빌드 & 실행

```bash
g++ -O2 -std=c++17 mysim.cpp -o mysim
# program_memory.bin + G_buffer_data.bin 을 현재 폴더에 두고:
./mysim
```

옵션:
```
--run  N     halt 없을 때 최대 명령 수 (기본 30; HALT opcode 0xFF가 더 먼저 멈춤)
--gout FILE  실행 후 G-buffer를 FP16로 FILE에 기록 (다음 run의 입력으로 체이닝)
--gn   N     write-back 엔트리 수 (기본 = 입력 엔트리 수)
--gbuf N     G-buffer 초기 용량(FP16 엔트리); 어차피 접근 시 자동 확장
```

## 전체 재검증

```bash
bash verify_all.sh        # -> PASS=55 FAIL=0
```

원본은 conda 라이브러리(`LD_LIBRARY_PATH`)가 필요하고 **무한 루프**(PC가 끝없이 증가하며 출력 폭주)라
`timeout`+`script`(pty)+`head` 트릭으로만 캡처 가능. mysim은 그냥 실행하면 깔끔히 종료됨.

---

## 원본 c-model에서 확인한 사실

| 항목 | 값 |
|------|-----|
| G-buffer | 고정 **8192 FP16** (동적 할당 없음; 바이너리에 `cmp 0x1fff` 상수) |
| Program memory | 고정 **32768 명령어** (`cmp 0x7fff`) |
| 파일 write-back | **없음** (출력은 stdout trace뿐, `ofstream` 0개) |
| 자체 종료 | **없음** — PC 무한 증가, 한도 초과 시 에러 대신 **조용한 메모리 오염** |
| 입력 G_buffer_data.bin | 8192개 FP16(2B LE) + 개행 = 16385바이트. 기본값 램프 `G[i]=i` |
| 입력 program_memory.bin | uint32 LE 명령어. 256개×4 + 개행 = 1025바이트 |
| **연산 정밀도** | **float32** 내부 연산, **FP16 반올림은 G-buffer 저장 시에만** |

> 연산이 float32라는 게 핵심: divide/sqrt/exp가 full 정밀도로 출력되고,
> `sub_w_act` 결과가 저장 후 재로드 시 `0.00202087 → 0.00202179`로 달라지는 게 증거.

trace 포맷 quirk:
- `instruction :` 값은 **16진수**, PE 배열 값은 **10진수**
- 헤더에서 `G_buffer size`는 10진수(바이트), `Program memory size`는 **16진수**(`std::hex` 잔존, 0x401=1025)
- 헤더는 명령어 20개를 16진수로 덤프, 블록마다 끝에 빈 줄 3개, NOP은 `NOP --- `

---

## 명령어 인코딩 (32비트 워드, 하위 바이트 = opcode)

공통 필드: `[31:30]` operand 모드 (0=immediate, 1=scalar, 2=vector/matrix),
`[29]` activation 플래그(행렬 연산), 값/주소는 `[23:8]`.

### 제어
| opcode | 의미 | 필드 |
|--------|------|------|
| `0x80` | set address | `[31:30]`operand(0=1st,1=2nd,2=dest) `[29]`hi/lo `[23:8]`값 |
| `0x82` | set vector length | `[23:8]`길이 |
| `0x88` | set matrix tile | `[31]`matrix(0/1) `[23:16]`dim2 `[15:8]`dim1 |
| `0x90` | load | `[31]`matrix `[30]`operand(0→PE_in_1,1→PE_in_2) |
| `0x98` | save | `[31]`matrix; PE_out → G[dest] (FP16 반올림) |
| `0xFF` | **HALT** (mysim 확장; 원본엔 없음) | — |

### 벡터 연산 (결과 = PE_out, raw 10진수로 출력)
| opcode | 연산 | 비고 |
|--------|------|------|
| `0x01` | add | `a + b` |
| `0x02` | sub | `a - b` |
| `0x08` | logical | `[29:27]` 0=AND 1=OR 2=NOT(`~a`) 3=XOR 4=NAND 5=NOR (정수 비트연산) |
| `0x09` | shift | `a * 2^s` (s = signed16 값; 0xFFFF=-1 → a/2, **float 결과**) |
| `0x0A` | multiply | `a * b` |
| `0x0B` | divide | `a / b` |
| `0x0C` | mul-add | `PE_out += a * b` (PE_out에 **누산**) |
| `0x0D` | move | `b` (operand 값으로 덮어씀) |
| `0x0E` | sqrt | `sqrt(a)` |
| `0x0F` | exp | `exp(a)` |
| `0x11` | compare | `(a == b) ? 1 : 0` |
| `0x12` | min/max | `[28]` 0=min 1=max |
| `0x13` | type convert | `[31]` 0=int→float 1=float→int (정수 데이터엔 항등) |

`b` = mode 0/1이면 `[23:8]` 상수, mode 2면 PE_in_data_2[i].

### 행렬 연산
| opcode | 연산 |
|--------|------|
| `0x40` | matrix add (imm/scalar=elementwise, matrix=elementwise) |
| `0x41` | matrix sub |
| `0x42` | matrix multiply — **imm/scalar 모드는 elementwise `a*const`, matrix 모드(2)만 진짜 matmul** |
| `0x43` | matrix move |

**activation** (`[29]`): `f(x) = x² · sigmoid(x)`
- 검증: `(-11)² · σ(-11) = 121 × 1.6702e-5 = 0.00202087` (정확 일치). 양수 입력에선 ≈ x².

immediate ≡ scalar (효과 동일). matmul: A(rA×cA)·B(cA×cB), `C[i][j]=Σ_k A[i*cA+k]·B[k*cB+j]`.

---

## 개선 옵션 (원본 한계 해소)

| 한계 | 해소 |
|------|------|
| 8192 / 32768 고정 캡 | 파일 크기 동적 할당 + 접근 시 자동 확장 (주소 무제한) |
| write-back 없음 | `--gout`으로 G-buffer를 FP16 파일 출력 (스테이지 간 체이닝) |
| 자체 종료 없음 | `0xFF` HALT opcode + `--run N` 한도 |

→ 풀 차원 Llama3 매핑을 막던 3대 블로커 제거. 단 **기존 55개 검증은 그대로 통과**(개선은 비파괴적).
