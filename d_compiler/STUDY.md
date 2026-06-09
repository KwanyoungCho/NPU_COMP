# NPU LLM 컴파일러 — 코드 학습 가이드

> 이 문서는 `d_compiler/`에 지금까지 구현한 컴파일러를 **처음 보는 사람이 스스로 이해**할 수 있게
> 개념 → 코드 → 실제 예제 순서로 풀어 쓴 것입니다. 설계 의도는 `PLAN.md`, 빌드/환경은 `README.md` 참고.

## 목차
0. [큰 그림 — 우리가 만든 것](#0-큰-그림)
1. [타깃: NPU와 mysim 시뮬레이터](#1-타깃-npu와-mysim)
2. [숫자: FP16과 G-buffer](#2-숫자-fp16과-g-buffer)
3. [ISA 인코딩 — `isa.py`](#3-isa-인코딩--isapy)
4. [실행 하네스 — `runtime.py`](#4-실행-하네스--runtimepy)
5. [TVM Relax 기초](#5-tvm-relax-기초)
6. [메모리 배치 — `memplan.py`](#6-메모리-배치--memplanpy)
7. [코드 생성 — `codegen.py`](#7-코드-생성--codegenpy)
8. [미지원 연산 우회 — `legalize.py`](#8-미지원-연산-우회--legalizepy)
9. [전체 연결 — `driver.py`](#9-전체-연결--driverpy)
10. [타일링과 FP16 누산 — B0.5](#10-타일링과-fp16-누산--b05)
11. [직접 실험해보기](#11-직접-실험해보기)
12. [용어집](#용어집)

---

## 0. 큰 그림

우리가 만드는 건 **컴파일러**입니다. 입력은 신경망(Llama 레이어), 출력은 NPU가 실행할 수 있는 **명령어 바이너리**입니다.

```
 신경망 (TVM Relax 그래프)
      │   ← memplan: 텐서마다 메모리 주소 정하기
      │   ← legalize: NPU에 없는 연산을 있는 연산 조합으로 바꾸기
      │   ← codegen: 각 연산을 NPU 명령어로 바꾸기
      ▼
 NPU 명령어 (program_memory.bin)  +  데이터 (G_buffer_data.bin)
      │   ← runtime: 주어진 시뮬레이터 mysim 실행
      ▼
 결과 (gout.bin)  →  numpy 참조와 비교해 검증
```

**중요한 원칙 2가지** (헷갈리기 쉬움):
1. **mysim은 우리가 만든 게 아니라 "주어진 것"**입니다. NPU의 동작을 정의하는 c-model(`_poc/mysim.cpp`)이고, 우리는 이걸 **타깃**으로 삼아 명령어를 생성할 뿐, 실행기를 새로 만들지 않습니다.
2. **`isa.py`는 시뮬레이터가 아니라 "인코더"**입니다. 사람이 읽는 명령("주소 5에서 8개 로드")을 mysim이 이해하는 32비트 숫자로 바꿔주는 번역기일 뿐입니다.

파일 지도:
```
d_compiler/npu_compiler/
  isa.py        명령어 → 32비트 숫자 (인코더)
  runtime.py    mysim 빌드·실행·결과 읽기
  memplan.py    텐서 → G-buffer 주소 (정적 배치)
  codegen.py    Relax 연산 → NPU 명령어
  legalize.py   미지원 연산을 우회 구현 (reduce, softmax, silu, rope...)
  driver.py     위 전부를 엮어 "Relax 모듈 → 실행 → 결과"
d_compiler/tests/   9개 테스트 (각 단계 검증)
```

학습 순서는 **아래에서 위로** 가는 게 쉽습니다: 먼저 NPU가 뭘 하는지(1~2장) → 명령어를 어떻게 만드는지(3~4장) → 그 위에 TVM을 얹는 법(5~10장).

---

## 1. 타깃: NPU와 mysim

NPU(Neural Processing Unit)는 행렬곱·벡터연산을 빠르게 하는 가속기입니다. 우리 타깃 NPU의 특징:

- **행렬 연산기(PE array)**: 64×64 크기. 한 번에 최대 64×64 행렬곱을 함.
- **벡터 연산기**: 한 번에 여러 원소를 동시에 더하기/곱하기 등.
- **G-buffer**: 모든 데이터가 사는 1차원 메모리 (그냥 큰 배열이라고 생각).
- 명령어로 동작: "G-buffer 주소 X에서 데이터 읽어 → 연산 → 주소 Y에 써라".

이 NPU의 동작은 `_poc/mysim.cpp`(C++ 시뮬레이터)에 정의되어 있습니다. mysim의 핵심 루프(`mysim.cpp:83` 부근)를 의사코드로 보면:

```
for pc in 0..maxRun:           # pc = program counter (명령어 번호)
    instr = program[pc]        # 32비트 명령어 하나
    op = instr & 0xFF          # 맨 아래 8비트 = opcode (무슨 연산인가)
    case op:
      0x80: 주소 설정          # 다음 load/save가 쓸 G-buffer 주소
      0x82: 벡터 길이 설정
      0x88: 타일(행렬) 크기 설정
      0x90: load  (G-buffer → 연산기 입력 버퍼)
      0x98: save  (연산기 출력 → G-buffer)  ← 이때 FP16 반올림!
      0x42: 행렬곱
      0x01: 벡터 덧셈
      0xFF: 정지(HALT)
      ...
```

즉 NPU 프로그램은 **"주소 정하기 → 로드 → 연산 → 저장"** 패턴의 반복입니다. 레지스터 머신처럼 상태(주소, 타일 크기, 벡터 길이)를 먼저 세팅하고 load/연산/save를 하는 구조입니다.

> mysim은 명령어마다, 그리고 읽고 쓰는 데이터 원소마다 전부 화면(stdout)에 출력합니다(`mysim.cpp:87,101,104`). 이게 원래는 디버그용이지만, 큰 모델에선 이 출력 비용 때문에 느려집니다(그래서 풀 3B는 "실행"보다 "비용 추정"이 현실적 — PLAN §4.1).

---

## 2. 숫자: FP16과 G-buffer

### FP16 (half precision)
16비트 부동소수점. 32비트(float)보다 정밀도가 낮지만 메모리·연산이 빠릅니다. NPU는 데이터를 FP16으로 저장합니다.

**가장 중요한 동작 하나**: mysim은 **연산은 float32로 정확히 하지만, G-buffer에 저장(save)할 때만 FP16으로 반올림**합니다 (`mysim.cpp:103` — `G[b+i]=fp16(pout[i])`).

이게 왜 중요하냐면: 같은 계산이라도 **언제 저장하느냐**에 따라 결과가 달라집니다.
- 행렬곱을 한 방에 하면 → 마지막에 한 번만 반올림.
- 행렬곱을 여러 조각으로 쪼개 누적하면 → 조각마다 저장→반올림이 끼어들어 결과가 미세하게 달라짐.

→ 이게 10장(타일링)에서 핵심이 됩니다. "타일로 쪼갠 결과 ≠ 한 방에 한 결과"인 게 **버그가 아니라 정상**입니다.

### G-buffer
그냥 FP16 숫자들의 1차원 배열입니다. 모든 텐서(입력, 가중치, 중간 결과, 출력)가 이 배열의 어딘가에 자리잡습니다. 2차원 행렬 `A[2][3]`은 **행 우선(row-major)**으로 펼쳐 저장: `A[0][0], A[0][1], A[0][2], A[1][0], ...`.

"동적 할당"이 없습니다. 그래서 **컴파일 타임에 모든 텐서의 주소를 미리 정해야** 합니다 (6장 memplan).

---

## 3. ISA 인코딩 — `isa.py`

ISA(Instruction Set Architecture) = 명령어 집합. `isa.py`는 사람이 읽는 명령을 32비트 숫자로 바꿉니다.

### 명령어의 비트 구조
32비트 숫자 한 개가 명령어 하나입니다. 비트 위치마다 의미가 있습니다:

```
비트:  31  30 | 29 | 28 ......... 8 | 7 .... 0
       [mode] |act |   값/주소/길이  | opcode
```
- 맨 아래 8비트(`[7:0]`) = **opcode** (무슨 명령인가)
- `[23:8]` = 16비트 값 (즉시값, 주소, 길이 등)
- `[31:30]` = **mode** (피연산자가 0=즉시값, 1=스칼라, 2=벡터)
- `[29]` = activation 플래그 (행렬 연산용)

이 규칙의 "정답"은 mysim의 디코드 코드입니다. 예를 들어 mysim은 `op=instr&0xFF`, `mode=(instr>>30)&3`로 읽으니, 우리는 거꾸로 그 위치에 비트를 넣어 인코딩합니다.

### 예제로 보기: "즉시값 3을 더하라"
`isa.py`의 벡터 덧셈 인코더(`enc_add`):
```python
def _enc_simple(op, mode, imm):
    return ((mode & 3) << 30) | (_u16(imm) << 8) | (op & 0xFF)
def enc_add(mode=VECTOR, imm=0):  return _enc_simple(0x01, mode, imm)
```
`enc_add(mode=IMM, imm=3)` 를 계산해보면 (IMM=0, op=0x01):
- `(0 << 30)` = 0
- `(3 << 8)` = 0x300
- `| 0x01` = 0x301

→ **0x301**. 이게 "각 원소에 3을 더하라(즉시값 모드)"는 32비트 명령. (실제 원본 예제 생성기가 만든 값과 정확히 일치 — 그래서 우리 인코더가 맞다는 걸 알 수 있음.)

### 주소는 왜 명령어 2개인가
주소는 16비트씩 끊어 **하위/상위 2개 명령**으로 보냅니다 (`addr` 메서드가 2개를 emit):
```python
def addr(self, operand, value):
    self._emit(enc_addr_lo(operand, value))   # 하위 16비트
    return self._emit(enc_addr_hi(operand, value))  # 상위 16비트
```
`operand`는 0=첫째 입력, 1=둘째 입력, 2=출력. 즉 "다음 load는 G-buffer 어디서 읽을지" 또는 "다음 save는 어디에 쓸지"를 정합니다.

### `Asm` 빌더
`Asm` 클래스는 명령어들을 리스트(`self.words`)에 차곡차곡 쌓습니다. 메서드 체이닝으로 프로그램을 짭니다:
```python
a = Asm()
a.addr(SRC1, 5).vlen(8).load(0, 0)   # 주소5 / 길이8 / 로드
a.v_add(IMM, 3)                       # +3
a.addr(DST, 32).save(0)               # 주소32에 저장
a.halt()
words = a.words                       # [0x..., 0x..., ...]
bytes_ = a.to_bytes()                 # program_memory.bin 내용
```

### 인코더가 맞다는 걸 어떻게 확신하나 (`tests/test_isa.py`)
두 가지로 검증했습니다:
1. **golden 상수**: 원본 예제 생성기(`b_program/inst_*/*.cpp`)에 적힌 비트 표현식과 우리 인코더 출력이 일치하는지 (21개).
2. **라운드트립**: 실제 예제 바이너리 55개 + a_npu의 명령어 **14,336개 전부**를, 한 번 디코드(`reencode`)했다가 다시 인코드하면 **원래 바이트와 1비트도 안 틀리는지**. → 비트 레이아웃이 진짜 NPU 바이너리와 똑같다는 강력한 증거.

`reencode(w)`는 32비트 워드에서 의미 있는 필드만 뽑아 다시 조립합니다. 실제 파일의 모든 명령에 대해 `reencode(w)==w`면 우리 필드 해석이 정확하다는 뜻입니다.

---

## 4. 실행 하네스 — `runtime.py`

컴파일한 프로그램을 **주어진 mysim**으로 돌리는 부분입니다. 하는 일:
1. `build_mysim()`: `_poc/mysim.cpp`를 `g++`로 컴파일 (이미 빌드돼 있으면 재사용).
2. 임시 폴더에 `program_memory.bin`(명령어)과 `G_buffer_data.bin`(초기 데이터)을 씀.
3. `mysim --run N --gout gout.bin` 실행.
4. `gout.bin`(결과 G-buffer, FP16)을 numpy 배열로 읽어 반환.

```python
def run(program, gbuf, gn=None, maxrun=None):
    build_mysim()
    # program → program_memory.bin,  gbuf(float) → FP16 → G_buffer_data.bin
    # mysim 실행, gout.bin 읽어서 float32 배열로 반환
```

핵심: **우리는 NPU 동작을 흉내내지 않습니다.** 입력 파일 만들고 → mysim 돌리고 → 출력 파일 읽기만 합니다. 정확성은 전적으로 mysim이 보장.

`tests/test_runtime.py`가 isa+runtime를 함께 검증: 손으로 인코딩한 "벡터 덧셈" 프로그램을 mysim에 돌려 `[8..15]`가 나오는지, 작은 행렬곱이 맞는지 확인.

---

## 5. TVM Relax 기초

여기서부터 TVM이 등장합니다. **Relax**는 TVM의 신경망 그래프 표현(IR, 중간표현)입니다. 신경망을 "연산들의 그래프"로 들고 있는 자료구조라고 보면 됩니다.

### IRModule과 함수
- **IRModule**: 컴파일 단위. 함수들을 담음. `mod["main"]`으로 함수를 꺼냄.
- **함수(relax.Function)**: 입력(params)과 본문(body)을 가짐.
- 본문은 **SeqExpr** → 그 안에 **DataflowBlock** → 그 안에 **바인딩(binding)들**.
- 바인딩 = `변수 = 연산(...)` 한 줄. 예: `y = matmul(x, w)`.

### 우리가 그래프를 만드는 법: BlockBuilder
신경망을 코드로 조립할 때 `relax.BlockBuilder`를 씁니다:
```python
bb = relax.BlockBuilder()
x = relax.Var("x", relax.TensorStructInfo([8, 64], "float16"))  # 입력 텐서 [8,64]
w = relax.Var("w", relax.TensorStructInfo([64, 64], "float16"))
with bb.function("main", [x, w]):
    with bb.dataflow():
        y = bb.emit(relax.op.matmul(x, w))   # y = x @ w  (바인딩 추가)
        gv = bb.emit_output(y)               # y를 출력으로
    bb.emit_func_output(gv)
mod = bb.finalize()                          # 완성된 IRModule
```
- `TensorStructInfo([shape], dtype)`: 텐서의 모양과 자료형.
- `bb.emit(연산)`: 그래프에 "변수 = 연산" 한 줄 추가, 그 변수를 반환.
- `relax.op.matmul / add / multiply / exp / ...`: Relax가 제공하는 연산들.

> 왜 TVMScript(`@R.function` 데코레이터)를 안 쓰나? TVMScript는 소스코드 텍스트를 파싱하는데, 차원(M,K,N)을 파이썬 변수로 넘기면 못 잡습니다. BlockBuilder는 차원을 평범한 파이썬 정수로 다룰 수 있어 파라메트릭 모듈에 적합합니다.

### 그래프 구조를 코드로 순회하기
컴파일러는 이 그래프를 읽어야 합니다:
```python
func = mod["main"]
for p in func.params:              # 입력 변수들
    ...
for block in func.body.blocks:     # DataflowBlock들
    for binding in block.bindings: # 각 "변수 = 연산"
        var  = binding.var         # 왼쪽 변수
        call = binding.value       # 오른쪽 연산 (relax.Call)
        opname = call.op.name      # 예: "relax.matmul"
        args = call.args           # 입력들
out = func.body.body               # 함수가 반환하는 변수
```
memplan과 codegen이 정확히 이 구조를 순회합니다.

### 왜 "coarse-grained" 매핑인가 (설계 핵심)
TVM은 보통 연산을 잘게 쪼개(TIR의 for 루프들) GPU/CPU에 맞춥니다. 하지만 **우리 NPU는 행렬곱·벡터연산이 통째로 명령어 1개**입니다. 그래서 잘게 쪼갰다가 다시 합치는 건 낭비 — 우리는 Relax 연산을 **그대로** NPU 명령에 매핑합니다. (PLAN §2, §4의 "operator-level codegen".)

---

## 6. 메모리 배치 — `memplan.py`

NPU엔 동적 할당이 없으니, **모든 텐서가 G-buffer의 어느 주소에 살지 컴파일 타임에 정해야** 합니다. `memplan.py`는 가장 단순한 방식 — **bump 할당기**(앞에서부터 차곡차곡)를 씁니다.

```python
class MemPlan:
    offset = {}   # 변수/상수 → G-buffer 주소
    shape  = {}   # 변수/상수 → 모양
    top    = 0    # 다음 빈 주소
    def alloc(self, var):
        off = self.top
        self.offset[var] = off
        self.top += (shape의 원소 개수)   # 그만큼 자리 차지
        return off
```

`plan(func)`이 하는 일:
1. **입력(params)**에 주소 배정.
2. 각 바인딩의 **상수**(예: ones 벡터)에 주소 배정 + 데이터 기억.
3. 각 바인딩의 **결과 변수**에 주소 배정.
4. **alias 처리**: BlockBuilder가 만드는 출력은 `gv = lv`(복사) 형태인데, 이건 새 자리를 안 주고 `lv`와 **같은 주소**를 가리키게 함.

```python
if isinstance(val, relax.Var):       # gv = lv 같은 alias
    mp.offset[binding.var] = mp.offset[val]   # 같은 주소 공유
else:
    mp.alloc(binding.var)
```

**상수 처리**가 중요합니다. legalize가 만드는 ones/zeros/mask/RoPE테이블 등은 `relax.Constant`로 그래프에 박힙니다. memplan은 이들에 주소를 주고, 그 데이터(`c.data.numpy()`)를 기억해뒀다가 driver가 초기 G-buffer에 써넣습니다.

> 지금은 버퍼 재사용을 안 합니다(중간 결과도 새 자리). 그래서 G-buffer가 큼. 최적화는 나중 과제.

---

## 7. 코드 생성 — `codegen.py`

`compile_func(func, mp)`이 Relax 그래프를 순회하며 각 연산을 NPU 명령으로 바꿉니다. 핵심은 연산 이름으로 분기:

```python
for binding in block.bindings:
    name = call.op.name
    if name == "relax.matmul":        emit_matmul(...)
    elif name == "relax.permute_dims": emit_transpose(...)
    elif name in EW2:  emit_ew(... 2개 입력 ...)   # add/sub/mul/div
    elif name in EW1:  emit_ew(... 1개 입력 ...)   # sqrt/exp
```

### 행렬곱 만들기 (`emit_matmul`)
`y = x @ w`, `x[M,K]`, `w[K,N]` 를 NPU 명령으로:
```python
a.tile(0, M, K)               # 첫째 행렬(A) 크기: M×K
a.tile(1, K, N)               # 둘째 행렬(B) 크기: K×N
a.addr(SRC1, off[x]); a.load(1, 0)   # A를 주소 off[x]에서 로드
a.addr(SRC2, off[w]); a.load(1, 1)   # B를 로드
a.m_mul(mode=VECTOR)          # 행렬곱 실행
a.addr(DST, off[y]); a.save(1)       # 결과를 off[y]에 저장
```
이게 "주소→로드→연산→저장" 패턴입니다. mysim이 `C[i][j]=Σ_k A[i,k]·B[k,j]`를 계산해 줍니다.

### 원소별 연산 (`emit_ew`)
`y = x + z` (둘 다 [8,16] → 128개 원소):
```python
a.vlen(128)                          # 벡터 길이 128
a.addr(SRC1, off[x]); a.load(0, 0)   # x 로드
a.addr(SRC2, off[z]); a.load(0, 1)   # z 로드
a.v_add(mode=VECTOR)                 # 원소별 덧셈
a.addr(DST, off[y]); a.save(0)
```
단항(sqrt/exp)은 둘째 입력 없이 `op_method()`만 호출.

### 전치 (`emit_transpose`) — 원소 복사
NPU엔 전치 명령이 없습니다. 그래서 `[R,C]→[C,R]`을 **한 원소씩 복사**합니다:
```python
for r in range(R):
    for c in range(C):
        a.vlen(1)
        a.addr(SRC1, src + r*C + c); a.load(0,0)
        a.v_add(mode=IMM, imm=0)              # a+0 = a (그냥 복사)
        a.addr(DST, dst + c*R + r); a.save(0)
```
주소 계산이 핵심: 원본의 `(r,c)`는 `r*C+c`, 전치본의 같은 값은 `(c,r)` 즉 `c*R+r`. 이게 O(R×C)개 명령이라 비쌉니다 — attention에서 측정해보니 **전체 명령의 68%**가 전치였습니다(전치 ISA가 필요하다는 정량 근거).

---

## 8. 미지원 연산 우회 — `legalize.py`

NPU ISA엔 없지만 신경망에 필요한 연산들을, **있는 연산 조합으로** 만드는 곳입니다. 이게 컴파일러의 진짜 알맹이입니다.

### 합(reduce-sum) = ones와의 행렬곱
"행마다 다 더하기"를 어떻게? `x[rows,k]`에 `ones[k,1]`을 곱하면:
`(x @ ones)[i,0] = Σ_k x[i,k]·1 = x의 i행 합`. 행렬곱이 곧 합이 됩니다.
```python
def reduce_sum_lastdim(bb, x, rows, k):
    return bb.emit(relax.op.matmul(x, _c(np.ones((k, 1)))))   # [rows,1]
```
`_c(...)`는 numpy 배열을 FP16 relax 상수로 만드는 헬퍼.

### 브로드캐스트 = ones와의 외적
스칼라 하나를 벡터로 복제: `x[rows,1] @ ones[1,n]` → `[rows,n]` (각 행의 값이 n번 복제). 역시 행렬곱.

### RMSNorm
수식: `y = x / sqrt(mean(x²)) · w`. 이걸 위 도구들로 조립(`rms_norm`):
```
sq    = x * x                         # 제곱 (원소별)
ssum  = sq @ ones[D,1]                # 행 합 = Σx²  (reduce)
mean  = ssum * (1/D)                  # 평균 (상수 곱)
rms   = sqrt(mean)
inv   = ones[seq,1] / rms             # 1/rms  (나눗셈)
scale = inv @ ones[1,D]               # [seq,1]→[seq,D] (broadcast)
xn    = x * scale                     # x/rms
y     = xn * (ones[seq,1] @ w)        # · weight (w도 broadcast)
```
`1/D`, `1/rms` 같은 건 정수 즉시값으로 못 넣으니 상수 텐서로 처리.

### SiLU (SwiGLU의 활성화)
NPU의 하드웨어 활성화는 `x²·sigmoid(x)`라 SiLU가 아닙니다. 그래서 `SiLU(z)=z·sigmoid(z)=z/(1+exp(-z))`를 직접 조립(`silu`):
```
neg = zeros - z          # -z  (음수화: 0에서 빼기)
den = exp(neg)           # exp(-z)
den = den + ones         # 1+exp(-z)
sig = ones / den         # sigmoid(z)
out = z * sig            # SiLU
```
negate(음수화)도 ISA에 없어서 "0에서 빼기"로 우회.

### Softmax (max 빼기 생략)
정석 softmax는 수치안정화를 위해 행 최댓값을 빼지만(`exp(x-max)`), NPU엔 reduce-max가 없습니다. 그래서 **생략**하고, 점수가 작을 때만 안전하게 씁니다(`softmax_lastdim`):
```
e      = exp(s)                  # max 안 뺌
rowsum = e @ ones[cols,1]        # 분모 (reduce)
denom  = rowsum @ ones[1,cols]   # broadcast
p      = e / denom
```

### RoPE (회전 위치 인코딩)
`q_embed = q·cos + rotate_half(q)·sin`. 두 가지 트릭:
1. **cos/sin은 NPU가 못 만듦**(삼각함수 없음) → 호스트(파이썬)에서 미리 계산한 표를 상수로 적재.
2. **rotate_half**(앞뒤 절반을 섞고 부호 바꾸기)를 **고정 순열행렬과의 행렬곱**으로! `rh = q @ Rot`, 여기서 `Rot[hd,hd]`은 적절한 위치에 ±1이 박힌 상수 행렬. 이러면 슬라이싱/concat 없이 행렬곱 하나로 끝.
```python
def rope(bb, q, cos_c, sin_c, rot_c):
    rh = bb.emit(relax.op.matmul(q, rot_c))   # rotate_half
    a  = bb.emit(relax.op.multiply(q, cos_c))
    b  = bb.emit(relax.op.multiply(rh, sin_c))
    return bb.emit(relax.op.add(a, b))
```

`rope_tables(seq, hd)`가 cos/sin 표와 Rot 행렬을 만듭니다. Rot 만드는 규칙은 코드 주석 참고.

---

## 9. 전체 연결 — `driver.py`

지금까지의 조각을 하나로 엮습니다:
```python
def run_module(mod, inputs, tile=None):
    func = mod["main"]
    asm, mp = compile_func(func, tile=tile)   # memplan + codegen
    gbuf = zeros(mp.top)                       # G-buffer 통째로
    # 1) 상수 데이터 써넣기 (ones, mask, rope표...)
    for c in mp.constants: gbuf[off:..] = c의 데이터
    # 2) 입력 데이터 써넣기 (이름으로 매칭)
    for p in mp.params:    gbuf[off:..] = inputs[p.name]
    # 3) 실행
    full = runtime.run(asm, gbuf, gn=mp.top)
    # 4) 출력 변수 위치에서 결과 잘라 모양 복원
    return full[out_off : out_off + n].reshape(out_shape)
```

즉 driver는 **"무엇을 어디에 두고, 무엇을 실행하고, 어디서 결과를 읽을지"**를 memplan의 주소표를 보고 처리합니다.

테스트(예: `tests/test_rmsnorm.py`)는 이걸 호출해 NPU 결과와 numpy 참조를 비교합니다.

---

## 10. 타일링과 FP16 누산 — B0.5

지금까지(B0)는 행렬곱을 **한 방에**(logical) 했습니다. 하지만 실제 PE는 64×64라서, 큰 행렬은 **64×64 조각(타일)으로 쪼개** 계산해야 합니다(hardware-legal).

### K 방향 타일링 (`codegen.py`의 `tile=64` 경로)
`C[M,N] = A[M,K] @ B[K,N]`에서 K가 64보다 크면, K를 64씩 끊어 부분곱을 **누적**합니다:
```
C = 0
for kk in range(0, K, 64):
    A_tile = A[:, kk:kk+64]          # 조각
    B_tile = B[kk:kk+64, :]
    partial = A_tile @ B_tile        # 64×64 이하 행렬곱
    C = C + partial                  # 누적
```
누적을 NPU에서 어떻게? **저장→다시 로드→덧셈**으로:
```python
if ti == 0:
    a.addr(DST, C); a.save(1)        # 첫 조각은 그냥 C에 저장
else:
    a.addr(DST, sP); a.save(1)       # 부분곱을 임시(sP)에 저장
    # C = C + sP
    a.vlen(M*N)
    a.addr(SRC1, C); a.load(0,0)
    a.addr(SRC2, sP); a.load(0,1); a.v_add(VECTOR)
    a.addr(DST, C); a.save(0)
```

### 왜 결과가 one-shot과 다른가 (가장 중요한 포인트)
조각마다 `save`가 일어나고, **save마다 FP16 반올림**(2장)이 끼어듭니다. 그래서:
- one-shot: 끝에 한 번 반올림
- 타일링: 조각마다 + 누적마다 반올림 → **결과가 미세하게 다름** (둘 다 정답에 가깝지만 비트가 다름)

이건 버그가 아닙니다. 그래서 타일링 결과를 검증할 땐 one-shot과 byte 비교하면 **안 되고**, 같은 반올림 순서를 흉내낸 `tiled_fp16_reference`와 비교해야 합니다:
```python
def tiled_fp16_ref(A, B, T=64):
    C = None
    for kk in range(0, K, T):
        part = fp16(A[:,kk:kk+T] @ B[kk:kk+T,:])   # 조각마다 반올림
        C = part if C is None else fp16(C + part)   # 누적마다 반올림
    return C
```
`tests/test_tiling.py`에서 이 둘이 **byte-exact**로 일치함을 확인했고(우리 모델이 정확), one-shot과는 절반쯤 다름을 보였습니다(반올림 효과 실증).

### A 조각은 왜 "gather"가 필요한가
B의 K조각 `B[kk:kk+64, :]`은 연속된 행들이라 메모리에서 **연속**입니다(그냥 오프셋만 더하면 됨). 하지만 A의 K조각 `A[:, kk:kk+64]`은 각 행의 일부라 **띄엄띄엄**(strided) 있습니다. NPU load는 연속만 읽으니, A 조각을 임시 버퍼에 **행마다 복사해 연속으로 모읍니다**(gather). 이 임시 버퍼는 `mp.scratch_alloc()`으로 G-buffer에 자리를 받습니다.

> 지금 버전은 **K만 타일링**(M,N≤64). M이나 N도 64를 넘으면 출력도 타일로 쪼개고 결과를 다시 흩어 쓰는(scatter) 작업이 필요한데, 그게 다음 단계(B1)입니다.

---

## 11. 직접 실험해보기

환경: `conda activate npu-tvm` (또는 `/home/chokwans99/anaconda3/envs/npu-tvm/bin/python` 직접 사용).

### ⭐ 먼저: 단계별 walkthrough 스크립트 (가장 좋은 학습 진입점)
각 스크립트는 파이프라인의 각 단계에서 **실제 값·명령어를 출력**하며 설명을 곁들입니다. 순서대로 돌려보세요:
```bash
P=/home/chokwans99/anaconda3/envs/npu-tvm/bin/python
$P d_compiler/walkthrough_matmul.py    # 기본 흐름: Relax→memplan→codegen→mysim (6단계)
$P d_compiler/walkthrough_rmsnorm.py   # legalize: 1개 연산 → 여러 primitive, reduce/broadcast=ones곱
$P d_compiler/walkthrough_tiling.py    # 타일링: K쪼개기+FP16누산, one-shot과 왜 다른지
```
- `walkthrough_matmul.py` → 본 문서 §6,§7,§9 (memplan/codegen/driver)와 대조.
- `walkthrough_rmsnorm.py` → §8 (legalize). 한 줄 `rms_norm()`이 10개 연산으로 펼쳐지는 걸 봄.
- `walkthrough_tiling.py` → §10 (타일링). hardware-legal 64×64 조각 + 누적 + FP16 반올림 효과.
- 공용 명령어 해석기는 `study_util.py`의 `disasm()`.

### 테스트 돌려보기
```bash
cd /home/chokwans99/NPU_cmodel
P=/home/chokwans99/anaconda3/envs/npu-tvm/bin/python
$P d_compiler/tests/test_matmul.py      # 가장 단순한 e2e부터
$P d_compiler/tests/test_rmsnorm.py     # legalize 등장
$P d_compiler/tests/test_layer.py       # 전체 레이어
$P d_compiler/tests/test_tiling.py      # 타일링
```

### 생성된 명령어를 눈으로 보기
파이썬에서 (레포 루트에서 실행):
```python
import sys; sys.path.insert(0, "d_compiler")
from tvm import relax
from npu_compiler.driver import compile_func

# 2x3 @ 3x2 행렬곱 모듈을 직접 짓기
bb = relax.BlockBuilder()
x = relax.Var("x", relax.TensorStructInfo([2, 3], "float16"))
w = relax.Var("w", relax.TensorStructInfo([3, 2], "float16"))
with bb.function("main", [x, w]):
    with bb.dataflow():
        y = bb.emit(relax.op.matmul(x, w)); gv = bb.emit_output(y)
    bb.emit_func_output(gv)
mod = bb.finalize()

asm, mp = compile_func(mod["main"])
for i, word in enumerate(asm.words):
    print(i, hex(word))         # 명령어들
print("주소표:", {k.name_hint: v for k, v in mp.offset.items() if hasattr(k, "name_hint")})
```
각 `hex(w)`를 3장의 비트 규칙으로 해석해보면 "아, 이게 tile 설정이고, 이게 load구나" 하고 읽힙니다.

### 추천 학습 실험
1. `test_matmul.py`의 행렬곱 차원을 바꿔보고 명령어가 어떻게 바뀌나 관찰.
2. `legalize.rms_norm`의 한 단계(예: broadcast)를 빼고 결과가 어떻게 깨지나 보기.
3. `tile=64`로 큰 K를 주고 명령어 수가 어떻게 늘어나는지(`len(asm.words)`) 측정 — 비용 감각 익히기.
4. mysim의 전체 출력 보기: `runtime.run(asm, gbuf, capture_trace=True)` → 반환된 trace 문자열에 PE 입출력이 다 찍힘.

---

## 용어집
- **NPU**: 신경망 가속기. 여기선 64×64 행렬 연산기 + 벡터 연산기.
- **mysim**: 주어진 NPU 시뮬레이터(`_poc/mysim.cpp`). 우리의 실행기 겸 정답 기준.
- **ISA**: 명령어 집합. 32비트 워드 = 명령 1개.
- **opcode**: 명령어 종류 (맨 아래 8비트).
- **G-buffer**: NPU의 1차원 데이터 메모리(FP16).
- **FP16**: 16비트 부동소수점. mysim은 **저장 시에만** FP16 반올림.
- **TVM / Relax**: 컴파일러 프레임워크 / 그 신경망 그래프 IR.
- **IRModule / BlockBuilder**: 컴파일 단위 / 그래프를 코드로 짓는 도구.
- **legalize**: 미지원 연산을 지원 연산 조합으로 바꾸기.
- **codegen**: IR를 명령어로 바꾸기.
- **타일링**: 큰 연산을 하드웨어 크기(64×64) 조각으로 쪼개기.
- **gather/scatter**: 띄엄띄엄한 데이터를 연속으로 모으기 / 그 반대.
- **logical vs hardware-legal**: 시뮬레이터가 받아주는(>64 타일) 코드 vs 실제 64×64 PE에 맞는 코드.
- **one-shot vs tiled**: 한 방에 한 행렬곱 vs 조각내 누적 (FP16 반올림 시점이 달라 결과가 다름).

---

## 다음 단계 (참고)
`PLAN.md §4` 로드맵 기준 현재 **B0 완성 + B0.5 K-타일링** 까지. 다음은 B1(M/N 타일링 → 임의 차원), 비용 모델(`cost.py`), 실제 Llama 3.2 3B 차원 코드 생성입니다.
```
