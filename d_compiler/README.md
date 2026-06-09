# d_compiler — NPU LLM Compiler

TVM(Relax) 기반으로 **Llama 3.2 3B**를 본 레포의 NPU c-model(`_poc/mysim`)에서 추론하기 위한 컴파일러.
설계·로드맵은 **[PLAN.md](./PLAN.md)** 참고.

---

## 환경 (구축 완료)

> ⚠️ 기존 env는 사용하지 않는다: `ssd`(타 작업용, 손대지 말 것), `tvm-study`(mlc nightly = 개명·일부 깨진 API).
> 본 프로젝트는 **소스 빌드한 표준 API TVM**을 전용 env `npu-tvm`에서 사용한다.

- **conda env**: `npu-tvm` (Python 3.11)
- **TVM**: apache/tvm **v0.19.0** 소스 빌드 (표준 `tvm.tir`/`tvm.script.tir` + Relax)
- **소스 위치**: `~/tvm-src` (레포 밖), 빌드 산출물 `~/tvm-src/build/libtvm.so`
- **핀 toolchain**(최신은 v0.19 빌드를 깸): `llvmdev=18.1.8`, `cmake=3.31.*`

### 사용
```bash
conda activate npu-tvm
python -c "import tvm; print(tvm.__version__)"   # -> 0.19.0
```

### 검증된 동작
- Relax: author → `LegalizeOps` → `relax.build(target="llvm")` → `VirtualMachine` 실행 (FP16 matmul relerr ~8e-4)
- TIR: `from tvm.script import tir as T`의 `T.block` 작성 + `tvm.tir.Schedule` + `split` + `tensorize` 사용 가능
- `tvm.tir`, `tvm.dlight`, `tvm.nd.array`, `tvm.tir.TensorIntrin` 모두 표준 경로로 존재

---

## 처음부터 재구축하는 법 (재현용)

```bash
# 1) 빌드 의존성 갖춘 env 생성
conda create -y -n npu-tvm -c conda-forge python=3.11 cmake ninja llvmdev numpy cython libstdcxx-ng
# 2) v0.19 호환 toolchain으로 핀 (최신 LLVM/cmake는 빌드 깨짐)
conda install -y -n npu-tvm -c conda-forge "cmake=3.31.*" "llvmdev=18.1.8"

# 3) TVM 소스 (서브모듈 포함)
git clone --recursive --branch v0.19.0 --depth 1 https://github.com/apache/tvm ~/tvm-src

# 4) 빌드 설정
cd ~/tvm-src && rm -rf build && mkdir build && cp cmake/config.cmake build/
cat >> build/config.cmake <<EOF
set(USE_LLVM "$CONDA_PREFIX/envs/npu-tvm/bin/llvm-config")
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(USE_RELAX ON)
EOF

# 5) 빌드 (npu-tvm 의 cmake/ninja/llvm 사용)
conda activate npu-tvm
cd ~/tvm-src/build && cmake .. -G Ninja && ninja        # 64코어서 ~4분

# 6) Python 패키지 (editable)
cd ~/tvm-src && pip install -e python
```

빌드 메모:
- LLVM 22 / cmake 4 면 실패 → 반드시 핀 버전 사용. (cmake4는 서브모듈 `rang`의 `cmake_minimum_required 2.8` 거부; LLVM22는 TVM 0.19 C++와 비호환)
- `cmake .. -G Ninja` 출력에 `Build with LLVM`, `TVM_LLVM_VERSION=181` 떠야 정상.

---

## 시뮬레이터(mysim) — NPU 실행기 (주어진 c-model)

컴파일러 결과는 **주어진** `_poc/mysim.cpp`로만 실행한다(우리가 실행기를 재구현하지 않음, PLAN §2.0).
```bash
g++ -O2 -std=c++17 ../_poc/mysim.cpp -o mysim   # runtime.py 픽스처가 자동 수행 예정
```

---

## 디렉토리 (예정)
`npu_compiler/`(frontend·legalize·memplan·isa·codegen·runtime), `tests/`. 상세는 PLAN.md §6.2.

## 다음 단계
PLAN.md §11: M1 `isa.py`(명령어 인코더) → M2 단일 matmul e2e(Relax→ISA→mysim).
