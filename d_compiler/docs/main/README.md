# Main backend: ver.08 최종 기준

[전체 문서](../README.md) · [동작 상세](pipeline.md) · [검증 결과](results.md)

이 문서는 `main` commit **1143abbc6bedddb1eed8d2bcd8ecae6b346d9fd1**의
최종 ver.08/source-0818 경로를 설명한다. 최신 SRAM/DRAM 계층 backend의 이전 단계로
섞어 읽지 않는다. 현재 작업 트리에는 이 compiler 파일들이 제거되어 있지만 해당
commit에 원본 코드와 보고서가 남아 있다. 이번 작업은 main branch를 변경하지 않았다.

## 어떤 실행기를 대상으로 하는가?

Vendor binary는 `0818_npu_update/a_npu/a.out`이다. 8192 FP16-entry의 고정 G-buffer가
있어 작은 kernel/quirk 비교 기준으로 사용한다. 전체 모델은 산술·명령의 vendor 호환성을
보존하면서 global 저장 공간과 program 수용량을 확장한 `_poc/mysim_0818.cpp`를 사용한다.
Vendor 원본을 전체 모델 실행에 맞춰 임의 수정하는 것이 아니다.

이 backend에서는 tensor가 row-major G-buffer에 있고 MAIN/PARTIAL descriptor로
최대64×64 sub-tile을 직접 LOAD/SAVE한다. 최신 v09의 명시적 GLOAD/GSTORE와 SRAM
보관 탐색을 이 경로에 적용한 것으로 설명하지 않는다.

## 두 최종 backend의 중요한 차이

| 항목 | main ver.08/source-0818 | 최신 v09 |
|---|---|---|
| 주된 lowering | Relax graph를 수작업 연산별 emitter로 연결 | LegalizeOps/TIR/Schedule을 거쳐 공통 codegen |
| 메모리 | Flat G-buffer와 host의 실행 경계 | DRAM↔SRAM↔연산장치 |
| Layer 반복 | 같은 layer 프로그램을 host가 weight를 바꾸어 실행 | 전체 entry stream, 컴파일 판단을 재사용 |
| Decode 길이 | Context별 정적 프로그램 cache | 고정 capacity+mask, 선택적 bucket |
| KV 상태 | Host가 반환된 K/V로 배열을 확장 | Persistent DRAM cache에 device DMA append |
| LM head | 큰 stride를 K×64 packed panel로 처리 | 일반 compiler의 지원되는 배치·명령 경로 |
| 종료/출력 | 0xF0 snapshot과 ver.08 실행 규약 | HALT와 v09 image 규약 |

둘의 words나 실행 시간을 숫자만 비교해 성능 순위를 만들 수 없다. 컴파일 단위,
메모리 경계, 실행기, 입력과 codegen 세대가 다르다.

## 원본을 확인하는 방법

현재 workspace를 checkout하지 않고도 읽을 수 있다.

```bash
git show 1143abb:d_compiler/npu_compiler/backend_0818.py
git show 1143abb:d_compiler/npu_compiler/v3_source_llama.py
git show 1143abb:d_compiler/npu_compiler/source_runtime_0818.py
git show 1143abb:report/report_0818.md
```

과거 0710/direct/hybrid 실험 경로와 개발 계획을 이 설명서의 별도 최종 backend로
유지하지 않는다. Main의 상세 재현은 해당 commit의 환경·entry를 사용하며, 현재
v09 runner에서 `backend="source-0818"` 옵션을 찾으면 안 된다.
