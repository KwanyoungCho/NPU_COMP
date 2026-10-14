# Compiler 설명 발표자료 — 상세 설명판

[문서 목차](../README.md) · [웹 뷰어](http://localhost:8765) · [PPTX](npu_compiler_explained.pptx) · [PDF](npu_compiler_explained.pdf) · [발표자 노트](speaker_notes.md)

## 89장 상세판 (npu_compiler_detailed.pptx)

[PPTX](npu_compiler_detailed.pptx) · [PDF](npu_compiler_detailed.pdf) · [발표자 노트](speaker_notes_detailed.md) · [생성기](build_detailed/README.md)

위 54장 자료와 별도로, 연구자 발표용으로 더 자세히 쓴 **89장 덱**이다 (약 90분).
`docs/current/` 13장의 설명을 실제 예제로 따라가며, 슬라이드마다 실제 IR·명령·주소·측정값을 넣었다.

- 예제 세 개를 끝까지 사용: SiLU(X[64,128]·Wᵀ+bias) (authored Relax → host transform → semantic TIR →
  선택된 fusion → scheduled TIR → ISA packet → 실행), 두 matmul 44 KiB (SRAM 주소·77-word 해독),
  Llama 3.2 3B (1-layer IR 발췌, 선택 기록, 3-token 생성 기록).
- 각 슬라이드에 근거 경로(footer)와 발표자 노트가 있으며, 도형·표·차트는 PowerPoint에서 편집할 수 있다
  (실험 그래프는 native chart). 한글 서체는 맑은 고딕, 16:9.
- 재생성: `build_detailed/`의 pptxgenjs 스크립트 (`npm install && node build.js ../npu_compiler_detailed.pptx`).
  PDF는 LibreOffice headless 변환본이며 나눔고딕으로 대체 렌더링했다 (자세한 절차는 생성기 README).
- 발표자료 생성기는 compiler를 실행하지 않는다. 28-layer·256-token prefill 실험(E10)은 메모리 증가로 중단됐으며, 병목 수정과 전체 재검증 미실행을 구분해서 표시한다.


## 13장 요약판 (npu_compiler_short.pptx)

[PPTX](npu_compiler_short.pptx) · [PDF](npu_compiler_short.pdf) · [발표자 노트](speaker_notes_short.md)

89장 상세판을 13장(표지 포함)으로 압축한 요약판이다 (약 25~30분). 각 장은 동작을 보여주는 그림 하나와 핵심 3~4줄로 구성하고,
IR은 2~4줄 요약만 둔다. 2번 슬라이드가 전체 pipeline을 작은 예제의 실제 산출물(tile → call_tir → SRAM panel →
DRAM/SRAM 주소 → 명령 stream → C-model)로 한 장에 담고, 이후 장치·Relax/host 분리·수치 계약·fusion·matmul panel·
vector/attention·메모리 계획·ISA·runtime·검증 순서로 이어진다. 재생성: `build_detailed/`에서
`node build.js ../npu_compiler_short.pptx short_deck`.

## 54장 기본 설명판과 웹 뷰어

현재 메모리 계층을 사용하는 **v09 FP16 backend**를 설명한다. 본문 48장과 부록 6장,
총 **54장**이다. 본문 약 60~80분을 예상하며 청중의 TVM 경험에 따라 조절한다.
필요하면 1~26장의 계산·schedule 설명과 27~48장의 메모리·실행·검증 설명으로 나눈다.

## 상세판에서 달라진 점

기존 자료의 짧은 표제와 발표자 노트 중심 구성을 바꿨다. 각 슬라이드는 왼쪽의 동작 그림,
오른쪽의 세 가지 본문 설명, 아래의 구체적인 예제와 해석으로 읽는다.
왜 필요한지, 어떻게 처리하는지, 어떤 조건에서 성립하는지를 화면에서 확인할 수 있다.

- Llama layer에는 Q/K/V shape, gate/up 분기와 두 residual 경로를 함께 표시했다.
- Matmul은 X/W panel의 영역, K 두 조각의 FP32 누산, 실제 폭 projection의 반복 횟수를 보여준다.
- `compute_at`, `reverse_compute_at`, cache 생성·접근 범위·compaction을 구분하고 전후 loop를 넣었다.
- SiLU, vector 256/VLEN, Softmax, RMSNorm, RoPE를 별도 슬라이드로 설명한다.
- 메모리는 tensor 수명 시간표, 실제 SRAM 주소 재사용, H의 열 tile과 row stride로 설명한다.
- Decode는 고정 capacity, 유효 길이, mask, 임시 위치와 최종 cache 슬롯을 함께 표시한다.
- 실험은 조건·비교값·검증 수준과 해석의 한계를 본문에 넣었다.

도형·화살표·격자·수명 막대·표·실험 막대그래프는 **PowerPoint에서 편집할 수 있다**.
격자는 논리 영역을 보여주는 구조도이며 모든 cell이 실제 원소 하나나 정확한 크기 비율을
뜻하지 않는다. 보관 주소 그림의 작은 상수 영역은 식별을 위해 확대했음을 표시했다.
양자화와 기존 main은 계속 부록으로 구분한다.

## 발표 순서

| 슬라이드 | 내용 | 그림과 본문에서 확인할 것 |
|---|---|---|
| 1~6 | 전체 구조와 Llama | 계산·배치·보관, compile/prepare/run, 세 메모리 층, layer와 TVM 역할 |
| 7~10 | 모델에서 TIR까지 | 작은 예제의 shape, host ABI 순서, attention 분해, Relax와 TIR의 공존 |
| 11~14 | 수치 계약과 fusion | K 누산, fusion 구성, TVM pattern과 외부 사용, 영역/전체 후보 비교 |
| 15~20 | Matmul scheduling | Tile/panel, 두 compute_at API, compaction, 탐색 축, 실제 폭 MLP |
| 21~26 | Vector와 attention | SiLU 수명, 256/VLEN, Softmax, RMSNorm, RoPE/GQA, query 분할 |
| 27~30 | 메모리 계획 | 세 분석 범위, 수명 막대, SRAM 주소 공유, 직접 SAVE와 DRAM 예약 제거 |
| 31~33 | ISA와 bundle | Tensorize/codegen, 실제 packet 단위, 명령·image·ABI 연결 |
| 34~37 | Llama runtime | Token 입력/예측 순서, capacity와 mask, KV append, layer 판단 재사용 |
| 38~43 | 과거 실험 | 지표별 의미, fusion, 용량별 DMA, 공간 개선, compile 관측, 전체 Llama 출력 |
| 44~46 | 현재 탐색 개선·새 실험 | MetaSchedule의 역할, E9 실제 커널 비교, E10 256-token 전체 prefill 진행 상태 |
| 47~48 | 현재 경계와 전체 연결 | 남은 검토의 규모와 근거, 같은 값의 pipeline 추적 |
| 49~54 | 부록 | TVM API, IR 읽기, 기존 main, 양자화, 긴 prefill, 근거 찾기 |

예를 들어 scheduling부터 검토하려면 [15번 슬라이드](http://localhost:8765/#15),
메모리 전달은 [29번](http://localhost:8765/#29), cache slot은
[35번](http://localhost:8765/#35), 전체 Llama 출력은 [43번](http://localhost:8765/#43)에서 시작한다.

## 발표자 노트와 근거

PPTX의 발표자 노트와 [speaker_notes.md](speaker_notes.md)는 원래 설명 흐름과
추가된 슬라이드 본문, 예제 해석, 원본 문서 경로를 함께 담는다.
[sources.json](sources.json)은 슬라이드별 출처와 원본 hash를 기록한다.

09-29 갱신에서는 현재의 후보 정규화·용량 검사·적응형 예산, 반복 layer의 재사용,
선택형 MetaSchedule trace DB와 ReplayFunc를 반영했다. 상세판 28·33·34·42장은 source 중복 제거 격자,
후보 정규화와 용량 필터(loop signature, SRAM 용량 막대), 검사 흐름과 종료 사유, ReplayFunc·JSONDatabase 연결을
실제 동작 그림으로 보여주고, 요약판 13장이 같은 내용을 한 장으로 요약하며 E10 상태를 함께 표시한다. E9의 실제 커널 비교 결과와
E10의 **중단된** 256-token 전체 모델 실험을 추가했다. 작은 예제 실행, 과거 7-token 전체
Llama 검증, 정적 MLP compile, 현재 커널 탐색 비교, 새 전체 모델 실험을 구분한다. DMA와 ISA words를 실장치 latency로
바꾸어 주장하지 않는다. 양자화 generation 완료로 확대하지 않는다.

## 브라우저에서 보기

[웹 뷰어](http://localhost:8765)를 새로고침하면 기본 설명판 54장을 읽는다.
이번 추가 내용은 [44번](http://localhost:8765/#44)부터, E10 상태는 [46번](http://localhost:8765/#46)이다.
89장 상세판과 13장 요약판은 위의 별도 PDF/PPTX 링크로 연다. 서버가 꺼져 있다면
저장소 루트에서 다음 명령을 실행한다. 기존 PDF와 노트를 읽으며 재컴파일하지 않는다.

```bash
python d_compiler/docs/presentation/serve.py --port 8765
```

왼쪽 목차에서 검색·이동할 수 있다. `←` / `→`는 슬라이드 이동, `N`은 노트,
`F`는 전체 화면, `Z`는 확대 보기다. 확대 상태에서는 슬라이드를 스크롤해 읽고,
다시 `Z`를 누르면 화면에 맞춘다. 화면이 작으면 전체 화면이나 확대 보기를 사용한다.
원격 VS Code에서는 **Ports에서 8765를 포워딩**한 다음 로컬 브라우저로 연다.
서버를 직접 실행한 터미널에서는 `Ctrl+C`로 종료한다.

- 16:9, 1440×810 기준 좌표다. 한글 서체는 나눔고딕(NanumGothic)이다.
- PPTX를 편집하는 PC에도 같은 서체를 설치하면 배치 차이를 줄일 수 있다.
- PDF에는 한글 서체를 포함했다. PPTX와 같은 도형 좌표로 ReportLab에서 별도 생성했다.
- PowerPoint 자체 렌더링과의 동일성까지 검증하지 않았다. 실제 발표 PC에서 PPTX를 확인한다.

## 편집과 재생성

[slide_content.json](slide_content.json)이 제목·설명·예제·노트·근거의 편집 원본이고,
[build_deck.py](build_deck.py)가 그림과 PPTX/PDF를 생성한다. 실험 막대는 원본 JSON을
읽어 도형으로 그린다. 상세판에는 차트 생성을 위한 matplotlib 의존성이 없다.

```bash
python d_compiler/docs/presentation/build_deck.py
```

다른 환경에서는 다음 의존성을 compiler의 TVM 환경과 분리해 설치할 수 있다.
웹 서버에는 `PyMuPDF`가 필요하다.

```bash
python -m pip install --target d_compiler/build/presentation_tools/deps \
  python-pptx==1.0.2 reportlab==5.0.1
```

나눔고딕 Regular/Bold TTF를 `d_compiler/build/presentation_tools/fonts`에 두거나
`--font-dir`로 지정한다. 파일명은 `NanumGothic-Regular.ttf`, `NanumGothic-Bold.ttf`다.
현재 폰트는 Google Fonts `google/fonts` 저장소의 `ofl/nanumgothic`에서 가져왔다.
`--out`으로 별도 결과 폴더를 지정할 수 있다.

[validation_detailed.json](validation_detailed.json)은 상세판·요약판의 페이지·노트·PDF 검사를 기록한다.
기본 설명판의 [validation.json](validation.json)은 글자 폭·한글 glyph·도형 경계·설명 영역 검사,
PPTX 재열기·노트·출처 연결과 PDF 페이지 렌더링 범위를 기록한다.
화면 검토용 이미지는 `d_compiler/build/presentation_tools/review_detailed`에 두며
문서 폴더에 중복 slide image를 저장하지 않는다.
