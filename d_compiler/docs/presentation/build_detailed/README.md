# 상세판 발표자료 생성기

`../npu_compiler_detailed.pptx`(89장)를 만드는 pptxgenjs 스크립트다. 내용은 `docs/current/` 13장,
`docs/examples/`의 실제 IR·명령·실행 결과, `docs/evidence/`의 보존 기록에서 가져왔고,
compiler나 모델을 실행하지 않는다.

## 구성

| 파일 | 내용 |
|---|---|
| `lib.js` | 색상·글꼴·도형·표·코드 블록 헬퍼(`Deck` 클래스)와 동작 그림용 헬퍼: `matrix`(tile/cell 격자 + 강조 영역), `lanes`(vector 행), `wordStrip`(명령 stream), `lifetimeChart`(buffer offset/수명), `membar`(메모리 주소 막대), `bitfield`(명령 word). `evidence(name)`은 `docs/evidence/current`의 JSON을 읽는다 (`NPU_EVIDENCE_DIR`로 위치 지정 가능) |
| `p1_intro.js` … `p10_limits.js` | 파트별 슬라이드. 각 슬라이드는 `D.add(meta, fn)`으로 제목·부제·근거·발표자 노트와 본문 도형을 정의 |
| `build.js` | 파트를 순서대로 모아 PPTX와 발표자 노트 markdown(`*.notes.md`)을 쓴다 (`NOTES_TITLE`/`NOTES_INTRO` 환경변수로 노트 제목 지정) |
| `search_update.js` | E9의 실제 커널 탐색 비교와 진행 중인 E10의 256-token 전체 prefill 구성 (상세판 78·79장). 요약판은 13장에 탐색 개선 동작과 E10 상태를 직접 그린다 |
| `short_deck.js` | 13장 요약판 (`node build.js ../npu_compiler_short.pptx short_deck`). 13장은 09-29 탐색 개선(후보 정규화·용량 필터·예산·ReplayFunc/DB)과 E10 상태 |

예제 그림은 도형 흐름도가 아니라 실제 동작을 보여주도록 그린다: 64×64 tile 격자와 panel, SRAM/DRAM 주소 막대(실제 byte 주소), 실제 word 단위 명령 stream(77-word 예제), USMP offset 차트, KV cache slot 격자, 탐색 후보의 loop signature·SRAM 용량 막대·후보 pool 순서(33·34·42장) 등. 수치와 주소는 `docs/examples`·`docs/evidence`의 기록에서 가져오고, 09-29 정규화 공간의 후보 수는 `npu_schedule_space.candidates()/prioritize()`로 계산한 값임을 슬라이드에 표시한다.

슬라이드 크기는 16:9 (13.333 × 7.5 in), 한글 본문 글꼴은 **맑은 고딕(Malgun Gothic)**, 코드는 Courier New다.
제목·부제는 길이에 따라 자동으로 글꼴 크기를 줄여 한 줄(부제는 두 줄)에 맞춘다.

## 재생성

```bash
cd d_compiler/docs/presentation/build_detailed
npm install                       # pptxgenjs 4.0.1
node build.js ../npu_compiler_detailed.pptx
#   → ../npu_compiler_detailed.pptx, ../npu_compiler_detailed.notes.md
```

일부 파트만 만들려면 파트 이름을 뒤에 붙인다: `node build.js test.pptx p4_schedule p5_vector_attn`.
이 디렉터리 밖에서 빌드할 때는 `NPU_EVIDENCE_DIR=<repo>/d_compiler/docs/evidence/current`를 지정한다.

## PDF와 시각 검사

PDF는 LibreOffice headless 변환으로 만들었다 (PowerPoint export가 아니다). 이 저장소 환경에는 LibreOffice가
설치되어 있지 않아 공식 tarball(`LibreOffice_25.8.7_Linux_x86-64_deb.tar.gz`)을 임시 디렉터리에 `dpkg -x`로
풀어 사용했고, 맑은 고딕이 없으므로 fontconfig alias로 `NanumGothic`(`../../../build/presentation_tools/fonts`)을
대체해 렌더링했다. 따라서 PDF의 글자 폭은 PowerPoint에서 여는 PPTX와 조금 다를 수 있다.

```bash
# 예: 추출한 LibreOffice로 변환
FONTCONFIG_FILE=<alias가 있는 fonts.conf> SAL_USE_VCLPLUGIN=svp \
  <lo>/program/soffice --headless --convert-to pdf --outdir . ../npu_compiler_detailed.pptx
```

검사 기록은 `../validation_detailed.json`에 남긴다. PPTX/PDF 페이지 수·노트·텍스트 추출과 갱신 슬라이드의 시각 검토를 구분한다.
실제 발표 PC의 PowerPoint에서 한 번 더 열어 확인한다.
