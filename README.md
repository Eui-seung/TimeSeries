# TimeSeries

서울 지하철 역별 일별 이용량과 기상(ASOS 실측, 기상특보)의 관계를 SARIMA/SARIMAX로 분석하는 시계열 프로젝트입니다.

## **`notebooks/`를 00번부터 순서대로 보면 됩니다.**

모든 노트북은 맨 위에 "하는 일 / 입력 파일 / 출력 파일"이 적혀 있고, 코드 셀마다 설명이 붙어 있습니다. 설정은 `00_settings.ipynb` 한 곳에만 있고, 다른 노트북은 첫 셀에서 `%run ./00_settings.ipynb`로 불러옵니다. 노트북은 위에서 아래로 한 번에 실행한 결과(표·그림)가 저장된 상태입니다.

| 번호 | 노트북 | 한 줄 설명 | 상태 |
|---|---|---|---|
| 00 | `00_settings.ipynb` | 경로, 대상 역 9개, 가리기·더미 열 이름, 기간, SARIMA 차수, 적합 옵션 등 모든 설정 | 완료 |
| 01 | `01_preprocessing.ipynb` | 원본(지하철, ASOS)에서 `model_input_9stations.csv`까지 만드는 전처리 (원본이 없으면 처리된 파일 읽기만 실행) | 완료 |
| 02 | `02_event_counts.ipynb` | 특보 종류별 이벤트 수와 겹침 제외 표 (깨끗한 이벤트 수) | 완료 |
| 03 | `03_sarimax.ipynb` | SARIMAX 모형 N / A / B / C 비교 (날씨 없음, 특보만, ASOS만, 특보+ASOS) | 완료 |
| 04 | `04_event_baseline.ipynb` | 이벤트 기준선, 충격·회복 계산, placebo (단일 / 복합 / 민감도) | 완료 |
| 05 | `05_model_compare.ipynb` | 단순 SARIMA(Naive) → N → A/B 단계별 비교 (하루 앞 롤링, 365일 한 번에) | 완료 |
| 06 | `06_figures_ppt.ipynb` | 발표용 그림 | 예정 |
| 07 | `07_test_2025.ipynb` | 2025년 test 평가 (설정이 모두 확정된 뒤) | 예정 |

> **2025년은 07번 전까지 모델링에 쓰지 않습니다.** (전처리 데이터에는 2025년 행이 들어 있지만 모델 입력에서 바로 버립니다.)

## 폴더 구조

```
notebooks/            분석 노트북 (00번부터 순서대로)
data/processed/       처리된 데이터와 문서
    model_input_9stations.csv   모델 입력 (9개 역 × 2014~2025 일별, 44열)
    weather_daily.csv           서울 일별 날씨 (ASOS)
    README_data.md              열 설명과 데이터 범위
    data_notes.md               전처리 결정·근거·이상 기록
results/              분석 결과 표 (csv)
legacy/               예전 스크립트와 config.py (노트북으로 대체됨, 기록용)
Naive_SARIMA.ipynb    팀원의 단순 SARIMA 기준선 (수정하지 않음)
서울_날씨특보_활성상태_일별_onehot.csv   특보 파일 (특보 담당 팀원이 만듦)
```

원본(`rawdata/`)과 용량이 큰 파일(`*.gz` 등)은 저장소에 없습니다. 아래 출처에서 받아 `rawdata/`에 두면 `01_preprocessing.ipynb`로 전부 다시 만들 수 있습니다.

## 데이터 출처

- **지하철 승하차 인원**: 서울 열린데이터광장 「서울교통공사_역별 일별 시간대별 승하차인원 정보」 https://data.seoul.go.kr/dataList/OA-12921/F/1/datasetView.do , 공공데이터포털의 같은 자료(2024년까지) https://www.data.go.kr/data/15048032/fileData.do
- **날씨**: 기상청 기상자료개방포털 종관기상관측(ASOS) 시간자료, 서울(108) 2014~2025
- **기상특보**: 팀 저장소의 `서울_날씨특보_활성상태_일별_onehot.csv` (서울 전체 기준, 권역·주의보/경보 단계 구분 없음)

## 실행 방법

- **로컬**: `notebooks/` 폴더에서 Jupyter를 열고 번호 순서대로 위에서 아래로 실행합니다. 필요한 패키지: pandas, numpy, scipy, statsmodels, joblib, matplotlib, openpyxl, chardet, holidays, factor_analyzer (뒤의 둘은 01번에서만 필요).
- **Colab**: 노트북 첫 셀이 저장소를 clone하고 경로를 맞춰 줍니다 (로컬이면 건너뜀). 01번의 원본 데이터는 따로 올려야 하므로, 원본이 없으면 처리된 파일을 읽는 부분만 실행됩니다.
- 경로는 `ROOT`(저장소 루트) 기준 상대경로입니다. 환경변수 `SUBWAY_PROC`, `SUBWAY_INTERIM`, `SUBWAY_FIG`, `SUBWAY_RESULTS`로 출력 위치를 바꿀 수 있습니다.

## 팀 메모 (기존 README에서 옮김)

해야 할 일

1. 전처리 끝내기
2. 조교님이 언급한 To improve 고쳐서 중간 발표 전까지 자료 만들기
3. SARIMA 만들어서 baseline모델 구축하기
4. 발표 ppt 클로드 아니고 직접 만들기

개선할 점 정리

- 날씨 종류(폭우/폭설/한파 등) × 역 유형(환승역/주거지역/업무지구 등) 표를 만들어 각 칸의 사용 가능한 사건 수를 적어 넣기
- 사건이 너무 적은 칸은 합치거나("제외" 또는 "참고용"으로 표시)
- 역 분류 기준을 명확히 설명하기 (예: 승하차 패턴, 주변 지역 특성 등으로 어떻게 나눴는지)
- 기준선과 회복의 정의를 명확히 + 정보 누수 점검
- **중요** 실제 날씨 강도 구분
