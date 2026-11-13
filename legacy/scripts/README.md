# legacy/scripts/ — 지하철 전처리 코드 (예전 방식)

> **이 코드는 `notebooks/`로 대체되었습니다.** 전처리는 `notebooks/01_preprocessing.ipynb`, SARIMAX 분석은 `notebooks/03_sarimax.ipynb`에 같은 로직이 들어 있습니다. 여기는 기록용으로 남겨 두었습니다.

원본(`rawdata/`)에서 `data/processed/` 최종 파일까지 단계별로 다시 만드는 코드입니다. 프로젝트 루트에서 실행합니다.

## 실행 순서
| 파일 | 하는 일 | 주요 산출물 |
|---|---|---|
| `common.py` | 경로·상수·원본 읽기·이름 정규화 (직접 실행하지 않음) | |
| `step1_load_raw.py` | 1단계: 파일 구조 확인, 연도별 표준화 | `data/interim/raw_{연도}.pkl`, `file_structure.csv` |
| `step2_station_map.py` | 2단계: 역명 통일 | `station_map.csv`, `rename_candidates.csv`, `new_closed_stations.csv` |
| `step3_apply_map.py` | 3단계: 표준역명 적용, 서울 외 역 제외, 시간대 테이블 | `station_date_hour.csv.gz`, `excluded_stations.csv`, `always_present_stations.csv`, `station_presence_by_year.csv` |
| `step4_classify.py` | 4단계: 역 유형 분류와 논문 비교, 연도별 검증 | `station_type.csv` 등 |
| `step5_daily_tables.py` | 5단계: 일별·그룹별 테이블(split, 공휴일 포함) | `station_daily.csv.gz`, `group_daily.csv` |
| `step5_quality_checks.py` | 5단계: 결측·0·집계 오류·급감일 점검 | `data_quality_*.csv`, `anomaly_*.csv` |
| `step6_figures.py` | 그림 3종 | `figures/*.png` |
| `step7_add_test_year.py` | 테스트 연도(2025) 추가 | `station_daily`/`group_daily`에 split=test 행 추가 |
| `step8_weather_daily.py` | 8단계: ASOS 서울 시간자료(`rawdata/asos/`) → 일별 날씨 (2014~2025) | `weather_daily.csv`, `weather_daily_check.csv` |
| `step9_model_input.py` | 9단계: 단독형 역 9개 모델 입력(가리기·계단 더미, 날씨 join, 빈 특보 열) | `model_input_9stations.csv` |
| `eda_weekly_pattern.py` | EDA: 한 해 전체 이용량 변화와 주별 반복성 (`--year 2024`) | `figures/fig_eda_weekly_pattern_{연도}.png`, `eda_weekly_pattern_{연도}.csv` |

```
python legacy/scripts/step1_load_raw.py        # (연도 지정: --years 2014 2015)
python legacy/scripts/step2_station_map.py
python legacy/scripts/step3_apply_map.py
python legacy/scripts/step4_classify.py
python legacy/scripts/step5_daily_tables.py
python legacy/scripts/step5_quality_checks.py
python legacy/scripts/step6_figures.py
python legacy/scripts/step7_add_test_year.py --year 2025                  # 점검만 하고, 새·변경된 역이 있으면 중단(종료 코드 2)
python legacy/scripts/step7_add_test_year.py --year 2025 --accept-names   # 확인 후 진행
python legacy/scripts/step8_weather_daily.py                              # 날씨 일별 (rawdata/asos 필요)
python legacy/scripts/step9_model_input.py                                # 모델 입력 9개 역 (step8 먼저)
```

## 테스트 연도 처리 규칙 (step7)
- 파일 형식(헤더, 시간대 20개, 연도 전체·누락일, 값 이상), 역 목록(새 역·이름 변경·부기명 변경·사라진 역)을 `station_map.csv`와 비교해 `new_year_check_{연도}.csv`에 기록하고, 하나라도 있으면 아무것도 바꾸지 않고 중단합니다.
- 유형은 다시 분류하지 않고 `station_type.csv`를 그대로 붙입니다(학습 때 없던 역은 `New (not in training)`).
- 새 역(기존에 없던 역번호)은 자동 처리하지 않고 오류로 중단합니다(소재지·유형 확인 필요).

## 필요한 패키지
pandas, numpy, scipy, openpyxl, chardet, factor_analyzer, holidays, matplotlib (그림은 macOS의 AppleGothic 글꼴 사용)
추가 패키지 경로는 환경변수 `SUBWAY_EXTRA_PATH`로 줄 수 있고, 출력 경로는 `SUBWAY_PROC`, `SUBWAY_INTERIM`, `SUBWAY_FIG`로 바꿀 수 있습니다.

## 검증
1~6단계를 별도 폴더에서 처음부터 실행해 현재 `data/processed/`의 파일과 비교했고, 17개 파일이 값까지 동일했습니다(차이: `file_structure.csv`의 xlsx 3개 형식 표기 오류를 바로잡음, `data_quality_summary.csv`의 설명 문구).
`step4_variable_definition_test.csv`(변수 정의 비교 진단표)는 스크립트로 재생성하지 않는 정적 기록입니다.
`data/interim/`의 파일은 중간 산출물이라 지워도 step1로 다시 만들 수 있습니다(전 연도 약 수백 MB).
