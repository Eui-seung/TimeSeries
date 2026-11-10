# 서울 지하철 전처리 결과 (README)

## 저장소에 포함된 파일 / 포함되지 않은 파일
**포함된 파일**
- `data/processed/model_input_9stations.csv` — 모델 입력(단독형 역 9개 × 2014~2025 일별)
- `data/processed/weather_daily.csv` — 서울 일별 날씨(2014~2025)
- `data/processed/README_data.md`, `data/processed/data_notes.md` — 이 문서와 전처리 기록
- `scripts/` — 원본에서 위 파일까지 다시 만드는 코드(단계별 설명은 `scripts/README.md`)
- `config.py` — 모델링 설정(역·열 이름·분할·SARIMA 차수)

**포함되지 않은 파일** (아래 표·본문에 이름이 나오지만 저장소에는 없음)
- 용량 때문에 제외: `station_daily.csv.gz`, `station_date_hour.csv.gz`, `rawdata/`(지하철 승하차 원본·ASOS 원본), `data/interim/`(`raw_*.pkl`)
- 그 밖에 올리지 않은 것: `group_daily.csv`, `station_map.csv`, `station_type.csv`, `weather_daily_check.csv` 등 `data/processed/`의 나머지 산출물과 `figures/`
- **원본 데이터 출처**: 서울 지하철 승하차 인원은 서울교통공사가 공개한 역별 일별 시간대별 승하차인원 자료(파일 목록은 `data_notes.md`의 '사용 파일'), 날씨는 기상청 기상자료개방포털의 종관기상관측(ASOS) 시간자료 서울(108)이다. 원본은 `rawdata/`와 `rawdata/asos/`에 두면 된다.
- **다시 만드는 방법**: 원본을 `rawdata/`에 두고 `scripts/README.md`의 순서(step1~step7 → step8 → step9)대로 실행하면 제외된 파일과 `model_input_9stations.csv`를 다시 만들 수 있다.

서울교통공사 1~8호선 역별·일별·시간대별 승하차 인원(원본: `rawdata/`, 수정하지 않음)을 정리한 결과물입니다.
상세한 결정·근거·이상 기록은 `data_notes.md`에 있습니다. 모든 CSV는 UTF-8(BOM), 열 이름은 영문, 역 이름 값은 한글입니다.

## 범위와 기준
- **기간**: 2014-01-01 ~ 2025-12-31. 2014~2024 = 학습(`split`=train), 2025 = 테스트(`split`=test). 2025년은 전처리를 마쳐 `station_daily`/`group_daily`에 포함됨(아래 '2025년 추가 내용').
- **대상 역**: 서울특별시 소재 역만. 서울 외 소재 24개 노선-역은 `excluded_stations.csv`. 표준역명 기준 224개 역(+값이 전부 0인 6호선 신내는 표에서 제외).
- **사용한 원본**: 2014~2021 연도별 파일, 2022 `…20221231`, 2023 `(23.1~23.12)`, 2024 `(24.1~24.12)`. 겹치는 2023~2024년 다른 파일은 제외(`data_notes.md`).
- **역 이름 통일**: 개명(신천→잠실새내 등)과 환승역(노선 합산)은 `station_map.csv`의 표준역명(`station`) 기준. 환승역은 노선을 합산.
- **역 유형 분류**: 논문(성현곤·김태현 2005) 방식. 2014~2019, 2023~2024년 3월의 평일·토·일(공휴일 제외) 일평균 이용객 수 30개 변수 → 주성분 + oblimin 3요인 → 요인점수 ≥ 1이면 그 기능. 3월에 존재한 연도가 2개 미만이면 분류 제외(Excluded).
- **집계 오류 처리**: 승차 또는 하차 한쪽만 0인 4건은 0인 쪽 값과 합계를 NaN으로 바꾸고 그룹 합계에서 해당 역-일을 뺌(`data_notes.md`).

## 최종 분석용 파일
| 파일 | 행 × 열 | 설명 |
|---|---|---|
| `station_daily.csv.gz` | 975,289 × 19 | **역×날짜 일별 최종 테이블.** `split`(train/test), boarding / alighting / total, 요일(day_of_week, is_weekend), 공휴일(is_public_holiday, holiday_name), 설날·추석 연휴(is_seollal_chuseok), 유형(type_kr, type_en), 분석 그룹(group), 요인점수 3개(score_commercial_leisure / score_residential / score_employment), 사용 연도 수(years_used), `flag_one_side_zero`(집계 오류 NaN 처리 4건=1). 224개 역, 결측 날짜 없음. train 893,529행(2014~2024) + test 81,760행(2025). |
| `group_daily.csv` | 21,915 × 7 | **그룹×날짜 일별 합계.** group(Employment / Residential / Commercial & Leisure / Mixed / Unclassified), date, split, boarding, alighting, total, n_stations(그날 합계에 포함된 역 수). Excluded 역(암사역사공원)과 집계 오류 NaN 역-일은 제외하고 합산. |
| `station_type.csv` | 224 × 8 | **역별 유형 결과.** 요인점수 3개, type_kr(7개 유형 + 미분류 + 분류제외), years_used, status, avg_daily_ridership(3월 선택일 평균 승+하차). |
| `station_date_hour.csv.gz` | 1,787,058 × 25 | 역×날짜×방향(direction: 승차/하차)별 **시간대 20개**(before_06, h06_07 … h23_24, after_24). 환승역은 노선 합산. 2021년은 h23_24에 '23시 이후' 합산 값이 들어 있고 after_24는 비어 있음(`flag_23plus_merged`=1). 집계 오류 4건은 해당 방향 행이 NaN(`flag_one_side_zero_set_nan`=1). |
| `weather_daily.csv` | 4,383 × 11 | **서울(ASOS 108) 일별 날씨, 2014-01-01~2025-12-31 (feels_like_max 외 결측 없음).** date, temp_mean / tmax / tmin(℃), rain_mm(일 합계), snow_depth_max(cm, 일 최대), snow_new_cm(3시간신적설 일 합계), wind_mean(m/s), rh_mean(%), feels_like_max(℃, 기상청 여름철 식으로 시간별 계산 후 일 최대, 5~9월만 값이 있고 10~4월은 NaN), precip_flag9_hours. 강수·적설·신적설 빈칸은 0 처리(`data_notes.md` 마지막 절). 날짜(date)로 `station_daily`/`group_daily`에 붙임. |
| `weather_daily_check.csv` | 12 × 10 | 연도별 시간 빈틈 점검표(기대·실제 시간 수, 12월 31일 시간 수, 1월 1일 0시 유무, 기온·습도 빈 시간 수, NaN 일수). |
| `model_input_9stations.csv` | 39,447 × 40 | **모델 입력: 단독형 역 9개 × 2014-01-01~2025-12-31 일별** (날짜 빈틈·`total` 결측 없음). 열 설명은 아래 '모델 입력 열 설명'. |

## 역 이름 통일·범위 (2~3단계)
| 파일 | 행 | 설명 |
|---|---|---|
| `station_map.csv` | 482 | 원본 이름(raw_name) × 노선 → 표준역명(station) 대응. 시작·종료일, 유효일수, 환승 그룹의 노선 수, 서울 소재 여부(in_seoul), 제외 사유. |
| `rename_candidates.csv` | 3 | 이름 변경 후보(신천→잠실새내, 동대문역사문화공원5→동대문역사문화공원, 뚝섬유원지→자양). 변경 전후 28일 일평균 승차 포함. |
| `new_closed_stations.csv` | 16 | 개명이 아닌 신설 역(5호선 하남 구간·8호선 남위례·암사역사공원 등)과 자료상 소멸 역(7호선 부천 구간 9개). |
| `excluded_stations.csv` | 24 | 서울 외 소재라 제외한 노선-역, 소재지와 확인 방식. |
| `always_present_stations.csv` | 222 | 2014~2024년 매년 연 300일 이상 값이 있는 역. |
| `station_presence_by_year.csv` | 224 | 역별 연도별 존재 일수. |
| `file_structure.csv` | 14 | 1단계 원본 파일 구조 요약(형식, 기간, 누락일, 시간대 수 등). xlsx 3개의 형식 표기 오류를 스크립트 정리 때 바로잡음. 2025 파일은 포함하지 않음. |

## 역 유형 분류 (4단계)
| 파일 | 행 | 설명 |
|---|---|---|
| `station_type_lists.csv` | 8 | 유형별 역 수와 역 목록(미분류 포함). |
| `step4_variable_definition_test.csv` | 5 | 변수 정의(비율 vs 이용객 수)별 고유값·누적분산 비교. 이용객 수 정의에서 논문과 같은 3요인 구조가 나와 채택. |
| `step4_table2_comparison.csv` | 10 | 고유값·%분산·%누적, 논문 표 2와 나란히. |
| `step4_table3_comparison.csv` | 30 | 변수 30개의 oblimin 패턴 행렬 적재량과 논문 표 3 일치 여부(23/30 일치). |
| `step4_table4_comparison.csv` | 54 | 논문 표 4의 역별 유형 비교(일치 41 / 부분 일치 5 / 불일치 8). |
| `yearly_station_types.csv` | 223 | 연도별(각 3월) 분류 결과와 역별 안정도. |
| `yearly_agreement_summary.csv` | 8 | 연도별 고유값, 분류 역 수, 통합 분류와의 일치율. |
| `yearly_agreement_pairwise.csv` | 8 | 연도쌍 유형 일치율(%) 행렬. |
| `station_type_change_vs_2005.csv` | 21 | 논문 그림 4 대비 영역이 바뀐 역(신규·이탈·이동). |

## 데이터 품질 점검 (5단계)
| 파일 | 행 | 설명 |
|---|---|---|
| `data_quality_summary.csv` | 18 | 결측일·0인 날·집계 오류·급감일 요약표. |
| `data_quality_missing_zero.csv` | 4 | 역별 결측·0인 날 상세(강일, 신내, 암사역사공원). |
| `anomaly_drop_days_station.csv` | 17,870 | 역 단위 급감(전주 같은 요일 대비 −50% 이하) 목록과 공휴일 여부. |
| `anomaly_drop_days_group.csv` | 358 | 그룹 단위 급감 목록. |
| `anomaly_drop_by_year.csv` | 11 | 연도별 급감 역-일 수. |
| `new_year_check_2025.csv` | 3 | 2025 파일 점검에서 나온 이름 변경 목록(확인 대기). |

## 그림 (`../../figures/`, PNG 300dpi, 텍스트는 영어·역 이름은 한글)
| 파일 | 설명 |
|---|---|
| `fig_hourly_profile_4stations.png` | 논문 그림 2 재현. 강남·강변·명동·아현의 평일/주말 승차·하차 시간대 분포(자기 방향 하루 합 대비 %). |
| `fig_station_type_venn.png` | 논문 그림 4 재현. 3기능 벤 다이어그램과 미분류 개수. |
| `fig_group_daily_trend.png` | 그룹별 일별 이용객 추이(7일 이동평균, 절대값과 2019년=100 지수). |
| `fig_eda_weekly_pattern_2024.png` | EDA: 2024년 한 해 전체 이용량 변화와 주별 반복성(일별 추이, 요일별 분포, 자기상관). 요일별 수치는 `eda_weekly_pattern_2024.csv`. |

## 사용 시 주의
- 분류된 역은 53개, 미분류는 170개(76%). 점수가 이용객 수 기반이라 규모가 큰 역이 분류되기 쉬움(분류된 역 일평균 이용객 중앙값은 미분류의 약 3.5배).
- 연도별(각 3월)로 돌렸을 때 28개 역은 연도에 따라 유형이 바뀜(`yearly_station_types.csv`).
- 급감일은 대부분 공휴일·명절이므로 이상값 처리 시 `is_public_holiday`, `is_seollal_chuseok`를 함께 볼 것.
- 이전 한글 열 이름과의 대응은 `data_notes.md` 맨 아래 대응표.
- 전처리 코드는 프로젝트의 `scripts/`에 있음(아래 '스크립트').

## 스크립트 (`../../scripts/`)
단계별 파이썬 파일(`step1_load_raw.py` … `step7_add_test_year.py`)과 공통 모듈(`common.py`), 실행 방법은 `scripts/README.md`에 있습니다.
1~6단계를 별도 폴더에서 처음부터 실행해 이 폴더의 파일과 비교했고, 17개 파일이 값까지 동일했습니다.

## 2025년 추가 내용 (테스트 기간, 완료)
- 원본: `rawdata/서울교통공사_역별 일별 시간대별 승하차인원_20251231.csv`. 형식은 2023년 파일과 같고, 파일 끝의 빈 행 134개를 제외하면 365일 완전(누락·중복·음수·NaN 없음).
- 같은 스크립트(`scripts/step7_add_test_year.py`)로 역명 통일 → 서울 외 역 제외 → 일별·그룹별 합계를 만들어 `station_daily`/`group_daily`에 `split=test`로 추가. 224개 역, 새로 생긴 역·사라진 역 없음. 유형은 다시 분류하지 않고 기존 `station_type.csv`를 그대로 붙임.
- **이름이 바뀐 역 3건**(2025-04-01부터, 목록은 `new_year_check_2025.csv`)
  - 4·6호선 삼각지 → 삼각지(전쟁기념관): 부기명을 지우면 같은 이름이라 표준역명 `삼각지` 그대로.
  - 4호선 당고개(409) → 불암산: 같은 역으로 보고 표준역명은 기존 `당고개`로 유지(학습 파일 변화 없음). 2025-04-01 이후 행도 `station`은 `당고개`이고 실제 이름은 `station_map.csv`의 raw_name `불암산`으로 확인 가능.

## 모델 입력 열 설명 (`model_input_9stations.csv`)
대상 역: Employment(시청·역삼·선릉), Commercial & Leisure(신촌·혜화·신사), Residential(서울대입구·수유·연신내). 근거와 날짜 확인 출처는 `data_notes.md` 마지막 절.

| 열 | 설명 |
|---|---|
| `station`, `district`, `date` | 역, 자치구(특보 권역 연결용), 날짜 |
| `split` | 2014~2023 train, 2024 validation, 2025 test |
| `boarding`, `alighting`, `total`, `flag_one_side_zero` | `station_daily`와 같음 (이 9개 역은 `total` 결측 0) |
| `day_of_week`, `is_weekend`, `group` | 요일, 주말(토·일) 여부, 역 유형 |
| `score_commercial_leisure`, `score_residential`, `score_employment` | 역별 요인점수(역마다 한 값) |
| `holiday` | 공휴일·대체·임시공휴일·선거일 = 1 (`station_daily.is_public_holiday`) |
| `myeongjeol_pm1` | 설날·추석 당일 ±1일 = 1 |
| `covid` / `mers` / `strike` | 2020-02-01~2022-04-30 / 2015-05-20~2015-07-28 / 2016-09-27~29, 2022-11-30, 2023-11-09~10 = 1 (전 역) |
| `rally_cityhall` | 시청역만: 2016-10-29~2017-04-29, 2024-12-07~2025-04-05 사이의 토요일 = 1 (그 외 역은 0) |
| `fare_2015`, `fare_2023`, `fare_2025`, `climate_card` | 요금 인상 2015-06-27 / 2023-10-07 / 2025-06-28, 기후동행카드 2024-01-27부터 1 (전 역) |
| `sillim_line`, `sinbundang_ext` | 2022-05-28부터 1: 서울대입구만 / 신사만 (그 외 역은 0) |
| `temp_mean` … `precip_flag9_hours` (10열) | `weather_daily.csv`를 `date`로 붙임. `feels_like_max`는 5~9월만 값 있음 |
| `alert_heat`, `alert_cold`, `alert_rain`, `alert_snow` | 기상특보. **지금은 전부 비어 있음**(나중에 `rawdata/alert/`를 `district` 기준으로 채움) |

역별 가리기 열이 1인 날 수 (전체 4,383일씩):

| 역 | holiday | myeongjeol_pm1 | covid | mers | strike | rally_cityhall | total 결측 |
|---|---|---|---|---|---|---|---|
| 시청 | 215 | 72 | 820 | 70 | 6 | 45 | 0 |
| 역삼 | 215 | 72 | 820 | 70 | 6 | 0 | 0 |
| 선릉 | 215 | 72 | 820 | 70 | 6 | 0 | 0 |
| 신촌 | 215 | 72 | 820 | 70 | 6 | 0 | 0 |
| 혜화 | 215 | 72 | 820 | 70 | 6 | 0 | 0 |
| 신사 | 215 | 72 | 820 | 70 | 6 | 0 | 0 |
| 서울대입구 | 215 | 72 | 820 | 70 | 6 | 0 | 0 |
| 수유 | 215 | 72 | 820 | 70 | 6 | 0 | 0 |
| 연신내 | 215 | 72 | 820 | 70 | 6 | 0 | 0 |
