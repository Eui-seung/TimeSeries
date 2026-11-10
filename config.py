"""지하철 전처리 결과(data/processed/model_input_9stations.csv)를 쓰는 모델링 설정.

열 이름은 model_input_9stations.csv의 실제 열 이름과 같다.
"""

# 단독형 역 9개 (Employment / Commercial & Leisure / Residential 순)
STATIONS = ['시청', '역삼', '선릉', '신촌', '혜화', '신사', '서울대입구', '수유', '연신내']

# 가리기용 0/1 열 (전 역 공통; rally_cityhall은 시청역만 값이 1)
MASK_COLS = ['holiday', 'labor_day', 'myeongjeol_pm2', 'covid', 'mers', 'strike', 'rally_cityhall']

# 계단 더미 (전 역 공통)
STEP_COLS = ['fare_2015', 'fare_2023', 'fare_2025', 'climate_card']

# 해당 역만 쓰는 계단 더미: 역 -> 열 이름
STATION_STEP = {'서울대입구': 'sillim_line', '신사': 'sinbundang_ext'}

# 분할: 2014~2023 train, 2024 validation, 2025 test (split 열과 같은 기준)
TRAIN_END = '2023-12-31'
VALID = 2024
TEST = 2025

# SARIMA 차수: (p, d, q), (P, D, Q, s)
ORDER = (1, 0, 1)
SEASONAL_ORDER = (1, 1, 1, 7)

# 기상특보 발효 후 가리는 일수
ALERT_MASK_AFTER_DAYS = 7

# 적합 설정 (수렴 문제로 최적화 방법과 최대 반복 횟수를 명시)
FIT_METHOD = 'lbfgs'
FIT_MAXITER = 1000
