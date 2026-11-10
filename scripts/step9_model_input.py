"""9단계: 모델 입력 데이터 (단독형 역 9개 × 2014-01-01~2025-12-31 일별).

- 대상: Employment(시청·역삼·선릉), Commercial & Leisure(신촌·혜화·신사), Residential(서울대입구·수유·연신내)
- station_daily.csv.gz에서 뽑고(날짜 빈틈 없이 재인덱스), 역별 요인점수 3개와 자치구를 붙임
- split: 2014~2023 train, 2024 validation, 2025 test
- 가리기용 0/1 열: holiday, labor_day(매년 5/1), myeongjeol_pm2(설날·추석 당일 ±2일), covid, mers, strike, rally_cityhall(시청역만)
- 계단 더미(시행일부터 1): fare_2015/fare_2023/fare_2025/climate_card(전 역), sillim_line(서울대입구만), sinbundang_ext(신사만)
- 계절 더미: summer_vacation(매년 7월 마지막 월요일 ~ 8월 둘째 일요일 = 1, 전 역)
- weather_daily.csv를 date로 left join
- 특보 열(alert_heat/cold/rain/snow)은 비워 둠(나중에 rawdata/alert/ 파일을 district 열 기준으로 붙임)

실행: python scripts/step8_weather_daily.py 후 python scripts/step9_model_input.py
"""
from common import *          # (먼저 import: 추가 패키지 경로 설정)

START, END = '2014-01-01', '2025-12-31'
# 역 -> 자치구 (특보 권역 연결용)
STATIONS = {'시청': '중구', '역삼': '강남구', '선릉': '강남구',
            '신촌': '서대문구', '혜화': '종로구', '신사': '강남구',
            '서울대입구': '관악구', '수유': '강북구', '연신내': '은평구'}
STRIKE = [('2016-09-27', '2016-09-29'), ('2022-11-30', '2022-11-30'), ('2023-11-09', '2023-11-10')]
RALLY = [('2016-10-29', '2017-04-29'), ('2024-12-07', '2025-04-05')]       # 시청역, 토요일만
STEPS = {'fare_2015': '2015-06-27', 'fare_2023': '2023-10-07', 'fare_2025': '2025-06-28', 'climate_card': '2024-01-27'}
STATION_STEPS = {'sillim_line': ('서울대입구', '2022-05-28'), 'sinbundang_ext': ('신사', '2022-05-28')}
MYEONGJEOL_DAYS = 2                                           # 명절 당일 앞뒤로 가리는 일수 (원래 1이었다가 연휴 직전 토요일까지 가리려고 2로 넓힘)
SEASONS = ['summer_vacation']
MASKS = ['holiday', 'labor_day', 'myeongjeol_pm2', 'covid', 'mers', 'strike', 'rally_cityhall']
WEATHER = ['temp_mean', 'tmax', 'tmin', 'rain_mm', 'snow_depth_max', 'snow_new_cm', 'wind_mean', 'rh_mean', 'feels_like_max', 'precip_flag9_hours']
ALERTS = ['alert_heat', 'alert_cold', 'alert_rain', 'alert_snow']


def summer_vacation_range(year):
    """7월의 마지막 월요일 ~ 8월의 두 번째 일요일"""
    jul = pd.date_range(f'{year}-07-01', f'{year}-07-31')
    start = jul[jul.dayofweek == 0].max()
    aug = pd.date_range(f'{year}-08-01', f'{year}-08-31')
    return start, aug[aug.dayofweek == 6][1]


def in_ranges(dates, ranges):
    m = pd.Series(False, index=dates.index)
    for a, b in ranges:
        m |= (dates >= a) & (dates <= b)
    return m.astype(int)


def build():
    sd = pd.read_csv(PROC / 'station_daily.csv.gz', parse_dates=['date'])
    sd = sd[sd.station.isin(STATIONS) & (sd.date >= START) & (sd.date <= END)]
    assert sd.station.nunique() == len(STATIONS) and not sd.duplicated(['station', 'date']).any()
    idx = pd.MultiIndex.from_product([list(STATIONS), pd.date_range(START, END)], names=['station', 'date'])
    D = sd.set_index(['station', 'date']).reindex(idx).reset_index()      # 날짜 빈틈 없이
    D['district'] = D.station.map(STATIONS)
    for c in ['group', 'score_commercial_leisure', 'score_residential', 'score_employment']:
        D[c] = D.groupby('station')[c].transform('first')
    y = D.date.dt.year
    D['split'] = np.where(y <= 2023, 'train', np.where(y == 2024, 'validation', 'test'))
    D['day_of_week'] = D.date.dt.day_name()
    D['is_weekend'] = (D.date.dt.dayofweek >= 5).astype(int)

    # ---- 가리기용 0/1
    cal = sd.drop_duplicates('date').set_index('date').holiday_name.reindex(pd.date_range(START, END)).fillna('')
    D['holiday'] = D.date.map(sd.drop_duplicates('date').set_index('date').is_public_holiday).fillna(0).astype(int)
    D['labor_day'] = ((D.date.dt.month == 5) & (D.date.dt.day == 1)).astype(int)         # 근로자의 날(법정 공휴일 아님, 업무지구는 사실상 휴일)
    day0 = cal.index[[any(t.strip() in ('설날', '추석') for t in n.split(';')) for n in cal]]      # 명절 당일
    near = set().union(*[set(day0 + pd.Timedelta(days=k)) for k in range(-MYEONGJEOL_DAYS, MYEONGJEOL_DAYS + 1)])
    D['myeongjeol_pm2'] = D.date.isin(near).astype(int)      # 당일 ±MYEONGJEOL_DAYS(=2)일
    D['covid'] = in_ranges(D.date, [('2020-02-01', '2022-04-30')])
    D['mers'] = in_ranges(D.date, [('2015-05-20', '2015-07-28')])
    D['strike'] = in_ranges(D.date, STRIKE)
    D['rally_cityhall'] = (in_ranges(D.date, RALLY) & (D.date.dt.dayofweek == 5) & (D.station == '시청')).astype(int)

    # ---- 계절 더미
    D['summer_vacation'] = in_ranges(D.date, [summer_vacation_range(y) for y in range(2014, 2026)])

    # ---- 계단 더미
    for k, d in STEPS.items():
        D[k] = (D.date >= d).astype(int)
    for k, (st, d) in STATION_STEPS.items():
        D[k] = ((D.date >= d) & (D.station == st)).astype(int)

    # ---- 날씨 (date로 left join), 특보(빈 열)
    W = pd.read_csv(PROC / 'weather_daily.csv', parse_dates=['date'])
    n = len(D)
    D = D.merge(W, on='date', how='left')
    assert len(D) == n
    for c in ALERTS:
        D[c] = np.nan

    cols = ['station', 'district', 'date', 'split', 'boarding', 'alighting', 'total', 'flag_one_side_zero', 'day_of_week', 'is_weekend', 'group',
            'score_commercial_leisure', 'score_residential', 'score_employment'] + MASKS + list(STEPS) + SEASONS + list(STATION_STEPS) + WEATHER + ALERTS
    return D[cols]


def summary(D):
    S = D.groupby('station', sort=False).agg(**{c: (c, 'sum') for c in MASKS}, total_nan=('total', lambda s: int(s.isna().sum())))
    S.insert(0, 'days', D.groupby('station', sort=False).size())
    return S


if __name__ == '__main__':
    ensure_dirs()
    D = build()
    out = D.copy()
    out['date'] = out.date.dt.date
    out.to_csv(PROC / 'model_input_9stations.csv', index=False, encoding='utf-8-sig')
    print(D.shape, D.date.min().date(), D.date.max().date())
    print(D.groupby('split').date.agg(['min', 'max', 'nunique']).to_string())
    print(summary(D).to_string())
    print('summer_vacation:', {y: (str(a.date()), str(b.date()), (b - a).days + 1) for y in range(2014, 2026) for a, b in [summer_vacation_range(y)]})
    print('NaN 열별:', D.isna().sum()[lambda s: s > 0].to_dict())
