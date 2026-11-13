"""8단계: 기상청 ASOS 서울(108) 시간자료 -> 일별 날씨 (2014~2025).

- 입력: rawdata/asos/OBS_ASOS_TIM_*.csv (cp949). 지점 108(서울)만 사용, 2026년 이후 시각은 버림
- 일별 변환은 개인 프로젝트의 build_merged.py와 같은 기준:
  강수량·적설 빈칸은 0으로 보고(rain_mm=일 합계, snow_depth_max=일 최대), 관측 시간 수가 20 미만인 날은 값을 NaN으로 둠
  (3시간신적설도 빈칸=0, snow_new_cm=일 합계)
- 습도(%)는 일 평균(rh_mean). 체감온도는 기상청 여름철 체감온도 식(Ta=기온, RH=습도)으로 시간별로 계산한 뒤 일 최대(feels_like_max):
    Tw = Ta*atan(0.151977*sqrt(RH+8.313659)) + atan(Ta+RH) - atan(RH-1.676331) + 0.00391838*RH**1.5*atan(0.023101*RH) - 4.686035
    체감온도 = -0.2442 + 0.55399*Tw + 0.45535*Ta - 0.0022*Tw**2 + 0.00278*Tw*Ta + 3.0
  (여름철 식이라 5~9월에만 값을 두고 10~4월은 NaN. 기온이나 습도가 빈 시간은 그 시간의 체감온도를 NaN으로 둠)
- 강수량 QC플래그 9인 시간 수는 precip_flag9_hours로 따로 남김(값은 그대로 사용)
- 연도별 시간 빈틈 점검표를 weather_daily_check.csv로 저장

실행: python scripts/step8_weather_daily.py
"""
from common import *          # (먼저 import: 추가 패키지 경로 설정)

ASOS = RAW / 'asos'
STATION = 108
START, END = '2014-01-01', '2025-12-31'
MIN_HOURS = 20                # 하루 관측 시간 수가 이보다 적으면 그날 값은 NaN
COLS = ['date', 'temp_mean', 'tmax', 'tmin', 'rain_mm', 'snow_depth_max', 'snow_new_cm', 'wind_mean', 'rh_mean', 'feels_like_max', 'precip_flag9_hours']


def feels_like(ta, rh):
    """기상청 여름철 체감온도(℃)."""
    tw = (ta * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + np.arctan(ta + rh) - np.arctan(rh - 1.676331)
          + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) - 4.686035)
    return -0.2442 + 0.55399 * tw + 0.45535 * ta - 0.0022 * tw ** 2 + 0.00278 * tw * ta + 3.0


def load_hourly():
    fs = sorted(f for f in ASOS.glob('OBS_ASOS_TIM_*.csv'))
    w = pd.concat([pd.read_csv(f, encoding='cp949') for f in fs], ignore_index=True)
    n0 = len(w)
    w = w[w['지점'] == STATION].copy()                        # 2015년 파일에 섞인 관악산(116) 행 등 제거
    w['dt'] = pd.to_datetime(w['일시'])
    w = w[(w.dt >= START) & (w.dt < pd.Timestamp(END) + pd.Timedelta(days=1))]
    dup = w.dt.duplicated().sum()
    if dup:
        raise ValueError(f'중복 시각 {dup}건')
    print(f'파일 {len(fs)}개, 읽은 행 {n0}, 지점 {STATION}·{START}~{END} 행 {len(w)}')
    return w


def to_daily(w):
    w = w.assign(date=w.dt.dt.normalize(),
                 rain=w['강수량(mm)'].fillna(0),             # 빈칸 = 강수 없음
                 snow=w['적설(cm)'].fillna(0),               # 빈칸 = 적설 없음
                 snow_new=w['3시간신적설(cm)'].fillna(0),    # 빈칸 = 신적설 없음
                 fl=feels_like(w['기온(°C)'], w['습도(%)']))
    g = w.groupby('date')
    wd = pd.DataFrame({
        'temp_mean': g['기온(°C)'].mean(), 'tmax': g['기온(°C)'].max(), 'tmin': g['기온(°C)'].min(),
        'rain_mm': g.rain.sum(), 'snow_depth_max': g.snow.max(), 'snow_new_cm': g.snow_new.sum(),
        'wind_mean': g['풍속(m/s)'].mean(), 'rh_mean': g['습도(%)'].mean(), 'feels_like_max': g.fl.max(),
        'precip_flag9_hours': g['강수량 QC플래그'].apply(lambda s: int((s == 9).sum())),
        'hours': g['기온(°C)'].count(),
    })
    wd = wd.reindex(pd.date_range(START, END, name='date'))
    wd['hours'] = wd.hours.fillna(0).astype(int)
    wd['precip_flag9_hours'] = wd.precip_flag9_hours.fillna(0).astype(int)
    partial = wd.hours < MIN_HOURS
    wd.loc[partial, COLS[1:-1]] = np.nan
    wd.loc[~wd.index.month.isin(range(5, 10)), 'feels_like_max'] = np.nan      # 여름철 식: 5~9월만
    return wd


def hour_check(w, wd):
    """연도별: 있어야 할 시간 수, 실제 시간 수, 12월 31일 시간 수, 1월 1일 0시 유무, 값이 NaN인 날 수"""
    rows = []
    for y in range(2014, 2026):
        wy = w[w.dt.dt.year == y]
        exp = len(pd.date_range(f'{y}-01-01', f'{y}-12-31 23:00', freq='h'))
        have = pd.DatetimeIndex(wy.dt)
        miss = pd.date_range(f'{y}-01-01', f'{y}-12-31 23:00', freq='h').difference(have)
        d31 = int((wy.dt.dt.strftime('%m-%d') == '12-31').sum())
        rows.append({'year': y, 'expected_hours': exp, 'present_hours': len(wy), 'missing_hours': len(miss),
                     'dec31_hours': d31, 'jan1_0000_present': int(pd.Timestamp(y, 1, 1) in have),
                     'missing_other_than_dec31_jan1_0000': int(sum(not ((m.month == 12 and m.day == 31) or (m.month == 1 and m.day == 1 and m.hour == 0)) for m in miss)),
                     'blank_temp_hours': int(wy['기온(°C)'].isna().sum()), 'blank_rh_hours': int(wy['습도(%)'].isna().sum()),
                     'days_nan': int(wd.loc[str(y)].temp_mean.isna().sum())})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    ensure_dirs()
    w = load_hourly()
    wd = to_daily(w)
    out = wd.reset_index()[COLS]
    out.to_csv(PROC / 'weather_daily.csv', index=False, encoding='utf-8-sig', float_format='%.4f')
    chk = hour_check(w, wd)
    chk.to_csv(PROC / 'weather_daily_check.csv', index=False, encoding='utf-8-sig')
    pd.set_option('display.width', 250)
    print(chk.to_string(index=False))
    print(out.shape, out.date.min().date(), out.date.max().date())
    print('NaN 열별:', out.isna().sum()[lambda s: s > 0].to_dict())
    print('NaN인 날 수:', int(out.temp_mean.isna().sum()), '| 전부 12/31인가:', bool((out[out.temp_mean.isna()].date.dt.strftime('%m-%d') == '12-31').all()))
