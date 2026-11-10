"""5단계: 최종 데이터 (역×날짜 일별, 그룹×날짜 일별).

- station_date_hour의 시간대를 합산해 역×날짜 일별 승차·하차·합계 생성
- station_type.csv의 유형·요인점수·분석 그룹 붙이기 (유형은 다시 분류하지 않고 기존 결과를 그대로 사용)
- 요일, 공휴일(holidays KR), 설날·추석 연휴 열 추가
- split 열: 2014~2024 = train, 2025 = test
- 승차 또는 하차 한쪽만 0이던 날(시간대 표에서 NaN 처리됨)은 일별 값·합계가 NaN이고 flag_one_side_zero=1,
  그룹 합계에서는 해당 역-일을 승차·하차·합계 모두 뺌
- 같은 함수들을 step7(테스트 연도 추가)이 재사용함

실행: python scripts/step5_daily_tables.py
"""
from common import *          # (먼저 import: 추가 패키지 경로 설정)
import holidays

TYPE_EN = {'고용형': 'Employment', '주거형': 'Residential', '상업·여가형': 'Commercial & Leisure', '복합형(상업·여가+고용)': 'Mixed (Commercial + Employment)',
           '복합형(상업·여가+주거)': 'Mixed (Commercial + Residential)', '복합형(상업·여가+주거+고용)': 'Mixed (All three)', '복합형(주거+고용)': 'Mixed (Residential + Employment)',
           '미분류': 'Unclassified', '분류제외(3월 존재 연도 2개 미만)': 'Excluded'}
SINGLE = ('Employment', 'Residential', 'Commercial & Leisure', 'Unclassified', 'Excluded')
EXTRA_CHUSEOK = [pd.Timestamp('2017-10-02'), pd.Timestamp('2017-10-06')]     # 이름에 '추석'이 없는 추석 연휴 임시공휴일
COLS = ['station', 'date', 'split', 'boarding', 'alighting', 'total', 'day_of_week', 'is_weekend', 'is_public_holiday', 'holiday_name', 'is_seollal_chuseok',
        'type_kr', 'type_en', 'group', 'score_commercial_leisure', 'score_residential', 'score_employment', 'years_used', 'flag_one_side_zero']


def split_of(year):
    return 'train' if year in TRAIN_YEARS else 'test'


def daily_from_hourly(T):
    """T: step3 형식의 시간대 표(date는 Timestamp) -> 역×날짜 승차·하차·합계 + flag"""
    v = T[HLAB].sum(axis=1, min_count=1)               # 한쪽 NaN 처리된 방향은 NaN 유지
    D = T.assign(v=v).groupby(['station', 'date', 'direction']).v.sum(min_count=1).unstack('direction').rename(columns={'승차': 'boarding', '하차': 'alighting'}).reset_index()
    D.columns.name = None
    D['total'] = D.boarding + D.alighting
    fl = T.groupby(['station', 'date']).flag_one_side_zero_set_nan.max().rename('flag_one_side_zero').reset_index()
    return D.merge(fl, on=['station', 'date'])


def calendar(D):
    years = range(D.date.dt.year.min(), D.date.dt.year.max() + 1)
    h = holidays.country_holidays('KR', years=years, language='ko')
    hs = pd.Series({pd.Timestamp(k): v for k, v in h.items()})
    D['day_of_week'] = D.date.dt.day_name()
    D['is_weekend'] = (D.date.dt.dayofweek >= 5).astype(int)
    D['holiday_name'] = D.date.map(hs).fillna('')
    D['is_public_holiday'] = (D.holiday_name != '').astype(int)
    ext = D.date.isin(EXTRA_CHUSEOK)
    D['is_seollal_chuseok'] = (D.holiday_name.str.contains('설날|추석') | ext).astype(int)
    D['holiday_name'] = np.where(ext, D.holiday_name + ' (Chuseok extension)', D.holiday_name)
    D['split'] = D.date.dt.year.map(split_of)
    return D


def attach_type(D, st):
    """station_type.csv(이미 분류된 결과)를 그대로 붙임. 학습 때 없던 역은 'New (not in training)'."""
    st = st.copy()
    st['type_en'] = st.type_kr.map(TYPE_EN)
    assert st.type_en.notna().all(), st[st.type_en.isna()].type_kr.unique()
    st['group'] = st.type_en.map(lambda t: t if t in SINGLE else 'Mixed')
    D = D.merge(st[['station', 'type_kr', 'type_en', 'group', 'score_commercial_leisure', 'score_residential', 'score_employment', 'years_used']], on='station', how='left')
    new = D.type_en.isna()
    D.loc[new, ['type_kr', 'type_en', 'group']] = ['신규역(학습 미포함)', 'New (not in training)', 'New']
    return D


def group_daily(D):
    ok = D[(D.group != 'Excluded') & (D.flag_one_side_zero == 0)]
    G = ok.groupby(['group', 'date']).agg(boarding=('boarding', 'sum'), alighting=('alighting', 'sum'), total=('total', 'sum'), n_stations=('station', 'nunique')).reset_index()
    G['split'] = G.date.dt.year.map(split_of)
    return G[['group', 'date', 'split', 'boarding', 'alighting', 'total', 'n_stations']]


def write(D, G):
    D = D[COLS].sort_values(['station', 'date']).copy()
    D['date'] = D.date.dt.date
    D.to_csv(PROC / 'station_daily.csv.gz', index=False, encoding='utf-8-sig', compression='gzip')
    G = G.sort_values(['group', 'date']).copy()
    G['date'] = G.date.dt.date
    G.to_csv(PROC / 'group_daily.csv', index=False, encoding='utf-8-sig')


def main():
    ensure_dirs()
    T = pd.read_csv(PROC / 'station_date_hour.csv.gz', parse_dates=['date'])
    D = daily_from_hourly(T)
    D = calendar(D)
    D = attach_type(D, pd.read_csv(PROC / 'station_type.csv'))
    G = group_daily(D)
    write(D, G)
    print('station_daily', D.shape, D.station.nunique(), '개 역 |', D.date.min().date(), '~', D.date.max().date(), '| group_daily', G.shape)
    print(G.groupby('group').n_stations.agg(['min', 'max']).to_dict('index'))


if __name__ == '__main__':
    main()
