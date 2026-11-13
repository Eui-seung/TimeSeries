"""3단계: 역명 통일 적용 + 서울 외 역 제외 + 시간대 테이블.

- station_map(표준역명)으로 전체 자료의 이름을 바꾸고, 환승역은 노선을 합산
- 서울 외 소재 역(EXCLUDED)을 제외하고 excluded_stations.csv 저장, station_map에 in_seoul 열 추가
- 역×날짜×방향별 20개 시간대 테이블(station_date_hour.csv.gz) 저장
- 승차 또는 하차 한쪽만 0인 날은 집계 오류로 보고 0인 방향의 시간대 값을 NaN으로 바꿈
- 매년 존재하는 역 목록(always_present_stations.csv), 연도별 존재 일수(station_presence_by_year.csv)

실행: python scripts/step3_apply_map.py
"""
from common import *

# 서울 외 소재 역: (노선, 표준역명) -> (소재지, 확인방식). 소재지는 역 주소(웹 문서)로 확인.
W, K = '웹 문서 확인(itwiki.kr)', '행정구역상 명확(개별 문서 미확인)'
EXCLUDED = {
    (3, '지축'): ('경기도 고양시 덕양구 지축동', '웹 검색 확인(지축동 소재)'),
    (5, '미사'): ('경기도 하남시', K), (5, '하남풍산'): ('경기도 하남시', K), (5, '하남시청'): ('경기도 하남시', K), (5, '하남검단산'): ('경기도 하남시', K),
    (7, '장암'): ('경기도 의정부시 장암동', W), (7, '철산'): ('경기도 광명시', W), (7, '광명사거리'): ('경기도 광명시', K),
    (7, '까치울'): ('경기도 부천시', W), (7, '부천종합운동장'): ('경기도 부천시', W), (7, '춘의'): ('경기도 부천시', W), (7, '신중동'): ('경기도 부천시', W),
    (7, '부천시청'): ('경기도 부천시', W), (7, '상동'): ('경기도 부천시', W),
    (7, '삼산체육관'): ('인천광역시 부평구', W), (7, '굴포천'): ('인천광역시 부평구', W), (7, '부평구청'): ('인천광역시 부평구', W),
    (8, '남위례'): ('경기도 성남시 수정구 복정동', W), (8, '산성'): ('경기도 성남시', W), (8, '남한산성입구'): ('경기도 성남시', W),
    (8, '단대오거리'): ('경기도 성남시', W), (8, '신흥'): ('경기도 성남시', W), (8, '수진'): ('경기도 성남시', W), (8, '모란'): ('경기도 성남시', W),
}


def write_excluded_and_flag(SM):
    SM = SM.drop(columns=[c for c in ('in_seoul', 'exclusion_reason') if c in SM.columns])
    SM['in_seoul'] = ['N' if (l, s) in EXCLUDED else 'Y' for l, s in zip(SM.line, SM.station)]
    SM['exclusion_reason'] = ['서울 외 소재' if (l, s) in EXCLUDED else '' for l, s in zip(SM.line, SM.station)]
    g = SM.groupby(['line', 'station']).agg(station_no=('station_no', 'first'), first_date=('start_date', 'min'), last_date=('end_date', 'max')).reset_index()
    rows = []
    for (l, s), (addr, how) in EXCLUDED.items():
        r = g[(g.line == l) & (g.station == s)]
        assert len(r) == 1, ('station_map에 없는 제외 역', l, s)
        r = r.iloc[0]
        rows.append(dict(station=s, line=l, station_no=r.station_no, location=addr, exclusion_reason='서울 외 소재', verification=how, first_date=r.first_date, last_date=r.last_date))
    pd.DataFrame(rows).sort_values(['line', 'station_no']).to_csv(PROC / 'excluded_stations.csv', index=False, encoding='utf-8-sig')
    return SM


def build_hourly(A, SM):
    """A(정리된 원자료, base 열 포함) -> 표준역명×날짜×방향 시간대 표 (서울 소재 역만, 환승 합산)."""
    mp = SM[['raw_name', 'line', 'station', 'in_seoul']].rename(columns={'raw_name': 'name_raw'})
    B = A.merge(mp, on=['name_raw', 'line'], how='left')
    unmapped = B[B.station.isna()][['line', 'name_raw']].drop_duplicates()
    if len(unmapped):
        raise RuntimeError(f'station_map에 없는 (노선, 원본명): {unmapped.values.tolist()}')
    S = B[B.in_seoul == 'Y']
    T = S.groupby(['station', 'date', 'io'])[HOURS].sum(min_count=1).reset_index()
    assert abs(T[HOURS].sum().sum() - S[HOURS].sum().sum()) < 1
    # 역-일 합계가 0인 행 제거 (개통 전·자리표시 행 등)
    dd = T.groupby(['station', 'date'])[HOURS].sum(min_count=1).sum(axis=1)
    zero = set(dd[dd == 0].index)
    T = T[[(a, b) not in zero for a, b in zip(T.station, T.date)]].copy()
    # 승차 또는 하차 한쪽만 0 (합계>0): 집계 오류 -> 0인 방향 시간대 값을 NaN
    tot = T.assign(v=T[HOURS].sum(axis=1)).pivot(index=['station', 'date'], columns='io', values='v')
    one = tot[((tot['승차'] == 0) ^ (tot['하차'] == 0)) & ((tot['승차'] + tot['하차']) > 0)]
    T['flag_one_side_zero_set_nan'] = 0
    for (s, dt), r in one.iterrows():
        io = '승차' if r['승차'] == 0 else '하차'
        m = (T.station == s) & (T.date == dt) & (T.io == io)
        T.loc[m, HOURS] = np.nan
        T.loc[m, 'flag_one_side_zero_set_nan'] = 1
    T['flag_23plus_merged'] = (T.date.dt.year == 2021).astype(int)
    T = T.rename(columns={**HOUR_LABELS, 'io': 'direction'})
    T = T[['station', 'date', 'direction'] + HLAB + ['flag_23plus_merged', 'flag_one_side_zero_set_nan']]
    return T.sort_values(['station', 'date', 'direction']), len(zero), one


def main():
    ensure_dirs()
    SM = pd.read_csv(PROC / 'station_map.csv')
    SM = write_excluded_and_flag(SM)
    SM.to_csv(PROC / 'station_map.csv', index=False, encoding='utf-8-sig')
    print('서울 외 제외', len(EXCLUDED), '개 노선-역 / 남은 표준역명', SM[SM.in_seoul == 'Y'].station.nunique())
    A = load_interim(TRAIN_YEARS)
    T, n_zero, one = build_hourly(A, SM)
    print('역-일 합계 0 제거:', n_zero, '| 한쪽만 0 NaN 처리:', len(one), '|', T.shape)
    out = T.copy()
    out['date'] = out.date.dt.date
    out.to_csv(PROC / 'station_date_hour.csv.gz', index=False, encoding='utf-8-sig', compression='gzip')
    # 존재 일수
    P = T[T.direction == '승차'].assign(year=lambda x: x.date.dt.year).groupby(['station', 'year']).size().unstack(fill_value=0)
    P.to_csv(PROC / 'station_presence_by_year.csv', encoding='utf-8-sig')
    full = P[(P >= 300).all(axis=1)]
    pd.DataFrame({'station': full.index, 'n_years': len(TRAIN_YEARS), 'min_days_per_year': full.min(axis=1).values}).to_csv(PROC / 'always_present_stations.csv', index=False, encoding='utf-8-sig')
    print('매년 존재하는 역:', len(full), '/', len(P))


if __name__ == '__main__':
    main()
