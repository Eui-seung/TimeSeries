"""2단계: 역명 통일.

- 괄호 부기명·공백·끝의 '역'을 제거한 정규화 이름으로 연도별 (노선, 이름)을 비교
- 이름 변경 후보(rename_candidates.csv), 신설·소멸 역(new_closed_stations.csv),
  환승역 통합까지 반영한 station_map.csv(원본명, 노선, 표준역명=가장 최근 이름, 시작·종료일) 생성
- 학습 기간(2014~2024)만 사용. 일 합계 > 0인 날을 '존재'로 봄.

실행: python scripts/step2_station_map.py
"""
from common import *

# 이름이 달라도 같은 환승역으로 묶을 쌍 (확정: 4호선 총신대입구(이수) = 7호선 이수).
# 6호선 새절(신사) ↔ 3호선 신사는 별개 역이라 묶지 않음.
MANUAL_TRANSFER = [((4, '총신대입구'), (7, '이수'))]
PREFERRED_STANDARD = {'이수'}          # 표준역명 동률일 때 우선할 이름
PLACEHOLDER_MIN_DAYS = 30              # 유효일수가 이보다 적으면 자리표시 행으로 봄
MAX_GAP_DAYS = 31                      # 이름 변경 후보: 사라진 날과 생긴 날의 최대 간격
WIN = 28                               # 이름 변경 전후 일평균 승차 계산 일수


def main():
    ensure_dirs()
    A = load_interim(TRAIN_YEARS)
    A['norm'] = A.base.map(strip_yeok)
    act = A[A.tot > 0]
    START, END = A.date.min(), A.date.max()

    # (노선, 원본명) 단위 요약
    raw_all = A.groupby(['line', 'name_raw']).agg(rows=('date', 'size'), first_all=('date', 'min'), last_all=('date', 'max')).reset_index()
    raw_act = act.groupby(['line', 'name_raw']).agg(first=('date', 'min'), last=('date', 'max'), days=('date', 'nunique')).reset_index()
    R = raw_all.merge(raw_act, on=['line', 'name_raw'], how='left')
    R['days'] = R.days.fillna(0).astype(int)
    nm = A.drop_duplicates(['line', 'name_raw'])[['line', 'name_raw', 'stn_no', 'base', 'norm']]
    R = R.merge(nm, on=['line', 'name_raw'])
    R['first'] = R['first'].fillna(R.first_all)
    R['last'] = R['last'].fillna(R.last_all)
    # (노선, 부기명 제거 이름) 단위
    N = R.groupby(['line', 'base', 'norm']).agg(first=('first', 'min'), last=('last', 'max'), days=('days', 'sum'), stn_no=('stn_no', 'first')).reset_index()
    N['placeholder'] = N.days < PLACEHOLDER_MIN_DAYS

    # ---- 이름 변경 후보: 같은 노선에서 사라진 이름과 새로 생긴 이름의 짝 (역번호가 같으면 우선)
    good = N[~N.placeholder]
    old = good[good['last'] < END]
    new = good[good['first'] > START]
    dayq = act[act.io == '승차'].groupby(['line', 'base', 'date']).tot.sum()

    def avg(l, b, a, z):
        s = dayq.loc[(l, b)]
        s = s[(s.index >= a) & (s.index <= z)]
        return round(s.mean(), 1) if len(s) else np.nan

    rows = []
    for _, o in old.iterrows():
        for _, n in new[new.line == o.line].iterrows():
            gap = (n['first'] - o['last']).days
            same = o.stn_no == n.stn_no
            if same or 0 <= gap <= MAX_GAP_DAYS:
                b = avg(o.line, o.base, o['last'] - pd.Timedelta(days=WIN - 1), o['last'])
                a = avg(n.line, n.base, n['first'], n['first'] + pd.Timedelta(days=WIN - 1))
                rows.append(dict(line=o.line, old_name=o.base, new_name=n.base, old_last_date=o['last'].date(), new_first_date=n['first'].date(),
                                 change_period=str(n['first'].date())[:7], gap_days=gap, old_station_no=o.stn_no, new_station_no=n.stn_no, same_station_no=same,
                                 basis=('역번호 동일' + (' + 시기 인접' if 0 <= gap <= MAX_GAP_DAYS else '')) if same else '같은 노선·시기 인접(역번호 다름)',
                                 avg_daily_boarding_28d_before=b, avg_daily_boarding_28d_after=a,
                                 change_rate=round(a / b - 1, 3) if b == b and a == a and b else np.nan,
                                 matches_file_boundary=(str(o['last'].date()) == '2016-12-31')))
    RC = pd.DataFrame(rows).sort_values(['same_station_no', 'change_period'], ascending=[False, True])
    RC.to_csv(PROC / 'rename_candidates.csv', index=False, encoding='utf-8-sig')

    # ---- 같은 역 묶기 (union-find): 역번호 동일(개명) / 다른 노선 같은 정규화 이름(환승) / 수동 지정
    nodes = list(zip(N.line, N.base))
    par = {n: n for n in nodes}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    def union(a, b):
        par[find(a)] = find(b)

    for _, d in N.groupby('stn_no'):
        ns = list(zip(d.line, d.base))
        for a, b in zip(ns, ns[1:]):
            union(a, b)
    for _, d in N.groupby('norm'):
        ns = list(zip(d.line, d.base))
        for a, b in zip(ns, ns[1:]):
            union(a, b)
    for a, b in MANUAL_TRANSFER:
        union(a, b)
    N['grp'] = [find(n) for n in nodes]
    std = {}
    for g, d in N.groupby('grp'):
        d2 = d[~d.placeholder] if (~d.placeholder).any() else d
        c = d2[d2['last'] == d2['last'].max()]                      # 표준역명 = 가장 최근까지 쓰인 이름
        pick = c[c.base.isin(PREFERRED_STANDARD)] if c.base.isin(PREFERRED_STANDARD).any() else c
        std[g] = pick.base.iloc[0]
    N['station'] = N.grp.map(std)

    SM = R.merge(N[['line', 'base', 'grp', 'station', 'placeholder']], on=['line', 'base'])
    SM['note'] = np.where(SM.placeholder, '값 전부(거의) 0인 자리표시 행 - 유효 데이터 없음',
                          np.where((SM.days < SM.rows / 2 * 0.99) & (SM.days > 0) & (SM['last_all'] > SM['last']), '종료일 이후 값 0인 잔여 행 존재', ''))
    SM = SM.rename(columns={'name_raw': 'raw_name', 'norm': 'normalized_name', 'stn_no': 'station_no', 'first': 'start_date', 'last': 'end_date', 'days': 'valid_days'})
    SM['transfer_group_line_count'] = SM.grp.map(N.groupby('grp').line.nunique())
    SM = SM[['raw_name', 'normalized_name', 'line', 'station_no', 'station', 'start_date', 'end_date', 'valid_days', 'transfer_group_line_count', 'note']]
    SM = SM.sort_values(['station', 'line', 'start_date'])
    SM['start_date'] = SM.start_date.dt.date
    SM['end_date'] = SM.end_date.dt.date
    SM.to_csv(PROC / 'station_map.csv', index=False, encoding='utf-8-sig')     # in_seoul 열은 step3에서 추가

    # ---- 신설·소멸 역 (이름 변경이 아닌 것)
    G = N[~N.placeholder]
    nc = []
    for (l, s), d in G.groupby(['line', 'station']):
        f0, l0 = d['first'].min(), d['last'].max()
        if f0 > START:
            nc.append(dict(station=s, line=l, category='신설', first_date=f0.date(), last_date=l0.date(), note='이전 이름 없음(이름 변경 아님)'))
        if l0 < END:
            nc.append(dict(station=s, line=l, category='자료상 소멸', first_date=f0.date(), last_date=l0.date(),
                           note='2022년 이후 값 없음(잔여 0값 행만 존재). 폐역이 아니라 집계 제외로 추정 - 원인 확인 필요' if l == 7 else ''))
    pd.DataFrame(nc).drop_duplicates().to_csv(PROC / 'new_closed_stations.csv', index=False, encoding='utf-8-sig')
    print('station_map', len(SM), '행 / 표준역명', SM.station.nunique(), '개 | 이름 변경 후보', len(RC), '건 | 신설·소멸', len(nc), '건')


if __name__ == '__main__':
    main()
