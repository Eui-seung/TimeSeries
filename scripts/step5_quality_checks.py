"""5단계(점검): 결측일·0인 날·집계 오류·급감일(전주 같은 요일 대비 -50% 이하) 점검표. 학습 기간(train) 기준.

산출: data_quality_missing_zero.csv, anomaly_drop_days_station.csv, anomaly_drop_days_group.csv,
      anomaly_drop_by_year.csv, data_quality_summary.csv

실행: python scripts/step5_quality_checks.py
"""
from common import *


def main():
    ensure_dirs()
    D = pd.read_csv(PROC / 'station_daily.csv.gz', parse_dates=['date'])
    D = D[D.split == 'train']
    G = pd.read_csv(PROC / 'group_daily.csv', parse_dates=['date'])
    G = G[G.split == 'train']
    cal = pd.date_range('2014-01-01', '2024-12-31')
    print('달력 전체에서 자료가 아예 없는 날:', len(cal.difference(pd.DatetimeIndex(D.date.unique()))), '일')

    # ---- 원자료(서울 소재 역) 기준 결측·0인 날
    SM = pd.read_csv(PROC / 'station_map.csv')
    A = load_interim(TRAIN_YEARS)
    mp = SM[['raw_name', 'line', 'station', 'in_seoul']].rename(columns={'raw_name': 'name_raw'})
    B = A.merge(mp, on=['name_raw', 'line'], how='left')
    B = B[B.in_seoul == 'Y']
    raw = B.groupby(['station', 'date']).agg(total=('tot', 'sum'), rows=('tot', 'size')).reset_index()
    rows = []
    for s, d in raw.groupby('station'):
        nz = d[d.total > 0].date
        if nz.empty:
            rows.append(dict(station=s, kind='all days zero (station has no valid data)', first_valid='', last_valid='', days=len(d), example_dates=f'{d.date.min().date()}~{d.date.max().date()}'))
            continue
        f, l = nz.min(), nz.max()
        w = pd.date_range(f, l)
        miss = w.difference(pd.DatetimeIndex(d.date))
        zero = d[(d.total == 0) & (d.date >= f) & (d.date <= l)].date
        before = d[(d.total == 0) & (d.date < f)].date
        after = d[(d.total == 0) & (d.date > l)].date
        for kind, ds in (('missing day inside active window', miss), ('zero day inside active window', zero), ('zero rows before first valid day', before), ('zero rows after last valid day', after)):
            if len(ds):
                rows.append(dict(station=s, kind=kind, first_valid=f.date(), last_valid=l.date(), days=len(ds), example_dates=','.join(str(x.date()) for x in ds[:5])))
    pd.DataFrame(rows).to_csv(PROC / 'data_quality_missing_zero.csv', index=False, encoding='utf-8-sig')

    # ---- 급감일
    tt = D.set_index(['date', 'station']).total.unstack().reindex(cal)
    chg = tt / tt.shift(7) - 1
    dr = chg.rename_axis('date').stack().rename('change').reset_index()
    dr = dr[np.isfinite(dr.change)]
    low = dr[dr.change <= -0.5].copy()
    low['total'] = [tt.at[d, s] for d, s in zip(low.date, low.station)]
    low['prev_week_total'] = [tt.shift(7).at[d, s] for d, s in zip(low.date, low.station)]
    cal_df = D[['date', 'is_public_holiday', 'holiday_name', 'is_seollal_chuseok', 'day_of_week']].drop_duplicates('date')
    low = low.merge(cal_df, on='date', how='left')
    pwc = cal_df.rename(columns={c: 'prev_' + c for c in cal_df.columns})
    pwc['date'] = pwc.prev_date + pd.Timedelta(days=7)
    low = low.merge(pwc[['date', 'prev_is_public_holiday']], on='date', how='left')
    low['group'] = low.station.map(D.drop_duplicates('station').set_index('station').group)
    low['change_pct'] = (low.change * 100).round(1)
    low = low[['station', 'group', 'date', 'day_of_week', 'total', 'prev_week_total', 'change_pct', 'is_public_holiday', 'holiday_name', 'is_seollal_chuseok', 'prev_is_public_holiday']].sort_values(['date', 'station'])
    low.to_csv(PROC / 'anomaly_drop_days_station.csv', index=False, encoding='utf-8-sig')
    g = G.pivot(index='date', columns='group', values='total').reindex(cal)
    g.index.name = 'date'
    gc = g / g.shift(7) - 1
    gl = gc.stack().rename('change').reset_index()
    gl = gl[gl.change <= -0.5].merge(cal_df, on='date', how='left')
    gl['change_pct'] = (gl.change * 100).round(1)
    gl.to_csv(PROC / 'anomaly_drop_days_group.csv', index=False, encoding='utf-8-sig')
    tot = g.sum(axis=1)
    tc = tot / tot.shift(7) - 1
    dd = tc[tc <= -.5].index
    nonh = [x for x in dd if cal_df.set_index('date').loc[x, 'is_public_holiday'] == 0]

    # ---- 요약표
    low['cause'] = np.where(low.is_seollal_chuseok == 1, 'Seollal/Chuseok', np.where(low.is_public_holiday == 1, 'Other public holiday', np.where(low.prev_is_public_holiday == 1, 'Previous-week day was a holiday', 'Not a holiday')))
    nh = low[low.cause == 'Not a holiday']
    nh = nh.assign(sub=np.where(nh.date.dt.strftime('%m-%d') == '05-01', 'May 1 (Labor Day, not a statutory holiday)', np.where(nh.day_of_week.isin(['Saturday', 'Sunday']), 'Weekend', 'Weekday (other)')))
    one = D[D.flag_one_side_zero == 1]
    gnh = gl[gl.is_public_holiday == 0]
    S = []
    add = lambda a, b, c, d: S.append(dict(category=a, item=b, count=c, detail=d))
    Q = pd.DataFrame(rows)
    add('Missing', 'Calendar dates with no data at all (2014-01-01~2024-12-31)', len(cal.difference(pd.DatetimeIndex(D.date.unique()))), '')
    m = Q[Q.kind == 'missing day inside active window']
    add('Missing', "Station-days missing inside a station's active window", int(m.days.sum()), '; '.join(f'{r.station} {r.days} day(s)' for r in m.itertuples()))
    z = Q[Q.kind == 'zero day inside active window']
    add('Zero', 'Station-days with zero total inside active window', int(z.days.sum()), '; '.join(f'{r.station} {r.example_dates}' for r in z.itertuples()))
    az = Q[Q.kind.str.startswith('all days zero')]
    add('Zero', 'Station with zero in all rows (removed from analysis tables)', int(az.days.sum()), '; '.join(f'{r.station} {r.example_dates}' for r in az.itertuples()))
    bz = Q[Q.kind == 'zero rows before first valid day']
    add('Zero', 'Zero rows before first valid day', int(bz.days.sum()), '; '.join(f'{r.station} (before opening, removed)' for r in bz.itertuples()))
    add('Aggregation error', 'Station-days where only boarding or only alighting is 0 (total>0) -> zero side and total set to NaN', len(one),
        '; '.join(f'{r.station} {r.date.date()}' for r in one.itertuples()) + '. flag_one_side_zero=1 in station_daily')
    fg = one[one.group != 'Excluded']
    gd = fg.groupby(['group', 'date']).size().reset_index()
    nst = G.set_index(['group', 'date']).n_stations
    add('Aggregation error', 'Group-days affected (stations left out of the group sum for that day)', len(gd),
        '; '.join(f'{r.group} {r.date.date()} ({int(nst[(r.group, r.date)])} stations in the sum)' for r in gd.itertuples()) + '; 암사역사공원 is in no group')
    add('Group', 'Group-days with total = 0', int((G.total == 0).sum()), '')
    for k, v in low.cause.value_counts().items():
        add('Drop (station, <= -50% vs same weekday last week)', k, int(v), '')
    for k, v in nh['sub'].value_counts().items():
        add('Drop (station, not a holiday)', k, int(v), '')
    add('Drop (station)', 'Total station-days / distinct stations / distinct dates', len(low), f'{low.station.nunique()} stations / {low.date.nunique()} dates')
    add('Drop (group)', 'Group-days (<= -50%)', len(gl), f'{gl.date.nunique()} distinct dates; not on a public holiday: {len(gnh)} (' + ', '.join(f'{r.group} {r.date.date()}' for r in gnh.itertuples()) + ')')
    add('Drop (all stations combined)', 'Dates (<= -50%)', len(dd), 'all are public holidays/Seollal/Chuseok' if not nonh else 'not holidays: ' + ','.join(str(x.date()) for x in nonh))
    pd.DataFrame(S).to_csv(PROC / 'data_quality_summary.csv', index=False, encoding='utf-8-sig')
    low.assign(year=low.date.dt.year).groupby('year').agg(station_days=('station', 'size'), stations=('station', 'nunique'), dates=('date', 'nunique')).to_csv(PROC / 'anomaly_drop_by_year.csv', encoding='utf-8-sig')
    print('급감 역-일', len(low), '| 그룹-일', len(gl), '| 한쪽만 0 NaN', len(one), '| 전 역 합계 급감일 중 공휴일 아님:', len(nonh))


if __name__ == '__main__':
    main()
