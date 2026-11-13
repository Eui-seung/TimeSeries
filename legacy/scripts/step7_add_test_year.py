"""테스트 연도(기본 2025) 추가: 파일 구조 확인 → 역명 통일 확인 → 서울 외 역 제외 → 일별 합계 → station_daily / group_daily에 붙임.

중단 조건: 형식이 이전과 다르거나, 새로 생기거나 이름이 바뀐 역이 있으면 아무것도 바꾸지 않고
new_year_check_{year}.csv 에 목록을 저장한 뒤 종료(종료 코드 2). 사용자 확인 후 --accept-names 로 다시 실행.
- 역 유형은 다시 분류하지 않고 station_type.csv를 그대로 붙임 (학습 때 없던 역은 'New (not in training)')
- split 열: 2014~2024 = train, 2025 = test

실행: python scripts/step7_add_test_year.py --year 2025 [--accept-names]
"""
import argparse
from common import *
from step3_apply_map import build_hourly
from step5_daily_tables import daily_from_hourly, calendar, attach_type, group_daily, write, COLS


def header_of(fname):
    if fname.endswith('.xlsx'):
        return None
    return tuple(pd.read_csv(RAW / fname, encoding='cp949', nrows=0).columns)


def check_format(year, s, n_raw):
    """형식 점검 결과(정보, 문제)"""
    info, issues = [], []
    f = raw_file(year)
    h = header_of(f)
    known = {y: header_of(raw_file(y)) for y in range(2014, year) if y in TRAIN_YEARS or y == year - 1}
    same = [y for y, hh in known.items() if hh is not None and hh == h]
    info.append(f'파일 {nfc(f)} | 컬럼 {len(h)}개 | 이전 연도 중 같은 헤더: {same or "없음"}')
    if not same:
        issues.append(dict(issue='새로운 파일 형식(헤더가 이전 어느 해와도 다름)', detail=str(h)))
    blank = n_raw - len(s)
    info.append(f'원본 {n_raw:,}행 중 빈 행 {blank}개 제외 -> {len(s):,}행')
    d0, d1 = s.date.min(), s.date.max()
    miss = pd.date_range(d0, d1).difference(pd.DatetimeIndex(s.date.unique()))
    info.append(f'기간 {d0.date()} ~ {d1.date()} ({s.date.nunique()}일, 누락일 {len(miss)})')
    if len(miss) or d1 != pd.Timestamp(year, 12, 31) or d0 != pd.Timestamp(year, 1, 1):
        issues.append(dict(issue='연도 전체가 아니거나 누락일 있음', detail=f'{d0.date()}~{d1.date()}, 누락 {len(miss)}일'))
    info.append(f'방향 {sorted(s.io.unique())} | 노선 {sorted(s.line.unique())} | 시간대 NaN {int(s[HOURS].isna().sum().sum())} | 음수 {int((s[HOURS] < 0).sum().sum())} | 중복 {int(s.duplicated(["date", "line", "stn_no", "io"]).sum())}')
    if s[HOURS].isna().any().any() or (s[HOURS] < 0).any().any() or s.duplicated(['date', 'line', 'stn_no', 'io']).any() or sorted(s.io.unique()) != ['승차', '하차']:
        issues.append(dict(issue='값 이상(NaN/음수/중복/방향 값)', detail=info[-1]))
    return info, issues


def check_names(year, A, SM):
    """station_map과 비교: 새 이름·이름 변경·부기명 변경·새 역번호·사라진 역"""
    S = A.groupby(['line', 'name_raw', 'stn_no']).agg(first_date=('date', 'min'), last_date=('date', 'max')).reset_index()
    SM = SM.assign(station_no=SM.station_no.astype(str))
    key = set(zip(SM.raw_name, SM.line))
    cur = SM.sort_values('end_date').drop_duplicates(['line', 'station_no'], keep='last').set_index(['line', 'station_no'])
    rows = []
    for r in S.itertuples():
        if (r.name_raw, r.line) in key:
            continue
        k = (r.line, r.stn_no)
        if k not in cur.index:
            rows.append(dict(issue='새로 생긴 역(기존에 없던 역번호)', line=r.line, station_no=r.stn_no, old_name='', new_name=r.name_raw, first_date=r.first_date.date(), last_date=r.last_date.date()))
            continue
        old = cur.loc[k]
        kind = '부기명 변경(정규화 이름 같음, 자동 대응 가능)' if base_name(old.raw_name) == base_name(r.name_raw) else '이름 변경(역번호 동일)'
        rows.append(dict(issue=kind, line=r.line, station_no=r.stn_no, old_name=old.raw_name, new_name=r.name_raw, first_date=r.first_date.date(), last_date=r.last_date.date(), standard_name_now=old.station))
    present = set(zip(S.name_raw, S.line))
    prev_end = pd.Timestamp(year - 1, 12, 31).date().isoformat()
    for r in SM[SM.end_date >= prev_end].itertuples():
        if (r.raw_name, r.line) not in present:
            rows.append(dict(issue='사라진 역(전년 말까지 있었는데 이 해에 없음)', line=r.line, station_no=r.station_no, old_name=r.raw_name, new_name='', first_date='', last_date=''))
    return S, pd.DataFrame(rows)


def extend_station_map(SM, S, policy):
    """새 원본 이름을 기존 역에 대응시켜 station_map에 추가하고, 이 해에 나온 기존 이름의 종료일을 갱신. (policy='keep': 표준역명은 그대로)"""
    if policy != 'keep':
        raise NotImplementedError('표준역명을 새 이름으로 바꾸는 정책은 기존 산출물의 역 이름도 함께 바꿔야 해서 아직 지원하지 않음')
    SM = SM.copy()
    SM['station_no'] = SM.station_no.astype(str)
    add = []
    for r in S.itertuples():
        m = (SM.raw_name == r.name_raw) & (SM.line == r.line)
        if m.any():
            idx = SM[m].index
            SM.loc[idx, 'end_date'] = max(str(SM.loc[idx, 'end_date'].iloc[0]), str(r.last_date.date()))
            continue
        same_no = SM[(SM.line == r.line) & (SM.station_no == r.stn_no)]
        if same_no.empty:
            raise RuntimeError(f'새 역은 자동 처리하지 않음(소재지·유형 확인 필요): {r.line}호선 {r.name_raw} ({r.stn_no})')
        o = same_no.sort_values('end_date').iloc[-1]
        add.append(dict(raw_name=r.name_raw, normalized_name=strip_yeok(base_name(r.name_raw)), line=r.line, station_no=r.stn_no, station=o.station,
                        start_date=r.first_date.date(), end_date=r.last_date.date(), valid_days=(r.last_date - r.first_date).days + 1,
                        transfer_group_line_count=o.transfer_group_line_count, note=f'{r.first_date.year}년 새 표기(이전: {o.raw_name})', in_seoul=o.in_seoul, exclusion_reason=o.exclusion_reason))
    return pd.concat([SM, pd.DataFrame(add)], ignore_index=True) if add else SM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, default=2025)
    ap.add_argument('--accept-names', action='store_true', help='새·변경된 이름을 확인했고 기존 역에 대응시켜 진행')
    ap.add_argument('--rename-policy', default='keep', choices=['keep', 'latest'])
    a = ap.parse_args()
    ensure_dirs()
    y = a.year
    s, n_raw = load_year(y)
    s.to_pickle(INTERIM / f'raw_{y}.pkl')
    info, issues = check_format(y, s, n_raw)
    print('\n'.join('  ' + i for i in info))
    SM = pd.read_csv(PROC / 'station_map.csv')
    A = load_interim([y])
    S, nm = check_names(y, A, SM)
    print(f'  역(노선·이름) {len(S)}개 | 이름 점검 결과 {len(nm)}건')
    out = pd.concat([pd.DataFrame(issues), nm], ignore_index=True)
    if len(out):
        out.to_csv(PROC / f'new_year_check_{y}.csv', index=False, encoding='utf-8-sig')
        print(out.to_string(index=False))
    if len(out) and not a.accept_names:
        print(f'\n중단: 위 항목을 확인한 뒤 --accept-names 로 다시 실행. 목록: {PROC / f"new_year_check_{y}.csv"}')
        sys.exit(2)
    if any(i['issue'].startswith(('새로운 파일 형식', '연도 전체', '값 이상')) for i in issues):
        print('\n중단: 파일 형식/값 문제는 --accept-names 로 넘어갈 수 없음')
        sys.exit(2)

    # ---- 이름 대응 확장 -> 서울 외 제외 -> 시간대/일별
    SM2 = extend_station_map(SM, S, a.rename_policy)
    SM2.to_csv(PROC / 'station_map.csv', index=False, encoding='utf-8-sig')
    T, n_zero, one = build_hourly(A, SM2)
    D = calendar(daily_from_hourly(T))
    D = attach_type(D, pd.read_csv(PROC / 'station_type.csv'))
    G = group_daily(D)
    D0 = pd.read_csv(PROC / 'station_daily.csv.gz', parse_dates=['date'])
    G0 = pd.read_csv(PROC / 'group_daily.csv', parse_dates=['date'])
    D0 = D0[D0.date.dt.year != y]
    G0 = G0[G0.date.dt.year != y]
    for c in COLS:
        if c not in D0.columns:
            raise RuntimeError(f'기존 station_daily에 {c} 열이 없음: step5를 먼저 실행')
    write(pd.concat([D0, D[COLS]], ignore_index=True), pd.concat([G0, G], ignore_index=True))
    new_st = sorted(set(D.station) - set(pd.read_csv(PROC / 'station_type.csv').station))
    print(f'\n{y}년 추가: station_daily {len(D):,}행 ({D.station.nunique()}개 역), group_daily {len(G)}행 | 역-일 합계 0 제거 {n_zero} | 한쪽만 0 NaN {len(one)} | 학습에 없던 역 {new_st}')
    print(D.groupby('group').station.nunique().to_dict())


if __name__ == '__main__':
    main()
