"""1단계: 원본 파일 구조 확인 + 연도별 표준화.

- 분석에 쓰는 연도별 원본을 읽어 표준 컬럼으로 정리해 data/interim/raw_{year}.pkl 로 저장
- 모든 원본 파일(분석 제외 파일 포함)의 구조 요약을 file_structure.csv 로 저장 (--years를 주지 않았을 때)

실행: python scripts/step1_load_raw.py [--years 2014 2015 ...]   (기본: 2014~2024 전체)
"""
import argparse
import pandas as pd
import chardet
from common import *


def profile_file(fname):
    d = read_raw(fname)
    s, hc = standardize(d.dropna(how='all'))
    s = clean(s, fname)
    ds = pd.DatetimeIndex(sorted(s.date.unique()))
    miss = pd.date_range(ds.min(), ds.max()).difference(ds)
    lines = sorted(s.line.dropna().astype(str).unique())
    return dict(file=nfc(fname)[:-4] if fname.endswith('.csv') else nfc(fname)[:-5],
                format='xlsx' if fname.endswith('.xlsx') else 'csv(cp949)', n_rows=len(d),
                start_date=ds.min().date(), end_date=ds.max().date(), n_days=len(ds), n_missing_days=len(miss),
                has_line_column=bool(s.line.notna().any()), line_list=','.join(lines) or '(없음)',
                n_station_names=s.name_raw.nunique(), n_time_bins=19 if s.attrs.get('merged_23_24') else 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=int, nargs='*')
    a = ap.parse_args()
    ensure_dirs()
    years = a.years or TRAIN_YEARS
    for y in years:
        s, n_raw = load_year(y)
        s.to_pickle(INTERIM / f'raw_{y}.pkl')
        print(f'{y}: 원본 {n_raw:,}행 -> 사용 {len(s):,}행 | {s.date.min().date()}~{s.date.max().date()} | 파일 {s.src.iloc[0][:40]}')
    if not a.years:
        files = sorted([f for f in os.listdir(RAW) if not f.startswith('.')], key=nfc)
        files = [f for f in files if not any(k in nfc(f) for k in ['20251231'])]   # 테스트 연도는 step7에서 다룸
        rows = [profile_file(f) for f in files]
        pd.DataFrame(rows).to_csv(PROC / 'file_structure.csv', index=False, encoding='utf-8-sig')
        print('file_structure.csv 저장 (파일', len(rows), '개)')


if __name__ == '__main__':
    main()
