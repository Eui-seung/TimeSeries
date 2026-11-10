"""공통 설정과 함수: 경로, 원본 읽기·정리, 이름 정규화.

경로는 환경변수로 바꿀 수 있음(검증 실행용): SUBWAY_PROC, SUBWAY_INTERIM, SUBWAY_FIG, SUBWAY_EXTRA_PATH.
필요 패키지: pandas, numpy, scipy, openpyxl, factor_analyzer, holidays, matplotlib
"""
import os, re, sys, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd

for _p in os.environ.get('SUBWAY_EXTRA_PATH', '').split(os.pathsep):
    if _p:
        sys.path.append(_p)          # 뒤에 붙여서 기존 패키지가 우선

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'rawdata'
PROC = Path(os.environ.get('SUBWAY_PROC', ROOT / 'data' / 'processed'))
INTERIM = Path(os.environ.get('SUBWAY_INTERIM', ROOT / 'data' / 'interim'))
FIG = Path(os.environ.get('SUBWAY_FIG', ROOT / 'figures'))

TRAIN_YEARS = list(range(2014, 2025))     # 학습 기간 2014~2024
TEST_YEARS = [2025]                       # 테스트 기간
HOURS = [f'h{i:02d}' for i in range(5, 25)]   # h05='06시 이전'(05~06 포함) … h24='24시 이후'
HOUR_LABELS = {'h05': 'before_06', 'h24': 'after_24', **{f'h{i:02d}': f'h{i:02d}_{i+1:02d}' for i in range(6, 24)}}
HLAB = [HOUR_LABELS[h] for h in HOURS]        # station_date_hour의 영문 시간대 열


def ensure_dirs():
    for p in (PROC, INTERIM, FIG):
        p.mkdir(parents=True, exist_ok=True)


def nfc(s):
    return unicodedata.normalize('NFC', s)


def raw_file(year):
    """분석에 쓰는 연도별 원본 파일 하나를 고름 (겹치는 2023~2024년 다른 파일은 제외)."""
    key = {2022: '20221231', 2023: '(23.1~23.12)', 2024: '(24.1~24.12)', 2025: '20251231'}.get(year, f'{year}년 일별 역별')
    hits = [f for f in os.listdir(RAW) if not f.startswith('.') and key in nfc(f)]
    if len(hits) != 1:
        raise FileNotFoundError(f'{year}년 원본 파일을 하나로 특정하지 못함: {hits}')
    return hits[0]


def read_raw(fname):
    """원본 파일을 컬럼 그대로 읽음 (csv: cp949, xlsx: 첫 '날짜' 행을 헤더로)."""
    p = RAW / fname
    if fname.endswith('.xlsx'):
        d = pd.read_excel(p, header=None, dtype=object)
        hr = [i for i in range(5) if '날짜' in [str(x).strip() for x in d.iloc[i]]][0]
        d.columns = [str(c).strip() for c in d.iloc[hr]]
        return d.iloc[hr + 1:].reset_index(drop=True)
    return pd.read_csv(p, encoding='cp949', dtype=object)


def standardize(d):
    """연도마다 다른 컬럼 구성을 위치 기준으로 표준화 -> date,line,stn_no,name,io,daytype,h05..h24"""
    cols = list(d.columns)
    io_idx = max(i for i, c in enumerate(cols) if c in ('구분', '승하차구분'))
    P = lambda i: d.iloc[:, i]
    out = pd.DataFrame()
    out['date'] = P([i for i, c in enumerate(cols) if c in ('날짜', '수송일자')][0])
    li = [i for i, c in enumerate(cols) if c == '호선']
    out['line'] = P(li[0]) if li else None
    out['stn_no'] = P([i for i, c in enumerate(cols) if c.startswith('역번호') or c.startswith('고유역번호')][0])
    out['name'] = P([i for i, c in enumerate(cols) if c == '역명'][0])
    out['io'] = P(io_idx)
    di = [i for i, c in enumerate(cols) if c == '구분' and i < io_idx]
    out['daytype'] = P(di[0]) if di else None
    hc = cols[io_idx + 1:io_idx + 21]
    # 2021년: '23시 이후'가 23~24시와 24시 이후를 합친 단일 컬럼 (다음 컬럼은 합계)
    merged = (len(cols) > io_idx + 20 and str(cols[io_idx + 19]).replace(' ', '').endswith('이후')
              and str(cols[io_idx + 20]).replace(' ', '') == '합계')
    for k, h in enumerate(HOURS):
        out[h] = pd.NA if (merged and k == 19) else P(io_idx + 1 + k)
    out.attrs['merged_23_24'] = merged
    return out, hc


def clean(s, src):
    """문자열 정리: 날짜·방향·이름·역번호·노선·숫자(쉼표 제거)"""
    s = s.copy()
    s['date'] = pd.to_datetime(s['date'].astype(str).str.strip().str[:10])
    s['io'] = s['io'].astype(str).str.strip()
    s['name_raw'] = s['name'].astype(str).str.strip()
    s = s.drop(columns=['name'])
    s['stn_no'] = s['stn_no'].astype(str).str.strip()
    s['line'] = s['line'].astype(str).str.replace('호선', '').str.strip().replace({'None': None})
    for h in HOURS:
        s[h] = pd.to_numeric(s[h].astype(str).str.replace(',', '').str.strip(), errors='coerce')
    if '20221231' in nfc(src):
        s['h24'] = s['h24'].fillna(0)          # 2022년 1~5월 '24시이후' 빈칸은 0
    s['year'] = s.date.dt.year
    s['src'] = nfc(src)
    return s


def load_year(year):
    """한 연도 원본 -> 정리된 표 (빈 행 제거). 반환: (표, 원본 컬럼 수, 원본 행 수)"""
    f = raw_file(year)
    d = read_raw(f)
    n_raw = len(d)
    d = d.dropna(how='all')                    # 2025 파일 끝의 빈 행 등
    s, _ = standardize(d)
    s = clean(s, f)
    s.attrs['merged_23_24'] = s.attrs.get('merged_23_24', False)
    return s, n_raw


def base_name(n):
    """괄호 부기명·공백 제거"""
    return re.sub(r'\s+', '', re.sub(r'\([^)]*\)', '', n))


def strip_yeok(n):
    """끝의 '역' 제거 (정규화 키용, 서울역 -> 서울)"""
    return n[:-1] if n.endswith('역') and len(n) >= 3 else n


def load_interim(years):
    """step1이 저장한 연도별 표를 합침. 2014~2016(노선 컬럼 없음)은 역번호로 노선을 채우고,
    빈 역번호는 같은 (노선, 이름)의 번호로 채움."""
    A = pd.concat([pd.read_pickle(INTERIM / f'raw_{y}.pkl') for y in years], ignore_index=True)
    ref = None
    for y in range(2017, 2030):
        p = INTERIM / f'raw_{y}.pkl'
        if p.exists():
            r = pd.read_pickle(p)[['stn_no', 'line']].dropna()
            ref = r.drop_duplicates('stn_no').set_index('stn_no').line
            break
    if ref is None and A.line.isna().any():
        raise RuntimeError('노선 보충용 2017년 이후 자료가 없음: step1을 먼저 실행')
    if ref is not None:
        A['line'] = A.line.fillna(A.stn_no.map(ref))
    assert A.line.notna().all(), '노선을 채우지 못한 행이 있음'
    A['line'] = A.line.astype(int)
    A['base'] = A.name_raw.map(base_name)
    blank = A.stn_no == ''
    if blank.any():
        fix = A[~blank].drop_duplicates(['line', 'base']).set_index(['line', 'base']).stn_no
        A.loc[blank, 'stn_no'] = [fix[(l, b)] for l, b in zip(A.loc[blank, 'line'], A.loc[blank, 'base'])]
    A['tot'] = A[HOURS].sum(axis=1)
    return A
