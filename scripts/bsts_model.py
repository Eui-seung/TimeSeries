"""BSTS(구조적 시계열) 기준선 모형: 9개 역 × 특보 가리기 기간(7일, 14일), 날씨 포함.

목적: 특보가 없을 때의 기준선을 만들고, 특보 구간의 실제값과 비교해 특보 효과를 봄.

- 데이터: data/processed/model_input_9stations.csv (2025 test는 읽자마자 버림)
- 특보 창: any_alert=1인 날부터 해제 후 AFTER일까지 (AFTER = 7 또는 14). 14일 창은 step10의 noalert 파일과 같은 정의
- y = log(total). config.MASK_COLS가 1인 날과 특보 창인 날은 y=NaN(날짜 유지, 칼만 필터가 건너뜀)
- 모형 (statsmodels UnobservedComponents, 최대우도):
    local level + AR(1) + 주간 계절성(주기 7, 확률적) + 회귀
    AR(1)이 없으면 하루짜리 충격(예: 역삼 2018-12-24)이 수준에 영구 반영돼 긴 가림 구간의 기준선이 15~45% 틀어졌음
    (역삼, 14일 창 기준 AIC -3942 -> AR(1) 추가 -4156 -> 연간 계절성 회귀로 이동 -4332)
- 회귀 X = 연간 계절성(365.25일 푸리에 sin/cos 4쌍) + STEP_COLS + summer_vacation + 그 역의 STATION_STEP
           + 날씨(sarimax_asos.py의 모형 B와 같은 정의)
    연간 계절성을 상태가 아닌 회귀로 둔 이유: 상태로 두면 첫해(2014)는 아직 추정이 안 돼 기준선이 크게 틀어짐
    heat_excess = max(0, feels_like_max-33), cold_excess = max(0, -12-tmin), rain = log(1+rain_mm), snow = snow_new_cm,
    각각 당일 + 1~3일 전(lag1~3). 학습 기간의 관측일(y가 NaN이 아닌 날)에 전부 0인 열은 뺌.
    날씨 변수는 0이 아닌 관측일이 MIN_NONZERO_OBS(30)일 미만이어도 뺌: 극한 날씨인 날은 거의 다 특보 구간이라 가려져서
    heat_excess는 0일, cold_excess는 1일(0.9℃), snow는 약 20일(최대 4.3cm)만 남음. 이렇게 적은 날로 추정한 계수를
    특보 구간의 극한 값(−17.8℃, 신적설 20cm)에 곱하면 기준선이 실제의 2~3배로 튐(신촌 2018-01, 2024-11 폭설에서 확인).
    결과적으로 남는 날씨 변수는 rain(lag 포함)
- 적합: 2014~2023(train). 2024(validation)는 계수를 고정하고 필터만 돌림
- 기준선 = 하루 앞 예측. 가려진 날은 그 전 마지막 관측일에서 이어서 예측한 값(특보 정보가 안 들어간 반사실 기준선).
  가리지 않은 날은 전날까지의 실제값을 쓴 하루 앞 예측
- 효과 = 실제/기준선 - 1 (%). 날 구분(kind): 특보일 / 해제 후 1~7일 / 해제 후 8~14일 / 평소(해제 후 15일 이상).
  두 설정을 같은 날끼리 비교할 수 있게 kind는 가리기 기간과 상관없이 같은 기준. masked 열은 그 설정에서 y를 가렸는지
- 관측 60일째까지는 초기화 구간이라 버림

비교용 SARIMAX 기준선(--model sarimax): 위와 같은 y, 같은 가리기, 같은 회귀 X(같은 규칙으로 뺀 열 포함)에
  config의 SARIMA 차수 (1,0,1)(1,1,1,7)을 씀. 상태공간 모형이라 가린 날을 같은 방식으로 건너뛰고 이어서 예측함

실행:
  python scripts/bsts_model.py                          # 9개 역 × 7일·14일 -> results/bsts_*.csv
  python scripts/bsts_model.py --model sarimax          # 같은 설정의 SARIMAX -> results/sarimaxbl_*.csv
  python scripts/bsts_model.py --station 역삼 --after 14   # 한 역, 한 설정만 (계수 출력)
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')                              # 프로세스를 나눠 돌리므로 행렬 연산 스레드는 1개

import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C                                             # noqa: E402

DATA = ROOT / 'data' / 'processed' / 'model_input_9stations.csv'
RESULTS = ROOT / 'results'
TRAIN_END = pd.Timestamp(C.TRAIN_END)
VALID_END = pd.Timestamp(f'{C.VALID}-12-31')
AFTER_OPTIONS = [7, 14]
HEAT_BASE, COLD_BASE = 33.0, -12.0
WEATHER = ['heat_excess', 'cold_excess', 'rain', 'snow']
LAGS = [1, 2, 3]
WEATHER_ALL = WEATHER + [f'{c}_lag{k}' for c in WEATHER for k in LAGS]
MIN_NONZERO_OBS = 30                                           # 날씨 변수는 학습 관측일 중 0이 아닌 날이 30일 미만이면 뺌
FOURIER_K = 4
BURN_IN_OBS = 60
WORKERS = 6
ALERT_NAMES = {'alert_heat': '폭염', 'alert_cold': '한파', 'alert_rain': '호우', 'alert_snow': '대설'}
KIND_ORDER = ['특보일', '해제 후 1~7일', '해제 후 8~14일', '평소']
MODELS = ['bsts', 'sarimax']
PREFIX = {'bsts': 'bsts', 'sarimax': 'sarimaxbl'}       # 결과 파일 이름 앞부분 (기존 sarimax_* 결과와 겹치지 않게)


def load(station):
    """한 역의 2014~2024 데이터 (날짜 빈틈 없음)"""
    d = pd.read_csv(DATA, encoding='utf-8-sig', parse_dates=['date'])
    s = d[(d.station == station) & (d.date <= VALID_END)].set_index('date').sort_index().asfreq('D')
    assert s.station.notna().all(), '날짜 빈틈'
    return s


def days_since_alert(s):
    """마지막 특보일로부터 지난 날 수 (특보일 0, 특보가 한 번도 없었으면 큰 값). any_alert NaN(2014-01-01~07)은 0으로 봄"""
    a = s.any_alert.fillna(0).values == 1
    out, last = np.full(len(a), 10 ** 6), None
    for i, on in enumerate(a):
        if on:
            last = i
        if last is not None:
            out[i] = i - last
    return pd.Series(out, index=s.index)


def make_xy(s, station, after):
    since = days_since_alert(s)
    window = since <= after
    masked = s[C.MASK_COLS].sum(axis=1) > 0
    y = np.log(s.total).where(~(masked | window))
    w = pd.DataFrame({
        'heat_excess': (s.feels_like_max - HEAT_BASE).clip(lower=0),
        'cold_excess': (COLD_BASE - s.tmin).clip(lower=0),
        'rain': np.log1p(s.rain_mm),
        'snow': s.snow_new_cm,
    }).fillna(0)
    lags = pd.DataFrame({f'{c}_lag{k}': w[c].shift(k).fillna(0) for c in WEATHER for k in LAGS})
    steps = list(C.STEP_COLS) + list(C.SEASON_COLS) + ([C.STATION_STEP[station]] if station in C.STATION_STEP else [])
    t = 2 * np.pi * (s.index.dayofyear - 1) / 365.25
    fourier = pd.DataFrame({f'{f}{k}': fn(k * t) for k in range(1, FOURIER_K + 1) for f, fn in (('sin', np.sin), ('cos', np.cos))},
                           index=s.index)
    X = pd.concat([fourier, s[steps], w, lags], axis=1).astype(float)
    return y, X, since, masked


def build(y, X, model='bsts'):
    if model == 'sarimax':
        return SARIMAX(y, exog=X, order=C.ORDER, seasonal_order=C.SEASONAL_ORDER)
    return UnobservedComponents(y, exog=X, level='local level', autoregressive=1, seasonal=7, stochastic_seasonal=True)


def columns_to_drop(y, X):
    """학습 관측일에 전부 0인 열, 그리고 0이 아닌 관측일이 MIN_NONZERO_OBS일 미만인 날씨 열"""
    ytr, Xtr = y[:TRAIN_END], X[:TRAIN_END]
    nonzero = (Xtr.loc[ytr.notna()] != 0).sum()
    return [c for c in X.columns if nonzero[c] < (MIN_NONZERO_OBS if c in WEATHER_ALL else 1)]


def fit(y, X, model='bsts'):
    dropped = columns_to_drop(y, X)
    X = X.drop(columns=dropped)
    ytr, Xtr = y[:TRAIN_END], X[:TRAIN_END]
    mod = build(ytr, Xtr, model)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = mod.fit(method=C.FIT_METHOD, maxiter=C.FIT_MAXITER, disp=False)
        if not res.mle_retvals.get('converged'):          # 수렴하지 않으면 그 결과에서 이어서 한 번 더
            caught.clear()
            res = mod.fit(start_params=res.params.values, method=C.FIT_METHOD, maxiter=5000, disp=False)
    msgs = sorted({str(w.message)[:110] for w in caught})
    return res, X, dropped, msgs


def coef_table(res, X):
    names = [f'beta.{c}' if f'beta.{c}' in res.params.index else c for c in X.columns]
    t = pd.DataFrame({'coef': res.params[names].values, 'p': res.pvalues[names].values}, index=X.columns)
    t['pct_change'] = (np.exp(t.coef) - 1) * 100
    return t


def baseline(res, y, X):
    """학습 계수로 2014~2024 전체를 필터링한 하루 앞 예측 (원래 단위). 관측 BURN_IN_OBS일 전까지는 버림"""
    pr = res.apply(y, exog=X, refit=False).get_prediction()
    keep = (y.notna().cumsum() > BURN_IN_OBS).values
    mean, ci = pr.predicted_mean[keep], pr.conf_int(alpha=0.05)[keep]
    return pd.DataFrame({'baseline': np.exp(mean), 'lo': np.exp(ci.iloc[:, 0]), 'hi': np.exp(ci.iloc[:, 1])})


def run_one(station, after, model='bsts'):
    """한 역, 한 설정: 적합 정보, 계수, 일별 결과(MASK_COLS 날 제외)"""
    s = load(station)
    y, X, since, masked = make_xy(s, station, after)
    res, X, dropped, msgs = fit(y, X, model)
    b = baseline(res, y, X)
    d = s[['group', 'total', *ALERT_NAMES]].join(b, how='inner')
    d['since'] = since
    d['y_masked'] = y.isna()
    d = d[~masked.reindex(d.index)]
    d['kind'] = pd.cut(d.since, [-1, 0, 7, 14, np.inf], labels=KIND_ORDER).astype(str)
    d['period'] = np.where(d.index <= TRAIN_END, 'train', 'validation')
    d['effect_pct'] = (d.total / d.baseline - 1) * 100
    d['outside95'] = (d.total < d.lo) | (d.total > d.hi)
    d.insert(0, 'after', after)
    d.insert(0, 'station', station)
    var = {n: res.params[n] for n in res.param_names if n.startswith(('sigma2', 'ar.', 'ma.'))}
    info = {'model': model, 'station': station, 'after': after, 'converged': bool(res.mle_retvals.get('converged')), 'aic': res.aic,
            'train_obs': int(y[:TRAIN_END].notna().sum()), 'dropped': ', '.join(dropped), 'warnings': ' | '.join(msgs), **var}
    coefs = coef_table(res, X).reset_index(names='var')
    coefs.insert(0, 'after', after)
    coefs.insert(0, 'station', station)
    coefs.insert(0, 'model', model)
    d.insert(0, 'model', model)
    return info, coefs, d.reset_index()


def summarize(daily):
    """역·설정·기간·날 구분별 오차와 효과, 그리고 특보 종류별 특보일 효과"""
    g = daily.groupby(['station', 'after', 'period', 'kind'])
    acc = g.agg(일수=('effect_pct', 'size'), MAPE=('effect_pct', lambda x: x.abs().mean()),
                평균효과=('effect_pct', 'mean'), 중앙값=('effect_pct', 'median'), 밖95=('outside95', 'mean'),
                가림비율=('y_masked', 'mean')).reset_index()
    rows = []
    for col, name in ALERT_NAMES.items():
        sub = daily[daily[col] == 1]
        t = sub.groupby(['station', 'after', 'period']).agg(
            일수=('effect_pct', 'size'), 평균효과=('effect_pct', 'mean'), 중앙값=('effect_pct', 'median'),
            밖95=('outside95', 'mean')).reset_index()
        t.insert(3, '특보', name)
        rows.append(t)
    return acc, pd.concat(rows, ignore_index=True)


def _run(args):
    return run_one(*args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--station', nargs='*', default=C.STATIONS)
    p.add_argument('--after', nargs='*', type=int, default=AFTER_OPTIONS)
    p.add_argument('--model', choices=MODELS, default='bsts')
    a = p.parse_args()
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 30)
    jobs = [(st, af, a.model) for st in a.station for af in a.after]

    if len(jobs) == 1:
        out = [run_one(*jobs[0])]
        print(out[0][1].round(4).to_string(index=False))
    else:
        with ProcessPoolExecutor(WORKERS) as ex:
            out = list(ex.map(_run, jobs))

    info = pd.DataFrame([o[0] for o in out])
    coefs = pd.concat([o[1] for o in out], ignore_index=True)
    daily = pd.concat([o[2] for o in out], ignore_index=True)
    acc, by_type = summarize(daily)

    print('적합 정보')
    print(info.drop(columns=['warnings']).round(6).to_string(index=False))
    if info.warnings.str.len().gt(0).any():
        print(info.loc[info.warnings.str.len() > 0, ['station', 'after', 'warnings']].to_string(index=False))

    v = acc[acc.period == 'validation']
    print('\nvalidation 2024: 날 구분별 평균 효과(%) = 실제/기준선 - 1')
    print(v.pivot_table(index='station', columns=['kind', 'after'], values='평균효과').reindex(C.STATIONS)[KIND_ORDER].round(2))
    print('\nvalidation 2024: 날 구분별 MAPE(%)')
    print(v.pivot_table(index='station', columns=['kind', 'after'], values='MAPE').reindex(C.STATIONS)[KIND_ORDER].round(2))
    t = acc[acc.period == 'train']
    print('\ntrain: 날 구분별 평균 효과(%)')
    print(t.pivot_table(index='station', columns=['kind', 'after'], values='평균효과').reindex(C.STATIONS)[KIND_ORDER].round(2))
    print('\n특보 종류별 특보일 평균 효과(%), train / validation')
    print(by_type.pivot_table(index='station', columns=['특보', 'period', 'after'], values='평균효과').reindex(C.STATIONS).round(1))

    RESULTS.mkdir(exist_ok=True)
    tag = '' if len(jobs) > 1 else f'_{jobs[0][0]}_{jobs[0][1]}'
    pre = PREFIX[a.model]
    info.to_csv(RESULTS / f'{pre}_fitinfo{tag}.csv', index=False, encoding='utf-8-sig')
    coefs.to_csv(RESULTS / f'{pre}_coef{tag}.csv', index=False, encoding='utf-8-sig')
    acc.to_csv(RESULTS / f'{pre}_accuracy{tag}.csv', index=False, encoding='utf-8-sig')
    by_type.to_csv(RESULTS / f'{pre}_alert_effect{tag}.csv', index=False, encoding='utf-8-sig')
    daily.to_csv(RESULTS / f'{pre}_daily{tag}.csv', index=False, encoding='utf-8-sig')
    print(f'\n저장: {RESULTS}/{pre}_*{tag}.csv')


if __name__ == '__main__':
    main()
