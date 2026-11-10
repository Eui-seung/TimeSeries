"""SARIMAX-B 시험 실행: 역별 log(total) ~ SARIMAX + 계단 더미 + ASOS 실측 날씨(기상특보 미사용).

- 데이터: data/processed/model_input_9stations.csv, 설정: config.py (적합은 FIT_METHOD, FIT_MAXITER)
- 기간: train 2014-01-01~2023-12-31(적합), validation 2024(예측 오차 확인용), test 2025는 읽자마자 버리고 쓰지 않음
- y = log(total), 하루 단위 빈틈 없음(asfreq D). config.MASK_COLS가 1인 날은 y=NaN(날짜 유지). 특보 관련 가리기는 하지 않음
- 날씨 변수: heat_excess = max(0, feels_like_max-33) (NaN은 0), cold_excess = max(0, -12-tmin),
  rain = log(1+rain_mm), snow = snow_new_cm, 각각 1~3일 전 값(lag1~3, 앞쪽 빈칸 0)
- 외생변수 X = STEP_COLS + 그 역의 STATION_STEP + 날씨 4개 + lag 12개.
  학습 기간에 값이 전부 0인 열(fare_2025, climate_card 등)은 자동으로 빼고 목록을 기록
- 변화율(%) = (exp(coef)-1)*100. 누적 효과 = lag0~3 계수의 합(선형 결합 t-검정으로 p-value 계산)
- validation: 학습 구간에서 구한 계수를 고정(refit=False)하고, 2024년 각 날을 전날까지의 y로 하루 앞 예측.
  가린 날을 뺀 날들로 원래 단위(exp) RMSE, MAPE 계산
- 차수 후보 두 개: (1,0,1)(1,1,1,7)(config.ORDER, SEASONAL_ORDER, 확정안)과 (2,0,1)(1,1,1,7).
  날씨 계수 표는 확정안으로 만들고, 후보는 2024 예측과 AIC, Ljung-Box, 수렴 비교용

실행:
  python scripts/sarimax_asos.py --stage 1 [--station 역삼]   # 한 역만, summary와 표 출력
  python scripts/sarimax_asos.py --stage 2                    # 9개 역 × 후보 2개 -> results/
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
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C                                             # noqa: E402

DATA = ROOT / 'data' / 'processed' / 'model_input_9stations.csv'
RESULTS = ROOT / 'results'
START = pd.Timestamp('2014-01-01')
TRAIN_END = pd.Timestamp(C.TRAIN_END)                          # 2023-12-31
VALID_START, VALID_END = pd.Timestamp(f'{C.VALID}-01-01'), pd.Timestamp(f'{C.VALID}-12-31')
HEAT_BASE, COLD_BASE = 33.0, -12.0
WEATHER = ['heat_excess', 'cold_excess', 'rain', 'snow']
LAGS = [1, 2, 3]
WEATHER_ALL = WEATHER + [f'{c}_lag{k}' for c in WEATHER for k in LAGS]
TYPE_ORDER = ['Employment', 'Commercial & Leisure', 'Residential']
CANDIDATES = [(C.ORDER, C.SEASONAL_ORDER), ((2, 0, 1), C.SEASONAL_ORDER)]      # 앞의 것이 확정안
WORKERS = 6


def label(order, seasonal):
    return f'{order}{seasonal}'.replace(' ', '')


MAIN = label(*CANDIDATES[0])


def load():
    d = pd.read_csv(DATA, parse_dates=['date'])
    return d[d.date <= VALID_END]                              # 2025(test)는 여기서 버림


def prepare(d, station):
    """한 역의 y, X(2014~2024 전체)와 유형. 학습·검증 분리는 호출하는 쪽에서 한다."""
    s = d[d.station == station].set_index('date').sort_index().asfreq('D')
    assert s.index.min() == START and s.index.max() == VALID_END and s.station.notna().all(), '날짜 빈틈'
    y = np.log(s.total)
    y[s[C.MASK_COLS].sum(axis=1) > 0] = np.nan                 # 가리기: 날짜는 유지, y만 NaN
    w = pd.DataFrame({
        'heat_excess': (s.feels_like_max - HEAT_BASE).clip(lower=0).fillna(0),
        'cold_excess': (COLD_BASE - s.tmin).clip(lower=0).fillna(0),
        'rain': np.log1p(s.rain_mm),
        'snow': s.snow_new_cm,
    }).fillna(0)
    lag = {f'{c}_lag{k}': w[c].shift(k).fillna(0) for c in WEATHER for k in LAGS}
    steps = list(C.STEP_COLS) + ([C.STATION_STEP[station]] if station in C.STATION_STEP else [])
    X = pd.concat([s[steps], w, pd.DataFrame(lag)], axis=1).astype(float)
    return y, X, s.group.iloc[0], s.total


def fit_station(y, X, order, seasonal_order):
    """2014~2023으로 적합. 반환: 결과, 경고 목록, 뺀 열, 사용한 X"""
    ytr, Xtr = y[:TRAIN_END], X[:TRAIN_END]
    dropped = [c for c in X.columns if (Xtr[c] == 0).all()]
    X = X.drop(columns=dropped); Xtr = Xtr.drop(columns=dropped)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = SARIMAX(ytr, exog=Xtr, order=order, seasonal_order=seasonal_order).fit(disp=False, method=C.FIT_METHOD, maxiter=C.FIT_MAXITER)
    msgs = sorted({f'{w.category.__name__}: {str(w.message)[:110]}' for w in caught})
    return res, msgs, dropped, X


def coef_table(res, X):
    t = pd.DataFrame({'coef': res.params[X.columns], 'p': res.pvalues[X.columns]})
    t['pct_change'] = (np.exp(t.coef) - 1) * 100
    return t


def cumulative(res, X):
    """날씨 변수별 누적 효과: lag0~3 계수의 합과 p-value (합 = 0 검정)"""
    names = list(res.param_names)
    rows = []
    for w in WEATHER:
        R = np.zeros(len(names))
        for c in [w] + [f'{w}_lag{k}' for k in LAGS]:
            R[names.index(c)] = 1.0
        t = res.t_test(R)
        eff, se, p = float(np.ravel(t.effect)[0]), float(np.ravel(t.sd)[0]), float(np.ravel(t.pvalue)[0])
        rows.append({'변수': w, 'cum_coef': eff, 'se': se, 'p': p, '변화율': (np.exp(eff) - 1) * 100})
    return pd.DataFrame(rows)


def ljung_box(res, order, seasonal_order):
    r = res.resid.iloc[2 * seasonal_order[3]:].dropna()        # 앞쪽 초기화 구간(14일)과 y가 NaN인 날 제외
    df = order[0] + order[2] + seasonal_order[0] + seasonal_order[2]
    return acorr_ljungbox(r, lags=[7, 14], model_df=df), len(r)


def validate(res, y, X, total):
    """계수 고정(refit=False)으로 2024년을 하루씩 앞 예측. 가린 날을 뺀 날의 RMSE, MAPE(원래 단위)"""
    full = res.apply(y, exog=X, refit=False)                    # 2014~2024 전체에 같은 계수 적용(2024 y는 전날까지만 예측에 쓰임)
    pred = np.exp(full.get_prediction(start=VALID_START, end=VALID_END, dynamic=False).predicted_mean)
    act = total[VALID_START:VALID_END]
    ok = y[VALID_START:VALID_END].notna()                      # 가리기된 날 제외
    err = act[ok] - pred[ok]
    naive = total.shift(7)[VALID_START:VALID_END][ok]            # 참고: 7일 전 같은 요일 값
    return {'n_days': int(ok.sum()), 'n_masked': int((~ok).sum()),
            'RMSE': float(np.sqrt((err ** 2).mean())), 'MAPE': float((err.abs() / act[ok]).mean() * 100),
            'naive7_RMSE': float(np.sqrt(((act[ok] - naive) ** 2).mean())), 'naive7_MAPE': float(((act[ok] - naive).abs() / act[ok]).mean() * 100)}


def run_station(d, st, order, seasonal_order):
    y, X, grp, total = prepare(d, st)
    res, msgs, dropped, Xu = fit_station(y, X, order, seasonal_order)
    lb, nlb = ljung_box(res, order, seasonal_order)
    return dict(station=st, group=grp, order=order, seasonal_order=seasonal_order, res=res, msgs=msgs, dropped=dropped, X=Xu, lb=lb, nlb=nlb,
                valid=validate(res, y, Xu, total), y=y)


def stage1(d, station):
    r = run_station(d, station, *CANDIDATES[0])
    res, X, v = r['res'], r['X'], r['valid']
    ytr = r['y'][:TRAIN_END]
    print(f'=== {station} ({r["group"]}) | order {MAIN} | 학습 {START.date()}~{TRAIN_END.date()} | y 관측 {int(ytr.notna().sum())}일 / 전체 {len(ytr)}일 (가린 날 {int(ytr.isna().sum())}일) | X {X.shape[1]}개 ===')
    print('학습 기간 값이 전부 0이라 뺀 열:', r['dropped'] if r['dropped'] else '없음')
    print(res.summary())
    pd.set_option('display.width', 200, 'display.float_format', lambda x: f'{x:,.4f}')
    print('\n[계수]'); print(coef_table(res, X).to_string())
    print('\n[누적 효과 lag0~3]'); print(cumulative(res, X).to_string(index=False))
    print('\n수렴:', res.mle_retvals.get('converged'), '| 경고:', r['msgs'] or '없음', f'| AIC {res.aic:,.2f}')
    print(r['lb'].to_string())
    print(f'validation {C.VALID}: RMSE {v["RMSE"]:,.0f} | MAPE {v["MAPE"]:.2f}% ({v["n_days"]}일)')


# ---------------- 9개 역 전체 ----------------
_DATA = None


def _task(args):
    global _DATA
    st, order, seasonal = args
    if _DATA is None:
        _DATA = load()
    r = run_station(_DATA, st, order, seasonal)
    res, v, lb = r['res'], r['valid'], r['lb']
    lab = label(order, seasonal)
    coefs = coef_table(res, r['X']).reset_index().rename(columns={'index': '변수', 'pct_change': '변화율'})
    cum = cumulative(res, r['X'])
    for t in (coefs, cum):
        t.insert(0, '차수', lab); t.insert(0, '유형', r['group']); t.insert(0, '역', st)
    metrics = {'역': st, '유형': r['group'], '차수': lab, 'RMSE': v['RMSE'], 'MAPE': v['MAPE'], 'AIC': res.aic,
               'LjungBox_p_lag7': float(lb.lb_pvalue.iloc[0]), 'LjungBox_p_lag14': float(lb.lb_pvalue.iloc[1]),
               '수렴': bool(res.mle_retvals.get('converged')), '반복': int(res.mle_retvals.get('iterations')), '경고수': len(r['msgs']),
               '평가일수': v['n_days'], 'naive7_MAPE': v['naive7_MAPE'], '뺀 열': ','.join(r['dropped'])}
    return metrics, coefs, cum


def stars(p):
    return '**' if p < 0.01 else ('*' if p < 0.05 else '')


def stage2():
    RESULTS.mkdir(exist_ok=True)
    jobs = [(st, o, so) for st in C.STATIONS for o, so in CANDIDATES]
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        out = list(ex.map(_task, jobs))
    V = pd.DataFrame([m for m, _, _ in out])
    A = pd.concat([c for _, c, _ in out], ignore_index=True)
    K = pd.concat([k for _, _, k in out], ignore_index=True)
    V.to_csv(RESULTS / 'sarimax_B_valid2024.csv', index=False, encoding='utf-8-sig')
    A.to_csv(RESULTS / 'sarimax_B_coefs.csv', index=False, encoding='utf-8-sig')
    K.to_csv(RESULTS / 'sarimax_B_cumulative.csv', index=False, encoding='utf-8-sig')

    pd.set_option('display.width', 250, 'display.max_columns', 40, 'display.float_format', lambda x: f'{x:,.4f}')
    order = [s for g in TYPE_ORDER for s in C.STATIONS if V[V.역 == s].유형.iloc[0] == g]
    # 후보 비교표 (역 × 차수)
    labs = [label(o, so) for o, so in CANDIDATES]
    W = V.pivot(index='역', columns='차수', values=['RMSE', 'MAPE', 'AIC', 'LjungBox_p_lag7', 'LjungBox_p_lag14', '수렴']).reindex(order)
    W.columns = [f'{m}|{c}' for m, c in W.columns]
    W.insert(0, '유형', [V[V.역 == s].유형.iloc[0] for s in W.index])
    W.to_csv(RESULTS / 'sarimax_B_order_compare.csv', encoding='utf-8-sig')
    print('\n[차수 후보 비교 (2024 롤링 예측, 가린 날 제외)]')
    show = V.set_index(['역', '차수']).reindex(pd.MultiIndex.from_product([order, labs], names=['역', '차수']))
    print(show[['유형', 'RMSE', 'MAPE', 'AIC', 'LjungBox_p_lag7', 'LjungBox_p_lag14', '수렴', '반복', '경고수', '평가일수', 'naive7_MAPE']].to_string())

    # 확정안: 유형별 날씨 계수와 누적 효과 나란히
    Am, Km = A[A.차수 == MAIN], K[K.차수 == MAIN]
    rows = {}
    for w in WEATHER:
        for name in [w] + [f'{w}_lag{k}' for k in LAGS]:
            rows[name] = {s: '{:+.4f}{}'.format(*Am[(Am.역 == s) & (Am.변수 == name)][['coef', 'p']].iloc[0].tolist()[:1] + [stars(Am[(Am.역 == s) & (Am.변수 == name)].p.iloc[0])]) for s in order}
        rows[f'{w} 누적(lag0~3)'] = {s: '{:+.4f}{}'.format(Km[(Km.역 == s) & (Km.변수 == w)].cum_coef.iloc[0], stars(Km[(Km.역 == s) & (Km.변수 == w)].p.iloc[0])) for s in order}
    T = pd.DataFrame(rows).T
    T.columns = pd.MultiIndex.from_tuples([(V[V.역 == s].유형.iloc[0], s) for s in order], names=['유형', '역'])
    T.to_csv(RESULTS / 'sarimax_B_type_compare.csv', encoding='utf-8-sig')
    print(f'\n[유형별 날씨 계수 — 확정안 {MAIN} (* p<0.05, ** p<0.01)]'); print(T.to_string())

    # 누적 효과 p-value 표와 유형 요약
    P = Km.pivot(index='변수', columns='역', values='p').reindex(index=WEATHER, columns=order)
    P.columns = pd.MultiIndex.from_tuples([(V[V.역 == s].유형.iloc[0], s) for s in order], names=['유형', '역'])
    P.to_csv(RESULTS / 'sarimax_B_cumulative_p.csv', encoding='utf-8-sig')
    print('\n[누적 효과 p-value]'); print(P.to_string())
    S = Km.groupby(['변수', '유형']).agg(평균_누적계수=('cum_coef', 'mean'), 양수_역수=('cum_coef', lambda s: int((s > 0).sum())),
                                       유의_역수=('p', lambda s: int((s < 0.05).sum())), 역수=('cum_coef', 'size')).reset_index()
    S['평균_변화율%'] = (np.exp(S.평균_누적계수) - 1) * 100
    S = S.set_index(['변수', '유형']).reindex(pd.MultiIndex.from_product([WEATHER, TYPE_ORDER], names=['변수', '유형']))
    S.to_csv(RESULTS / 'sarimax_B_cumulative_by_type.csv', encoding='utf-8-sig')
    print('\n[유형별 누적 효과 요약]'); print(S.to_string())
    sn = Am[Am.변수 == 'snow'].set_index('역').reindex(order)
    print('\n[snow(lag0) 계수 부호] 양수 역 수:', int((sn.coef > 0).sum()), '/ 9 | 유의(p<0.05) 양수:', int(((sn.coef > 0) & (sn.p < 0.05)).sum()))
    print(sn[['유형', 'coef', 'p', '변화율']].to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=int, choices=[1, 2], required=True)
    ap.add_argument('--station', default='역삼')
    a = ap.parse_args()
    stage1(load(), a.station) if a.stage == 1 else stage2()
