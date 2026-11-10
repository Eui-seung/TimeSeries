"""SARIMAX-B 시험 실행: 역별 log(total) ~ SARIMAX + 계단 더미 + ASOS 실측 날씨(기상특보 미사용).

- 데이터: data/processed/model_input_9stations.csv, 설정: config.py (적합은 FIT_METHOD, FIT_MAXITER)
- 기간: train 2014-01-01~2023-12-31(적합), validation 2024(예측 오차 확인용), test 2025는 읽자마자 버리고 쓰지 않음
- y = log(total), 하루 단위 빈틈 없음(asfreq D). config.MASK_COLS가 1인 날은 y=NaN(날짜 유지). 특보 관련 가리기는 하지 않음
- 날씨 변수: heat_excess = max(0, feels_like_max-33) (NaN은 0), cold_excess = max(0, -12-tmin),
  rain = log(1+rain_mm), snow = snow_new_cm, 각각 1~3일 전 값(lag1~3, 앞쪽 빈칸 0)
- 외생변수 X = STEP_COLS + (stage 3·4는 SEASON_COLS) + 그 역의 STATION_STEP + 모형별 변수:
  B(ASOS만) = 날씨 4개 + lag 12개, A(특보만) = 특보 4개(alert_heat/cold/rain/snow) + lag 12개, C = 둘 다,
  N(날씨 없음) = 계단 더미·summer_vacation만.
  B_lag, C_lag = B, C에서 당일 ASOS(lag0)를 빼고 lag1~3만 쓴 버전(예측 시점에 실제로 아는 정보만). 특보는 미리 발표되므로 당일 값을 그대로 씀.
  B, C(당일 ASOS 포함)는 사후 상한선으로 봄.
  특보는 서울 전체 기준 0/1이라 9개 역이 같은 값, 2014-01-01~07(특보 여부 모름)은 y를 가리고(MASK_COLS의 alert_unknown) X는 0으로 둠
  학습 기간에 값이 전부 0인 열(fare_2025, climate_card 등)은 자동으로 빼고 목록을 기록
- 변화율(%) = (exp(coef)-1)*100. 누적 효과 = lag0~3 계수의 합(선형 결합 t-검정으로 p-value 계산)
- validation: 학습 구간에서 구한 계수를 고정(refit=False)하고, 2024년 각 날을 전날까지의 y로 하루 앞 예측.
  가린 날을 뺀 날들로 원래 단위(exp) RMSE, MAPE 계산
- 차수 후보 두 개: (1,0,1)(1,1,1,7)(config.ORDER, SEASONAL_ORDER, 확정안)과 (2,0,1)(1,1,1,7).
  날씨 계수 표는 확정안으로 만들고, 후보는 2024 예측과 AIC, Ljung-Box, 수렴 비교용

참고: results/sarimax_B_*.csv(stage 2·3)는 alert_unknown 마스크를 넣기 전(2014-01-01~07을 가리지 않은) 설정으로 만든 결과이고,
results/sarimax_ABC_*.csv(stage 4)는 alert_unknown 마스크를 포함한 설정으로 A·B·C를 모두 다시 적합한 결과이다.

실행:
  python scripts/sarimax_asos.py --stage 1 [--station 역삼]   # 한 역만, summary와 표 출력
  python scripts/sarimax_asos.py --stage 2                    # 9개 역 × 후보 2개 -> results/
  python scripts/sarimax_asos.py --stage 3                    # summer_vacation 없음/있음 비교(확정 차수, 미수렴 재시도 포함) -> results/
  python scripts/sarimax_asos.py --stage 4                    # 모형 A(특보만)·B(ASOS만)·C(특보+ASOS) 9개 역 비교 -> results/sarimax_ABC_*.csv
  python scripts/sarimax_asos.py --stage 5                    # N(날씨 없음)·A·B·C, 수유·연신내 C 재시도, 전체/특보일+3일 예측 오차 -> results/sarimax_NABC_*.csv
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
ALERTS = list(C.ALERT_COLS)                                    # alert_heat, alert_cold, alert_rain, alert_snow
ALERT_ALL = ALERTS + [f'{c}_lag{k}' for c in ALERTS for k in LAGS]
TYPE_ORDER = ['Employment', 'Commercial & Leisure', 'Residential']
CANDIDATES = [(C.ORDER, C.SEASONAL_ORDER), ((2, 0, 1), C.SEASONAL_ORDER)]      # 앞의 것이 확정안
WORKERS = 6
RETRY_C_FROM_B = ['수유', '연신내']                              # stage 4에서 C가 미수렴이었던 역: B 계수를 초기값으로 다시 시도


def label(order, seasonal):
    return f'{order}{seasonal}'.replace(' ', '')


MAIN = label(*CANDIDATES[0])


def load():
    d = pd.read_csv(DATA, parse_dates=['date'])
    return d[d.date <= VALID_END]                              # 2025(test)는 여기서 버림


def prepare(d, station, season=False, model='B'):
    """한 역의 y, X(2014~2024 전체)와 유형. 학습·검증 분리는 호출하는 쪽에서 한다.
    model: 'B' = ASOS 실측만, 'A' = 특보만, 'C' = 특보 + ASOS, 'N' = 날씨 없음, 'B_lag'/'C_lag' = B/C에서 당일 ASOS를 뺀 버전"""
    s = d[d.station == station].set_index('date').sort_index().asfreq('D')
    assert s.index.min() == START and s.index.max() == VALID_END and s.station.notna().all(), '날짜 빈틈'
    y = np.log(s.total)
    y[s[C.MASK_COLS].sum(axis=1) > 0] = np.nan                 # 가리기: 날짜는 유지, y만 NaN
    blocks = []
    if model in ('B', 'C', 'B_lag', 'C_lag'):
        w = pd.DataFrame({
            'heat_excess': (s.feels_like_max - HEAT_BASE).clip(lower=0).fillna(0),
            'cold_excess': (COLD_BASE - s.tmin).clip(lower=0).fillna(0),
            'rain': np.log1p(s.rain_mm),
            'snow': s.snow_new_cm,
        }).fillna(0)
        blocks += ([] if model.endswith('_lag') else [w]) + [pd.DataFrame({f'{c}_lag{k}': w[c].shift(k).fillna(0) for c in WEATHER for k in LAGS})]
    if model in ('A', 'C', 'C_lag'):
        a = s[ALERTS].fillna(0)                                 # NaN은 2014-01-01~07뿐(y도 가려짐)
        blocks += [a, pd.DataFrame({f'{c}_lag{k}': a[c].shift(k).fillna(0) for c in ALERTS for k in LAGS})]
    steps = list(C.STEP_COLS) + (list(C.SEASON_COLS) if season else []) + ([C.STATION_STEP[station]] if station in C.STATION_STEP else [])
    X = pd.concat([s[steps]] + blocks, axis=1).astype(float)
    return y, X, s.group.iloc[0], s.total


def _fit(model, start=None, maxiter=None, method=None, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = model.fit(start_params=start, disp=False, method=method or C.FIT_METHOD, maxiter=maxiter or C.FIT_MAXITER, **kw)
    return res, sorted({f'{w.category.__name__}: {str(w.message)[:110]}' for w in caught})


def _converged(res):
    return bool(res.mle_retvals.get('converged'))


def alt_start(model, ytr, Xtr):
    """다른 초기값: 7일 차분한 y를 7일 차분한 X에 회귀한 계수 + 고정한 ARMA 초기값"""
    dy, dX = ytr - ytr.shift(7), Xtr - Xtr.shift(7)
    ok = dy.notna() & dX.notna().all(axis=1)
    beta = np.linalg.lstsq(dX[ok].values, dy[ok].values, rcond=None)[0]
    sig2 = float(np.var(dy[ok].values - dX[ok].values @ beta))
    cols, out = list(Xtr.columns), []
    for n in model.param_names:
        if n in cols: out.append(beta[cols.index(n)])
        elif n.startswith('ar.S'): out.append(0.1)
        elif n.startswith('ma.S'): out.append(-0.8)
        elif n == 'ar.L1': out.append(0.5)
        elif n.startswith(('ar.', 'ma.')): out.append(0.0)
        else: out.append(sig2)
    return np.array(out)


def fit_station(y, X, order, seasonal_order, retry=False, hint=None):
    """2014~2023으로 적합. 반환: 결과, 경고 목록, 뺀 열, 사용한 X, 적합 방법 메모.
    retry=True면 기본 설정(lbfgs, maxiter=1000)이 수렴하지 않을 때 순서대로 다시 시도:
    (1) 기본 결과의 계수에서 이어서 maxiter=5000, (2) 다른 초기값에서 maxiter=5000. 첫 수렴 결과를 쓰고, 없으면 loglike가 가장 큰 것을 씀.
    hint(다른 모형의 계수 {이름: 값})를 주면 그 계수를 초기값으로 lbfgs(maxiter=5000)를 먼저 돌리고, 수렴하지 않으면
    그 초기값에서 method='nm'을 먼저 돌린 뒤 그 결과로 lbfgs(maxiter=5000)를 다시 돌린다(없는 변수는 0). 그래도 안 되면 미수렴 표시"""
    ytr, Xtr = y[:TRAIN_END], X[:TRAIN_END]
    dropped = [c for c in X.columns if (Xtr[c] == 0).all()]
    X = X.drop(columns=dropped); Xtr = Xtr.drop(columns=dropped)
    model = SARIMAX(ytr, exog=Xtr, order=order, seasonal_order=seasonal_order)
    if hint is not None:
        start = np.array([hint.get(n, 0.0) for n in model.param_names])
        r1, m1 = _fit(model, start=start, maxiter=5000)
        tried = [('B 계수를 초기값으로(lbfgs, maxiter=5000)', r1, m1)]
        if not _converged(r1):
            rn, _ = _fit(model, start=start, method='nm', maxiter=3000, maxfun=6000)
            r2, m2 = _fit(model, start=rn.params.values, maxiter=5000)
            tried.append(('B 계수 초기값, nm 후 lbfgs(maxiter=5000)', r2, m2))
    else:
        res, msgs = _fit(model)
        tried = [('기본(lbfgs, maxiter=1000)', res, msgs)]
        if retry and not _converged(res):
            r1, m1 = _fit(model, start=res.params.values, maxiter=5000)
            tried.append(('기본 결과에서 이어서(maxiter=5000)', r1, m1))
            if not _converged(r1):
                r2, m2 = _fit(model, start=alt_start(model, ytr, Xtr), maxiter=5000)
                tried.append(('다른 초기값(차분 회귀, maxiter=5000)', r2, m2))
    best = next((t for t in tried if _converged(t[1])), None) or max(tried, key=lambda t: t[1].llf)
    note = best[0] + ('' if _converged(best[1]) else ' [미수렴, loglike 최대 결과 사용]') + ('' if len(tried) == 1 else f' (시도 {len(tried)}회)')
    return best[1], best[2], dropped, X, note


def coef_table(res, X):
    t = pd.DataFrame({'coef': res.params[X.columns], 'p': res.pvalues[X.columns]})
    t['pct_change'] = (np.exp(t.coef) - 1) * 100
    return t


def cumulative(res, X):
    """변수별 누적 효과: lag0~3 계수의 합과 p-value (합 = 0 검정). X에 있는 ASOS·특보 변수 모두"""
    names = list(res.param_names)
    rows = []
    for w in WEATHER + ALERTS:
        cols = [c for c in [w] + [f'{w}_lag{k}' for k in LAGS] if c in names]      # lag만 쓰는 버전은 lag1~3 합
        if not cols:
            continue
        R = np.zeros(len(names))
        for c in cols:
            R[names.index(c)] = 1.0
        t = res.t_test(R)
        eff, se, p = float(np.ravel(t.effect)[0]), float(np.ravel(t.sd)[0]), float(np.ravel(t.pvalue)[0])
        rows.append({'변수': w, 'cum_coef': eff, 'se': se, 'p': p, '변화율': (np.exp(eff) - 1) * 100})
    return pd.DataFrame(rows)


def ljung_box(res, order, seasonal_order):
    r = res.resid.iloc[2 * seasonal_order[3]:].dropna()        # 앞쪽 초기화 구간(14일)과 y가 NaN인 날 제외
    df = order[0] + order[2] + seasonal_order[0] + seasonal_order[2]
    return acorr_ljungbox(r, lags=[7, 14], model_df=df), len(r)


def alert_window(d, st):
    """any_alert=1인 날과 그다음 3일 (2024년 구간, 불리언 Series)"""
    a = d[d.station == st].set_index('date').sort_index().asfreq('D').any_alert.fillna(0)
    return ((a + a.shift(1) + a.shift(2) + a.shift(3)).fillna(0) > 0)[VALID_START:VALID_END]


def validate(res, y, X, total, window=None):
    """계수 고정(refit=False)으로 2024년을 하루씩 앞 예측. 가린 날을 뺀 날의 RMSE, MAPE(원래 단위).
    window를 주면 그 날들(특보일+3일) 중 가리지 않은 날만으로도 계산"""
    full = res.apply(y, exog=X, refit=False)                    # 2014~2024 전체에 같은 계수 적용(2024 y는 전날까지만 예측에 쓰임)
    pred = np.exp(full.get_prediction(start=VALID_START, end=VALID_END, dynamic=False).predicted_mean)
    act = total[VALID_START:VALID_END]
    ok = y[VALID_START:VALID_END].notna()                      # 가리기된 날 제외
    err = act[ok] - pred[ok]
    naive = total.shift(7)[VALID_START:VALID_END][ok]            # 참고: 7일 전 같은 요일 값
    out = {'n_days': int(ok.sum()), 'n_masked': int((~ok).sum()),
           'RMSE': float(np.sqrt((err ** 2).mean())), 'MAPE': float((err.abs() / act[ok]).mean() * 100),
           'naive7_RMSE': float(np.sqrt(((act[ok] - naive) ** 2).mean())), 'naive7_MAPE': float(((act[ok] - naive).abs() / act[ok]).mean() * 100)}
    detail = pd.DataFrame({'actual': act, 'pred': pred, 'ok': ok})
    if window is not None:
        m = ok & window
        e = act[m] - pred[m]
        out.update(n_alert=int(m.sum()), RMSE_alert=float(np.sqrt((e ** 2).mean())), MAPE_alert=float((e.abs() / act[m]).mean() * 100))
        detail['window'] = window
    out['detail'] = detail
    return out


def run_station(d, st, order, seasonal_order, season=False, retry=False, model='B', hint=None):
    y, X, grp, total = prepare(d, st, season, model)
    res, msgs, dropped, Xu, note = fit_station(y, X, order, seasonal_order, retry, hint)
    lb, nlb = ljung_box(res, order, seasonal_order)
    return dict(station=st, group=grp, order=order, seasonal_order=seasonal_order, res=res, msgs=msgs, dropped=dropped, X=Xu, lb=lb, nlb=nlb, note=note,
                valid=validate(res, y, Xu, total, alert_window(d, st)), y=y)


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


def _task_full(args):
    global _DATA
    st, order, seasonal, season, retry, tag, model, *rest = args
    if _DATA is None:
        _DATA = load()
    hint = None
    if rest and rest[0]:                                        # 다른 모형(rest[0], B 또는 B_lag)의 계수를 초기값으로 쓰는 재시도 (C 모형, 수유·연신내)
        hint = run_station(_DATA, st, order, seasonal, season, True, rest[0])['res'].params.to_dict()
    r = run_station(_DATA, st, order, seasonal, season, retry, model, hint)
    res, v, lb = r['res'], r['valid'], r['lb']
    lab = label(order, seasonal)
    coefs = coef_table(res, r['X']).reset_index().rename(columns={'index': '변수', 'pct_change': '변화율'})
    cum = cumulative(res, r['X'])
    for t in (coefs, cum):
        if tag:
            t.insert(0, '버전', tag)
        t.insert(0, '차수', lab); t.insert(0, '유형', r['group']); t.insert(0, '역', st)
    metrics = {'역': st, '유형': r['group'], '차수': lab, 'RMSE': v['RMSE'], 'MAPE': v['MAPE'], 'AIC': res.aic,
               'LjungBox_p_lag7': float(lb.lb_pvalue.iloc[0]), 'LjungBox_p_lag14': float(lb.lb_pvalue.iloc[1]),
               '수렴': bool(res.mle_retvals.get('converged')), '반복': int(res.mle_retvals.get('iterations') or 0), '경고수': len(r['msgs']),
               '평가일수': v['n_days'], 'naive7_MAPE': v['naive7_MAPE'], '뺀 열': ','.join(r['dropped'])}
    if tag:
        metrics = {**{'역': st, '유형': r['group'], '버전': tag}, **{k: m for k, m in metrics.items() if k not in ('역', '유형')}, '적합': r['note']}
    detail = v['detail'].reset_index().rename(columns={'index': 'date'})
    detail.insert(0, '모형', tag); detail.insert(0, '역', st)
    extra = {k: v[k] for k in ('n_alert', 'RMSE_alert', 'MAPE_alert')}
    return metrics, coefs, cum, detail, extra


def _task(args):
    m, c, k, _, _ = _task_full(args)
    return m, c, k


def stars(p):
    return '**' if p < 0.01 else ('*' if p < 0.05 else '')


def stage2():
    RESULTS.mkdir(exist_ok=True)
    jobs = [(st, o, so, False, False, None, 'B') for st in C.STATIONS for o, so in CANDIDATES]       # summer_vacation 없음, 재시도 없음(기존 결과 재현)
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


def stage3():
    """summer_vacation 없음/있음 비교 (확정 차수). 미수렴이면 초기값·maxiter를 바꿔 다시 시도"""
    RESULTS.mkdir(exist_ok=True)
    versions = [('없음', False), ('있음', True)]
    o, so = CANDIDATES[0]
    jobs = [(st, o, so, season, True, tag, 'B') for st in C.STATIONS for tag, season in versions]
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        out = list(ex.map(_task, jobs))
    V = pd.DataFrame([m for m, _, _ in out])
    A = pd.concat([c for _, c, _ in out], ignore_index=True)
    K = pd.concat([k for _, _, k in out], ignore_index=True)
    V.to_csv(RESULTS / 'sarimax_B_season_compare.csv', index=False, encoding='utf-8-sig')
    A.to_csv(RESULTS / 'sarimax_B_season_coefs.csv', index=False, encoding='utf-8-sig')
    K.to_csv(RESULTS / 'sarimax_B_season_cumulative.csv', index=False, encoding='utf-8-sig')

    pd.set_option('display.width', 250, 'display.max_columns', 40, 'display.max_colwidth', 60, 'display.float_format', lambda x: f'{x:,.4f}')
    order_st = [s_ for g in TYPE_ORDER for s_ in C.STATIONS if V[V.역 == s_].유형.iloc[0] == g]
    print(f'[summer_vacation 없음/있음 — {MAIN}, 2024 롤링 예측]')
    show = V.set_index(['역', '버전']).reindex(pd.MultiIndex.from_product([order_st, ['없음', '있음']], names=['역', '버전']))
    print(show[['유형', 'RMSE', 'MAPE', 'AIC', 'LjungBox_p_lag7', 'LjungBox_p_lag14', '수렴', '반복', '경고수', '적합']].to_string())
    print('\n[summer_vacation 계수]')
    sv = A[(A.버전 == '있음') & (A.변수 == 'summer_vacation')].set_index('역').reindex(order_st)
    print(sv[['유형', 'coef', 'p', '변화율']].to_string())

    # 유형별 요약
    rows = []
    for ver in ['없음', '있음']:
        for g in TYPE_ORDER:
            v = V[(V.버전 == ver) & (V.유형 == g)]
            row = {'버전': ver, '유형': g, '역수': len(v), 'MAPE평균': v.MAPE.mean(), 'AIC평균': v.AIC.mean(), 'RMSE평균': v.RMSE.mean()}
            for w in WEATHER:
                k = K[(K.버전 == ver) & (K.유형 == g) & (K.변수 == w)]
                row[f'{w}_누적평균'] = k.cum_coef.mean(); row[f'{w}_변화율%'] = (np.exp(k.cum_coef.mean()) - 1) * 100
                row[f'{w}_유의역수'] = int((k.p < 0.05).sum())
            rows.append(row)
    S = pd.DataFrame(rows)
    S.to_csv(RESULTS / 'sarimax_B_season_by_type.csv', index=False, encoding='utf-8-sig')
    print('\n[유형별 평균: MAPE, AIC와 날씨 누적 효과(lag0~3 합)]'); print(S.to_string(index=False))

    # 폭염 효과 유형 비교
    print('\n[폭염 누적 효과 (heat_excess lag0~3 합)  없음 → 있음]')
    H = K[K.변수 == 'heat_excess'].pivot(index='역', columns='버전', values='cum_coef').reindex(order_st)
    Hp = K[K.변수 == 'heat_excess'].pivot(index='역', columns='버전', values='p').reindex(order_st)
    H.insert(0, '유형', [V[V.역 == x].유형.iloc[0] for x in H.index])
    H['p(있음)'] = Hp['있음']
    print(H.to_string())
    hs = K[(K.버전 == '있음') & (K.변수 == 'heat_excess')]
    emp, oth = hs[hs.유형 == 'Employment'].cum_coef, hs[hs.유형 != 'Employment'].cum_coef
    print(f'\n있음 버전: Employment 평균 {emp.mean():+.4f} (범위 {emp.min():+.4f}~{emp.max():+.4f}) | 나머지 6개 역 평균 {oth.mean():+.4f} (범위 {oth.min():+.4f}~{oth.max():+.4f})')
    print('Employment가 나머지 역 전부보다 더 큰(더 음수) 역:', [x for x in emp.index.map(lambda i: hs.loc[i, "역"])] if False else
          [hs.loc[i, '역'] for i in emp.index if emp[i] < oth.min()])


def stage4():
    """모형 A(특보만) · B(ASOS만) · C(특보+ASOS): 확정 차수, summer_vacation 포함, 9개 역, 미수렴 재시도"""
    RESULTS.mkdir(exist_ok=True)
    o, so = CANDIDATES[0]
    models = ['A', 'B', 'C']
    jobs = [(st, o, so, True, True, m, m) for st in C.STATIONS for m in models]
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        out = list(ex.map(_task, jobs))
    V = pd.DataFrame([m for m, _, _ in out]).rename(columns={'버전': '모형'})
    A = pd.concat([c for _, c, _ in out], ignore_index=True).rename(columns={'버전': '모형'})
    K = pd.concat([k for _, _, k in out], ignore_index=True).rename(columns={'버전': '모형'})
    V.to_csv(RESULTS / 'sarimax_ABC_compare.csv', index=False, encoding='utf-8-sig')
    A.to_csv(RESULTS / 'sarimax_ABC_coefs.csv', index=False, encoding='utf-8-sig')
    K.to_csv(RESULTS / 'sarimax_ABC_cumulative.csv', index=False, encoding='utf-8-sig')

    pd.set_option('display.width', 250, 'display.max_columns', 40, 'display.max_colwidth', 60, 'display.float_format', lambda x: f'{x:,.4f}')
    order_st = [x for g in TYPE_ORDER for x in C.STATIONS if V[V.역 == x].유형.iloc[0] == g]
    print(f'[모형 A(특보만) / B(ASOS만) / C(특보+ASOS) — {MAIN}, summer_vacation 포함, 2024 롤링 예측]')
    show = V.set_index(['역', '모형']).reindex(pd.MultiIndex.from_product([order_st, models], names=['역', '모형']))
    print(show[['유형', 'RMSE', 'MAPE', 'AIC', 'LjungBox_p_lag7', 'LjungBox_p_lag14', '수렴', '반복', '경고수', '적합']].to_string())

    M = V.groupby(['유형', '모형']).agg(MAPE평균=('MAPE', 'mean'), AIC평균=('AIC', 'mean'), RMSE평균=('RMSE', 'mean'), 수렴역수=('수렴', 'sum')).reindex(
        pd.MultiIndex.from_product([TYPE_ORDER, models], names=['유형', '모형']))
    M.to_csv(RESULTS / 'sarimax_ABC_metrics_by_type.csv', encoding='utf-8-sig')
    print('\n[유형별 평균 지표]'); print(M.to_string())

    # 누적 효과: 유형별 평균과 유의한 역 수 (ASOS는 B vs C, 특보는 A vs C)
    G = K.groupby(['변수', '유형', '모형']).agg(평균_누적계수=('cum_coef', 'mean'), 양수_역수=('cum_coef', lambda v: int((v > 0).sum())),
                                              유의_역수=('p', lambda v: int((v < 0.05).sum())), 역수=('cum_coef', 'size')).reset_index()
    G['평균_변화율%'] = (np.exp(G.평균_누적계수) - 1) * 100
    G.to_csv(RESULTS / 'sarimax_ABC_cumulative_by_type.csv', index=False, encoding='utf-8-sig')
    for title, vars_, ms in [('ASOS 날씨 누적 효과 (lag0~3 합): B vs C', WEATHER, ['B', 'C']), ('특보 누적 효과 (lag0~3 합): A vs C', ALERTS, ['A', 'C'])]:
        T = G[G.변수.isin(vars_) & G.모형.isin(ms)].pivot(index=['변수', '유형'], columns='모형', values=['평균_누적계수', '평균_변화율%', '유의_역수'])
        T = T.reindex(pd.MultiIndex.from_product([vars_, TYPE_ORDER], names=['변수', '유형']))
        print(f'\n[{title} — 유형별 평균, 유의(p<0.05) 역 수 /3]'); print(T.to_string())
    # 특보: 역별 누적 효과(A, C)와 p-value, 그리고 lag0 계수
    for v in ALERTS:
        P = K[K.변수 == v].pivot(index='역', columns='모형', values=['cum_coef', 'p']).reindex(order_st)
        P.insert(0, ('유형', ''), [V[V.역 == x].유형.iloc[0] for x in P.index])
        print(f'\n[{v} 누적 효과, 역별 (A: 특보만, C: 특보+ASOS)]'); print(P.to_string())
    L = A[A.변수.isin(ALERTS) & A.모형.isin(['A', 'C'])].groupby(['변수', '모형']).agg(
        lag0_양수_역수=('coef', lambda v: int((v > 0).sum())), lag0_음수_역수=('coef', lambda v: int((v < 0).sum())), lag0_유의_역수=('p', lambda v: int((v < 0.05).sum()))).reindex(
        pd.MultiIndex.from_product([ALERTS, ['A', 'C']], names=['변수', '모형']))
    print('\n[특보 lag0 계수: 부호와 유의 역 수 /9]'); print(L.to_string())


def stage5():
    """N(날씨 없음) · A · B · B_lag · C · C_lag: 확정 차수, summer_vacation 포함, 9개 역.
    2024 롤링 예측 오차를 (1) 전체 평가일, (2) any_alert=1인 날과 그다음 3일로 계산. 수유·연신내의 C는 B 계수를 초기값으로 재시도"""
    from scipy.stats import wilcoxon
    RESULTS.mkdir(exist_ok=True)
    o, so = CANDIDATES[0]
    models = ['N', 'A', 'B', 'B_lag', 'C', 'C_lag']
    base = {'C': 'B', 'C_lag': 'B_lag'}
    jobs = [(st, o, so, True, True, m, m, base.get(m) if st in RETRY_C_FROM_B else None) for st in C.STATIONS for m in models]
    jobs.sort(key=lambda j: j[7] is None)                       # 오래 걸리는 재시도 작업을 먼저 시작
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        out = list(ex.map(_task_full, jobs))
    V = pd.DataFrame([{**m, **e} for m, _, _, _, e in out]).rename(columns={'버전': '모형', 'RMSE': 'RMSE_전체', 'MAPE': 'MAPE_전체', 'n_alert': '일수_특보', 'RMSE_alert': 'RMSE_특보', 'MAPE_alert': 'MAPE_특보', '평가일수': '일수_전체'})
    A = pd.concat([c for _, c, _, _, _ in out], ignore_index=True).rename(columns={'버전': '모형'})
    K = pd.concat([k for _, _, k, _, _ in out], ignore_index=True).rename(columns={'버전': '모형'})
    D = pd.concat([d for _, _, _, d, _ in out], ignore_index=True)
    V['_o'] = V.역.map({x: i for i, x in enumerate(C.STATIONS)}); V['_m'] = V.모형.map({x: i for i, x in enumerate(models)})
    V = V.sort_values(['_o', '_m']).drop(columns=['_o', '_m']).reset_index(drop=True)
    V.to_csv(RESULTS / 'sarimax_NABC_compare.csv', index=False, encoding='utf-8-sig')
    A.to_csv(RESULTS / 'sarimax_NABC_coefs.csv', index=False, encoding='utf-8-sig')
    K.to_csv(RESULTS / 'sarimax_NABC_cumulative.csv', index=False, encoding='utf-8-sig')
    D.to_csv(RESULTS / 'sarimax_NABC_pred2024.csv', index=False, encoding='utf-8-sig')

    pd.set_option('display.width', 280, 'display.max_columns', 40, 'display.max_colwidth', 70, 'display.float_format', lambda x: f'{x:,.4f}')
    order_st = [x for g in TYPE_ORDER for x in C.STATIONS if V[V.역 == x].유형.iloc[0] == g]
    print(f'[모형별 지표 — {MAIN}, summer_vacation 포함, 2024 롤링 예측. 전체 평가일 / 특보일+3일]')
    show = V.set_index(['역', '모형']).reindex(pd.MultiIndex.from_product([order_st, models], names=['역', '모형']))
    print(show[['유형', 'MAPE_전체', 'MAPE_특보', 'RMSE_전체', 'RMSE_특보', 'AIC', '수렴', '반복', '일수_전체', '일수_특보']].to_string())
    print('\n[수렴 / 적합 방법 메모 (기본이 아닌 것)]')
    print(V[V.적합.ne('기본(lbfgs, maxiter=1000)') | ~V.수렴][['역', '모형', '수렴', '반복', '적합']].to_string(index=False))

    M = V.groupby(['유형', '모형']).agg(MAPE_전체=('MAPE_전체', 'mean'), MAPE_특보=('MAPE_특보', 'mean'), RMSE_전체=('RMSE_전체', 'mean'), RMSE_특보=('RMSE_특보', 'mean'),
                                       AIC평균=('AIC', 'mean'), 수렴역수=('수렴', 'sum')).reindex(pd.MultiIndex.from_product([TYPE_ORDER, models], names=['유형', '모형']))
    M.to_csv(RESULTS / 'sarimax_NABC_by_type.csv', encoding='utf-8-sig')
    print('\n[유형별 평균 지표]'); print(M.to_string())

    # N 대비 개선 (역별 + 유형 평균), 두 평가 구간
    rows = []
    for st in order_st:
        for part, (mc, rc, sub) in {'전체': ('MAPE_전체', 'RMSE_전체', 'ok'), '특보일+3일': ('MAPE_특보', 'RMSE_특보', 'window')}.items():
            dn = D[(D.역 == st) & (D.모형 == 'N')].set_index('date')
            sel = dn.ok if sub == 'ok' else (dn.ok & dn.window)
            ape_n = ((dn.actual - dn.pred).abs() / dn.actual)[sel]
            vn = V[(V.역 == st) & (V.모형 == 'N')].iloc[0]
            for m in models[1:]:
                dm = D[(D.역 == st) & (D.모형 == m)].set_index('date')
                ape_m = ((dm.actual - dm.pred).abs() / dm.actual)[sel]
                vm = V[(V.역 == st) & (V.모형 == m)].iloc[0]
                diff = (ape_n - ape_m).dropna()
                try:
                    pw = float(wilcoxon(diff).pvalue) if (diff != 0).any() else np.nan
                except ValueError:
                    pw = np.nan
                rows.append({'역': st, '유형': vn.유형, '모형': m, '구간': part, '일수': int(sel.sum()), 'MAPE_N': vn[mc], 'MAPE_모형': vm[mc],
                             'MAPE_개선%p': vn[mc] - vm[mc], 'MAPE_개선%': (vn[mc] - vm[mc]) / vn[mc] * 100,
                             'RMSE_N': vn[rc], 'RMSE_모형': vm[rc], 'RMSE_개선%': (vn[rc] - vm[rc]) / vn[rc] * 100, 'wilcoxon_p': pw})
    R = pd.DataFrame(rows)
    T = R.groupby(['유형', '모형', '구간']).agg(일수=('일수', 'mean'), MAPE_N=('MAPE_N', 'mean'), MAPE_모형=('MAPE_모형', 'mean'), **{'MAPE_개선%p': ('MAPE_개선%p', 'mean'), 'MAPE_개선%': ('MAPE_개선%', 'mean'),
                                                                                  'RMSE_개선%': ('RMSE_개선%', 'mean'), '모형이_나은_역수': ('MAPE_개선%p', lambda v: int((v > 0).sum())),
                                                                                  'p<0.05_역수': ('wilcoxon_p', lambda v: int((v < 0.05).sum()))}).reset_index()
    T.insert(0, '역', '(유형 평균)')
    allr = pd.concat([R, T], ignore_index=True)
    allr.to_csv(RESULTS / 'sarimax_NABC_vs_N.csv', index=False, encoding='utf-8-sig')
    print('\n[N 대비 개선 — 유형별 평균 (MAPE_개선%p = N의 MAPE − 모형의 MAPE, 양수면 모형이 나음)]')
    TT = T.set_index(['구간', '유형', '모형']).reindex(pd.MultiIndex.from_product([['전체', '특보일+3일'], TYPE_ORDER, models[1:]], names=['구간', '유형', '모형']))
    print(TT[['일수', 'MAPE_N', 'MAPE_모형', 'MAPE_개선%p', 'MAPE_개선%', 'RMSE_개선%', '모형이_나은_역수', 'p<0.05_역수']].to_string())
    for m in ['B', 'B_lag']:
        print(f'\n[역별: {m}가 N보다 얼마나 좋아지나 — 특보일+3일 / 전체]')
        X = R[R.모형 == m].pivot(index='역', columns='구간', values=['일수', 'MAPE_N', 'MAPE_모형', 'MAPE_개선%p', 'MAPE_개선%', 'wilcoxon_p']).reindex(order_st)
        print(X.to_string())

    # 특보 누적 효과 (C, C_lag, A) 유의 역 수와 수유·연신내 재시도 결과
    G = K[K.변수.isin(ALERTS)].groupby(['변수', '모형']).agg(유의_역수=('p', lambda v: int((v < 0.05).sum())), 음수_역수=('cum_coef', lambda v: int((v < 0).sum()))).reindex(
        pd.MultiIndex.from_product([ALERTS, ['A', 'C', 'C_lag']], names=['변수', '모형']))
    G.to_csv(RESULTS / 'sarimax_NABC_alert_significance.csv', encoding='utf-8-sig')
    print('\n[특보 누적 효과 p<0.05인 역 수 /9]'); print(G.to_string())
    print('\n[수유·연신내 C 재시도 후: 특보 누적 효과]')
    print(K[K.역.isin(RETRY_C_FROM_B) & K.모형.isin(['C', 'C_lag']) & K.변수.isin(ALERTS)][['역', '모형', '변수', 'cum_coef', 'p']].to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=int, choices=[1, 2, 3, 4, 5], required=True)
    ap.add_argument('--station', default='역삼')
    a = ap.parse_args()
    {1: lambda: stage1(load(), a.station), 2: stage2, 3: stage3, 4: stage4, 5: stage5}[a.stage]()
