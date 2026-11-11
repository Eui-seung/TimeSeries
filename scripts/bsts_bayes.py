"""베이지안 BSTS 기준선 (R bsts로 MCMC, 파이썬으로 예측): 9개 역 × 특보 가리기 기간(7일, 14일).

bsts_model.py의 BSTS(최대우도)와 y, 가리기, 회귀 X(같은 규칙으로 뺀 열 포함), 상태 구조가 모두 같고 추정 방식만 다름.
- spike-and-slab 사전분포: 통제 변수(푸리에·계단 더미·여름휴가)는 포함 확률 1로 고정, 날씨 변수는 0.5에서 데이터로 고름.
  변수별 사후 포함 확률이 bstsbayes_coef.csv의 incl_prob
- 분산·AR 계수는 bsts 기본 사전분포(y 표준편차 기준). 데이터가 적은 성분이 극단값으로 가지 않게 잡아 줌
- MCMC: NITER번 반복, 앞 BURN번 버림, THIN번마다 하나 → 사후분포 표본 400개 (scripts/bsts_bayes.R)
- 예측: 표본마다 그 모수로 칼만 필터(statsmodels UnobservedComponents, 같은 구조)를 돌려 하루 앞 예측분포를 구하고,
  표본들을 섞어(혼합) 기준선과 95% 구간을 만듦 → 모수 불확실성이 구간에 들어감. 2024 validation도 2014~2023으로 뽑은 표본만 씀
- 결과 형식은 bsts_model.py와 같아서 bsts_report.py, bsts_impact.py에서 같이 비교됨

GPU는 쓰지 않음: R bsts는 CPU 전용이고, 이 모형은 날짜 순서대로 계산하는 구조라 GPU 이득이 작음. 모형 18개를 CPU 6개로 병렬 실행.

필요: R(https://cran.r-project.org) + install.packages('bsts'). Rscript 경로는 환경변수 RSCRIPT로 바꿀 수 있음.
  conda-forge의 Windows용 r-bsts는 반복 2번도 끝나지 않고 멈춰서 쓰지 않음(2026-11-11 확인)

실행:
  python scripts/bsts_bayes.py                    # MCMC + 예측 -> results/bstsbayes_*.csv, results/bayes/ (표본)
  python scripts/bsts_bayes.py --skip-mcmc        # 저장된 표본으로 예측만 다시
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import subprocess
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bsts_model as B                                         # noqa: E402

C = B.C
RSCRIPT = os.environ.get('RSCRIPT', r'C:\Program Files\R\R-4.6.2\bin\Rscript.exe')
R_SCRIPT = HERE / 'bsts_bayes.R'
OUT = B.RESULTS / 'bayes'
NITER, BURN, THIN = 5000, 1000, 10
SAMPLES_PER_DRAW = 10
SEED = 1


def tag(station, after):
    """R에 넘길 파일 이름 (한글 경로 문제를 피하려고 역 번호 사용)"""
    return f'st{C.STATIONS.index(station)}_{after}'


def prep(station, after):
    s = B.load(station)
    y, X, since, masked = B.make_xy(s, station, after)
    dropped = B.columns_to_drop(y, X)
    return s, y, X.drop(columns=dropped), since, masked, dropped


def run_mcmc(station, after):
    _, y, X, _, _, _ = prep(station, after)
    OUT.mkdir(parents=True, exist_ok=True)
    inp = OUT / f'in_{tag(station, after)}.csv'
    d = X[:B.TRAIN_END].copy()
    d.insert(0, 'y', y[:B.TRAIN_END])
    d.to_csv(inp, index_label='date')
    cmd = [RSCRIPT, str(R_SCRIPT), str(inp), str(OUT / tag(station, after)), str(NITER), str(BURN), str(THIN), str(SEED)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f'{station} {after}: {r.stderr[-1500:]}')
    return r.stdout.strip()


def load_draws(station, after, mod):
    draws = pd.read_csv(OUT / f'{tag(station, after)}_draws.csv')
    missing = set(mod.param_names) - set(draws.columns)
    assert not missing, f'표본에 없는 모수: {missing}'
    return draws[mod.param_names]


def filtered(station, after):
    """역·설정의 데이터와, 사후분포 표본마다 칼만 필터를 돌린 결과를 하나씩 내 주는 생성기"""
    s, y, X, since, masked, dropped = prep(station, after)
    mod = B.build(y, X, 'bsts')
    draws = load_draws(station, after, mod)

    def gen():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for p in draws.values:
                yield mod.filter(p)
    return s, y, X, since, masked, dropped, draws, gen


def mixture_quantiles(means, sds, rng, k=SAMPLES_PER_DRAW, q=(0.5, 0.025, 0.975)):
    """means, sds: (일수, 표본수) 정규분포들의 혼합에서 날마다 분위수 (log 단위)"""
    z = rng.standard_normal((means.shape[0], means.shape[1], k)).astype(np.float32)
    x = (means[:, :, None] + sds[:, :, None] * z).reshape(means.shape[0], -1)
    return np.quantile(x, q, axis=1)


def run_one(station, after):
    s, y, X, since, masked, dropped, draws, gen = filtered(station, after)
    means, sds = [], []
    for r in gen():
        f = r.filter_results
        means.append(f.forecasts[0])
        sds.append(np.sqrt(f.forecasts_error_cov[0, 0]))
    means, sds = np.column_stack(means), np.column_stack(sds)
    keep = (y.notna().cumsum() > B.BURN_IN_OBS).values          # 초기화 구간은 버림 (exp 전에 잘라 오버플로 방지)
    med, lo, hi = mixture_quantiles(means[keep], sds[keep], np.random.default_rng(SEED))
    b = pd.DataFrame({'baseline': np.exp(med), 'lo': np.exp(lo), 'hi': np.exp(hi)}, index=y.index[keep])

    d = s[['group', 'total', *B.ALERT_NAMES]].join(b, how='inner')
    d['since'] = since
    d['y_masked'] = y.isna()
    d = d[~masked.reindex(d.index)]
    d['kind'] = pd.cut(d.since, [-1, 0, 7, 14, np.inf], labels=B.KIND_ORDER).astype(str)
    d['period'] = np.where(d.index <= B.TRAIN_END, 'train', 'validation')
    d['effect_pct'] = (d.total / d.baseline - 1) * 100
    d['outside95'] = (d.total < d.lo) | (d.total > d.hi)
    for c, v in (('after', after), ('station', station), ('model', 'bayes')):
        d.insert(0, c, v)

    post = draws.mean()
    info = {'model': 'bayes', 'station': station, 'after': after, 'converged': np.nan, 'aic': np.nan,
            'train_obs': int(y[:B.TRAIN_END].notna().sum()), 'dropped': ', '.join(dropped), 'n_draws': len(draws),
            **{n: post[n] for n in post.index if n.startswith(('sigma2', 'ar.'))}}
    coefs = pd.read_csv(OUT / f'{tag(station, after)}_coef.csv')
    coefs['pct_change'] = (np.exp(coefs['mean']) - 1) * 100
    for c, v in (('after', after), ('station', station), ('model', 'bayes')):
        coefs.insert(0, c, v)
    return info, coefs, d.reset_index()


def pre_and_paths(station, after, pre_start, start, end, n_sim, seed=0):
    """특보 전후 그림용: 구간 앞 하루 앞 예측(혼합 분위수)과 구간 안 기준선 경로(표본마다 시뮬레이션, 원래 단위)"""
    s, y, X, since, masked, dropped, draws, gen = filtered(station, after)
    rng = np.random.default_rng(seed)
    n = (end - start).days + 1
    per = max(1, n_sim // len(draws))
    pre_idx = pd.date_range(pre_start, start - pd.Timedelta(days=1))
    i0, i1 = y.index.get_loc(pre_idx[0]), y.index.get_loc(pre_idx[-1]) + 1
    means, sds, paths = [], [], []
    for r in gen():
        f = r.filter_results
        means.append(f.forecasts[0, i0:i1])
        sds.append(np.sqrt(f.forecasts_error_cov[0, 0, i0:i1]))
        sim = r.simulate(n, anchor=start, repetitions=per, random_state=rng)
        paths.append(np.asarray(sim).reshape(n, -1))
    med, lo, hi = mixture_quantiles(np.column_stack(means), np.column_stack(sds), rng)
    pre = pd.DataFrame({'pred': np.exp(med), 'lo': np.exp(lo), 'hi': np.exp(hi)}, index=pre_idx)
    return pre, np.exp(np.column_stack(paths))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--station', nargs='*', default=C.STATIONS)
    p.add_argument('--after', nargs='*', type=int, default=B.AFTER_OPTIONS)
    p.add_argument('--skip-mcmc', action='store_true')
    a = p.parse_args()
    jobs = [(st, af) for st in a.station for af in a.after]
    pd.set_option('display.width', 220, 'display.max_columns', 30)

    with ProcessPoolExecutor(B.WORKERS) as ex:
        if not a.skip_mcmc:
            for line in ex.map(run_mcmc, *zip(*jobs)):
                print(line, flush=True)
        out = list(ex.map(run_one, *zip(*jobs)))

    info = pd.DataFrame([o[0] for o in out])
    coefs = pd.concat([o[1] for o in out], ignore_index=True)
    daily = pd.concat([o[2] for o in out], ignore_index=True)
    acc, by_type = B.summarize(daily)

    print('\n날씨 변수 사후 포함 확률 (spike-and-slab)')
    w = coefs[coefs.weather]
    print(w.pivot_table(index='station', columns=['var', 'after'], values='incl_prob').reindex(C.STATIONS).round(2).to_string())
    v = acc[acc.period == 'validation']
    print('\nvalidation 2024 MAPE(%)')
    print(v.pivot_table(index='station', columns=['kind', 'after'], values='MAPE').reindex(C.STATIONS)[B.KIND_ORDER].round(2))

    tg = '' if len(jobs) == len(C.STATIONS) * len(B.AFTER_OPTIONS) else '_partial'
    info.to_csv(B.RESULTS / f'bstsbayes_fitinfo{tg}.csv', index=False, encoding='utf-8-sig')
    coefs.to_csv(B.RESULTS / f'bstsbayes_coef{tg}.csv', index=False, encoding='utf-8-sig')
    acc.to_csv(B.RESULTS / f'bstsbayes_accuracy{tg}.csv', index=False, encoding='utf-8-sig')
    by_type.to_csv(B.RESULTS / f'bstsbayes_alert_effect{tg}.csv', index=False, encoding='utf-8-sig')
    daily.to_csv(B.RESULTS / f'bstsbayes_daily{tg}.csv', index=False, encoding='utf-8-sig')
    print(f'\n저장: {B.RESULTS}/bstsbayes_*{tg}.csv')


if __name__ == '__main__':
    main()
