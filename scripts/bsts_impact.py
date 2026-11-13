"""특보 전후 비교 그림 (CausalImpact 방식): 실제 vs 기준선, 날마다의 효과, 누적 효과. 열마다 모형 하나(BSTS 최대우도, BSTS 베이지안, SARIMAX).

- 모형은 bsts_model.py와 같음(역마다 2014~2023으로 적합, 계수 고정 후 2014~2024 필터링).
  세 모형은 y, 가리기, 회귀 X(통제 변수)가 같음. 베이지안 BSTS는 scripts/bsts_bayes.py를 먼저 실행해 사후분포 표본이 있어야 함
- 특보 구간(episode) = 특보 창(특보일 ~ 해제 후 AFTER일)이 끊기지 않고 이어진 기간. 이 기간은 y를 가렸으므로
  기준선은 구간 시작 직전까지의 정보만으로 이어서 예측한 반사실 값
- 구간 앞 PRE_DAYS일: 하루 앞 예측과 95% 구간 (평소 오차가 어느 정도인지 보여 주는 용도)
- 구간 안: 구간 첫날의 예측 상태에서 기준선 경로를 N_SIM번 시뮬레이션해 날마다의 효과와 누적 효과의 95% 구간을 구함
  (날마다의 구간을 단순히 더하면 날 사이 상관을 무시하게 되므로 경로 단위로 누적)
- 효과 = 실제 - 기준선 (명). 공휴일 등 MASK_COLS 날은 효과에서 뺌(누적에도 0으로)

실행:
  python scripts/bsts_impact.py                                   # 기본 예시 5개
  python scripts/bsts_impact.py --station 신촌 --after 7 --date 2018-01-25   # 그 날짜가 들어 있는 구간
출력: results/figures/impact_<역>_<구간 시작일>_<after>d.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bsts_model as B                                       # noqa: E402
C = B.C
FIG = B.RESULTS / 'figures'
PRE_DAYS = 56
N_SIM = 2000
EXAMPLES = [('신촌', 7, '2018-01-25'), ('역삼', 7, '2018-01-25'), ('신촌', 7, '2024-01-23'), ('신촌', 7, '2024-07-15'),
            ('신촌', 7, '2024-11-28')]

INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e6e5e1', '#fcfcfb'
C_ACT, C_PRED, BAND, ALERT = '#0b0b0b', '#2a78d6', '#2a78d6', '#f3d9cc'
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'figure.facecolor': SURFACE,
                     'axes.facecolor': SURFACE, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8})
_CACHE = {}
MODELS = ['bsts', 'bayes', 'sarimax']
MODEL_LABEL = {'bsts': 'BSTS 최대우도', 'bayes': 'BSTS 베이지안 (MCMC, spike-and-slab)', 'sarimax': 'SARIMAX (1,0,1)(1,1,1,7)'}


def fitted(station, after, model='bsts'):
    """역·설정·모형별 적합 결과와 2014~2024 필터링 결과 (한 번만 적합). 베이지안은 full 대신 None
    (표본마다 필터를 돌리므로 bsts_bayes.pre_and_paths에서 따로 계산)"""
    key = (station, after, model)
    if key not in _CACHE:
        s = B.load(station)
        y, X, since, masked = B.make_xy(s, station, after)
        full = None
        if model != 'bayes':
            res, Xu, _, _ = B.fit(y, X, model)
            full = res.apply(y, exog=Xu, refit=False)
        _CACHE[key] = (s, y, since, masked, full)
    return _CACHE[key]


def episode(since, after, date):
    """date가 들어 있는 특보 구간의 첫날과 마지막 날"""
    w = since <= after
    date = pd.Timestamp(date)
    if not w.get(date, False):
        raise ValueError(f'{date.date()}는 특보 구간이 아님')
    run = (w != w.shift()).cumsum()
    days = w.index[(run == run[date]) & w]
    return days[0], days[-1]


def impact(station, after, date, model='bsts', seed=0):
    s, y, since, masked, full = fitted(station, after, model)
    start, end = episode(since, after, date)
    pre_start = start - pd.Timedelta(days=PRE_DAYS)
    idx = pd.date_range(pre_start, end)
    actual = s.total.reindex(idx)
    excl = masked.reindex(idx)

    n = (end - start).days + 1
    if model == 'bayes':
        # 사후분포 표본마다 필터·시뮬레이션 (모수 불확실성 포함)
        import bsts_bayes
        pre, paths = bsts_bayes.pre_and_paths(station, after, pre_start, start, end, N_SIM, seed)
    else:
        # 구간 앞: 하루 앞 예측 (원래 단위)
        pr = full.get_prediction(start=pre_start, end=start - pd.Timedelta(days=1))
        ci = pr.conf_int(alpha=0.05)
        pre = pd.DataFrame({'pred': np.exp(pr.predicted_mean), 'lo': np.exp(ci.iloc[:, 0]), 'hi': np.exp(ci.iloc[:, 1])})
        # 구간 안: 구간 첫날의 예측 상태에서 기준선 경로 시뮬레이션
        sims = full.simulate(n, anchor=start, repetitions=N_SIM, random_state=seed)
        paths = np.exp(np.asarray(sims).reshape(n, -1))             # (일수, 반복)
    post_idx = pd.date_range(start, end)
    a = actual[post_idx].values[:, None]
    keep = ~excl[post_idx].values[:, None]
    eff_paths = np.where(keep, a - paths, 0.0)
    cum_paths = eff_paths.cumsum(axis=0)
    q = lambda m: np.quantile(m, [0.5, 0.025, 0.975], axis=1)
    post = pd.DataFrame(np.column_stack([*q(paths), *q(np.where(keep, a - paths, np.nan)), *q(cum_paths)]), index=post_idx,
                        columns=['pred', 'lo', 'hi', 'eff', 'eff_lo', 'eff_hi', 'cum', 'cum_lo', 'cum_hi'])
    post.loc[excl[post_idx].values, ['eff', 'eff_lo', 'eff_hi']] = np.nan

    pre['eff'] = actual[pre.index] - pre.pred
    pre['eff_lo'] = actual[pre.index] - pre.hi
    pre['eff_hi'] = actual[pre.index] - pre.lo
    pre.loc[excl[pre.index].values, ['eff', 'eff_lo', 'eff_hi']] = np.nan

    tot_act = float(np.nansum(np.where(keep[:, 0], actual[post_idx].values, 0)))
    tot_pred = np.where(keep, paths, 0).sum(axis=0)
    rel = (tot_act / tot_pred - 1) * 100
    summary = {'model': model, 'station': station, 'after': after, 'start': start.date(), 'end': end.date(), 'days': n,
               'alert_days': int((since[post_idx] == 0).sum()),
               'cum_effect': float(np.median(cum_paths[-1])), 'cum_lo': float(np.quantile(cum_paths[-1], 0.025)),
               'cum_hi': float(np.quantile(cum_paths[-1], 0.975)),
               'rel_pct': float(np.median(rel)), 'rel_lo': float(np.quantile(rel, 0.025)), 'rel_hi': float(np.quantile(rel, 0.975)),
               'p_tail': float(min((cum_paths[-1] > 0).mean(), (cum_paths[-1] < 0).mean()))}
    summary['types'] = '+'.join(B.ALERT_NAMES[c] for c in B.ALERT_NAMES if (s.loc[post_idx, c] == 1).any())
    return actual, pre, post, since[idx], excl, summary


def plot(station, after, date):
    """열마다 모형 하나 (BSTS 최대우도, BSTS 베이지안, SARIMAX). 같은 행은 y축을 공유해 크기를 바로 비교할 수 있게 함"""
    results = {m: impact(station, after, date, m) for m in MODELS}
    k = 1e3
    fig, axes = plt.subplots(3, len(MODELS), figsize=(7.5 * len(MODELS), 10), sharex=True, sharey='row',
                             gridspec_kw={'height_ratios': [1.3, 1, 1]})

    for col, m in enumerate(MODELS):
        actual, pre, post, since, excl, sm = results[m]
        start = post.index[0]
        both = pd.concat([pre, post])
        ax_col = axes[:, col]
        for ax in ax_col:
            for dt in since.index[since == 0]:
                ax.axvspan(dt - pd.Timedelta(hours=12), dt + pd.Timedelta(hours=12), color=ALERT, lw=0, zorder=0)
            ax.axvline(start - pd.Timedelta(hours=12), color=INK, ls='--', lw=1.2, zorder=3)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)

        ax = ax_col[0]
        ax.fill_between(both.index, both.lo / k, both.hi / k, color=BAND, alpha=0.18, lw=0, label='기준선 95% 구간')
        ax.plot(both.index, both.pred / k, color=C_PRED, ls='--', lw=1.6, label='기준선 (특보 없을 때)')
        ax.plot(actual.index, actual / k, color=C_ACT, lw=1.6, label='실제')
        ax.scatter(actual.index[excl], actual[excl] / k, s=22, facecolor=SURFACE, edgecolor=INK2, zorder=4,
                   label='공휴일 등 (효과 계산 제외)')
        ax.set_title(f"{MODEL_LABEL[m]}\n누적 {sm['cum_effect'] / k:+,.1f}천 명 (95% {sm['cum_lo'] / k:+,.1f} ~ {sm['cum_hi'] / k:+,.1f})\n"
                     f"기준선 대비 {sm['rel_pct']:+.1f}% (95% {sm['rel_lo']:+.1f} ~ {sm['rel_hi']:+.1f}%)",
                     loc='left', fontsize=10.5, color=INK)

        ax = ax_col[1]
        ax.axhline(0, color=INK, lw=1)
        ax.fill_between(both.index, both.eff_lo / k, both.eff_hi / k, color=BAND, alpha=0.18, lw=0)
        ax.plot(both.index, both.eff / k, color=C_PRED, ls='--', lw=1.4, label='날마다의 효과 (실제 - 기준선)')

        ax = ax_col[2]
        ax.axhline(0, color=INK, lw=1)
        ax.fill_between(post.index, post.cum_lo / k, post.cum_hi / k, color=BAND, alpha=0.18, lw=0)
        ax.plot(post.index, post.cum / k, color=C_PRED, ls='--', lw=1.6, label='누적 효과')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        for ax in ax_col[1:]:
            ax.legend(loc='lower left', frameon=False, fontsize=9)

    axes[0, 0].set_ylabel('일 이용객 (천 명)')
    axes[1, 0].set_ylabel('효과 (천 명)')
    axes[2, 0].set_ylabel('누적 효과 (천 명)')
    h, l = axes[0, 0].get_legend_handles_labels()
    h += [plt.Rectangle((0, 0), 1, 1, color=ALERT), plt.Line2D([], [], color=INK, ls='--')]
    l += ['특보일', '특보 구간 시작']
    fig.legend(h, l, loc='upper left', ncol=6, frameon=False, fontsize=9, bbox_to_anchor=(0.005, 0.955))
    sm = results['bsts'][-1]
    fig.suptitle(f"{station}  {sm['types']} 특보 구간 {sm['start']} ~ {sm['end']} ({sm['days']}일, 특보일 {sm['alert_days']}일)  |  "
                 f"가리기 {after}일, 같은 통제 변수", x=0.01, ha='left', fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"impact_{station}_{sm['start']}_{after}d.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, [results[m][-1] for m in MODELS]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--station')
    p.add_argument('--after', type=int, default=7)
    p.add_argument('--date')
    a = p.parse_args()
    jobs = [(a.station, a.after, a.date)] if a.station else EXAMPLES
    rows = []
    for st, af, dt in jobs:
        out, sms = plot(st, af, dt)
        rows += sms
        print('저장:', out)
    pd.set_option('display.width', 200)
    print(pd.DataFrame(rows).round(1).to_string(index=False))


if __name__ == '__main__':
    main()
