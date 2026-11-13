"""기준선 모형 결과 정리: BSTS vs SARIMAX, 역 × 특보 가리기 기간(7일, 14일)별 train 적합 vs validation(2024) 비교표와 그림.

먼저 실행: python scripts/bsts_model.py  (bsts_*.csv),  python scripts/bsts_model.py --model sarimax  (sarimaxbl_*.csv),
          python scripts/bsts_bayes.py  (bstsbayes_*.csv, 베이지안 BSTS)
세 모형은 y, 가리기, 회귀 X, 학습·검증 기간, 초기화 구간이 모두 같고 시계열 구조만 다름
(BSTS: local level + AR(1) + 확률적 주간 계절성을 최대우도 또는 MCMC(spike-and-slab)로 추정, SARIMAX: (1,0,1)(1,1,1,7))

- 정확도는 '평소' 날(마지막 특보일로부터 15일 이상 지난 날, MASK_COLS 날 제외)로 계산.
  이 날들은 7일·14일 두 설정 모두 y를 가리지 않은 날이라 두 설정을 같은 날끼리 비교할 수 있음.
  train은 적합에 쓴 날의 하루 앞 예측(in-sample), validation은 계수를 고정한 2024년 하루 앞 예측
- 참고로 각 설정에서 가리지 않은 모든 날(관측일)의 MAPE도 함께 둠
- 특보일 효과(실제/기준선 - 1)는 특보일의 평균. 여기서 기준선은 가린 구간을 이어서 예측한 값

출력: results/baseline_train_vs_valid.csv (세 모형), bsts_train_vs_valid.csv (BSTS만),
      results/figures/baseline_train_vs_valid_mape.png, baseline_valid2024_series_<after>d.png
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
FIG = RES / 'figures'
STATIONS = ['시청', '역삼', '선릉', '신촌', '혜화', '신사', '서울대입구', '수유', '연신내']

INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e6e5e1', '#fcfcfb'
C_ACT = '#3d3d3a'                                  # 실제값 = 진한 회색
MODEL_COLOR = {'bsts': '#2a78d6', 'bayes': '#1baf7a', 'sarimax': '#eb6834'}
MODEL_LABEL = {'bsts': 'BSTS (최대우도)', 'bayes': 'BSTS (베이지안)', 'sarimax': 'SARIMAX'}
PREFIX = {'bsts': 'bsts', 'bayes': 'bstsbayes', 'sarimax': 'sarimaxbl'}

plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'figure.facecolor': SURFACE,
                     'axes.facecolor': SURFACE, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8})


def metrics(g):
    e = g.total - g.baseline
    return pd.Series({'일수': len(g), 'MAPE': (e.abs() / g.total).mean() * 100, 'RMSE': np.sqrt((e ** 2).mean()),
                      'MAE': e.abs().mean(), '95%밖': g.outside95.mean() * 100})


def table(d, info):
    normal = d[d.kind == '평소'].groupby(['station', 'after', 'period']).apply(metrics, include_groups=False).unstack('period')
    normal.columns = [f'{p}_{m}' for m, p in normal.columns]
    obs = d[~d.y_masked].groupby(['station', 'after', 'period']).apply(
        lambda g: (g.total - g.baseline).abs().div(g.total).mean() * 100, include_groups=False).unstack('period')
    obs.columns = [f'{p}_MAPE_관측일전체' for p in obs.columns]
    eff = d[d.kind == '특보일'].groupby(['station', 'after', 'period']).effect_pct.mean().unstack('period')
    eff.columns = [f'{p}_특보일효과' for p in eff.columns]
    t = normal.join(obs).join(eff)
    t['MAPE_비(valid/train)'] = t.validation_MAPE / t.train_MAPE
    t = t.join(info.set_index(['station', 'after'])[['train_obs', 'aic', 'converged']])
    cols = ['train_obs', 'aic', 'converged',
            'train_일수', 'train_MAPE', 'train_RMSE', 'train_MAE', 'train_95%밖',
            'validation_일수', 'validation_MAPE', 'validation_RMSE', 'validation_MAE', 'validation_95%밖', 'MAPE_비(valid/train)',
            'train_MAPE_관측일전체', 'validation_MAPE_관측일전체', 'train_특보일효과', 'validation_특보일효과']
    return t[cols].reset_index()


def fig_dumbbell(T):
    """역별 train(빈 점) vs validation(채운 점) MAPE, 평소 날. 모형별 색. 왼쪽 7일, 오른쪽 14일"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True, sharex=True)
    ys = np.arange(len(STATIONS))[::-1].astype(float)
    off = {'bsts': 0.25, 'bayes': 0.0, 'sarimax': -0.25}
    for ax, after in zip(axes, [7, 14]):
        for m, c in MODEL_COLOR.items():
            s = T[(T.after == after) & (T.model == m)].set_index('station').reindex(STATIONS)
            yy = ys + off[m]
            for y, (tr, va) in zip(yy, s[['train_MAPE', 'validation_MAPE']].values):
                ax.plot([tr, va], [y, y], color=c, alpha=0.35, lw=2, zorder=1)
            ax.scatter(s.train_MAPE, yy, s=48, facecolor=SURFACE, edgecolor=c, lw=1.8, zorder=2, label=f'{MODEL_LABEL[m]} train')
            ax.scatter(s.validation_MAPE, yy, s=52, color=c, edgecolor=SURFACE, lw=1.5, zorder=3, label=f'{MODEL_LABEL[m]} validation')
            for y, tr, va in zip(yy, s.train_MAPE, s.validation_MAPE):
                right = va >= tr
                ax.text(va + (0.12 if right else -0.12), y, f'{va:.1f}', va='center', ha='left' if right else 'right',
                        fontsize=8, color=INK2)
        ax.set_title(f'특보 가리기 {after}일', loc='left', fontsize=12, color=INK)
        ax.set_yticks(ys, STATIONS)
        ax.set_xlabel('MAPE (%), 평소 날 하루 앞 예측')
        ax.grid(axis='y', visible=False)
        ax.set_xlim(0, T[['train_MAPE', 'validation_MAPE']].max().max() + 0.9)
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper right', ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.99, 0.96))
    fig.suptitle('기준선 모형 예측 오차: train(빈 점) vs validation(채운 점)', x=0.01, ha='left', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / 'baseline_train_vs_valid_mape.png', dpi=150)
    plt.close(fig)


def fig_valid_series(D, after):
    """2024년 평일만: 역별 실제값 vs 두 모형 기준선 (같은 가리기 기간). 특보일은 옅은 띠.
    주말을 빼야 요일 진폭에 묻히지 않음. 빠진 날(공휴일 등)은 선이 끊김. 축 위로 벗어난 기준선은 위쪽 ▲로 표시"""
    v = D[(D.period == 'validation') & (D.after == after)].copy()
    v['date'] = pd.to_datetime(v.date)
    days = pd.date_range('2024-01-01', '2024-12-31')
    weekdays = days[days.dayofweek < 5]
    fig, axes = plt.subplots(3, 3, figsize=(15, 9.5), sharex=True)
    for ax, st in zip(axes.flat, STATIONS):
        a = v[(v.station == st) & (v.model == 'bsts')].set_index('date').reindex(weekdays)
        for dt in a.index[a.kind == '특보일']:
            ax.axvspan(dt - pd.Timedelta(hours=12), dt + pd.Timedelta(hours=12), color='#f3d9cc', lw=0, zorder=0)
        lo, hi = np.nanpercentile(a.total / 1e3, [1, 99])
        pad = (hi - lo) * 0.25
        top = hi + pad
        ax.plot(a.index, a.total / 1e3, color=C_ACT, lw=1.2, label='실제')
        for m, c in MODEL_COLOR.items():
            s = v[(v.station == st) & (v.model == m)].set_index('date').reindex(weekdays)
            ax.plot(s.index, s.baseline / 1e3, color=c, lw=1, alpha=0.9, label=f'{MODEL_LABEL[m]} 기준선')
            over = s.baseline / 1e3 > top
            ax.scatter(s.index[over], np.full(over.sum(), top), marker='^', s=30, color=c, zorder=4, clip_on=False)
        ax.set_ylim(lo - pad, top)
        ax.set_title(st, loc='left', fontsize=11)
        ax.grid(axis='x', visible=False)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel('일 이용객 (천 명)')
    for ax in axes[-1]:
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m월'))
    h, l = axes[0, 0].get_legend_handles_labels()
    h.append(plt.Rectangle((0, 0), 1, 1, color='#f3d9cc')); l.append('특보일')
    h.append(plt.Line2D([], [], marker='^', ls='', color=INK2)); l.append('축 밖 기준선')
    fig.legend(h, l, loc='upper right', ncol=5, frameon=False, fontsize=10)
    fig.suptitle(f'2024 validation (평일): 실제 이용객과 기준선, 특보 가리기 {after}일', x=0.01, ha='left', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / f'baseline_valid2024_series_{after}d.png', dpi=150)
    plt.close(fig)


def load(model):
    d = pd.read_csv(RES / f'{PREFIX[model]}_daily.csv', encoding='utf-8-sig')
    info = pd.read_csv(RES / f'{PREFIX[model]}_fitinfo.csv', encoding='utf-8-sig')
    d['model'] = model
    return d, info


def main():
    pd.set_option('display.width', 250, 'display.max_columns', 30)
    tables, dailies = [], []
    for m in MODEL_COLOR:
        d, info = load(m)
        t = table(d, info)
        t.insert(0, 'model', m)
        tables.append(t)
        dailies.append(d)
    T = pd.concat(tables, ignore_index=True)
    D = pd.concat(dailies, ignore_index=True)
    T['station'] = pd.Categorical(T.station, STATIONS, ordered=True)
    T = T.sort_values(['station', 'after', 'model']).reset_index(drop=True)
    T.to_csv(RES / 'baseline_train_vs_valid.csv', index=False, encoding='utf-8-sig')
    T[T.model == 'bsts'].drop(columns='model').to_csv(RES / 'bsts_train_vs_valid.csv', index=False, encoding='utf-8-sig')

    W = T.pivot_table(index=['station', 'after'], columns='model', observed=True,
                      values=['train_MAPE', 'validation_MAPE', 'validation_RMSE', 'train_특보일효과', 'validation_특보일효과'])
    W.columns = [f'{v}|{m}' for v, m in W.columns]
    print(W.round(2).to_string())
    bad = T.loc[T.converged.eq(False) | T.converged.eq('False'), ['model', 'station', 'after']]   # 베이지안은 수렴 표시 없음(NaN)
    print('\n수렴하지 않은 적합:', bad.to_dict('records') if len(bad) else '없음')
    FIG.mkdir(parents=True, exist_ok=True)
    fig_dumbbell(T)
    for after in (7, 14):
        fig_valid_series(D, after)
    print(f'\n저장: {RES / "baseline_train_vs_valid.csv"}, {FIG}/baseline_*.png')


if __name__ == '__main__':
    main()
