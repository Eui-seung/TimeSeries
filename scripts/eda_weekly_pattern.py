"""EDA 1: 한 해 전체 지하철 이용량 변화와 주별 반복성.

- (a) 일별 전체 이용량(224개 역 합계)과 7일 이동평균, 요일 유형(평일/주말/공휴일) 표시
- (b) 요일별 분포(공휴일 제외): 평균과 평일 평균 대비 비율
- (c) 일별 이용량에서 7일 이동평균을 뺀 값의 자기상관(ACF): 7·14·21·28일 주기 확인
텍스트는 영어. 실행: python scripts/eda_weekly_pattern.py [--year 2024]
"""
import argparse
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import acf

INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
BLUE, ORANGE, GRAY = '#2a78d6', '#eb6834', '#8a8a85'
DOW = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, default=2024)
    y = ap.parse_args().year
    ensure_dirs()
    D = pd.read_csv(PROC / 'station_daily.csv.gz', usecols=['station', 'date', 'total', 'day_of_week', 'is_public_holiday', 'is_seollal_chuseok', 'holiday_name'], parse_dates=['date'])
    D = D[D.date.dt.year == y]
    day = D.groupby('date').agg(total=('total', 'sum'), n_stations=('station', 'nunique'), dow=('day_of_week', 'first'), holiday=('is_public_holiday', 'max'),
                                seollal_chuseok=('is_seollal_chuseok', 'max'), holiday_name=('holiday_name', lambda s: next((x for x in s if isinstance(x, str) and x), '')))
    assert len(day) == (366 if y % 4 == 0 else 365)
    day['kind'] = np.where(day.holiday == 1, 'Public holiday', np.where(day.dow.isin(['Saturday', 'Sunday']), 'Weekend', 'Weekday'))
    ma = day.total.rolling(7, center=True).mean()

    # ---- 요일별 통계 (공휴일 제외)
    nh = day[day.holiday == 0]
    st = nh.groupby('dow').total.agg(n_days='size', mean='mean', std='std', min='min', max='max').reindex(DOW)
    wk_mean = st.loc[DOW[:5], 'mean'].mean()
    st['ratio_to_weekday_mean'] = (st['mean'] / wk_mean).round(3)
    st['cv_pct'] = (st['std'] / st['mean'] * 100).round(1)
    st.round(0).assign(ratio_to_weekday_mean=st.ratio_to_weekday_mean, cv_pct=st.cv_pct).to_csv(PROC / f'eda_weekly_pattern_{y}.csv', encoding='utf-8-sig', index_label='day_of_week')
    # ---- ACF
    resid = (day.total - ma).dropna()
    a = acf(resid, nlags=35, fft=False)
    print(f'{y}년 일 평균 {day.total.mean()/1e6:.2f}M명 | 요일별(평일 평균=1):', st.ratio_to_weekday_mean.to_dict())
    print('ACF lag 7/14/21/28:', [round(a[k], 2) for k in (7, 14, 21, 28)], '| lag 3.5 부근 최소:', round(a[3:5].min(), 2))

    fig = plt.figure(figsize=(14, 10), facecolor=SURF)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.34, wspace=0.16)
    ax = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    for a_ in (ax, ax2, ax3):
        a_.set_facecolor(SURF)
        a_.grid(axis='y', color=GRID, lw=1)
        a_.set_axisbelow(True)
        a_.tick_params(length=0, labelsize=10)
        for sp in ('top', 'right', 'left'):
            a_.spines[sp].set_visible(False)
    # (a)
    ax.plot(day.index, day.total / 1e6, color='#c9c8c2', lw=0.9, zorder=2)
    for kind, col in (('Weekday', BLUE), ('Weekend', ORANGE), ('Public holiday', GRAY)):
        m = day.kind == kind
        ax.scatter(day.index[m], day.total[m] / 1e6, s=16, color=col, label=kind, zorder=4, lw=0)
    ax.plot(day.index, ma / 1e6, color=INK, lw=2, label='7-day moving average', zorder=3)
    ax.set_ylim(0, day.total.max() / 1e6 * 1.1)
    ax.set_ylabel('Daily ridership (millions, boarding + alighting)', fontsize=10.5)
    ax.set_title(f'(a) Daily ridership, all stations, {y}', loc='left', fontsize=12.5, color=INK)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.set_xlim(day.index.min() - pd.Timedelta(days=3), day.index.max() + pd.Timedelta(days=3))
    ax.legend(loc='lower left', ncol=4, frameon=False, fontsize=10, handletextpad=0.3, columnspacing=1.6)
    for nm, key in (('Seollal', '설날'), ('Chuseok', '추석')):
        d = day[day.holiday_name.str.contains(key)]
        if len(d):
            ax.annotate(nm, (d.index[len(d) // 2], d.total.min() / 1e6 if False else d.total.iloc[len(d) // 2] / 1e6), xytext=(0, -26), textcoords='offset points', ha='center', fontsize=9.5, color=INK2, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8))
    # (b)
    for i, dname in enumerate(DOW):
        v = nh[nh.dow == dname].total / 1e6
        col = BLUE if i < 5 else ORANGE
        rng = np.random.default_rng(i)
        ax2.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=14, color=col, alpha=0.55, lw=0, zorder=3)
        ax2.hlines(v.mean(), i - 0.34, i + 0.34, color=INK, lw=2.2, zorder=4)
        ax2.text(i, v.max() + 0.12, f'{st.loc[dname, "ratio_to_weekday_mean"]:.2f}×', ha='center', fontsize=9.5, color=INK2)
    ax2.set_xticks(range(7))
    ax2.set_xticklabels([d[:3] for d in DOW])
    ax2.set_ylim(0, nh.total.max() / 1e6 * 1.2)
    ax2.set_ylabel('Daily ridership (millions)', fontsize=10.5)
    ax2.set_title('(b) By day of week (public holidays excluded)', loc='left', fontsize=12.5, color=INK)
    ax2.text(0.99, 0.04, 'Bar = mean; label = ratio to the Mon–Fri mean', transform=ax2.transAxes, ha='right', fontsize=9, color=INK2)
    # (c)
    lags = np.arange(1, 36)
    ax3.bar(lags, a[1:], color=[ORANGE if l % 7 == 0 else '#b7d3f6' for l in lags], width=0.72, zorder=3)
    ax3.axhline(0, color=INK2, lw=0.8)
    ax3.set_xticks([7, 14, 21, 28, 35])
    ax3.set_xlabel('Lag (days)', fontsize=10.5)
    ax3.set_ylabel('Autocorrelation', fontsize=10.5)
    ax3.set_title('(c) Autocorrelation of ridership minus its 7-day average', loc='left', fontsize=12.5, color=INK)
    ax3.text(0.99, 0.96, 'Orange = multiples of 7 days', transform=ax3.transAxes, ha='right', va='top', fontsize=9, color=INK2)
    fig.suptitle(f'Weekly Repetition in Daily Subway Ridership, {y}', fontsize=16, y=0.97, color=INK)
    fig.text(0.5, 0.012, f'All {day.n_stations.max()} Seoul stations summed. Public holidays from the Korean public holiday calendar.', ha='center', fontsize=8.8, color=INK2)
    fig.subplots_adjust(top=0.9, bottom=0.07, left=0.07, right=0.98)
    fig.savefig(FIG / f'fig_eda_weekly_pattern_{y}.png', dpi=300, facecolor=SURF)
    print('saved', FIG / f'fig_eda_weekly_pattern_{y}.png')


if __name__ == '__main__':
    main()
