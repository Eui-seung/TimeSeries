"""그림: 논문 그림 2·4 재현, 그룹별 일별 추이 (텍스트는 영어, 역 이름만 한글, 글꼴 AppleGothic).

산출: figures/fig_hourly_profile_4stations.png, fig_station_type_venn.png, fig_group_daily_trend.png (300dpi)
실행: python scripts/step6_figures.py
"""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager as fm
from matplotlib.patches import Circle

CLASS_YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2023, 2024]
fm.fontManager.addfont('/System/Library/Fonts/Supplemental/AppleGothic.ttf')
plt.rcParams.update({'font.family': 'AppleGothic', 'axes.unicode_minus': False, 'savefig.dpi': 300, 'text.color': '#0b0b0b',
                     'axes.edgecolor': '#9a9a95', 'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e'})
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
C = {'Employment': '#2a78d6', 'Residential': '#eb6834', 'Commercial & Leisure': '#1baf7a', 'Mixed': '#e87ba4', 'Unclassified': '#8a8a85'}
EN = {'고용형': 'Employment', '주거형': 'Residential', '상업·여가형': 'Commercial & Leisure', '복합형(상업·여가+고용)': 'Mixed (Commercial + Employment)',
      '복합형(상업·여가+주거)': 'Mixed (Commercial + Residential)', '복합형(상업·여가+주거+고용)': 'Mixed (all three)', '복합형(주거+고용)': 'Mixed (Residential + Employment)', '미분류': 'Unclassified'}


def hourly_profile(stations):
    """3월 분류 연도, 3/1 제외. 주말 = 토·일 평균. 각 선 = 자기 방향(승차/하차)의 하루 합 대비 비율(%), 연도별로 계산해 평균."""
    T = pd.read_csv(PROC / 'station_date_hour.csv.gz', parse_dates=['date'])
    T = T[T.station.isin(stations) & (T.date.dt.month == 3) & T.date.dt.year.isin(CLASS_YEARS) & ~((T.date.dt.month == 3) & (T.date.dt.day == 1))].copy()
    T['y'] = T.date.dt.year
    wd = T.date.dt.dayofweek
    T['dt'] = np.where(wd < 5, 'weekday', np.where(wd == 5, 'sat', 'sun'))
    m = T.groupby(['station', 'y', 'dt', 'direction'])[HLAB].mean()
    out = {}
    for s in stations:
        shares = []
        for y in CLASS_YEARS:
            x = m.loc[s].loc[y]
            for nm, v in (('Weekday', x.loc['weekday']), ('Weekend', (x.loc['sat'] + x.loc['sun']) / 2)):
                for io, lab in (('승차', 'boarding'), ('하차', 'alighting')):
                    shares.append(dict(y=y, line=f'{nm} {lab}', **{h: v.loc[io, h] / v.loc[io].sum() * 100 for h in HLAB}))
        out[s] = pd.DataFrame(shares).groupby('line')[HLAB].mean()
    return out


def fig2(st):
    ST = ['강남', '강변', '명동', '아현']
    prof = hourly_profile(ST)
    hrs, cols = list(range(5, 24)), HLAB[:-1]            # 05~23시 (24시이후 제외)
    fig, axs = plt.subplots(2, 2, figsize=(13, 9.2), sharey=True, facecolor=SURF)
    STY = {'Weekday boarding': dict(c=C['Employment'], ls='-', m='o'), 'Weekday alighting': dict(c=C['Residential'], ls='-', m='s'),
           'Weekend boarding': dict(c=C['Employment'], ls=(0, (5, 2.5)), m='o'), 'Weekend alighting': dict(c=C['Residential'], ls=(0, (5, 2.5)), m='s')}
    for i, (ax, s) in enumerate(zip(axs.ravel(), ST)):
        ax.set_facecolor(SURF)
        for ln, sty in STY.items():
            ax.plot(hrs, prof[s].loc[ln, cols].values, color=sty['c'], ls=sty['ls'], lw=2, marker=sty['m'], ms=5.5, mfc=sty['c'] if 'Weekday' in ln else SURF, mec=sty['c'], mew=1.4, label=ln, zorder=3)
        t = EN[st.loc[st.station == s, 'type_kr'].iloc[0]]
        ax.set_title(f'({"abcd"[i]}) {s} – {t}', fontsize=13, loc='left', color=INK, pad=8)
        ax.set_xlim(4.6, 23.4)
        ax.set_ylim(0, 18)
        ax.set_xticks(hrs)
        ax.set_yticks(range(0, 19, 3))
        ax.tick_params(labelsize=10, length=0)
        ax.grid(axis='y', color=GRID, lw=1)
        ax.set_axisbelow(True)
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
        if i >= 2:
            ax.set_xlabel('Hour', fontsize=11)
        if i % 2 == 0:
            ax.set_ylabel('Share of daily ridership (%)', fontsize=11)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.925), ncol=4, frameon=False, fontsize=11, handlelength=3.2, columnspacing=2.2)
    fig.suptitle('Hourly Ridership Profiles of Selected Stations (Weekday vs. Weekend)', fontsize=16, y=0.985, color=INK)
    fig.text(0.5, 0.005, 'March, 2014–2019 and 2023–2024; public holidays excluded; weekend = mean of Saturday and Sunday. Each line sums to 100% of its own daily total (hours 05–23 shown; hour 05 includes earlier trips).', ha='center', fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.9))
    fig.savefig(FIG / 'fig_hourly_profile_4stations.png', facecolor=SURF)
    plt.close(fig)


def fig4(st):
    cls = st[st.status != '분류제외']
    f = lambda t: sorted(cls[cls.type_kr == t].station.tolist())
    reg = {'E': f('고용형'), 'R': f('주거형'), 'C': f('상업·여가형'), 'CE': f('복합형(상업·여가+고용)'), 'CR': f('복합형(상업·여가+주거)'), 'RE': f('복합형(주거+고용)'), 'ALL': f('복합형(상업·여가+주거+고용)')}
    n_un = int((st.type_kr == '미분류').sum())
    r = 3.4
    cen = {'E': (0, 1.55), 'R': (-1.95, -1.6), 'C': (1.95, -1.6)}
    fig, ax = plt.subplots(figsize=(13, 10.2), facecolor=SURF)
    ax.set_facecolor(SURF)
    ax.set_aspect('equal')
    ax.axis('off')
    colmap = {'E': C['Employment'], 'R': C['Residential'], 'C': C['Commercial & Leisure']}
    for k, (x, y) in cen.items():
        ax.add_patch(Circle((x, y), r, fc=colmap[k], alpha=0.13, ec='none', zorder=1))
        ax.add_patch(Circle((x, y), r, fc='none', ec=colmap[k], lw=2.2, zorder=2))
    gx, gy = np.meshgrid(np.linspace(-6, 6, 1200), np.linspace(-5.2, 5.3, 1050))
    inn = {k: (gx - x) ** 2 + (gy - y) ** 2 <= r * r for k, (x, y) in cen.items()}
    masks = {'E': inn['E'] & ~inn['R'] & ~inn['C'], 'R': inn['R'] & ~inn['E'] & ~inn['C'], 'C': inn['C'] & ~inn['E'] & ~inn['R'], 'CE': inn['C'] & inn['E'] & ~inn['R'],
             'CR': inn['C'] & inn['R'] & ~inn['E'], 'RE': inn['R'] & inn['E'] & ~inn['C'], 'ALL': inn['E'] & inn['R'] & inn['C']}
    per = {'E': 4, 'R': 3, 'C': 3, 'CE': 2, 'CR': 2, 'RE': 1, 'ALL': 2}
    fs = {'E': 11, 'R': 11, 'C': 11, 'CE': 9.5, 'CR': 9.5, 'RE': 9.5, 'ALL': 9.5}
    shift = {'E': (0, 0.35), 'R': (-0.25, 0.0), 'C': (0.7, 0.15), 'CE': (-0.05, -0.05), 'CR': (0, -0.05), 'RE': (-0.1, 0.05), 'ALL': (0, 0.0)}
    wrap = lambda names, k: '\n'.join(', '.join(names[i:i + k]) + (',' if i + k < len(names) else '') for i in range(0, len(names), k))
    for k, names in reg.items():
        m = masks[k]
        cx, cy = gx[m].mean(), gy[m].mean()
        dx, dy = shift[k]
        txt = wrap(names, per[k])
        if k == 'CE' and '동대문역사문화공원' in names:        # 긴 이름은 단독 줄로
            rest = [n for n in names if n != '동대문역사문화공원']
            txt = ', '.join(rest[:3]) + ',\n동대문역사문화공원,\n' + wrap(rest[3:], 2)
        ax.text(cx + dx, cy + dy, txt, ha='center', va='center', fontsize=fs[k], color=INK, zorder=5, linespacing=1.45)
    ax.text(0, 1.55 + r + 0.2, 'Employment', ha='center', va='bottom', fontsize=18, color=INK)
    ax.text(-1.95 - r * 0.80, -1.6 - r * 0.80 - 0.15, 'Residential', ha='right', va='top', fontsize=18, color=INK)
    ax.text(1.95 + r * 0.80, -1.6 - r * 0.80 - 0.15, 'Commercial & Leisure', ha='left', va='top', fontsize=18, color=INK)
    ax.text(5.6, 4.0, f'Unclassified ({n_un} stations)', ha='center', va='center', fontsize=14, color=INK2, style='italic')
    ax.set_xlim(-7.6, 8.0)
    ax.set_ylim(-5.4, 5.5)
    fig.suptitle('Station Types by Mixed Functions', fontsize=20, y=0.99, color=INK)
    fig.text(0.5, 0.03, f'{len(cls) - n_un} classified stations (factor score ≥ 1). Overlapping areas = mixed-function stations.', ha='center', fontsize=10.5, color=INK2)
    fig.savefig(FIG / 'fig_station_type_venn.png', facecolor=SURF, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)


def fig_trend():
    G = pd.read_csv(PROC / 'group_daily.csv', parse_dates=['date'])
    G = G[G.split == 'train']
    P = G.pivot(index='date', columns='group', values='total')[list(C)]
    n = G.groupby('group').n_stations.max()
    MA = P.rolling(7, center=True).mean()
    IDX = MA / P.loc['2019'].mean() * 100
    fig, axs = plt.subplots(2, 1, figsize=(13, 9.6), sharex=True, facecolor=SURF, gridspec_kw={'hspace': 0.12})
    for ax in axs:
        ax.set_facecolor(SURF)
        ax.axvspan(pd.Timestamp('2020-01-01'), pd.Timestamp('2022-12-31'), color='#ececE8', zorder=0, lw=0)
        ax.grid(axis='y', color=GRID, lw=1)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, labelsize=10)
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
    for g, col in C.items():
        axs[0].plot(MA.index, MA[g] / 1e6, color=col, lw=1.6, label=f'{g} ({n[g]} stations)', zorder=3)
        axs[1].plot(IDX.index, IDX[g], color=col, lw=1.6, zorder=3)
    axs[0].set_ylabel('Daily ridership (millions, boarding + alighting)', fontsize=10.5)
    axs[1].set_ylabel('Index (2019 daily mean = 100)', fontsize=10.5)
    axs[0].set_title('(a) Daily ridership', loc='left', fontsize=12.5, color=INK)
    axs[1].set_title('(b) Relative to the 2019 level', loc='left', fontsize=12.5, color=INK)
    axs[0].set_ylim(0, None)
    axs[1].set_ylim(30, 135)
    axs[1].axhline(100, color=INK2, lw=0.8, ls=(0, (2, 3)), zorder=2)
    axs[0].text(pd.Timestamp('2021-07-01'), axs[0].get_ylim()[1] * 0.97, 'COVID-19 period (2020–2022)', ha='center', va='top', fontsize=9.5, color=INK2)
    axs[1].xaxis.set_major_locator(mdates.YearLocator())
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axs[1].set_xlim(pd.Timestamp('2014-01-01'), pd.Timestamp('2024-12-31'))
    axs[1].set_xlabel('Date', fontsize=10.5)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.93), ncol=5, frameon=False, fontsize=10, handlelength=2.4, columnspacing=1.6)
    fig.suptitle('Daily Ridership Trends by Station Group, 2014–2024 (7-day moving average)', fontsize=15.5, y=0.985, color=INK)
    fig.text(0.5, 0.012, 'Unclassified includes 강일 from 2021 (169 → 170 stations). 암사역사공원 (excluded, 2024 only) is not included.', ha='center', fontsize=8.8, color=INK2)
    fig.subplots_adjust(top=0.84, bottom=0.07, left=0.075, right=0.985)
    fig.savefig(FIG / 'fig_group_daily_trend.png', facecolor=SURF)
    plt.close(fig)


def main():
    ensure_dirs()
    st = pd.read_csv(PROC / 'station_type.csv')
    fig2(st)
    fig4(st)
    fig_trend()
    print('그림 3종 저장:', FIG)


if __name__ == '__main__':
    main()
