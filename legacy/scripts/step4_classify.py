"""4단계: 역 유형 분류 (성현곤·김태현 2005 방식).

- 자료: 2014~2019, 2023~2024년 3월의 공휴일(3/1) 아닌 평일·토·일 → 역별 요일×5시간대×승/하차 = 30개 변수(일평균 이용객 수)
  연도별 일평균을 만든 뒤 사용 연도들의 평균을 역별 값으로 사용
- 분석: PCA(고유값>1 → 3요인) + oblimin 회전(패턴 행렬), 요인점수는 회귀 방식(SPSS 기본)
- 분류: 요인점수 ≥ 1인 요인이 그 역의 기능. 2개 이상이면 복합형, 모두 1 미만이면 미분류.
  3월 존재 연도가 2개 미만이면 분류 제외
- 출력: station_type.csv, station_type_lists.csv, step4_table2/3/4_comparison.csv, yearly_*.csv, station_type_change_vs_2005.csv

실행: python scripts/step4_classify.py   (station_date_hour.csv.gz 필요)
"""
import itertools
from common import *          # (먼저 import: 추가 패키지 경로 설정)
from scipy import stats
from factor_analyzer.rotator import Rotator

CLASS_YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2023, 2024]
BINS = {'09시이전': ['before_06', 'h06_07', 'h07_08', 'h08_09'], '09-12': ['h09_10', 'h10_11', 'h11_12'],
        '12-18': ['h12_13', 'h13_14', 'h14_15', 'h15_16', 'h16_17', 'h17_18'], '18-21': ['h18_19', 'h19_20', 'h20_21'],
        '21시이후': ['h21_22', 'h22_23', 'h23_24', 'after_24']}
BO = list(BINS)
DAYS = ['평일', '토', '일']
IO = ['승차', '하차']
VARS = [f'{d}_{b}_{i}' for d in DAYS for b in BO for i in IO]
FN = ['상업·여가', '주거', '고용']
# 논문 표 3 (PDF 8쪽, 0.5 이상 패턴 행렬 적재량): 변수 -> {요인: 적재량}
PAPER = {
    '일_21시이후_승차': {1: 1.011}, '일_18-21_승차': {1: .971}, '일_12-18_하차': {1: .953}, '토_21시이후_승차': {1: .925}, '토_12-18_하차': {1: .821},
    '토_18-21_승차': {1: .805}, '일_18-21_하차': {1: .683, 2: .507}, '토_18-21_하차': {1: .676}, '일_12-18_승차': {1: .651}, '일_09-12_하차': {1: .558}, '토_12-18_승차': {1: .52},
    '평일_09시이전_승차': {2: 1.084}, '토_09시이전_승차': {2: 1.057}, '평일_21시이후_하차': {2: .978}, '일_09시이전_승차': {2: .978}, '일_09-12_승차': {2: .952}, '토_09-12_승차': {2: .9},
    '일_21시이후_하차': {2: .883}, '평일_09-12_승차': {2: .874}, '토_21시이후_하차': {2: .851}, '평일_18-21_하차': {2: .622},
    '평일_09시이전_하차': {3: 1.09}, '토_09시이전_하차': {3: .949}, '평일_18-21_승차': {3: .857}, '평일_09-12_하차': {3: .844}, '평일_12-18_승차': {3: .641}, '토_09-12_하차': {3: .603},
    '평일_21시이후_승차': {1: .51, 3: .553}, '평일_12-18_하차': {3: .501}}
PAPER_EIG = [21.947, 5.496, 1.14, 0.533, 0.219, 0.15, 0.114, 0.08, 0.065, 0.056]
PAPER_CUM = [73.156, 91.475, 95.273, 97.05, 97.779, 98.28, 98.661, 98.928, 99.146, 99.333]
# 논문 표 4 (PDF 8쪽): 요인별 점수 1 이상 역 {요인: {역: 점수}}
T4 = {
    1: {'강남': 3.5, '강변': 2.9, '건대입구': 2.0, '고속터미널': 5.0, '노원': 1.52, '동대문': 2.0, '동대문운동장': 3.7, '명동': 3.4, '사당': 1.7, '삼성': 3.0, '서울대입구': 1.2, '서울역': 2.3, '신림': 2.1, '신촌': 3.9, '압구정': 1.3, '을지로입구': 2.0, '잠실': 3.4, '종각': 2.1, '종로3가': 3.2, '천호': 1.26, '청량리': 1.6, '혜화': 3.1, '홍대입구': 1.7},
    2: {'강남': 1.3, '강변': 2.2, '건대입구': 2.2, '고속터미널': 1.6, '광명사거리': 1.3, '구로디지털단지': 2.6, '까치산': 1.8, '낙성대': 1.2, '노원': 2.1, '당산': 1.7, '대림': 2.1, '미아삼거리': 2.2, '봉천': 1.0, '사당': 3.2, '상계': 1.3, '상봉': 1.1, '서울대입구': 2.5, '서울역': 1.4, '수유': 3.2, '신도림': 1.2, '신림': 5.1, '신촌': 1.3, '쌍문': 2.3, '연신내': 2.2, '이수': 1.8, '잠실': 2.9, '창동': 1.8, '천호': 1.3, '청량리': 2.3, '홍대입구': 1.3, '화곡': 1.7},
    3: {'강남': 4.6, '고속터미널': 2.2, '광화문': 1.5, '교대': 2.8, '구로디지털단지': 1.0, '남부터미널': 1.4, '동대문': 2.0, '동대문운동장': 1.3, '사당': 1.8, '삼성': 4.2, '서울역': 2.3, '선릉': 3.4, '시청': 2.9, '신촌': 1.0, '압구정': 2.0, '양재': 2.5, '여의도': 1.6, '역삼': 3.2, '을지로3가': 1.5, '을지로4가': 1.0, '을지로입구': 2.1, '잠실': 2.5, '종각': 1.7, '종로3가': 2.6, '종로5가': 1.1, '청담': 1.1, '충무로': 2.1, '학동': 1.2, '회현': 1.4}}
ALIAS = {'동대문운동장': '동대문역사문화공원', '미아삼거리': '미아사거리'}      # 논문의 옛 이름 -> 현재 이름
# 논문 그림 4 (PDF 9쪽) 영역별 역
FIG4 = {frozenset('E'): ['광화문', '교대', '남부터미널', '선릉', '시청', '양재', '여의도', '역삼', '을지로3가', '을지로4가', '종로5가', '청담', '충무로', '학동', '회현'],
        frozenset('R'): ['광명사거리', '까치산', '낙성대', '당산', '대림', '미아삼거리', '봉천', '상계', '상봉', '수유', '신도림', '쌍문', '연신내', '이수', '창동', '화곡'],
        frozenset('C'): ['명동', '혜화'], frozenset('RE'): ['구로디지털단지'],
        frozenset('CE'): ['동대문', '동대문운동장', '삼성', '압구정', '을지로입구', '종각', '종로3가'],
        frozenset('CR'): ['강변', '건대입구', '노원', '서울대입구', '신림', '천호', '청량리', '홍대입구'],
        frozenset('CER'): ['강남', '고속터미널', '사당', '서울역', '신촌', '잠실']}


# ---------------------------------------------------------------- 변수 만들기
def march_hourly():
    T = pd.read_csv(PROC / 'station_date_hour.csv.gz', parse_dates=['date'])
    return T[(T.date.dt.month == 3) & T.date.dt.year.isin(CLASS_YEARS)]


def build_features(T):
    """3월 평일·토·일(3/1 제외) -> (역, 연도)별 요일×시간대×승하차 일평균 이용객 수 (long)"""
    T = T[~((T.date.dt.month == 3) & (T.date.dt.day == 1))].copy()
    for b, cols in BINS.items():
        T[b] = T[cols].sum(axis=1)
    T['y'] = T.date.dt.year
    wd = T.date.dt.dayofweek
    T['요일'] = np.where(wd < 5, '평일', np.where(wd == 5, '토', '일'))
    T['구분'] = T.direction
    m = T.groupby(['station', 'y', '요일', '구분'])[BO].mean().reset_index()
    long = m.melt(id_vars=['station', 'y', '요일', '구분'], value_vars=BO, var_name='시간대', value_name='평균이용')
    long['변수'] = long.요일 + '_' + long.시간대 + '_' + long.구분
    return long, T


def kmo(X):
    R = np.corrcoef(X, rowvar=False)
    Ri = np.linalg.pinv(R)
    Pm = -Ri / np.sqrt(np.outer(np.diag(Ri), np.diag(Ri)))
    np.fill_diagonal(Pm, 0)
    np.fill_diagonal(R, 0)
    r2, p2 = (R ** 2).sum(axis=0), (Pm ** 2).sum(axis=0)
    return r2.sum() / (r2.sum() + p2.sum()), r2 / (r2 + p2)


def bartlett(X):
    n, p = X.shape
    ev = np.linalg.eigvalsh(np.corrcoef(X, rowvar=False))
    chi = -(n - 1 - (2 * p + 5) / 6) * np.log(ev[ev > 1e-10]).sum()
    df = p * (p - 1) / 2
    return chi, df, stats.chi2.sf(chi, df)


def solve(X, ref=None):
    """3요인 PCA + oblimin. 회귀 점수(R^-1 S)와 최소제곱 점수(PCA에서는 동일). ref가 있으면 요인 순서·부호를 ref에 맞춤."""
    Z = (X - X.mean(0)) / X.std(0, ddof=1)
    R = np.corrcoef(X, rowvar=False)
    w, v = np.linalg.eigh(R)
    o = np.argsort(w)[::-1]
    w, v = w[o], v[:, o]
    A = v[:, :3] * np.sqrt(w[:3])
    A = A * np.sign(A.sum(0))
    rot = Rotator(method='oblimin')
    Ar = rot.fit_transform(A)
    phi = rot.phi_
    if ref is None:
        sets_p = {f: set(v_ for v_, d in PAPER.items() if f in d) for f in (1, 2, 3)}
        so = [set(np.array(VARS)[np.abs(Ar[:, j]) >= .5]) for j in range(3)]
        best = max(itertools.permutations(range(3)), key=lambda pm: sum(len(so[pm[f - 1]] & sets_p[f]) for f in (1, 2, 3)))
    else:
        best = max(itertools.permutations(range(3)), key=lambda pm: sum(abs(np.corrcoef(Ar[:, pm[f]], ref[:, f])[0, 1]) for f in range(3)))
    best = list(best)
    Ar, phi = Ar[:, best], phi[np.ix_(best, best)]
    sg = np.sign(Ar.sum(0)) if ref is None else np.array([np.sign(np.corrcoef(Ar[:, f], ref[:, f])[0, 1]) for f in range(3)])
    Ar, phi = Ar * sg, phi * np.outer(sg, sg)
    F_reg = Z @ np.linalg.solve(R, Ar @ phi)
    F_ls = Z @ Ar @ np.linalg.inv(Ar.T @ Ar)
    return dict(Ar=Ar, A=A, phi=phi, F_reg=F_reg, F_ls=F_ls, eig=w)


def typ(sc):
    f = [FN[j] for j in range(3) if sc[j] >= 1]
    return '미분류' if not f else (f[0] + '형' if len(f) == 1 else '복합형(' + '+'.join(f) + ')')


# ---------------------------------------------------------------- 메인
def main():
    ensure_dirs()
    T = march_hourly()
    long, Tsel = build_features(T)
    W = long.pivot_table(index=['station', 'y'], columns='변수', values='평균이용')[VARS]
    S = W.groupby(level=0).mean()
    ny = W.groupby(level=0).size()
    X = S.values
    k, kv = kmo(X)
    chi, df_, p = bartlett(X)
    print(f'KMO {k:.3f} (변수별 최소 {kv.min():.3f}) | Bartlett chi2={chi:.0f} df={df_:.0f} p={p:.3g}')
    r = solve(X)
    ref = r['Ar']
    print('두 점수 방식 최대 차이 %.1e' % np.abs(r['F_reg'] - r['F_ls']).max(), '| 고유값>1 개수', int((r['eig'] > 1).sum()))
    df = pd.DataFrame(r['F_reg'], index=S.index, columns=['score_commercial_leisure', 'score_residential', 'score_employment'])
    df['years_used'] = ny
    df['type_kr'] = [typ(s) for s in r['F_reg']]
    # 일평균 이용객(3월 선택일 승+하차)
    Tsel['d'] = Tsel[HLAB].sum(axis=1)
    vol = Tsel.groupby(['station', 'y', 'date']).d.sum().groupby(level=[0, 1]).mean().groupby(level=0).mean()
    df['avg_daily_ridership'] = vol.reindex(df.index).round(0)
    out = df.reset_index()
    out['status'] = np.where(out.type_kr == '미분류', '미분류(요인점수 모두 1 미만)', '분류')
    out = out[['station', 'score_commercial_leisure', 'score_residential', 'score_employment', 'type_kr', 'years_used', 'status', 'avg_daily_ridership']]
    # 3월 존재 연도 2개 미만인 역(전체 역 목록에서 찾음)은 분류 제외로 표시
    all_st = pd.read_csv(PROC / 'always_present_stations.csv').station.tolist()
    presence = pd.read_csv(PROC / 'station_presence_by_year.csv').station.tolist()
    excl = [s for s in presence if s not in set(out.station)]
    extra = pd.DataFrame([dict(station=s, score_commercial_leisure=np.nan, score_residential=np.nan, score_employment=np.nan,
                               type_kr='분류제외(3월 존재 연도 2개 미만)', years_used=int(0), status='분류제외', avg_daily_ridership=np.nan) for s in excl])
    out = pd.concat([out, extra])
    out[['score_commercial_leisure', 'score_residential', 'score_employment']] = out[['score_commercial_leisure', 'score_residential', 'score_employment']].round(3)
    out.sort_values(['type_kr', 'station']).to_csv(PROC / 'station_type.csv', index=False, encoding='utf-8-sig')
    g = out[out.status != '분류제외'].groupby('type_kr').station.agg(['count', lambda x: ', '.join(sorted(x))])
    g.columns = ['n_stations', 'stations']
    g.to_csv(PROC / 'station_type_lists.csv', encoding='utf-8-sig')
    print(df.type_kr.value_counts().to_dict(), '| 분류제외:', excl)

    # ---- 논문 표 2·3·4 비교
    eig = r['eig']
    pd.DataFrame({'component': range(1, 11), 'paper_eigenvalue': PAPER_EIG, 'paper_pct_variance': [round(e / 30 * 100, 3) for e in PAPER_EIG],
                  'paper_pct_cumulative': PAPER_CUM, 'our_eigenvalue': np.round(eig[:10], 3), 'our_pct_variance': np.round(eig[:10] / 30 * 100, 3),
                  'our_pct_cumulative': np.round(np.cumsum(eig[:10]) / 30 * 100, 3)}).to_csv(PROC / 'step4_table2_comparison.csv', index=False, encoding='utf-8-sig')
    Ar = ref
    comm = (r['A'] ** 2).sum(1)
    fmt = lambda d: ' / '.join(f'요인{f}:{x:.3f}' for f, x in sorted(d.items())) or '-'
    rows = []
    for i, v in enumerate(VARS):
        pp = PAPER.get(v, {})
        o = {f: Ar[i, f - 1] for f in (1, 2, 3) if abs(Ar[i, f - 1]) >= .5}
        rows.append(dict(variable=v, paper=fmt(pp), our_factor1=round(Ar[i, 0], 3), our_factor2=round(Ar[i, 1], 3), our_factor3=round(Ar[i, 2], 3), our_loading_ge_0_5=fmt(o),
                         match='O' if set(pp) == set(o) else 'X', communality=round(comm[i], 3),
                         max_loading_diff=round(max(abs(o.get(f, 0) - pp.get(f, 0)) for f in (1, 2, 3)), 3)))
    pd.DataFrame(rows).to_csv(PROC / 'step4_table3_comparison.csv', index=False, encoding='utf-8-sig')
    F = ['score_commercial_leisure', 'score_residential', 'score_employment']
    paper = {}
    for f, d in T4.items():
        for s, v in d.items():
            paper.setdefault(ALIAS.get(s, s), {})[f] = v
    rows = []
    for s, d in sorted(paper.items()):
        if s not in df.index:
            continue
        o = {j + 1 for j in range(3) if df.loc[s, F[j]] >= 1}
        pset = set(d)
        rows.append(dict(station=s, paper_factors=','.join(map(str, sorted(pset))), paper_scores=' / '.join(f'{f}:{d[f]}' for f in sorted(d)),
                         our_factors=','.join(map(str, sorted(o))) or '-', our_scores=' / '.join(f'{df.loc[s, F[j]]:.1f}' for j in range(3)),
                         n_common=len(pset & o), n_paper=len(pset), n_ours=len(o), verdict='일치' if pset == o else ('부분일치' if pset & o else '불일치')))
    pd.DataFrame(rows).to_csv(PROC / 'step4_table4_comparison.csv', index=False, encoding='utf-8-sig')

    # ---- 논문 그림 4 대비 변경 역
    NAME = {'E': 'Employment', 'R': 'Residential', 'C': 'Commercial & Leisure'}
    lab = lambda fs: 'Unclassified / not shown' if not fs else (' + '.join(NAME[k] for k in 'ECR' if k in fs) + (' (all three)' if len(fs) == 3 else ' only' if len(fs) == 1 else ''))
    Pf = {}
    for fs, l in FIG4.items():
        for s in l:
            Pf[ALIAS.get(s, s)] = fs
    m = {'고용형': 'E', '주거형': 'R', '상업·여가형': 'C', '복합형(상업·여가+고용)': 'CE', '복합형(상업·여가+주거)': 'CR', '복합형(주거+고용)': 'RE', '복합형(상업·여가+주거+고용)': 'CER'}
    Of = {r_.station: frozenset(m[r_.type_kr]) for r_ in out.itertuples() if r_.type_kr in m}
    rows = []
    for s in sorted(set(Pf) | set(Of)):
        pp, o = Pf.get(s, frozenset()), Of.get(s, frozenset())
        if pp == o:
            continue
        if s == '광명사거리':
            o_lab, ct, note = 'Out of scope (outside Seoul)', 'Dropped', 'Excluded from our analysis because the station is located outside Seoul (Gwangmyeong-si)'
        elif not pp:
            o_lab, ct, note = lab(o), 'Added (newly classified)', 'Not classified in the 2005 figure'
        elif not o:
            o_lab, ct, note = 'Unclassified', 'Dropped (now unclassified)', 'All three factor scores below 1 in our analysis'
        else:
            o_lab, ct, note = lab(o), 'Moved to a different region', ''
        gained = [NAME[k] for k in 'ECR' if k in o and k not in pp]
        lost = [NAME[k] for k in 'ECR' if k in pp and k not in o]
        if ct.startswith('Moved'):
            note = ('Gained: ' + ', '.join(gained) if gained else '') + ('; ' if gained and lost else '') + ('Lost: ' + ', '.join(lost) if lost else '')
        alias = [k for k, v in ALIAS.items() if v == s]
        if alias:
            note = (note + '; ' if note else '') + f'Name in the 2005 paper: {alias[0]} (now {s})'
        rows.append(dict(station=s, region_in_2005_figure4=lab(pp) if pp else 'Not shown (unclassified)', region_in_our_figure4=o_lab, change_type=ct,
                         functions_gained=', '.join(gained), functions_lost=', '.join(lost), note=note))
    C = pd.DataFrame(rows)
    order = {'Added (newly classified)': 0, 'Dropped (now unclassified)': 1, 'Dropped': 2, 'Moved to a different region': 3}
    C = C.sort_values(['change_type', 'station'], key=lambda s: s.map(order) if s.name == 'change_type' else s)
    C.to_csv(PROC / 'station_type_change_vs_2005.csv', index=False, encoding='utf-8-sig')

    # ---- 연도별(각 3월) 분류 검증
    Y, info = {}, []
    for y in CLASS_YEARS:
        Wy = W.xs(y, level=1)
        ry = solve(Wy.values, ref)
        Y[y] = pd.Series([typ(s) for s in ry['F_reg']], index=Wy.index)
        ev = ry['eig']
        info.append(dict(year=y, n_stations=len(Wy), eigenvalue_3=round(ev[2], 3), eigenvalue_4=round(ev[3], 3), cumulative_3_pct=round(ev[:3].sum() / 30 * 100, 1),
                         n_classified=int((Y[y] != '미분류').sum()), unclassified_rate_pct=round((Y[y] == '미분류').mean() * 100, 1),
                         agreement_with_pooled_pct=round((Y[y] == df.type_kr.reindex(Wy.index)).mean() * 100, 1),
                         loading_correlation_with_pooled=round(np.mean([np.corrcoef(ry['Ar'][:, j], ref[:, j])[0, 1] for j in range(3)]), 3)))
    pd.DataFrame(info).to_csv(PROC / 'yearly_agreement_summary.csv', index=False, encoding='utf-8-sig')
    M = pd.DataFrame(Y)
    M['pooled_type'] = df.type_kr
    pm = pd.DataFrame(index=CLASS_YEARS, columns=CLASS_YEARS, dtype=float)
    for a in CLASS_YEARS:
        for b in CLASS_YEARS:
            c = M[[a, b]].dropna() if a != b else M[[a]].dropna()
            pm.loc[a, b] = round((c[a] == c[b]).mean() * 100, 1) if a != b else 100.0
    pm.index.name = 'year'
    pm.to_csv(PROC / 'yearly_agreement_pairwise.csv', encoding='utf-8-sig')
    M['modal_type'] = M[CLASS_YEARS].apply(lambda r_: r_.dropna().value_counts().index[0], axis=1)
    M['modal_share'] = M[CLASS_YEARS].apply(lambda r_: r_.dropna().value_counts().iloc[0] / len(r_.dropna()), axis=1)
    M['n_years'] = M[CLASS_YEARS].notna().sum(axis=1)
    M['n_distinct_types'] = M[CLASS_YEARS].nunique(axis=1)
    M['n_years_matching_pooled'] = M.apply(lambda r_: int((r_[CLASS_YEARS].dropna() == r_['pooled_type']).sum()), axis=1)
    M.index.name = 'station'
    M[CLASS_YEARS + ['pooled_type', 'modal_type', 'modal_share', 'n_years', 'n_distinct_types', 'n_years_matching_pooled']].reset_index().to_csv(PROC / 'yearly_station_types.csv', index=False, encoding='utf-8-sig')
    print('연도별 일치율(%):', {i['year']: i['agreement_with_pooled_pct'] for i in info}, '| 유형이 바뀌는 역', int((M.n_distinct_types > 1).sum()))


if __name__ == '__main__':
    main()
