"""특보가 없을 때 기준선(baseline)용 데이터 만들기.

- 입력: data/processed/model_input_9stations.csv (step9 결과)
- 특보일(any_alert=1)과 그 뒤 14일을 alert_window=1로 표시하고 boarding/alighting/total을 NaN으로 가림 (날짜는 유지)
- any_alert가 비어 있는 2014-01-01~07(alert_unknown)은 특보 없음으로 보고 창을 만들지 않음 (그 7일은 MASK_COLS로 따로 가려짐)
- 출력: data/processed/model_input_9stations_noalert.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'data' / 'processed' / 'model_input_9stations.csv'
OUT = ROOT / 'data' / 'processed' / 'model_input_9stations_noalert.csv'
AFTER_DAYS = 14
Y_COLS = ['boarding', 'alighting', 'total']


def alert_window(alert: pd.Series, after: int = AFTER_DAYS) -> pd.Series:
    """특보일부터 after일 뒤까지 1. alert는 날짜 순으로 빈틈 없이 정렬돼 있어야 함."""
    a = alert.fillna(0).astype(int)
    return (a.rolling(after + 1, min_periods=1).max() > 0).astype(int)


def main():
    d = pd.read_csv(SRC, encoding='utf-8-sig', parse_dates=['date'])
    d = d.sort_values(['station', 'date']).reset_index(drop=True)

    # 역마다 날짜 빈틈이 없는지 확인 (rolling이 '일' 단위가 되도록)
    gaps = d.groupby('station')['date'].apply(lambda x: x.diff().dropna().dt.days.ne(1).sum())
    assert gaps.sum() == 0, f'날짜 빈틈 있음: {gaps[gaps > 0]}'

    d['alert_window'] = d.groupby('station', group_keys=False)['any_alert'].apply(alert_window)
    d.loc[d['alert_window'] == 1, Y_COLS] = np.nan

    # 정수 열이 float로 바뀌지 않게 특보 열은 Int64로 유지
    for c in ['alert_heat', 'alert_cold', 'alert_rain', 'alert_snow', 'any_alert']:
        d[c] = d[c].astype('Int64')

    d['date'] = d['date'].dt.strftime('%Y-%m-%d')
    d.to_csv(OUT, index=False, encoding='utf-8-sig')

    one = d[d['station'] == d['station'].iloc[0]]
    print(f'저장: {OUT}  ({len(d):,}행)')
    print(f'역당 가린 날: {one.alert_window.sum():,} / {len(one):,}일 ({one.alert_window.mean():.1%})')
    print(one.groupby('split')['alert_window'].agg(['sum', 'count']).assign(ratio=lambda x: (x['sum'] / x['count']).round(3)))


if __name__ == '__main__':
    main()
