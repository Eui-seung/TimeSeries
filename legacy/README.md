# legacy/

노트북(`notebooks/`) 중심으로 정리하기 전에 쓰던 코드입니다. 기록용으로 남겨 두었고, 새 작업은 `notebooks/`에서 합니다.

- `config.py`: 설정 → `notebooks/00_settings.ipynb`로 옮겼습니다.
- `scripts/step1~9`: 전처리 → `notebooks/01_preprocessing.ipynb`로 옮겼습니다.
- `scripts/sarimax_asos.py`: SARIMAX 분석 → `notebooks/03_sarimax.ipynb`의 N/A/B/C 분석(stage 5)으로 옮겼습니다. 그 전 단계(stage 1~4)와 `results/sarimax_B_*.csv`, `results/sarimax_ABC_*.csv`는 이 스크립트로 만든 결과입니다.
- `scripts/step6_figures.py`, `scripts/eda_weekly_pattern.py`: 그림 코드. step6은 01번 노트북에 있고 `eda_weekly_pattern.py`는 노트북으로 옮기지 않았습니다.

저장소 루트에서 실행합니다: `python legacy/scripts/step1_load_raw.py` 등. (경로는 `legacy/` 위치 기준으로 고쳐 두었습니다.)
