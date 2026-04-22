# %% [markdown]
# # 02 — Model Training (Ablation Matrix)
#
# **Faz:** F5 — Model training (PLAN §4.3)
#
# Ablation boyutlari:
#   - Gorev: T1 commit, T2 bug (keyword / szz), T3 smell
#   - Feature seti: Static / +Derived / +Process / All
#   - Model: LR, RF, SVM, XGB, LGBM, AutoGluon, MLP, CNN, LSTM
#   - Split: project-based 70/15/15 (primary), time-based (secondary)
#   - Stacking: tekil / RF+AG meta LR

# %% Hucre 1 — Imports
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from pipeline.config import FEATURES_BUG, FEATURES_COMMIT, FEATURES_SMELL
from pipeline.model_utils import classification_metrics, project_based_split

logging.basicConfig(level=logging.INFO)

# %% Hucre 2 — Filtered dataset yukle
# TODO F5: son dataset_model_filtered_*.parquet otomatik bul
# df = pd.read_parquet(...)

# %% Hucre 3 — T1 Commit baseline (F5'te aktiflesir)
# train, val, test = project_based_split(df)
# ...

# %% Hucre 4 — T2 Bug ablation
# %% Hucre 5 — T3 Smell ablation
# %% Hucre 6 — Sonuclari CSV'ye yaz
