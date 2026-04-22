# %% [markdown]
# # 03 — Results Exploration
#
# **Faz:** F5/F8 (PLAN §4.4)
#
# Model karsilastirma, feature importance, hata analizi. Plotlar
# `output/figures/` altina kaydedilir.

# %% Hucre 1 — Imports
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from pipeline.config import FIGURES_DIR

logging.basicConfig(level=logging.INFO)

# %% Hucre 2 — Ablation sonuclarini oku
# TODO F5: output/ablation_results.csv otomatik bul

# %% Hucre 3 — Bar chart per model
# %% Hucre 4 — Feature importance
# %% Hucre 5 — Yanlis siniflandirma analizi
