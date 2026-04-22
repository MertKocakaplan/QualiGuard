# %% [markdown]
# # 01 — Filter & Categorize
#
# **Faz:** F4 — Filter & threshold (PLAN §4.2)
#
# Interactive analiz. VS Code Jupyter extension ile `# %%` hucreleri
# hucre hucre calistirilir. Plotlar inline gorunur + `output/figures/`
# altina PNG olarak kaydedilir.
#
# Calistirmadan once: `python -m scripts.collect --phase all` ile
# `output/dataset_full_*.parquet` uretilmis olmali.

# %% Hucre 1 — Imports + proje koku
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from pipeline.config import FIGURES_DIR, OUTPUT_DIR, SMELL_BINARY_PERCENTILE
from pipeline.dataset_builder import (
    add_commit_label,
    add_dynamic_smell_binary,
    load_project_parquets,
)

logging.basicConfig(level=logging.INFO)

# %% Hucre 2 — Veri yukleme
df = load_project_parquets()
print(f"Toplam satir: {len(df):,}")
print(f"Proje sayisi: {df['project_name'].nunique() if 'project_name' in df else 0}")

# %% Hucre 3 — Etiket uretimi (F4'te sensitivity analysis burada)
# TODO F4: Agresif 25/80 filtresi vs filtresiz karsilastirmasi
df = add_dynamic_smell_binary(df, percentile=SMELL_BINARY_PERCENTILE)
df = add_commit_label(df)
print(df[["label_commit", "smell_binary"]].describe())

# %% Hucre 4 — Filtered parquet yaz
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out = OUTPUT_DIR / f"dataset_model_filtered_{ts}.parquet"
# df.to_parquet(out, index=False)  # F4'te aktiflesir
print(f"(F4) yazilacak: {out}")
