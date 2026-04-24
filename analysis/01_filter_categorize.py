# %% [markdown]
# # 01 - Filter & Categorize
#
# **Faz:** F4 - Filter & threshold (PLAN §4.2, §15 F4 DoD)
#
# Amac: `scripts.collect --phase all` ile uretilen `output/dataset_full_*.parquet`
# dosyasini yukle, projeleri kategorilere ayir, sensitivity analizi yap
# (filtresiz vs min=10/max=100 vs min=25/max=80), dinamik smell esiklerini
# uygula ve `output/dataset_model_filtered_<ts>.parquet` olarak yaz.
#
# Kullanim: VS Code + Python/Jupyter extension ile hucre hucre calistirin.
# Plotlar inline gorunur, ayrica `output/figures/` altina kaydedilir.
#
# Calistirmadan once:
#
#   python -m scripts.collect --phase all    (veya asamali discovery/process/build)
#
# Bagimlilik: pandas, matplotlib. Keyword kategorizasyonu ek modul
# gerekmez; topics/description opsiyoneldir (yoksa proje_adi yeterli).

# %% Hucre 1 - Imports + ortam kurulumu
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from pipeline.categories import CATEGORY_KEYWORDS, OTHER_CATEGORY
from pipeline.config import (
    CHECKPOINT_DIR,
    FIGURES_DIR,
    OUTPUT_DIR,
    SMELL_BINARY_PERCENTILE,
)
from pipeline.dataset_builder import (
    add_commit_label,
    add_dynamic_smell_binary,
    add_project_categories,
    apply_commit_filter,
    load_project_parquets,
    sensitivity_summary,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("analysis.01")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# %% Hucre 2 - Dataset_full'i yukle (en guncel)
full_files = sorted(OUTPUT_DIR.glob("dataset_full_*.parquet"))
if full_files:
    DATA_PATH = full_files[-1]
    print(f"dataset_full bulundu: {DATA_PATH.name}")
    df_full = pd.read_parquet(DATA_PATH)
else:
    # Fallback: henuz `collect build` calismamissa per-project parquet'lerden
    # canli birlestir — ayni DataFrame'i verir.
    print("dataset_full_*.parquet yok. Per-project parquet'lerden birlestiriyorum.")
    df_full = load_project_parquets()
    DATA_PATH = None

print(f"Toplam satir : {len(df_full):,}")
if "project_name" in df_full.columns:
    print(f"Proje sayisi : {df_full['project_name'].nunique()}")

# %% Hucre 3 - Opsiyonel discovery metadatasi
# discovery.json icinde topics/description ya da description varsa
# kategorilendirme isabetini artirir. V2 discovery bu alanlari henuz
# kaydetmiyor olabilir; yoksa proje adi tek basina kullanilir.
disc_path = CHECKPOINT_DIR / "discovery.json"
project_meta: dict[str, dict] = {}
if disc_path.exists():
    try:
        disc_payload = json.loads(disc_path.read_text(encoding="utf-8"))
        for item in disc_payload.get("found", []):
            name = item.get("full_name")
            if not name:
                continue
            project_meta[name] = {
                "topics":      list(item.get("topics", []) or []),
                "description": item.get("description", "") or "",
            }
        print(f"discovery.json: {len(project_meta)} proje meta yuklendi")
    except (OSError, json.JSONDecodeError) as exc:
        print(f"discovery.json okunamadi: {exc}. Yalnizca project_name ile kategori atanacak.")
else:
    print("discovery.json yok. Yalnizca project_name ile kategori atanacak.")

# %% Hucre 4 - Kategori atama (project_name + opsiyonel meta)
df = df_full.copy()
df = add_project_categories(df, project_meta=project_meta)

# Proje bazinda kategori dagilimi (dosya degil)
if "project_name" in df.columns:
    proj_cats = df.drop_duplicates("project_name")[["project_name", "category_primary", "categories_all"]]
    primary_counts = proj_cats["category_primary"].value_counts()
else:
    proj_cats = pd.DataFrame(columns=["project_name", "category_primary", "categories_all"])
    primary_counts = pd.Series(dtype="int64")

print("Birincil kategori (proje bazinda):")
for cat, n in primary_counts.items():
    pct = (n / len(proj_cats) * 100.0) if len(proj_cats) else 0.0
    print(f"  {cat:15s} : {n:4d} proje ({pct:5.1f}%)")

other_pct = (primary_counts.get(OTHER_CATEGORY, 0) / len(proj_cats) * 100.0) if len(proj_cats) else 0.0
if other_pct > 40:
    print(
        f"UYARI: '{OTHER_CATEGORY}' orani %{other_pct:.1f}. discovery asamasinda "
        "topics/description kaydi yoksa normal; ek enrichment dusunulebilir."
    )

# %% Hucre 5 - Kategori dagilimi grafigi
if len(primary_counts):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    primary_counts.sort_values().plot(
        kind="barh", ax=ax, color="steelblue", edgecolor="white",
    )
    for i, v in enumerate(primary_counts.sort_values().values):
        ax.text(v + max(primary_counts) * 0.01, i, str(v), va="center")
    ax.set_title("Birincil Kategori - Proje Sayisi", fontweight="bold")
    ax.set_xlabel("Proje Sayisi")
    plt.tight_layout()
    out_fig = FIGURES_DIR / "sensitivity_category_distribution.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"kaydedildi: {out_fig}")
else:
    print("Kategori dagilimi bos (muhtemel: df_full bos). Plot atlaniyor.")

# %% Hucre 6 - Etiket kolonlarini ekle (label_commit + smell_binary)
df = add_dynamic_smell_binary(df, percentile=SMELL_BINARY_PERCENTILE)
df = add_commit_label(df)

if "label_commit" in df.columns and len(df):
    print("Sinif dagilimi (label_commit):")
    for v, n in df["label_commit"].value_counts().sort_index().items():
        print(f"  {v}: {n:,} ({n/len(df):.1%})")

if "smell_binary" in df.columns and len(df):
    print("smell_binary dagilimi:")
    for v, n in df["smell_binary"].value_counts().sort_index().items():
        print(f"  {v}: {n:,} ({n/len(df):.1%})")

# %% Hucre 7 - Sensitivity analysis (filtresiz vs 10/100 vs 25/80)
# PLAN §4.2: "Agresif 25/80 filtresi kaldirilir. Filtresiz vs min=10/max=100
# vs min=25/max=80 karsilastirilir; dramatik fark yoksa filtresiz tercih
# edilir."
SENS_FILTERS = [
    (None, None),
    (10, 100),
    (25, 80),
]
summary = sensitivity_summary(df, filters=SENS_FILTERS)
summary["label"] = summary.apply(
    lambda r: "filtresiz" if pd.isna(r["min_commits"]) and pd.isna(r["max_commits"])
              else f"{int(r['min_commits'])}-{int(r['max_commits'])}",
    axis=1,
)
print("Sensitivity ozet:")
print(summary.to_string(index=False))

# CSV olarak da yaz (makaleye ek tablo icin)
summary_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
summary_csv = FIGURES_DIR / f"sensitivity_summary_{summary_ts}.csv"
summary.to_csv(summary_csv, index=False)
print(f"kaydedildi: {summary_csv}")

# %% Hucre 8 - Sensitivity plot (satir sayisi + pozitif sinif orani)
if len(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Sol: satir sayisi
    axes[0].bar(summary["label"], summary["files"],
                color=["#2ecc71", "#3498db", "#e67e22"], edgecolor="white")
    for i, v in enumerate(summary["files"]):
        axes[0].text(i, v + max(summary["files"]) * 0.01,
                     f"{v:,}", ha="center", fontweight="bold")
    axes[0].set_title("Filtreye gore dosya sayisi", fontweight="bold")
    axes[0].set_ylabel("Dosya")

    # Sag: pozitif sinif orani (label_commit + smell_binary karsilastirmali)
    x = range(len(summary))
    width = 0.35
    axes[1].bar([i - width/2 for i in x], summary["pct_label_pos"],
                width=width, label="label_commit=1 (%)", color="#9b59b6", edgecolor="white")
    axes[1].bar([i + width/2 for i in x], summary["pct_smell_pos"],
                width=width, label="smell_binary=1 (%)", color="#e67e22", edgecolor="white")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(summary["label"])
    axes[1].set_title("Pozitif sinif orani (%)", fontweight="bold")
    axes[1].set_ylabel("%")
    axes[1].legend()

    plt.suptitle("F4 Sensitivity Analizi", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_fig = FIGURES_DIR / f"sensitivity_commit_filters_{summary_ts}.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"kaydedildi: {out_fig}")
else:
    print("df bos, sensitivity plot atlaniyor.")

# %% Hucre 9 - Tercih edilen filtre + filtered dataset
# PLAN §4.2 tercihi: dramatik fark yoksa filtresiz.
# Bu hucrede kullanici KARAR verir — default "filtresiz" olarak yaziyoruz,
# gerekirse (10, 100) veya (25, 80)'a cevirin.
CHOSEN_MIN: int | None = None      # kullanici degistirir
CHOSEN_MAX: int | None = None      # kullanici degistirir

df_filtered = apply_commit_filter(df, CHOSEN_MIN, CHOSEN_MAX)
print(f"Secilen filtre: min={CHOSEN_MIN}, max={CHOSEN_MAX}")
print(f"Filtre sonrasi : {len(df_filtered):,} satir, "
      f"{df_filtered['project_name'].nunique() if 'project_name' in df_filtered else 0} proje")

# %% Hucre 10 - Filtered parquet yaz
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = OUTPUT_DIR / f"dataset_model_filtered_{ts}.parquet"
df_filtered.to_parquet(out_path, index=False)
print(f"kaydedildi: {out_path}")
print(f"  {len(df_filtered):,} satir, {len(df_filtered.columns)} sutun")

# %% Hucre 11 (opsiyonel) - Kategori sayilari ozeti (referans icin)
# CATEGORY_KEYWORDS konfigurasyonu hizli kontrol: hangi keyword seti
# kac projeyi hangi kategoriye sokmus, manuel inceleme icin kisa bir tablo.
print("Kategori keyword setleri:")
for cat, kws in CATEGORY_KEYWORDS.items():
    print(f"  {cat:15s}: {len(kws)} keyword")
print("\nOrnek proje atama (ilk 10):")
if "project_name" in df_filtered.columns:
    sample = df_filtered.drop_duplicates("project_name").head(10)
    for _, row in sample.iterrows():
        print(f"  {row['project_name']:35s} -> {row['category_primary']}  "
              f"(tum: {row['categories_all']})")
