# %% [markdown]
# # 03 - Results Exploration
#
# **Faz:** F5 / F8 (PLAN §4.4, §15 F5 DoD)
#
# Ablation sonuclarini inceler:
#
#   - Model x gorev bar chartlari (F1, PR-AUC, MCC)
#   - Feature seti heatmap
#   - En iyi modelin confusion matrix + PR curve'u
#   - Random Forest feature importance (en iyi modellerden biriyse)
#   - Yanlis siniflandirma analizi — hata paternleri
#
# Calistirmadan once:
#   `analysis/02_model_training.py` hucreleri -> `output/ablation_results_*.csv`

# %% Hucre 1 - Imports
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from pipeline.config import FIGURES_DIR, OUTPUT_DIR
from pipeline.model_utils import (
    classification_metrics,
    confusion_quadrants,
    extract_xy,
    fit_scaler,
    get_feature_set,
    pr_curve,
    project_based_split,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("analysis.03")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

# %% Hucre 2 - En guncel ablation sonuclarini oku
abl_files = sorted(OUTPUT_DIR.glob("ablation_results_*.csv"))
if not abl_files:
    raise FileNotFoundError(
        "ablation_results_*.csv yok. Once `analysis/02_model_training.py` calistirin."
    )
ABL_PATH = abl_files[-1]
print(f"Sonuc dosyasi: {ABL_PATH.name}")
results = pd.read_csv(ABL_PATH)
print(f"Kayit: {len(results)}")
ok = results[results["status"] == "ok"].copy()
print(f"'ok' statuslu: {len(ok)} / {len(results)}")

# %% Hucre 3 - Bar chart: model bazinda ortalama F1 / PR-AUC / MCC
if len(ok):
    agg = (
        ok.groupby("model")[["f1", "pr_auc", "mcc", "accuracy"]]
          .mean()
          .sort_values("f1", ascending=False)
    )
    print("Model bazli ortalamalar:")
    print(agg.round(3).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, metric, color in zip(
        axes,
        ("f1", "pr_auc", "mcc"),
        ("#2ecc71", "#3498db", "#9b59b6"),
    ):
        series = agg[metric].dropna().sort_values()
        series.plot(kind="barh", ax=ax, color=color, edgecolor="white")
        ax.set_title(f"Model bazli ortalama {metric.upper()}")
        ax.set_xlim(0, 1)
        for i, v in enumerate(series.values):
            ax.text(v + 0.01, i, f"{v:.2f}", va="center")
    plt.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = FIGURES_DIR / f"model_bars_{ts}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"kaydedildi: {out}")
else:
    print("'ok' satir yok, bar chart atlandi.")

# %% Hucre 4 - Task x FeatureSet heatmap (F1)
if len(ok):
    for task_name in sorted(ok["task"].unique()):
        sub = ok[ok["task"] == task_name]
        pivot = sub.pivot_table(
            index="model", columns="feature_set", values="f1", aggfunc="mean",
        )
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis",
                       vmin=0, vmax=1)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f"Task: {task_name} — F1 heatmap")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if v < 0.5 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        out = FIGURES_DIR / f"heatmap_{task_name}_{datetime.now():%Y%m%d_%H%M%S}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"kaydedildi: {out}")

# %% Hucre 5 - Her task icin en iyi model + label/feature kombinasyonu
if len(ok):
    best_per_task = (
        ok.sort_values("f1", ascending=False)
          .drop_duplicates("task")
          .reset_index(drop=True)
    )
    print("Her task icin en iyi konfigurasyon:")
    print(best_per_task[[
        "task", "label_variant", "feature_set", "split", "model",
        "f1", "pr_auc", "mcc", "accuracy",
    ]].to_string(index=False))
else:
    best_per_task = pd.DataFrame()
    print("'ok' satir yok, en iyi konfigurasyon cikarilamadi.")

# %% Hucre 6 - Confusion matrix + PR curve (en iyi modellerden birini elle yeniden egit)
# Hizli bir sanity check: en iyi task=commit model=rf satirini ornek al,
# dataseti yeniden yukleyip RF'yi egit, confusion matrix + PR curve'u ciz.
# Bu hucre sadece RF/LR/SVM gibi sklearn uyumlu modeller icin tasarlanmistir;
# digerleri icin agir bagimliliklari atlar.

filtered = sorted(OUTPUT_DIR.glob("dataset_model_filtered_*.parquet"))
if len(best_per_task) and filtered:
    df = pd.read_parquet(filtered[-1])
    row = best_per_task.iloc[0]
    print(f"Ornek yeniden egitim: {row['task']} / {row['model']} / {row['feature_set']}")

    task = str(row["task"])
    feat = str(row["feature_set"])
    label_col_map = {
        ("commit", "median"):  "label_commit",
        ("bug",    "keyword"): "bug_keyword",
        ("bug",    "szz"):     "bug_szz",
        ("smell",  "p80"):     "smell_binary",
    }
    label_col = label_col_map.get((task, str(row["label_variant"])), None)

    if label_col and label_col in df.columns and row["model"] in ("rf", "lr", "svm"):
        features = get_feature_set(task, feat)
        train, val, test = project_based_split(df)
        Xtr, ytr = extract_xy(train, features, label_col)
        Xte, yte = extract_xy(test,  features, label_col)
        scaler = fit_scaler(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(Xte)

        # Basitlik icin RF'yi kullan (LR/SVM icin ayni kalip)
        rf = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                    random_state=RANDOM_STATE)
        rf.fit(Xtr_s, ytr)
        y_pred  = rf.predict(Xte_s)
        y_proba = rf.predict_proba(Xte_s)[:, 1]

        cm = confusion_quadrants(yte, y_pred)
        metrics = classification_metrics(yte, y_pred, y_proba)
        print(f"  Test metrikleri: {metrics}")
        print(f"  Confusion: {cm}")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        # Confusion matrix
        ax = axes[0]
        mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        im = ax.imshow(mat, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                        color="black", fontsize=14, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["True 0", "True 1"])
        ax.set_title(f"Confusion ({row['task']}, {row['model']})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # PR curve
        ax2 = axes[1]
        prec, rec, _ = pr_curve(yte, y_proba)
        ax2.plot(rec, prec, color="#e67e22", linewidth=2)
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title(f"PR curve (AP={metrics['pr_auc']:.3f})")
        ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        out = FIGURES_DIR / f"confusion_pr_{row['task']}_{row['model']}_{datetime.now():%Y%m%d_%H%M%S}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"kaydedildi: {out}")
    else:
        print("Secilen en iyi model sklearn degil; bu hucre atlandi "
              "(AutoGluon/DL yeniden egitim pahali). Gerekirse elle duzenleyin.")
else:
    print("Yeniden egitim icin filtered parquet ya da best_per_task yok.")

# %% Hucre 7 - Random Forest feature importance (task=commit ornek)
if len(filtered):
    df = pd.read_parquet(filtered[-1])
    task = "commit"
    label_col = "label_commit"
    if label_col in df.columns:
        features = get_feature_set(task, "all")
        train, _, _ = project_based_split(df)
        Xtr, ytr = extract_xy(train, features, label_col)
        rf = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                    random_state=RANDOM_STATE)
        rf.fit(Xtr, ytr)
        fi = pd.Series(rf.feature_importances_, index=features).sort_values()

        fig, ax = plt.subplots(figsize=(7, 8))
        fi.tail(20).plot(kind="barh", ax=ax, color="#2ecc71", edgecolor="white")
        ax.set_title(f"RF feature importance (task={task}, top 20)")
        plt.tight_layout()
        out = FIGURES_DIR / f"feature_importance_{task}_{datetime.now():%Y%m%d_%H%M%S}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"kaydedildi: {out}")
    else:
        print(f"label sutunu yok: {label_col}")
else:
    print("filtered parquet yok, feature importance atlandi.")

# %% Hucre 8 - Yanlis siniflandirma analizi
# Test split'inde pozitif tahmin ama gercekte 0 olanlari (FP) veya tersi
# (FN) inceler — yanlis tahminli dosyalarin proje dagilimi ve metrik
# karakteristiklerini tablolar.
if len(filtered):
    df = pd.read_parquet(filtered[-1])
    if "label_commit" in df.columns:
        train, val, test = project_based_split(df)
        features = get_feature_set("commit", "all")
        Xtr, ytr = extract_xy(train, features, "label_commit")
        Xte, yte = extract_xy(test,  features, "label_commit")
        scaler = fit_scaler(Xtr)
        rf = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                    random_state=RANDOM_STATE)
        rf.fit(scaler.transform(Xtr), ytr)
        y_pred = rf.predict(scaler.transform(Xte))

        err = test.copy().reset_index(drop=True)
        err["y_true"] = yte
        err["y_pred"] = y_pred
        err["err_type"] = "tn"
        err.loc[(err.y_true == 0) & (err.y_pred == 1), "err_type"] = "fp"
        err.loc[(err.y_true == 1) & (err.y_pred == 0), "err_type"] = "fn"
        err.loc[(err.y_true == 1) & (err.y_pred == 1), "err_type"] = "tp"

        # Hata siniflari ozeti
        print("Hata sinifi dagilimi:")
        print(err["err_type"].value_counts().to_string())

        # Hata dosyalarinin LOC / commit_count ortalamalari
        summary = (
            err.groupby("err_type")[["loc", "commit_count", "cc_mean"]]
               .mean().round(2)
        )
        print("\nHata sinifi bazli ortalama metrikler:")
        print(summary.to_string())

        # Hata dosyalarini CSV'ye yaz
        out_csv = OUTPUT_DIR / f"misclassification_{datetime.now():%Y%m%d_%H%M%S}.csv"
        err[["project_name", "file_path", "y_true", "y_pred", "err_type",
             "loc", "commit_count"]].to_csv(out_csv, index=False)
        print(f"kaydedildi: {out_csv}")
    else:
        print("label_commit yok, hata analizi atlandi.")
