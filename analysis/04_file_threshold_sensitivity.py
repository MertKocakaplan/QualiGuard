# %% [markdown]
# # 04 - File-count Threshold Sensitivity Analysis (V2.1 — Task 1)
#
# Proje basina minimum/maksimum dosya sayisi esiklerinin model performansina
# etkisini sistematik analiz eder. V1 port + V2.1 uyarlamalari:
#   - Kolon: "project" → "project_name"
#   - Hedef: "label"/"has_bug" → "bug_keyword"/"smell_binary"
#   - Feature: FEATURES_STATIC → FEATURES_BUG / FEATURES_SMELL (lean)
#   - Grid: 12×13=156 kombinasyon × 2 hedef (V1 ile birebir)
#
# **Metrik (V2.1 kritik duzeltme): MCC birincil, F1 referans.**
#   Agresif dosya cap'leme pozitif sinif oranini degistirir (ornek: bug %11.3→%13.9
#   max=20'de). F1 taban-orana DUYARLI oldugundan bu, gercek bir iyilesme olmadan
#   F1'i sisirir (confound). MCC taban-orana dayaniklidir; gercek etkiyi gosterir.
#   Heatmap her iki metrigi de cizer → confound gorsel olarak kanitlanir.
#
# **Karar politikasi:** Optimal (min,max) MCC ile secilir. Dataset KIRPILMAZ
#   (1000 proje korunur) — bu script analiz/figur uretir. Cap uygulamak istenirse
#   `--write-parquet` ile MCC-optimal esikte filtrelenmis parquet yazilir.
#
# Kullanim:
#   python analysis/04_file_threshold_sensitivity.py            # full grid (156×2), analiz-only
#   python analysis/04_file_threshold_sensitivity.py --quick    # pruned (49×2)
#   python analysis/04_file_threshold_sensitivity.py --write-parquet  # cap uygula + parquet yaz
#
# Ciktilar:
#   output/figures/file_threshold_sensitivity_<ts>.png  (2×2: MCC + F1)
#   output/figures/file_threshold_sensitivity_<ts>.csv  (mcc + f1, bug + smell)
#   (--write-parquet ile) output/dataset_model_filtered_filesens_<ts>.parquet

# %% Hucre 1 - Imports + CLI
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# Standalone calistirma icin proje kokunu path'e ekle (python analysis/04_*.py)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# pipeline imports — proje kokunden calistirilmali
from pipeline.config import FEATURES_BUG, FEATURES_SMELL, FIGURES_DIR, OUTPUT_DIR
from pipeline.dataset_builder import apply_file_threshold

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("analysis.04")

RANDOM_STATE = 42
N_CV_SPLITS = 5

# %% Hucre 2 - Grid tanimlari (V1 ile birebir uyum)
# Full grid: 12 × 13 = 156 kombinasyon × 2 hedef = ~312 RF fit
FULL_MIN_VALUES: list[Optional[int]] = [1, 3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50]
FULL_MAX_VALUES: list[Optional[int]] = [None, 20, 30, 40, 50, 60, 75, 80, 100, 120, 150, 200, 300]

# Quick grid: 7 × 7 = 49 kombinasyon (--quick flag)
QUICK_MIN_VALUES: list[Optional[int]] = [1, 5, 10, 15, 20, 30, 50]
QUICK_MAX_VALUES: list[Optional[int]] = [None, 20, 40, 60, 80, 100, 200]


# %% Hucre 3 - Tek-kombinasyon degerlendirici (MCC + F1)
def evaluate_threshold(
    df: pd.DataFrame,
    features: tuple[str, ...],
    target_col: str,
    min_files: Optional[int],
    max_files: Optional[int],
) -> tuple[Optional[float], Optional[float], int, int]:
    """
    Verilen esiklerle filtreleyip GroupKFold ile MCC + binary F1 hesapla.

    Returns:
        (mcc_mean, f1_mean, n_projects, n_files) — metrikler None ise yetersiz veri.
    """
    temp = apply_file_threshold(df, min_files=min_files, max_files=max_files, seed=RANDOM_STATE)

    n_projects = int(temp["project_name"].nunique()) if "project_name" in temp.columns else 0
    n_files = len(temp)

    # V1 koruma: yetersiz veri
    if n_projects < 10 or n_files < 100:
        return None, None, n_projects, n_files

    if target_col not in temp.columns:
        return None, None, n_projects, n_files

    temp = temp.dropna(subset=[target_col])
    if temp[target_col].nunique() < 2:
        return None, None, n_projects, n_files

    avail_feats = [f for f in features if f in temp.columns]
    if not avail_feats:
        return None, None, n_projects, n_files

    X = temp[avail_feats].fillna(0.0).to_numpy(dtype="float64")
    y = temp[target_col].to_numpy(dtype="int64")
    groups = temp["project_name"].to_numpy()

    n_splits = min(N_CV_SPLITS, n_projects // 2)
    if n_splits < 2:
        return None, None, n_projects, n_files

    gkf = GroupKFold(n_splits=n_splits)
    mcc_scores: list[float] = []
    f1_scores: list[float] = []

    for tr_idx, te_idx in gkf.split(X, y, groups):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        if y_tr.sum() == 0 or y_te.sum() == 0:
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        model = RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        f1_scores.append(float(f1_score(y_te, y_pred, zero_division=0)))
        mcc_scores.append(float(matthews_corrcoef(y_te, y_pred)))

    if not f1_scores:
        return None, None, n_projects, n_files

    return float(np.mean(mcc_scores)), float(np.mean(f1_scores)), n_projects, n_files


# %% Hucre 4 - Grid kosumu
def run_grid(
    df: pd.DataFrame,
    bug_col: str,
    min_values: list[Optional[int]],
    max_values: list[Optional[int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tum (min, max) kombinasyonlari icin bug + smell MCC + F1 hesapla.

    Returns:
        (results_bug_df, results_smell_df) — her birinde 'mcc' + 'f1' sutunlari.
    """
    total = len(min_values) * len(max_values)
    logger.info(
        "Grid basliyor: %d min × %d max = %d kombinasyon × 2 hedef (metrik: MCC + F1)",
        len(min_values), len(max_values), total,
    )

    results_bug: list[dict] = []
    results_smell: list[dict] = []
    done = 0

    for min_f in min_values:
        for max_f in max_values:
            done += 1
            max_label = str(max_f) if max_f is not None else "None"

            # V1 koruma: gecersiz kombinasyon
            if max_f is not None and min_f is not None and min_f >= max_f:
                skip_b = {"min_files": min_f, "max_files": max_f, "max_label": max_label,
                          "mcc": float("nan"), "f1": float("nan"), "n_projects": 0, "n_files": 0}
                results_bug.append(dict(skip_b))
                results_smell.append(dict(skip_b))
                logger.debug("[%d/%d] min=%s max=%s ATLANDI (min>=max)", done, total, min_f, max_label)
                continue

            mcc_b, f1_b, np_b, nf_b = evaluate_threshold(df, FEATURES_BUG,   bug_col,        min_f, max_f)
            mcc_s, f1_s, np_s, nf_s = evaluate_threshold(df, FEATURES_SMELL, "smell_binary", min_f, max_f)

            results_bug.append({
                "min_files": min_f, "max_files": max_f, "max_label": max_label,
                "mcc": mcc_b if mcc_b is not None else float("nan"),
                "f1":  f1_b  if f1_b  is not None else float("nan"),
                "n_projects": np_b, "n_files": nf_b,
            })
            results_smell.append({
                "min_files": min_f, "max_files": max_f, "max_label": max_label,
                "mcc": mcc_s if mcc_s is not None else float("nan"),
                "f1":  f1_s  if f1_s  is not None else float("nan"),
                "n_projects": np_s, "n_files": nf_s,
            })

            sb = f"mcc={mcc_b:.3f}/f1={f1_b:.3f}" if mcc_b is not None else "YETERSIZ"
            ss = f"mcc={mcc_s:.3f}/f1={f1_s:.3f}" if mcc_s is not None else "YETERSIZ"
            logger.info(
                "[%d/%d] min=%s max=%s -> bug[%s] smell[%s] | %d proj / %d files",
                done, total, min_f, max_label, sb, ss, np_b, nf_b,
            )

    return pd.DataFrame(results_bug), pd.DataFrame(results_smell)


# %% Hucre 5 - Heatmap cizimi (2×2: MCC ust, F1 alt)
def _make_pivot(results_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """results_df -> pivot (index=min_files, columns=max_label, values=metric)."""
    pivot = results_df.pivot(index="min_files", columns="max_label", values=metric)
    ordered_cols = []
    if "None" in pivot.columns:
        ordered_cols.append("None")
    numeric_cols = sorted((c for c in pivot.columns if c != "None"), key=lambda x: int(x))
    ordered_cols.extend(numeric_cols)
    return pivot[ordered_cols]


def _draw_panel(ax, pivot: pd.DataFrame, title: str) -> None:
    data = pivot.to_numpy(dtype=float)
    if np.all(np.isnan(data)):
        ax.set_title(f"{title}\n(veri yok)"); ax.axis("off"); return
    vmin = float(np.nanmin(data))
    vmax = float(np.nanmax(data))
    span = max(vmax - vmin, 1e-6)
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=6,
                        color="black" if (val - vmin) / span < 0.5 else "white")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(v) for v in pivot.index], fontsize=7)
    ax.set_xlabel("Max files / project", fontsize=9)
    ax.set_ylabel("Min files / project", fontsize=9)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8)


def plot_heatmaps(
    results_bug: pd.DataFrame,
    results_smell: pd.DataFrame,
    out_path: Path,
) -> None:
    """2×2 heatmap: ust sira MCC (birincil), alt sira F1 (referans/confound)."""
    fig, axes = plt.subplots(2, 2, figsize=(22, 15))
    _draw_panel(axes[0, 0], _make_pivot(results_bug,   "mcc"),
                "Bug — MCC (taban-orana dayanikli, BIRINCIL)")
    _draw_panel(axes[0, 1], _make_pivot(results_smell, "mcc"),
                "Smell — MCC (BIRINCIL)")
    _draw_panel(axes[1, 0], _make_pivot(results_bug,   "f1"),
                "Bug — F1 (taban-orana DUYARLI; cap'le siser = confound)")
    _draw_panel(axes[1, 1], _make_pivot(results_smell, "f1"),
                "Smell — F1 (referans)")
    plt.suptitle(
        "Dosya Esigi Sensitivity — MCC (gercek etki) vs F1 (confound)\n"
        "RF n=100, GroupKFold=5",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Heatmap kaydedildi (2×2 MCC+F1): %s", out_path)


# %% Hucre 6 - Optimal bulma + Pareto (MCC birincil)
def find_optimal(results_df: pd.DataFrame, metric: str = "mcc") -> Optional[dict]:
    """En yuksek `metric`'li gecerli (min, max) satiri dondur (mcc+f1 birlikte)."""
    valid = results_df.dropna(subset=[metric])
    if valid.empty:
        return None
    row = valid.loc[valid[metric].idxmax()]
    return {
        "min_files":  int(row["min_files"]) if row["min_files"] is not None else None,
        "max_files":  int(row["max_files"]) if row["max_files"] is not None and not pd.isna(row["max_files"]) else None,
        "mcc":        float(row["mcc"]),
        "f1":         float(row["f1"]),
        "n_projects": int(row["n_projects"]),
        "n_files":    int(row["n_files"]),
    }


def find_pareto(
    results_bug: pd.DataFrame,
    results_smell: pd.DataFrame,
    metric: str = "mcc",
) -> Optional[dict]:
    """Joint Pareto: bug + smell `metric` ortalamasini maximize eden (min, max)."""
    merged = results_bug[["min_files", "max_files", "mcc", "f1"]].merge(
        results_smell[["min_files", "max_files", "mcc", "f1"]],
        on=["min_files", "max_files"], suffixes=("_bug", "_smell"),
    )
    merged = merged.dropna(subset=[f"{metric}_bug", f"{metric}_smell"])
    if merged.empty:
        return None
    merged["joint"] = (merged[f"{metric}_bug"] + merged[f"{metric}_smell"]) / 2.0
    row = merged.loc[merged["joint"].idxmax()]
    return {
        "min_files":  int(row["min_files"]) if row["min_files"] is not None else None,
        "max_files":  int(row["max_files"]) if row["max_files"] is not None and not pd.isna(row["max_files"]) else None,
        "mcc_bug":    float(row["mcc_bug"]),   "mcc_smell": float(row["mcc_smell"]),
        "f1_bug":     float(row["f1_bug"]),    "f1_smell":  float(row["f1_smell"]),
        "joint":      float(row["joint"]),
    }


# %% Hucre 7 - main
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="File-count threshold sensitivity (V2.1 Task 1) — MCC birincil"
    )
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Filtered parquet (default: en guncel dataset_model_filtered_*)")
    parser.add_argument("--quick", action="store_true",
                        help="Pruned 7×7=49 grid (hizli)")
    parser.add_argument("--bug-label", choices=("keyword", "szz"), default="keyword",
                        help="Bug hedef etiketi")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--write-parquet", action="store_true",
                        help="MCC-optimal esikte filtrelenmis parquet YAZ (default: yazma — analiz-only)")
    args = parser.parse_args(argv)

    # Veri yukle
    if args.dataset:
        data_path = args.dataset
    else:
        files = sorted(args.output_dir.glob("dataset_model_filtered_*.parquet"))
        if not files:
            logger.error("dataset_model_filtered_*.parquet bulunamadi.")
            return 1
        data_path = files[-1]

    logger.info("Veri: %s", data_path)
    df = pd.read_parquet(data_path)
    logger.info("Kayit: %d | Proje: %d", len(df), df["project_name"].nunique() if "project_name" in df else 0)

    bug_col = f"bug_{args.bug_label}"
    if bug_col not in df.columns:
        alt = "bug_szz" if args.bug_label == "keyword" else "bug_keyword"
        if alt in df.columns:
            logger.warning("%s yok, %s kullaniliyor", bug_col, alt); bug_col = alt
        else:
            logger.error("Ne %s ne de %s mevcut.", bug_col, alt); return 1
    if "smell_binary" not in df.columns:
        logger.error("smell_binary sutunu bulunamadi."); return 1

    # Grid secimi
    if args.quick:
        min_values, max_values = QUICK_MIN_VALUES, QUICK_MAX_VALUES
    else:
        min_values, max_values = FULL_MIN_VALUES, FULL_MAX_VALUES
    logger.info("Grid: %d×%d=%d kombinasyon", len(min_values), len(max_values),
                len(min_values) * len(max_values))

    results_bug, results_smell = run_grid(df, bug_col, min_values, max_values)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # CSV (mcc + f1, bug + smell)
    combined = results_bug[["min_files", "max_files", "max_label", "n_projects", "n_files"]].copy()
    combined["mcc_bug"]   = results_bug["mcc"].values
    combined["f1_bug"]    = results_bug["f1"].values
    combined["mcc_smell"] = results_smell["mcc"].values
    combined["f1_smell"]  = results_smell["f1"].values
    csv_path = FIGURES_DIR / f"file_threshold_sensitivity_{ts}.csv"
    combined.to_csv(csv_path, index=False)
    logger.info("CSV kaydedildi: %s", csv_path)

    # Heatmap (2×2)
    png_path = FIGURES_DIR / f"file_threshold_sensitivity_{ts}.png"
    plot_heatmaps(results_bug, results_smell, png_path)

    # Optimal/Pareto — MCC birincil; F1 referans/contrast
    mcc_bug   = find_optimal(results_bug,   "mcc")
    mcc_smell = find_optimal(results_smell, "mcc")
    mcc_pareto = find_pareto(results_bug, results_smell, "mcc")
    f1_bug    = find_optimal(results_bug,   "f1")
    f1_smell  = find_optimal(results_smell, "f1")

    print("\n" + "=" * 72)
    print("SENSITIVITY ANALIZ SONUCLARI — MCC birincil (taban-orana dayanikli)")
    print("=" * 72)
    if mcc_bug:
        print(f"\n[MCC] Bug   en iyi: min={mcc_bug['min_files']}, max={mcc_bug['max_files']} "
              f"→ MCC={mcc_bug['mcc']:.4f} (F1={mcc_bug['f1']:.4f}) "
              f"| {mcc_bug['n_projects']} proje, {mcc_bug['n_files']} dosya")
    if mcc_smell:
        print(f"[MCC] Smell en iyi: min={mcc_smell['min_files']}, max={mcc_smell['max_files']} "
              f"→ MCC={mcc_smell['mcc']:.4f} (F1={mcc_smell['f1']:.4f}) "
              f"| {mcc_smell['n_projects']} proje, {mcc_smell['n_files']} dosya")
    if mcc_pareto:
        print(f"[MCC] Joint Pareto: min={mcc_pareto['min_files']}, max={mcc_pareto['max_files']} "
              f"→ MCC_bug={mcc_pareto['mcc_bug']:.4f}, MCC_smell={mcc_pareto['mcc_smell']:.4f}")

    print("\n--- F1 (REFERANS — confound gosterimi) ---")
    if f1_bug:
        print(f"[F1]  Bug   F1-max: min={f1_bug['min_files']}, max={f1_bug['max_files']} "
              f"→ F1={f1_bug['f1']:.4f} (ayni noktada MCC={f1_bug['mcc']:.4f})")
    if f1_smell:
        print(f"[F1]  Smell F1-max: min={f1_smell['min_files']}, max={f1_smell['max_files']} "
              f"→ F1={f1_smell['f1']:.4f} (ayni noktada MCC={f1_smell['mcc']:.4f})")
    print("\nNOT: F1 kucuk cap'lerde siser (sinif dengesi kayar). MCC ~duzse → "
          "cap GERCEK iyilesme saglamiyor, kirpma gereksiz (1000 proje korunur).")

    # Parquet — yalnizca --write-parquet ile
    if args.write_parquet:
        best = mcc_pareto or mcc_bug or mcc_smell
        if best is None:
            logger.warning("Optimal bulunamadi; parquet yazilmadi.")
        else:
            df_filtered = apply_file_threshold(
                df, min_files=best.get("min_files"), max_files=best.get("max_files"),
                seed=RANDOM_STATE,
            )
            parquet_path = args.output_dir / f"dataset_model_filtered_filesens_{ts}.parquet"
            df_filtered.to_parquet(parquet_path, index=False)
            logger.info("Filtrelenmis parquet yazildi: %s (%d satir, %d proje)",
                        parquet_path.name, len(df_filtered),
                        df_filtered["project_name"].nunique() if "project_name" in df_filtered.columns else 0)
    else:
        print("\n(Analiz-only: parquet YAZILMADI — dataset 1000 proje korunuyor. "
              "Cap uygulamak istersen --write-parquet ekle.)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
