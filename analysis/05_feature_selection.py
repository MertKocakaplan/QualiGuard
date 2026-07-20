# %% [markdown]
# # 05 - Feature Selection (V2.1 — Task 2)
#
# Iki asamali algoritmik feature secimi:
#   Aşama A — Correlation filtering: |r| > 0.9 esleri bul, dusuk F1'liyi at.
#   Aşama B — RF importance pruning: OOF importance ile stable-low ozellikleri at.
#
# Validasyon: lean vs full RF GroupKFold(5) CV.
#   Pass kriteri: F1 dusus <= 0.005.
#
# Kullanim:
#   python analysis/05_feature_selection.py
#   python analysis/05_feature_selection.py --apply     # config.py + testi guncelle
#   python analysis/05_feature_selection.py --dataset output/dataset_model_filtered_X.parquet
#
# Ciktilar:
#   output/figures/correlation_heatmap_<ts>.png
#   output/figures/feature_importance_pruning_<ts>.png
#   output/feature_selection_proposal_<ts>.json

# %% Hucre 1 - Imports
from __future__ import annotations

import argparse
import json
import logging
import re
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
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# Standalone calistirma icin proje kokunu path'e ekle (python analysis/05_*.py)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.config import (
    FEATURES_BUG,
    FEATURES_SMELL,
    FIGURES_DIR,
    OUTPUT_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("analysis.05")

RANDOM_STATE = 42
N_CV_SPLITS = 5
CORR_THRESHOLD = 0.9
IMPORTANCE_MIN = 0.005


# %% Hucre 2 - Yardimci: univariate F1
def _univariate_f1(
    df: pd.DataFrame,
    feature: str,
    target_col: str,
    n_splits: int = 3,
) -> float:
    """Tek feature ile RF binary F1 (GroupKFold(n_splits) ortalamasi)."""
    if feature not in df.columns or target_col not in df.columns:
        return 0.0
    temp = df.dropna(subset=[target_col, feature])
    if temp[target_col].nunique() < 2:
        return 0.0
    X = temp[[feature]].fillna(0.0).to_numpy(dtype="float64")
    y = temp[target_col].to_numpy(dtype="int64")
    groups = temp["project_name"].to_numpy()
    n_proj = temp["project_name"].nunique()
    n_sp = min(n_splits, max(2, n_proj // 2))
    if n_sp < 2:
        return 0.0
    gkf = GroupKFold(n_splits=n_sp)
    scores = []
    for tr, te in gkf.split(X, y, groups):
        if y[tr].sum() == 0 or y[te].sum() == 0:
            continue
        sc = StandardScaler()
        clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=RANDOM_STATE)
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        scores.append(float(f1_score(y[te], clf.predict(sc.transform(X[te])), zero_division=0)))
    return float(np.mean(scores)) if scores else 0.0


# %% Hucre 3 - Aşama A: Korelasyon filtreleme
def phase_a_correlation(
    df: pd.DataFrame,
    features: tuple[str, ...],
    target_col: str,
) -> tuple[list[str], list[dict]]:
    """
    |r| > CORR_THRESHOLD esleri bul. Her esden univariate F1 dusuk olani at.

    Returns:
        (kept_features, dropped_info_list)
    """
    avail = [f for f in features if f in df.columns]
    if not avail:
        return list(features), []

    X_df = df[avail].fillna(0.0)

    pearson  = X_df.corr(method="pearson").abs()
    spearman = X_df.corr(method="spearman").abs()
    combined = (pearson + spearman) / 2.0

    # Ust ucgen maskesi
    mask = np.triu(np.ones(combined.shape, dtype=bool), k=1)
    high_corr_pairs = [
        (combined.index[i], combined.columns[j])
        for i in range(len(combined.index))
        for j in range(len(combined.columns))
        if mask[i, j] and combined.iloc[i, j] > CORR_THRESHOLD
    ]

    if not high_corr_pairs:
        logger.info("Asama A: |r|>%.1f esigi gecen cift bulunamadi.", CORR_THRESHOLD)
        return list(avail), []

    logger.info("Asama A: %d yuksek korelasyon ciftine univariate F1 hesaplaniyor...", len(high_corr_pairs))

    dropped: set[str] = set()
    drop_info: list[dict] = []

    for fa, fb in high_corr_pairs:
        if fa in dropped or fb in dropped:
            continue
        r = float(combined.loc[fa, fb])
        f1_a = _univariate_f1(df, fa, target_col)
        f1_b = _univariate_f1(df, fb, target_col)
        loser = fb if f1_a >= f1_b else fa
        winner = fa if loser == fb else fb
        dropped.add(loser)
        drop_info.append({
            "feature_a": fa, "feature_b": fb,
            "combined_r": round(r, 4),
            "f1_a": round(f1_a, 4), "f1_b": round(f1_b, 4),
            "dropped": loser, "kept": winner,
            "reason": f"|r|={r:.3f} > {CORR_THRESHOLD}",
        })
        logger.info(
            "  Korelasyon: %s <-> %s (r=%.3f) → at: %s (f1=%.4f), tut: %s (f1=%.4f)",
            fa, fb, r, loser, f1_a if loser == fa else f1_b, winner,
            f1_b if loser == fa else f1_a,
        )

    kept = [f for f in avail if f not in dropped]
    logger.info("Asama A: %d/%d feature tutuldu (%d atildi)", len(kept), len(avail), len(dropped))
    return kept, drop_info


# %% Hucre 4 - Aşama B: RF importance pruning
def phase_b_importance(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
) -> tuple[list[str], list[dict], np.ndarray, np.ndarray]:
    """
    RF(n_estimators=400) OOF importance ile stable-low feature'lari at.

    Drop kriteri: importance < IMPORTANCE_MIN AND std < importance (stable-low).

    Returns:
        (kept_features, dropped_info_list, mean_importances, std_importances)
    """
    avail = [f for f in features if f in df.columns]
    temp = df.dropna(subset=[target_col])
    if temp[target_col].nunique() < 2 or not avail:
        return list(features), [], np.array([]), np.array([])

    X = temp[avail].fillna(0.0).to_numpy(dtype="float64")
    y = temp[target_col].to_numpy(dtype="int64")
    groups = temp["project_name"].to_numpy()
    n_proj = temp["project_name"].nunique()
    n_sp = min(N_CV_SPLITS, max(2, n_proj // 2))

    if n_sp < 2:
        return list(avail), [], np.zeros(len(avail)), np.zeros(len(avail))

    gkf = GroupKFold(n_splits=n_sp)
    fold_importances: list[np.ndarray] = []

    logger.info("Asama B: RF OOF importance (%d fold, n=%d feature)...", n_sp, len(avail))
    for fold_idx, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        if y[tr].sum() == 0:
            continue
        sc = StandardScaler()
        clf = RandomForestClassifier(
            n_estimators=400, n_jobs=-1, random_state=RANDOM_STATE
        )
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        fold_importances.append(clf.feature_importances_)
        logger.info("  Fold %d/%d tamamlandi", fold_idx, n_sp)

    if not fold_importances:
        return list(avail), [], np.zeros(len(avail)), np.zeros(len(avail))

    imp_arr = np.vstack(fold_importances)
    mean_imp = imp_arr.mean(axis=0)
    std_imp  = imp_arr.std(axis=0)

    # Drop: importance < IMPORTANCE_MIN AND std < importance (stable-low)
    drop_mask = (mean_imp < IMPORTANCE_MIN) & (std_imp < mean_imp)

    dropped_info: list[dict] = []
    kept: list[str] = []
    for i, feat in enumerate(avail):
        if drop_mask[i]:
            dropped_info.append({
                "feature": feat,
                "mean_importance": round(float(mean_imp[i]), 6),
                "std_importance":  round(float(std_imp[i]), 6),
                "reason": f"importance={mean_imp[i]:.6f} < {IMPORTANCE_MIN} AND std={std_imp[i]:.6f} < importance",
            })
            logger.info("  AT: %s (imp=%.6f, std=%.6f)", feat, mean_imp[i], std_imp[i])
        else:
            kept.append(feat)

    logger.info("Asama B: %d/%d feature tutuldu (%d atildi)", len(kept), len(avail), len(dropped_info))
    return kept, dropped_info, mean_imp, std_imp


# %% Hucre 5 - Validasyon: lean vs full CV
def validate_lean(
    df: pd.DataFrame,
    full_features: tuple[str, ...],
    lean_features: list[str],
    target_col: str,
) -> tuple[float, float]:
    """
    Lean ve full feature setiyle RF GroupKFold(5) CV F1 karsilastir.

    Returns:
        (f1_full, f1_lean)
    """
    def _cv_f1(feats: list[str]) -> float:
        avail = [f for f in feats if f in df.columns]
        temp = df.dropna(subset=[target_col])
        if temp[target_col].nunique() < 2 or not avail:
            return 0.0
        X = temp[avail].fillna(0.0).to_numpy(dtype="float64")
        y = temp[target_col].to_numpy(dtype="int64")
        groups = temp["project_name"].to_numpy()
        n_proj = temp["project_name"].nunique()
        n_sp = min(N_CV_SPLITS, max(2, n_proj // 2))
        if n_sp < 2:
            return 0.0
        gkf = GroupKFold(n_splits=n_sp)
        scores = []
        for tr, te in gkf.split(X, y, groups):
            if y[tr].sum() == 0 or y[te].sum() == 0:
                continue
            sc = StandardScaler()
            clf = RandomForestClassifier(
                n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE
            )
            clf.fit(sc.fit_transform(X[tr]), y[tr])
            scores.append(float(f1_score(y[te], clf.predict(sc.transform(X[te])), zero_division=0)))
        return float(np.mean(scores)) if scores else 0.0

    f1_full = _cv_f1(list(full_features))
    f1_lean = _cv_f1(lean_features)
    logger.info("Validasyon (%s): full F1=%.4f | lean F1=%.4f | Δ=%.4f",
                target_col, f1_full, f1_lean, f1_lean - f1_full)
    return f1_full, f1_lean


# %% Hucre 6 - Importance fallback: geriye dogru ekle
def _restore_by_importance(
    lean: list[str],
    dropped_b: list[dict],
    mean_imp: np.ndarray,
    avail_after_a: list[str],
    f1_diff: float,
) -> list[str]:
    """
    F1 dususu > 0.005 ise; importance sirasiyla at ati geriye ekle.
    En onemli atilandan baslayarak ekle; her adimda tekrar test et.
    Bu fonksiyon sadece cagiriyi geri bildirir; caller CV'yi tekrar calistirir.
    """
    if not dropped_b or f1_diff >= -0.005:
        return lean
    # Importance azalarak sirala
    sorted_drops = sorted(dropped_b, key=lambda d: -d["mean_importance"])
    restored = list(lean)
    for item in sorted_drops:
        feat = item["feature"]
        if feat in avail_after_a and feat not in restored:
            restored.append(feat)
            logger.info("  Restore: %s (imp=%.6f)", feat, item["mean_importance"])
            break
    return restored


# %% Hucre 7 - Korelasyon heatmap
def plot_correlation_heatmap(
    df: pd.DataFrame,
    features: tuple[str, ...],
    out_path: Path,
) -> None:
    avail = [f for f in features if f in df.columns]
    if not avail:
        return
    corr = df[avail].fillna(0.0).corr(method="pearson").abs()

    fig, ax = plt.subplots(figsize=(max(10, len(avail) * 0.4), max(8, len(avail) * 0.35)))
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(avail)))
    ax.set_xticklabels(avail, rotation=90, fontsize=6)
    ax.set_yticks(range(len(avail)))
    ax.set_yticklabels(avail, fontsize=6)
    ax.set_title("Feature Korelasyon Matrisi (Pearson |r|)", fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Korelasyon heatmap: %s", out_path)


# %% Hucre 8 - Importance plot
def plot_importance(
    features: list[str],
    mean_imp: np.ndarray,
    std_imp: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    if mean_imp.size == 0:
        return
    order = np.argsort(mean_imp)[::-1]
    sorted_feats = [features[i] for i in order]
    sorted_mean  = mean_imp[order]
    sorted_std   = std_imp[order]

    # Kumulatif importance
    cumsum = np.cumsum(sorted_mean)
    k95 = int(np.searchsorted(cumsum, 0.95 * cumsum[-1])) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(5, len(features) * 0.25)))

    # Bar chart
    colors = ["#d62728" if m < IMPORTANCE_MIN else "#1f77b4" for m in sorted_mean]
    ax1.barh(range(len(sorted_feats)), sorted_mean, xerr=sorted_std,
             color=colors, ecolor="gray", capsize=2)
    ax1.set_yticks(range(len(sorted_feats)))
    ax1.set_yticklabels(sorted_feats, fontsize=7)
    ax1.invert_yaxis()
    ax1.axvline(IMPORTANCE_MIN, color="red", linestyle="--", linewidth=1, label=f"threshold={IMPORTANCE_MIN}")
    ax1.set_xlabel("Mean OOF Importance")
    ax1.set_title(f"{title}\n(kırmızı = atılacak)", fontsize=10)
    ax1.legend(fontsize=8)

    # Kumulatif
    ax2.plot(range(1, len(sorted_feats) + 1), cumsum / cumsum[-1], marker=".", markersize=4)
    ax2.axhline(0.95, color="orange", linestyle="--", linewidth=1, label="95% threshold")
    ax2.axvline(k95, color="green", linestyle="--", linewidth=1, label=f"top-{k95} features")
    ax2.set_xlabel("Feature sayisi (onem sirasinda)")
    ax2.set_ylabel("Kumulatif importance orani")
    ax2.set_title("Kumulatif Importance", fontsize=10)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Importance plot: %s", out_path)


# %% Hucre 9 - config.py guncelleme (AST tabanli — guvenli)
def _update_config_features(
    config_path: Path,
    lean_bug: list[str],
    lean_smell: list[str],
) -> None:
    r"""
    pipeline/config.py icindeki FEATURES_BUG ve FEATURES_SMELL tanimlarini
    AST ile lokalize edip lean setlerle yeniden yazar.

    NOT (eski regex tabanli versiyondaki bug duzeltildi):
      - Lazy regex `.*?` ile `(?=\n\n|\Z)` aralarinda bos satir olmayinca
        FEATURES_BUG bloğu FEATURES_SMELL'i de yutuyor → smell silinirdi.
      - AST yaklasimi node.lineno / node.end_lineno ile tam satir araligini hedefler;
        yorumlari korur, diger tanimlara dokunmaz.

    Yapisal duzeltme (eski smell_extra bug'i):
      - Lean feature secimi bug/smell icin BAGIMSIZ yapilir. Eski kod
        `FEATURES_BUG + (smell_extra,)` formuyla yaziyordu — bu lean_bug bir
        feature'i atip lean_smell'in tutmasi halinde yanlis sonuc verirdi.
      - Yeni: hem BUG hem SMELL bagimsiz duz tuple olarak yazilir.
    """
    import ast

    text = config_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        logger.error("config.py parse hatasi: %s — elle guncelleyin.", exc)
        return

    # FEATURES_BUG ve FEATURES_SMELL AnnAssign node'larini bul.
    # ast'de lineno/end_lineno 1-indexed, end_lineno son satir INCLUSIVE.
    targets: dict[str, tuple[int, int]] = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id in {"FEATURES_BUG", "FEATURES_SMELL"}):
            start_idx = node.lineno - 1            # 0-indexed slice start
            end_idx   = node.end_lineno or node.lineno  # exclusive icin end_lineno
            targets[node.target.id] = (start_idx, end_idx)

    if "FEATURES_BUG" not in targets or "FEATURES_SMELL" not in targets:
        logger.warning(
            "config.py'de FEATURES_BUG/SMELL bulunamadi (bulunan: %s) — elle guncelleyin.",
            list(targets),
        )
        return

    def _format_decl(name: str, features: list[str]) -> str:
        body = "".join(f'    "{f}",\n' for f in features)
        return f"{name}: Final[tuple[str, ...]] = (\n{body})\n"

    lines = text.splitlines(keepends=True)
    # Asagidan yukariya yaz ki onceki satir indexleri kaymasin.
    for name in sorted(targets, key=lambda n: -targets[n][0]):
        start, end = targets[name]
        features = lean_bug if name == "FEATURES_BUG" else lean_smell
        new_decl = _format_decl(name, features)
        lines[start:end] = [new_decl]

    new_text = "".join(lines)

    # AST validation: yeni dosya parse ediliyor mu?
    try:
        ast.parse(new_text)
    except SyntaxError as exc:
        logger.error("Yeni config.py parse edilemiyor (%s) — yazma iptal edildi.", exc)
        return

    config_path.write_text(new_text, encoding="utf-8")
    logger.info(
        "config.py guncellendi: FEATURES_BUG=%d, FEATURES_SMELL=%d (AST tabanli, guvenli)",
        len(lean_bug), len(lean_smell),
    )


def _update_test_assertion(test_path: Path, lean_smell_count: int) -> None:
    """
    tests/test_train_final.py:113 icindeki `len(fn["smell"]) == 48`
    ifadesini yeni count ile guncelle.
    """
    if not test_path.exists():
        logger.warning("Test dosyasi bulunamadi: %s", test_path)
        return
    text = test_path.read_text(encoding="utf-8")
    pattern = re.compile(r'assert\s+len\(fn\["smell"\]\)\s*==\s*\d+')
    new_assert = f'assert len(fn["smell"]) == {lean_smell_count}'
    new_text = pattern.sub(new_assert, text)
    if new_text == text:
        logger.warning("Test assertion guncellenmedi — elle guncelleyin: len(fn[\"smell\"]) == %d", lean_smell_count)
    else:
        test_path.write_text(new_text, encoding="utf-8")
        logger.info("Test assertion guncellendi: len(fn[\"smell\"]) == %d", lean_smell_count)


# %% Hucre 10 - main
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Feature selection: correlation + RF importance pruning (V2.1 Task 2)"
    )
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Filtered parquet (default: en guncel filesens veya model_filtered)")
    parser.add_argument("--apply", action="store_true",
                        help="config.py ve test dosyasini otomatik guncelle")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    # Veri yukle
    if args.dataset:
        data_path = args.dataset
    else:
        candidates = sorted(args.output_dir.glob("dataset_model_filtered_filesens_*.parquet"))
        if not candidates:
            candidates = sorted(args.output_dir.glob("dataset_model_filtered_*.parquet"))
        if not candidates:
            logger.error("Filtered parquet bulunamadi.")
            return 1
        data_path = candidates[-1]

    logger.info("Veri: %s", data_path)
    df = pd.read_parquet(data_path)
    logger.info("Kayit: %d | Proje: %d", len(df),
                df["project_name"].nunique() if "project_name" in df.columns else 0)

    # Bug label
    bug_col = "bug_keyword" if "bug_keyword" in df.columns else "bug_szz"
    if bug_col not in df.columns:
        logger.error("Bug label sutunu bulunamadi.")
        return 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Korelasyon heatmap ---
    corr_png = FIGURES_DIR / f"correlation_heatmap_{ts}.png"
    plot_correlation_heatmap(df, FEATURES_SMELL, corr_png)

    proposal: dict = {
        "created_at": datetime.now().isoformat(),
        "source_dataset": str(data_path),
        "bug": {"full_count": len(FEATURES_BUG), "phase_a_dropped": [], "phase_b_dropped": [],
                "lean_features": [], "f1_full": 0.0, "f1_lean": 0.0, "delta": 0.0, "accepted": False},
        "smell": {"full_count": len(FEATURES_SMELL), "phase_a_dropped": [], "phase_b_dropped": [],
                  "lean_features": [], "f1_full": 0.0, "f1_lean": 0.0, "delta": 0.0, "accepted": False},
    }

    # ── Bug ─────────────────────────────────────────────────────────
    logger.info("\n=== BUG feature selection (%s) ===", bug_col)

    # Asama A
    bug_after_a, bug_drop_a = phase_a_correlation(df, FEATURES_BUG, bug_col)
    proposal["bug"]["phase_a_dropped"] = bug_drop_a

    # Asama B
    bug_lean, bug_drop_b, bug_mean_imp, bug_std_imp = phase_b_importance(df, bug_after_a, bug_col)
    proposal["bug"]["phase_b_dropped"] = bug_drop_b

    # Importance plot
    if bug_mean_imp.size:
        imp_bug_png = FIGURES_DIR / f"feature_importance_pruning_bug_{ts}.png"
        plot_importance(bug_after_a, bug_mean_imp, bug_std_imp, "Bug Feature Importance", imp_bug_png)

    # Validasyon
    f1_full_b, f1_lean_b = validate_lean(df, FEATURES_BUG, bug_lean, bug_col)
    delta_b = f1_lean_b - f1_full_b

    # Fallback: geriye ekle eger F1 < threshold
    if delta_b < -0.005 and bug_drop_b:
        logger.warning("Bug lean F1 dususu %.4f > 0.005 — feature geri ekleniyor.", -delta_b)
        restored = list(bug_lean)
        for item in sorted(bug_drop_b, key=lambda d: -d["mean_importance"]):
            feat = item["feature"]
            if feat in bug_after_a and feat not in restored:
                restored.append(feat)
                f1_full_b2, f1_lean_b2 = validate_lean(df, FEATURES_BUG, restored, bug_col)
                logger.info("  +%s → lean F1=%.4f (Δ=%.4f)", feat, f1_lean_b2, f1_lean_b2 - f1_full_b2)
                if (f1_lean_b2 - f1_full_b2) >= -0.005:
                    bug_lean = restored
                    f1_lean_b = f1_lean_b2
                    delta_b = f1_lean_b - f1_full_b
                    break

    bug_accepted = delta_b >= -0.005
    proposal["bug"].update({
        "lean_features": bug_lean,
        "lean_count": len(bug_lean),
        "f1_full": round(f1_full_b, 4),
        "f1_lean": round(f1_lean_b, 4),
        "delta": round(delta_b, 4),
        "accepted": bug_accepted,
    })
    logger.info("Bug lean: %d feature, F1: %.4f → %.4f (Δ=%.4f) [%s]",
                len(bug_lean), f1_full_b, f1_lean_b, delta_b,
                "KABUL" if bug_accepted else "RED — tam set kullanilacak")

    # ── Smell ───────────────────────────────────────────────────────
    logger.info("\n=== SMELL feature selection (smell_binary) ===")

    smell_after_a, smell_drop_a = phase_a_correlation(df, FEATURES_SMELL, "smell_binary")
    proposal["smell"]["phase_a_dropped"] = smell_drop_a

    smell_lean, smell_drop_b, smell_mean_imp, smell_std_imp = phase_b_importance(df, smell_after_a, "smell_binary")
    proposal["smell"]["phase_b_dropped"] = smell_drop_b

    if smell_mean_imp.size:
        imp_smell_png = FIGURES_DIR / f"feature_importance_pruning_smell_{ts}.png"
        plot_importance(smell_after_a, smell_mean_imp, smell_std_imp, "Smell Feature Importance", imp_smell_png)

    f1_full_s, f1_lean_s = validate_lean(df, FEATURES_SMELL, smell_lean, "smell_binary")
    delta_s = f1_lean_s - f1_full_s

    if delta_s < -0.005 and smell_drop_b:
        logger.warning("Smell lean F1 dususu %.4f > 0.005 — feature geri ekleniyor.", -delta_s)
        restored = list(smell_lean)
        for item in sorted(smell_drop_b, key=lambda d: -d["mean_importance"]):
            feat = item["feature"]
            if feat in smell_after_a and feat not in restored:
                restored.append(feat)
                f1_full_s2, f1_lean_s2 = validate_lean(df, FEATURES_SMELL, restored, "smell_binary")
                logger.info("  +%s → lean F1=%.4f (Δ=%.4f)", feat, f1_lean_s2, f1_lean_s2 - f1_full_s2)
                if (f1_lean_s2 - f1_full_s2) >= -0.005:
                    smell_lean = restored
                    f1_lean_s = f1_lean_s2
                    delta_s = f1_lean_s - f1_full_s
                    break

    smell_accepted = delta_s >= -0.005
    proposal["smell"].update({
        "lean_features": smell_lean,
        "lean_count": len(smell_lean),
        "f1_full": round(f1_full_s, 4),
        "f1_lean": round(f1_lean_s, 4),
        "delta": round(delta_s, 4),
        "accepted": smell_accepted,
    })
    logger.info("Smell lean: %d feature, F1: %.4f → %.4f (Δ=%.4f) [%s]",
                len(smell_lean), f1_full_s, f1_lean_s, delta_s,
                "KABUL" if smell_accepted else "RED — tam set kullanilacak")

    # JSON ciktisi
    json_path = args.output_dir / f"feature_selection_proposal_{ts}.json"
    json_path.write_text(json.dumps(proposal, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Proposal JSON: %s", json_path)

    # Ozet
    print("\n" + "=" * 70)
    print("FEATURE SELECTION SONUCLARI")
    print("=" * 70)
    print(f"Bug:   {len(FEATURES_BUG)} → {len(bug_lean)} feature | Δ F1={delta_b:+.4f} [{('KABUL' if bug_accepted else 'RED')}]")
    print(f"Smell: {len(FEATURES_SMELL)} → {len(smell_lean)} feature | Δ F1={delta_s:+.4f} [{('KABUL' if smell_accepted else 'RED')}]")

    # --apply: config.py ve test dosyasini guncelle
    if args.apply:
        final_bug   = bug_lean   if bug_accepted   else list(FEATURES_BUG)
        final_smell = smell_lean if smell_accepted else list(FEATURES_SMELL)

        config_path = Path(__file__).resolve().parents[1] / "pipeline" / "config.py"
        _update_config_features(config_path, final_bug, final_smell)

        test_path = Path(__file__).resolve().parents[1] / "tests" / "test_train_final.py"
        _update_test_assertion(test_path, len(final_smell))
        print(f"\nconfig.py guncellendi: BUG={len(final_bug)}, SMELL={len(final_smell)}")
        print("pytest tests/test_train_final.py ile dogrulayin.")
    else:
        print(f"\nKonfigurasyonu uygulamak icin: python analysis/05_feature_selection.py --apply")

    return 0


if __name__ == "__main__":
    sys.exit(main())
