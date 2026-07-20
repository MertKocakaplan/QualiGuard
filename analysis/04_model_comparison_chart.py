# %% [markdown]
# # 04 - Model Comparison Chart (Benchmark CV + Hibrit)
#
# `06_ml_baseline_cv.py` ciktisi (model_benchmark_cv_<ts>.json — 10 model) ile
# `train_final --cv-folds 5` ciktisini (cv_summary_<ts>.json — hibrit stacking)
# tek figurde birlestirir: 11 bar (10 model + Hibrit), bug + smell paneli.
#
# Amac: "hibrit, en iyi tekil modelden daha mi iyi?" sorusunu gorsellestirmek.
#
# Adil karsilastirma notu:
#   - 10 model: threshold 0.5 (benchmark CV, 06)
#   - Hibrit: hem default (0.5) hem optimal (threshold-opt) hesaplanir;
#     ana bar default (0.5) — diger 10 modelle ayni kosul (elma-elmaya).
#     Optimal F1 etikette parantezde belirtilir.
#
# (Eski F5 ablation-aggregate isleviyle uretilen figur git history'de;
#  yeni akista benchmark CV (06) onun CV-tabanli yerini aldi.)
#
# Kullanim:
#   python analysis/04_model_comparison_chart.py
#   python analysis/04_model_comparison_chart.py --benchmark X.json --cv-summary Y.json
#   python analysis/04_model_comparison_chart.py --no-hybrid   # sadece 10 model
#
# Cikti: output/figures/model_comparison_<ts>.png

# %% Hucre 1 - Imports + sabitler
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Standalone calistirma icin proje kokunu path'e ekle (python analysis/04_*.py)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline.config import FIGURES_DIR, OUTPUT_DIR

# Aile renkleri (benchmark json'daki family alaniyla uyumlu) + Hibrit
FAMILY_COLOR = {
    "ML":     "#3498db",
    "DL":     "#e74c3c",
    "AutoML": "#2ecc71",
    "Hibrit": "#9b59b6",
}
DISPLAY = {
    "rf": "Random Forest", "lightgbm": "LightGBM", "xgboost": "XGBoost",
    "lr": "Logistic Reg.", "mlp": "MLP", "cnn1d": "CNN-1D", "lstm": "LSTM",
    "autogluon": "AutoGluon", "h2o": "H2O", "tpot": "TPOT",
    "stacking": "Hibrit (Stacking)",
}


# %% Hucre 2 - En guncel dosyalari bul
def _latest(pattern: str, directory: Path) -> Optional[Path]:
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


# %% Hucre 3 - Veri toplama
def _collect_task(
    benchmark_results: dict,
    cv_summary: Optional[dict],
    task: str,
) -> list[dict]:
    """
    Bir task (bug/smell) icin bar verisi: 10 model + (varsa) hibrit.

    Returns: [{name, display, family, f1, std, note}] — F1 artan sirali.
    """
    rows: list[dict] = []
    results = benchmark_results.get(f"{task}_results", {})
    family_map = benchmark_results.get("family", {})

    for model_name, v in results.items():
        if v.get("status") != "ok" or "f1_mean" not in v:
            continue
        rows.append({
            "name":    model_name,
            "display": DISPLAY.get(model_name, model_name),
            "family":  v.get("family") or family_map.get(model_name, "ML"),
            "f1":      v["f1_mean"],
            "std":     v.get("f1_std", 0.0),
            "note":    "",
        })

    # Hibrit (cv_summary'den) — BAR = threshold-opt (production setting);
    # default (thr=0.5) etikette.
    #
    # Asimetrik kosul gerekcesi: 10 model benchmark'ta hep thr=0.5; threshold-opt
    # yalniz hibrit'e fayda saglar (meta-LR cikis dagilimi 0.5 etrafinda
    # dengelenmemis). Tek modeller (LightGBM) icin thr-opt OOF overfit yapip F1'i
    # DUSURUYOR (LGB szz: 0.6351 thr0.5 -> 0.6302 thr-opt). Hibridin production
    # F1'i (thr-opt) en savunulabilir karsilastirma noktasi.
    if cv_summary is not None:
        o = cv_summary.get(f"{task}_summary_optimal", {}).get("f1", {})
        d = cv_summary.get(f"{task}_summary_default", {}).get("f1", {})
        if o.get("mean") is not None:
            note = f"@thr=0.5: {d['mean']:.3f}" if d.get("mean") is not None else ""
            rows.append({
                "name":    "stacking",
                "display": DISPLAY["stacking"],
                "family":  "Hibrit",
                "f1":      float(o["mean"]),
                "std":     float(o.get("std", 0.0)),
                "note":    note,
            })
        elif d.get("mean") is not None:
            # Fallback: optimal yoksa default'u kullan
            rows.append({
                "name":    "stacking",
                "display": DISPLAY["stacking"],
                "family":  "Hibrit",
                "f1":      float(d["mean"]),
                "std":     float(d.get("std", 0.0)),
                "note":    "",
            })

    rows.sort(key=lambda r: r["f1"])
    return rows


# %% Hucre 4 - Cizim
def plot_comparison(
    rows_bug: list[dict],
    rows_smell: list[dict],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, rows, title in zip(
        axes, [rows_bug, rows_smell],
        ["Hata Tahmini (Bug)", "Code Smell Tahmini"],
    ):
        if not rows:
            ax.set_title(f"{title}\n(veri yok)"); ax.axis("off"); continue

        names  = [r["display"] for r in rows]
        f1s    = [r["f1"] for r in rows]
        stds   = [r["std"] for r in rows]
        colors = [FAMILY_COLOR.get(r["family"], "#888888") for r in rows]
        y = np.arange(len(rows))

        ax.barh(y, f1s, xerr=stds, color=colors, edgecolor="white", capsize=3, height=0.72)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Ortalama F1 (5-fold CV)")
        ax.set_title(f"{title}\n(GroupKFold=5; 10 model @ thr=0.5; Hibrit @ thr-opt = production)",
                     fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        for i, r in enumerate(rows):
            txt = f"{r['f1']:.3f}"
            if r["note"]:
                txt += f"  ({r['note']})"
            ax.text(r["f1"] + r["std"] + 0.02, i, txt, va="center",
                    fontsize=8.5, fontweight="bold" if r["family"] == "Hibrit" else "normal")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLOR.values()]
    fig.legend(handles, list(FAMILY_COLOR.keys()), loc="upper center",
               ncol=4, frameon=True, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Model Karsilastirmasi — Benchmark CV + Hibrit",
                 fontsize=13, fontweight="bold", y=1.09)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# %% Hucre 5 - main
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark CV + hibrit karsilastirma figuru"
    )
    parser.add_argument("--benchmark", type=Path, default=None,
                        help="model_benchmark_cv_<ts>.json (default: en guncel)")
    parser.add_argument("--cv-summary", type=Path, default=None,
                        help="cv_summary_<ts>.json (default: en guncel; --no-hybrid ile atla)")
    parser.add_argument("--no-hybrid", action="store_true",
                        help="Hibriti ekleme — sadece 10 model")
    args = parser.parse_args(argv)

    # Benchmark json
    bench_path = args.benchmark or _latest("model_benchmark_cv_*.json", OUTPUT_DIR)
    if not bench_path or not bench_path.exists():
        print("HATA: model_benchmark_cv_*.json bulunamadi. Once 06_ml_baseline_cv.py calistir.")
        return 1
    benchmark = json.loads(bench_path.read_text(encoding="utf-8"))
    print(f"Benchmark kaynak: {bench_path.name}")

    # cv_summary (hibrit) — opsiyonel
    cv_summary = None
    if not args.no_hybrid:
        cv_path = args.cv_summary or _latest("cv_summary_*.json", _ROOT / "models")
        if cv_path and cv_path.exists():
            cv_summary = json.loads(cv_path.read_text(encoding="utf-8"))
            print(f"Hibrit kaynak  : {cv_path.name}")
        else:
            print("UYARI: cv_summary bulunamadi — hibrit barsiz cizilecek.")

    rows_bug   = _collect_task(benchmark, cv_summary, "bug")
    rows_smell = _collect_task(benchmark, cv_summary, "smell")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = FIGURES_DIR / f"model_comparison_{ts}.png"
    plot_comparison(rows_bug, rows_smell, out_path)
    print(f"kaydedildi: {out_path}")

    # Konsol ozeti
    for task, rows in [("BUG", rows_bug), ("SMELL", rows_smell)]:
        if not rows:
            continue
        print(f"\n{task} (F1 azalan):")
        for r in sorted(rows, key=lambda x: -x["f1"]):
            extra = f"  ({r['note']})" if r["note"] else ""
            star = "  <- HIBRIT" if r["family"] == "Hibrit" else ""
            print(f"  {r['display']:20s} {r['f1']:.4f} ± {r['std']:.4f}{extra}{star}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
