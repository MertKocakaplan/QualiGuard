# %% [markdown]
# # 06 - Model Benchmark CV (V2.1 — birlesik 10 model)
#
# rf / lightgbm / xgboost / lr (ML) + mlp / cnn1d / lstm (DL) +
# autogluon / h2o / tpot (AutoML) adaylarini AYNI GroupKFold(5) CV
# protokolunde egitip karsilastirir. F5 ablation'in tek-split zayifligini giderir.
#
# Otomatik kazanan secimi YOK — tablo + figur uretir, hibrit base'i ELLE secilir.
# (Hibrit, tek seferlik bir karar; sonra train_final --stacking-base/--automl ile kurulur.)
#
# Tum modeller ayni kosulda (StandardScaler + 5 fold + threshold 0.5):
#   - ML: class imbalance handling (balanced / is_unbalance / scale_pos_weight)
#   - DL: Keras class_weight dict (dengesiz veri icin)
#   - AutoML: kendi preprocessing; H2O JVM her fold init/shutdown; TPOT sanity-then-promote
#
# Kullanim:
#   python analysis/06_ml_baseline_cv.py --dataset output/dataset_model_filtered_X.parquet
#   python analysis/06_ml_baseline_cv.py --dataset ... --skip-tpot   # TPOT atla (hizli)
#   python analysis/06_ml_baseline_cv.py --dataset ... --skip-dl     # DL atla
#
# Ciktilar:
#   output/model_benchmark_cv_<ts>.json
#   output/figures/model_benchmark_<ts>.png

# %% Hucre 1 - Imports + CLI
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# Standalone calistirma icin proje kokunu path'e ekle (python analysis/06_*.py)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.config import FEATURES_BUG, FEATURES_SMELL, FIGURES_DIR, OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("analysis.06")

RANDOM_STATE = 42
N_CV_SPLITS = 5

# Model sirasi (tablo + figur bu sirayla) ve aile etiketleri
MODELS = ["rf", "lightgbm", "xgboost", "lr",
          "mlp", "cnn1d", "lstm",
          "autogluon", "h2o", "tpot"]
FAMILY = {
    "rf": "ML", "lightgbm": "ML", "xgboost": "ML", "lr": "ML",
    "mlp": "DL", "cnn1d": "DL", "lstm": "DL",
    "autogluon": "AutoML", "h2o": "AutoML", "tpot": "AutoML",
}
FAMILY_COLOR = {"ML": "#3498db", "DL": "#e74c3c", "AutoML": "#2ecc71"}

# DL hiperparametreleri (02 ablation ile tutarli)
DL_EPOCHS = 10
DL_BATCH = 64

# AutoML butceleri
AUTOGLUON_TIME_LIMIT = 300       # sn/fold
H2O_MAX_RUNTIME_SEC = 300        # sn/fold — AutoGluon (300) ile esit butce: adil + hizli
TPOT_MAX_TIME_MINS = 5           # dk/fold — AutoGluon (300s=5dk) ile esit butce: adil + hizli
TPOT_SANITY_FOLDS = 2            # once 2 fold dene
TPOT_PROMOTE_F1 = 0.45           # bu uzerinde full CV'ye gec


# %% Hucre 2 - sklearn ML fabrikalari
def _build_rf(n_neg: int = 1, n_pos: int = 1) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400, n_jobs=-1, class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def _build_lgbm(n_neg: int = 1, n_pos: int = 1):
    from lightgbm import LGBMClassifier  # type: ignore[import-not-found]
    return LGBMClassifier(
        n_estimators=400, num_leaves=31, n_jobs=-1,
        is_unbalance=True, verbose=-1, random_state=RANDOM_STATE,
    )


def _build_xgb(n_neg: int = 1, n_pos: int = 1):
    from xgboost import XGBClassifier  # type: ignore[import-not-found]
    scale_pos = float(n_neg) / max(n_pos, 1)
    return XGBClassifier(
        n_estimators=400, n_jobs=-1, scale_pos_weight=scale_pos,
        eval_metric="logloss", random_state=RANDOM_STATE, tree_method="hist",
    )


def _build_lr(n_neg: int = 1, n_pos: int = 1) -> LogisticRegression:
    return LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE,
    )


SKLEARN_FACTORIES = {
    "rf": _build_rf, "lightgbm": _build_lgbm,
    "xgboost": _build_xgb, "lr": _build_lr,
}


# %% Hucre 3 - Keras DL builder'lari (02 ablation'dan port, class_weight'li)
def _keras_dense(input_dim: int):
    from tensorflow import keras  # type: ignore[import-not-found]
    m = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m


def _keras_cnn1d(input_dim: int):
    from tensorflow import keras  # type: ignore[import-not-found]
    m = keras.Sequential([
        keras.layers.Input(shape=(input_dim, 1)),
        keras.layers.Conv1D(32, 3, activation="relu", padding="same"),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m


def _keras_lstm(input_dim: int):
    from tensorflow import keras  # type: ignore[import-not-found]
    m = keras.Sequential([
        keras.layers.Input(shape=(input_dim, 1)),
        keras.layers.LSTM(32),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m


KERAS_BUILDERS = {
    "mlp":   (_keras_dense, False),  # (builder, reshape_3d)
    "cnn1d": (_keras_cnn1d, True),
    "lstm":  (_keras_lstm,  True),
}


def _class_weight_dict(ytr: np.ndarray) -> Optional[dict]:
    """sklearn 'balanced' semantigine esdeger Keras class_weight dict."""
    n = len(ytr)
    n_pos = int(ytr.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    return {0: n / (2.0 * n_neg), 1: n / (2.0 * n_pos)}


# %% Hucre 4 - Fold-level degerlendiriciler (her model tipi icin)
def _metrics(yte: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "f1":        float(f1_score(yte, y_pred, zero_division=0)),
        "precision": float(precision_score(yte, y_pred, zero_division=0)),
        "recall":    float(recall_score(yte, y_pred, zero_division=0)),
    }


def _eval_sklearn(model_name, Xtr, ytr, Xte, yte, n_neg, n_pos) -> dict:
    clf = SKLEARN_FACTORIES[model_name](n_neg=n_neg, n_pos=n_pos)
    clf.fit(Xtr, ytr)
    return {"status": "ok", **_metrics(yte, clf.predict(Xte))}


def _eval_keras(model_name, Xtr, ytr, Xte, yte, n_neg, n_pos) -> dict:
    try:
        from tensorflow import keras  # noqa: F401
    except ImportError as exc:
        return {"status": "skipped", "error": f"tensorflow yok: {exc}"}
    builder, reshape_3d = KERAS_BUILDERS[model_name]
    Xtr_in = Xtr[..., None] if reshape_3d else Xtr
    Xte_in = Xte[..., None] if reshape_3d else Xte
    cw = _class_weight_dict(ytr)
    model = builder(Xtr.shape[1])
    model.fit(Xtr_in, ytr, epochs=DL_EPOCHS, batch_size=DL_BATCH,
              class_weight=cw, verbose=0)
    y_proba = model.predict(Xte_in, verbose=0).ravel()
    y_pred = (y_proba >= 0.5).astype("int64")
    return {"status": "ok", **_metrics(yte, y_pred)}


def _eval_autogluon(model_name, Xtr, ytr, Xte, yte, n_neg, n_pos) -> dict:
    try:
        from autogluon.tabular import TabularPredictor  # type: ignore[import-not-found]
    except ImportError as exc:
        return {"status": "skipped", "error": f"autogluon yok: {exc}"}
    import tempfile
    feat_cols = [f"f{i}" for i in range(Xtr.shape[1])]
    tr_df = pd.DataFrame(Xtr, columns=feat_cols); tr_df["target"] = ytr
    te_df = pd.DataFrame(Xte, columns=feat_cols)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            p = TabularPredictor(
                label="target", path=tmp, problem_type="binary", verbosity=0,
            ).fit(tr_df, time_limit=AUTOGLUON_TIME_LIMIT)
            y_pred = p.predict(te_df).to_numpy().astype("int64")
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "error": str(exc)[:200]}
    return {"status": "ok", **_metrics(yte, y_pred)}


def _eval_h2o(model_name, Xtr, ytr, Xte, yte, n_neg, n_pos) -> dict:
    try:
        import h2o  # type: ignore[import-not-found]
        from h2o.automl import H2OAutoML  # type: ignore[import-not-found]
    except ImportError as exc:
        return {"status": "skipped", "error": f"h2o yok: {exc}"}
    feat_cols = [f"f{i}" for i in range(Xtr.shape[1])]
    train_df = pd.DataFrame(Xtr, columns=feat_cols); train_df["target"] = ytr
    test_df = pd.DataFrame(Xte, columns=feat_cols)
    try:
        h2o.init(nthreads=-1, max_mem_size="4G", verbose=False)
        h_train = h2o.H2OFrame(train_df)
        h_test = h2o.H2OFrame(test_df)
        h_train["target"] = h_train["target"].asfactor()
        aml = H2OAutoML(max_runtime_secs=H2O_MAX_RUNTIME_SEC, max_models=20,
                        seed=RANDOM_STATE, verbosity="warn")
        aml.train(x=feat_cols, y="target", training_frame=h_train)
        y_pred = aml.leader.predict(h_test).as_data_frame()["predict"].astype(int).to_numpy()
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "error": str(exc)[:200]}
    finally:
        try:
            h2o.remove_all()
            h2o.cluster().shutdown(prompt=False)
        except Exception:  # noqa: BLE001
            pass
    return {"status": "ok", **_metrics(yte, y_pred)}


def _eval_tpot(model_name, Xtr, ytr, Xte, yte, n_neg, n_pos) -> dict:
    try:
        from tpot import TPOTClassifier  # type: ignore[import-not-found]
    except ImportError as exc:
        return {"status": "skipped", "error": f"tpot yok: {exc}"}
    try:
        # TPOT 1.x API (search_space/scorers/verbose) — train_final ile birebir tutarli.
        # NOT: tpot 0.x'in 'scoring'/'verbosity' kwarg'lari 1.x'te YOK (TypeError verir).
        tpot = TPOTClassifier(
            search_space="linear-light", scorers=["f1"], scorers_weights=[1],
            max_time_mins=TPOT_MAX_TIME_MINS, max_eval_time_mins=1,
            cv=2, n_jobs=-1, verbose=0, random_state=RANDOM_STATE,
        )
        tpot.fit(Xtr, ytr)
        y_pred = tpot.predict(Xte)
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "error": str(exc)[:200]}
    return {"status": "ok", **_metrics(yte, y_pred)}


def _eval_one_fold(model_name, Xtr, ytr, Xte, yte, n_neg, n_pos) -> dict:
    """model_name'e gore dogru degerlendiriciye yonlendir."""
    if model_name in SKLEARN_FACTORIES:
        return _eval_sklearn(model_name, Xtr, ytr, Xte, yte, n_neg, n_pos)
    if model_name in KERAS_BUILDERS:
        return _eval_keras(model_name, Xtr, ytr, Xte, yte, n_neg, n_pos)
    if model_name == "autogluon":
        return _eval_autogluon(model_name, Xtr, ytr, Xte, yte, n_neg, n_pos)
    if model_name == "h2o":
        return _eval_h2o(model_name, Xtr, ytr, Xte, yte, n_neg, n_pos)
    if model_name == "tpot":
        return _eval_tpot(model_name, Xtr, ytr, Xte, yte, n_neg, n_pos)
    return {"status": "skipped", "error": f"bilinmeyen model: {model_name}"}


# %% Hucre 5 - Tek-model CV (tum fold'lar)
def run_cv_for_model(
    df: pd.DataFrame,
    features: tuple[str, ...],
    target_col: str,
    model_name: str,
    task_name: str,
    sanity_only: bool = False,
) -> dict:
    """
    Tek model icin GroupKFold(N) CV. Her fold'da StandardScaler fit/transform.
    sanity_only: TPOT_SANITY_FOLDS kadar fold (TPOT promote kontrolu).
    """
    if target_col not in df.columns:
        return {"status": "skipped", "reason": "no_target"}
    temp = df.dropna(subset=[target_col])
    if temp[target_col].nunique() < 2:
        return {"status": "skipped", "reason": "single_class"}
    avail = [f for f in features if f in temp.columns]
    if not avail:
        return {"status": "skipped", "reason": "no_features"}

    X = temp[avail].fillna(0.0).to_numpy(dtype="float64")
    y = temp[target_col].to_numpy(dtype="int64")
    groups = temp["project_name"].to_numpy()
    n_proj = temp["project_name"].nunique()
    n_sp = min(N_CV_SPLITS, max(2, n_proj // 2))
    if n_sp < 2:
        return {"status": "skipped", "reason": "low_projects"}

    gkf = GroupKFold(n_splits=n_sp)
    fold_indices = list(gkf.split(X, y, groups))
    fold_limit = TPOT_SANITY_FOLDS if sanity_only else len(fold_indices)

    fold_metrics: list[dict] = []
    t0 = time.monotonic()

    for fold_idx, (tr_idx, te_idx) in enumerate(fold_indices[:fold_limit], 1):
        Xtr_raw, Xte_raw = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        if ytr.sum() == 0 or yte.sum() == 0:
            continue

        n_neg = int((ytr == 0).sum())
        n_pos = int((ytr == 1).sum())

        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr_raw)
        Xte_s = sc.transform(Xte_raw)

        logger.info("  [%s/%s] fold %d/%d...", task_name, model_name, fold_idx, fold_limit)
        res = _eval_one_fold(model_name, Xtr_s, ytr, Xte_s, yte, n_neg, n_pos)
        if res["status"] != "ok":
            logger.warning("  [%s/%s] fold %d: %s", task_name, model_name, fold_idx,
                           res.get("error", res.get("reason", "?")))
            # Ilk fold skipped (ImportError) → tum model atlanir
            if res["status"] == "skipped" and not fold_metrics:
                return res
            continue
        fold_metrics.append(res)
        logger.info("  [%s/%s] fold %d: f1=%.4f prec=%.4f rec=%.4f",
                    task_name, model_name, fold_idx, res["f1"], res["precision"], res["recall"])

    dur = time.monotonic() - t0
    if not fold_metrics:
        return {"status": "no_valid_folds", "duration_secs": round(dur, 2)}

    f1s   = [m["f1"]        for m in fold_metrics]
    precs = [m["precision"] for m in fold_metrics]
    recs  = [m["recall"]    for m in fold_metrics]
    summary = {
        "status":         "ok",
        "family":         FAMILY.get(model_name, "?"),
        "f1_mean":        round(float(np.mean(f1s)), 4),
        "f1_std":         round(float(np.std(f1s)),  4),
        "precision_mean": round(float(np.mean(precs)), 4),
        "recall_mean":    round(float(np.mean(recs)),  4),
        "fold_f1s":       [round(float(v), 4) for v in f1s],
        "n_folds":        len(fold_metrics),
        "duration_secs":  round(dur, 2),
        "sanity_only":    sanity_only,
    }
    logger.info("[%s] %s: F1=%.4f ± %.4f (%d folds, %.1fs)%s",
                task_name, model_name, summary["f1_mean"], summary["f1_std"],
                summary["n_folds"], dur, " [SANITY]" if sanity_only else "")
    return summary


def run_cv_for_task(
    df: pd.DataFrame,
    features: tuple[str, ...],
    target_col: str,
    task_name: str,
    skip_tpot: bool = False,
    skip_dl: bool = False,
    models: Optional[list[str]] = None,
) -> dict[str, dict]:
    """Verilen modeller icin CV (model-disinda-loop — H2O/AutoGluon/TF izolasyonu).

    models=None → tum MODELS; aksi halde sadece verilen alt-kume (MODELS sirasi korunur).
    """
    run_models = [m for m in MODELS if m in set(models)] if models else list(MODELS)
    results: dict[str, dict] = {}
    for model_name in run_models:
        if skip_dl and model_name in KERAS_BUILDERS:
            results[model_name] = {"status": "skipped", "reason": "skip_dl_flag"}
            continue
        if model_name == "tpot":
            if skip_tpot:
                results[model_name] = {"status": "skipped", "reason": "skip_tpot_flag"}
                continue
            # TPOT: once sanity, umit verirse full
            logger.info("\n--- [%s] tpot (sanity %d fold) ---", task_name, TPOT_SANITY_FOLDS)
            sanity = run_cv_for_model(df, features, target_col, "tpot", task_name, sanity_only=True)
            if sanity.get("status") == "ok" and sanity.get("f1_mean", 0) >= TPOT_PROMOTE_F1:
                logger.info("[%s] TPOT sanity F1=%.4f >= %.4f → full CV",
                            task_name, sanity["f1_mean"], TPOT_PROMOTE_F1)
                results["tpot"] = run_cv_for_model(df, features, target_col, "tpot", task_name)
            else:
                results["tpot"] = sanity
                if sanity.get("status") == "ok":
                    logger.info("[%s] TPOT sanity F1=%.4f < %.4f → full atlandi",
                                task_name, sanity.get("f1_mean", 0), TPOT_PROMOTE_F1)
            continue

        logger.info("\n--- [%s] %s ---", task_name, model_name)
        results[model_name] = run_cv_for_model(df, features, target_col, model_name, task_name)
    return results


# %% Hucre 6 - Benchmark bar chart (10 model, aile renkli, kazanan vurgusu YOK)
def plot_benchmark(
    results_bug: dict[str, dict],
    results_smell: dict[str, dict],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, results, label in zip(
        axes, [results_bug, results_smell],
        ["Bug (binary F1)", "Smell (binary F1)"],
    ):
        ok = {k: v for k, v in results.items() if v.get("status") == "ok" and "f1_mean" in v}
        if not ok:
            ax.set_title(f"{label}\n(veri yok)"); ax.axis("off"); continue

        # F1'e gore artan sirala (en iyi ustte gozuksun barh ile)
        items = sorted(ok.items(), key=lambda kv: kv[1]["f1_mean"])
        names  = [k for k, _ in items]
        means  = [v["f1_mean"] for _, v in items]
        stds   = [v["f1_std"]  for _, v in items]
        colors = [FAMILY_COLOR[FAMILY[k]] for k in names]
        y = np.arange(len(names))

        ax.barh(y, means, xerr=stds, color=colors, edgecolor="white", capsize=3, height=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Ortalama F1 (5-fold CV)")
        ax.set_title(f"{label}\n(GroupKFold=5, threshold=0.5)", fontsize=11)
        ax.grid(axis="x", alpha=0.3)
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(m + s + 0.02, i, f"{m:.3f}", va="center", fontsize=9, fontweight="bold")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLOR.values()]
    fig.legend(handles, list(FAMILY_COLOR.keys()), loc="upper center",
               ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Model Benchmark — 5-fold CV (hibrit base secimi icin)",
                 fontsize=13, fontweight="bold", y=1.08)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Benchmark figuru: %s", out_path)


# %% Hucre 7 - main
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Model benchmark CV — 10 model, otomatik kazanan secimi YOK (V2.1)"
    )
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--skip-tpot", action="store_true", help="TPOT'u atla (cok yavas)")
    parser.add_argument("--skip-dl",   action="store_true", help="DL modellerini atla (mlp/cnn1d/lstm)")
    parser.add_argument("--skip-bug",   action="store_true")
    parser.add_argument("--skip-smell", action="store_true")
    parser.add_argument("--models", type=str, default=None,
                        help="Sadece bu modelleri kostur (virgullu, or: h2o,tpot). Default: 10 model.")
    parser.add_argument("--merge-into", type=str, default=None,
                        help="Mevcut model_benchmark_cv_*.json'a birlestir ('auto'=en guncel). "
                             "Yeni kosulan modeller eskileri gunceller; tum tablo tek JSON'da toplanir.")
    parser.add_argument("--bug-label", choices=("keyword", "szz"), default="szz",
                        help="Bug hedef etiketi: bug_keyword (keyword heuristic) veya bug_szz "
                             "(SZZ algoritmasi — rigorous; train_final default). Benchmark ile "
                             "hibrit CV ayni etikette olmali (paper figuru tutarliligi).")
    args = parser.parse_args(argv)

    if args.dataset:
        data_path = args.dataset
    else:
        cands = sorted(args.output_dir.glob("dataset_model_filtered_filesens_*.parquet"))
        if not cands:
            cands = sorted(args.output_dir.glob("dataset_model_filtered_*.parquet"))
        if not cands:
            logger.error("Filtered parquet bulunamadi.")
            return 1
        data_path = cands[-1]

    logger.info("Veri: %s", data_path)
    df = pd.read_parquet(data_path)
    logger.info("Kayit: %d | Proje: %d", len(df),
                df["project_name"].nunique() if "project_name" in df.columns else 0)

    bug_col = f"bug_{args.bug_label}"
    if bug_col not in df.columns:
        alt = "bug_szz" if args.bug_label == "keyword" else "bug_keyword"
        if alt in df.columns:
            logger.warning("%s yok, %s kullaniliyor", bug_col, alt)
            bug_col = alt
        else:
            logger.error("Ne %s ne de %s mevcut.", bug_col, alt); return 1
    logger.info("Bug etiketi: %s (--bug-label %s)", bug_col, args.bug_label)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --models filtresi
    sel_models: Optional[list[str]] = None
    if args.models:
        req = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in req if m not in MODELS]
        if unknown:
            logger.error("Bilinmeyen model(ler): %s. Gecerli: %s", unknown, list(MODELS))
            return 1
        sel_models = req
        logger.info("Sadece secilen modeller kosturulacak: %s", sel_models)

    results_bug: dict = {}
    results_smell: dict = {}

    if not args.skip_bug:
        logger.info("\n=== BUG benchmark CV (%s) ===", bug_col)
        results_bug = run_cv_for_task(df, FEATURES_BUG, bug_col, "bug",
                                       skip_tpot=args.skip_tpot, skip_dl=args.skip_dl,
                                       models=sel_models)
    if not args.skip_smell:
        logger.info("\n=== SMELL benchmark CV ===")
        results_smell = run_cv_for_task(df, FEATURES_SMELL, "smell_binary", "smell",
                                         skip_tpot=args.skip_tpot, skip_dl=args.skip_dl,
                                         models=sel_models)

    # --merge-into: mevcut benchmark JSON ile birlestir (yeni kosulanlar eskileri ezer)
    if args.merge_into:
        if args.merge_into == "auto":
            existing = sorted(args.output_dir.glob("model_benchmark_cv_*.json"))
            merge_path = existing[-1] if existing else None
        else:
            merge_path = Path(args.merge_into)
        if merge_path and merge_path.exists():
            base = json.loads(merge_path.read_text(encoding="utf-8"))
            mb = dict(base.get("bug_results", {}));   mb.update(results_bug)
            ms = dict(base.get("smell_results", {})); ms.update(results_smell)
            results_bug, results_smell = mb, ms
            logger.info("Merge: %s ile birlestirildi (%d bug, %d smell model)",
                        merge_path.name, len(results_bug), len(results_smell))
        else:
            logger.warning("--merge-into hedefi bulunamadi (%s); merge atlandi.", args.merge_into)

    if results_bug or results_smell:
        fig_path = FIGURES_DIR / f"model_benchmark_{ts}.png"
        plot_benchmark(results_bug, results_smell, fig_path)

    output = {
        "created_at":     datetime.now().isoformat(),
        "source_dataset": str(data_path),
        "bug_label":      bug_col,   # paper tutarliligi — hibrit CV ayni etikette olmali
        "cv_folds":       N_CV_SPLITS,
        "models":         MODELS,
        "family":         FAMILY,
        "note":           "Otomatik kazanan secimi yok — hibrit base'i elle secilir.",
        "bug_results":    results_bug,
        "smell_results":  results_smell,
    }
    json_path = args.output_dir / f"model_benchmark_cv_{ts}.json"
    json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("JSON: %s", json_path)

    # Ozet tablo (F1 azalan, kazanan isareti YOK)
    print("\n" + "=" * 72)
    print("MODEL BENCHMARK CV SONUCLARI (5-fold) — hibrit base'i elle sec")
    print("=" * 72)
    for task, results in [("BUG", results_bug), ("SMELL", results_smell)]:
        if not results:
            continue
        print(f"\n{task}:")
        print(f"  {'model':12s} {'aile':8s} {'F1':>8s} {'±std':>7s} {'prec':>7s} {'rec':>7s}")
        print("  " + "-" * 56)
        for name, v in sorted(results.items(), key=lambda kv: -kv[1].get("f1_mean", -1)):
            if v.get("status") == "ok":
                sanity = " [sanity]" if v.get("sanity_only") else ""
                print(f"  {name:12s} {v.get('family','?'):8s} "
                      f"{v['f1_mean']:8.4f} {v['f1_std']:7.4f} "
                      f"{v['precision_mean']:7.4f} {v['recall_mean']:7.4f}{sanity}")
            else:
                err = (v.get("error") or v.get("reason") or "")[:40]
                print(f"  {name:12s} {'-':8s}  [{v.get('status','?')}] {err}")

    print("\nSonraki adim: tabloya bak, en iyi ML + en iyi AutoML'i SEC, sonra:")
    print("  python -m scripts.train_final --cv-folds 5 --tasks bug,smell --threshold-opt \\")
    print("    --stacking-base-bug <ml> --stacking-base-smell <ml> --stacking-automl <automl> \\")
    print(f"    --dataset {data_path}")
    print("Hibrit cv_summary'i bu tabloya eklemek icin: python analysis/04_model_comparison_chart.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
