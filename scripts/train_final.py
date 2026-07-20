"""
scripts/train_final.py — Final model egitimi CLI (F6).

CLI contract (PLAN §12.2):

    python -m scripts.train_final [OPTIONS]

Isleyis (PLAN §15 F6 DoD):
  1. En guncel `output/dataset_model_filtered_*.parquet`'i yukle (veya --dataset)
  2. Her gorev icin:
       - bug    → Stacking (RF + AutoGluon, LR meta) -> bug_rf_base + bug_ag_base + bug_meta_lr
       - smell  → RandomForestClassifier           -> smell_rf.joblib
     + scaler (StandardScaler) joblib'e yaz
  3. `feature_names.json` — egitilen gorev(ler) icin feature isimleri
  4. `project_stats.json` — pipeline.project_stats.compute_project_stats
  5. Sanity: app.predictor ile tek satir tahmin — modellerin yuklendigi
     dogrulanir.

NOT (V2.1): T1 commit standalone task'i kaldirildi. `label_commit` sutunu
dataset'te kalir ama bu script egitmez. `FEATURES_COMMIT` artik sadece
FEATURES_BUG/SMELL'in temel altkumesi.

Exit codes:
    0   basarili
    1   genel hata
    2   config hatasi
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from pipeline.config import (
    FEATURES_BUG,
    LOG_DATEFMT,
    LOG_FORMAT,
    LOGS_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    ensure_runtime_dirs,
)
from sklearn.model_selection import GroupKFold

from pipeline.model_utils import (
    TwoStageSplit,
    apply_smote_train_only,
    classification_metrics,
    extract_xy,
    fit_scaler,
    two_stage_split,
)
from pipeline.project_stats import write_project_stats

logger = logging.getLogger("train_final")


RANDOM_STATE = 42  # Default; --seed CLI ile override edilir
CLASS_WEIGHT = None  # Default; --class-weight CLI ile override edilir
THRESHOLD_OPT = False  # Default; --threshold-opt ile override edilir
# Task 3c: stacking base model + AutoML secimi (06 + 07 CV ciktilarindan)
STACKING_BASE_BUG   = "rf"        # rf | lightgbm | xgboost
STACKING_BASE_SMELL = "rf"        # rf | lightgbm | xgboost
STACKING_AUTOML     = "autogluon" # autogluon | h2o | tpot
TASK_CHOICES = ("bug", "smell")
AUTOGLUON_TIME_LIMIT = 600  # PLAN §4.3 notu


# ── CLI ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scripts.train_final",
        description="QualiGuard V2 — Final model egitimi (F6).",
    )
    p.add_argument("--dataset", type=Path, default=None,
                   help="Filtered parquet (varsayilan: en guncel dataset_model_filtered_*)")
    p.add_argument("--tasks", type=str, default="bug,smell",
                   help="Virgulle ayrilmis: bug,smell  (T1 commit V2.1'de kaldirildi)")
    p.add_argument("--bug-label", choices=("keyword", "szz"), default="szz",
                   help="Bug etiket kaynagi")
    p.add_argument("--smell-label", choices=("binary", "count"), default="binary",
                   help="Smell etiket turu (V2'de count Phase B, su an binary)")
    p.add_argument("--models-dir", type=Path, default=MODELS_DIR,
                   help="Model artifact cikti dizini")
    p.add_argument("--autogluon-time-limit", type=int, default=AUTOGLUON_TIME_LIMIT,
                   help="AutoGluon fit time budget (saniye)")
    p.add_argument("--smote", action="store_true",
                   help="SMOTE'u aktifle (default: KAPALI, organik veri V2.1)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (split + RF + LR). Variance analizi icin degistirilebilir.")
    p.add_argument("--max-projects", type=int, default=None,
                   help="Egitimden once dataset'i N projeye downsample et (seed-controlled).")
    p.add_argument("--cv-folds", type=int, default=1,
                   help="N>=2 ise GroupKFold(N) CV modu (mean+/-std raporlar, artifact yazmaz). "
                        "Default 1: tek-shot 70/15/15 production training.")
    p.add_argument("--class-weight", choices=("none", "balanced"), default="none",
                   help="RF base + LR meta + Smell RF icin class_weight ayari. "
                        "'balanced': azinlik sinifa loss'ta agirlik (SMOTE alternatifi).")
    p.add_argument("--threshold-opt", action="store_true",
                   help="Validation fold uzerinde F1-max threshold bul, test'e uygula. "
                        "CV modunda her fold icin ayri threshold; default 0.5 yerine optimal.")
    p.add_argument("--stacking-base-bug",
                   choices=("rf", "lightgbm", "xgboost"), default="rf",
                   help="Bug stacking base model. 06_ml_baseline_cv.py ciktisindan belirlenir.")
    p.add_argument("--stacking-base-smell",
                   choices=("rf", "lightgbm", "xgboost"), default="rf",
                   help="Smell standalone base model. 06_ml_baseline_cv.py ciktisindan belirlenir.")
    p.add_argument("--stacking-automl",
                   choices=("autogluon", "h2o", "tpot"), default="autogluon",
                   help="Stacking AutoML katmani. 07_automl_baseline_cv.py ciktisindan belirlenir.")
    p.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"),
                   default="INFO")
    p.add_argument("--dry-run", action="store_true",
                   help="Sadece config raporu; egitim yapma")
    return p


def _setup_logging(level: str, log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level),
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        handlers=handlers,
        force=True,
    )


def _validate(args: argparse.Namespace) -> list[str]:
    tasks = [t.strip().lower() for t in args.tasks.split(",") if t.strip()]
    invalid = [t for t in tasks if t not in TASK_CHOICES]
    if invalid:
        raise SystemExit(f"HATA: bilinmeyen task(lar): {invalid} (exit 2)")
    if args.autogluon_time_limit <= 0:
        raise SystemExit("HATA: --autogluon-time-limit pozitif olmali (exit 2)")
    return tasks


def _print_dry_run(args: argparse.Namespace, tasks: list[str]) -> None:
    print("=" * 60)
    print("  QualiGuard V2 — scripts.train_final  [--dry-run]")
    print("=" * 60)
    print(f"  dataset              : {args.dataset or '(son filtered)'}")
    print(f"  tasks                : {tasks}")
    print(f"  bug-label            : {args.bug_label}")
    print(f"  smell-label          : {args.smell_label}")
    print(f"  models-dir           : {args.models_dir}")
    print(f"  autogluon-time-limit : {args.autogluon_time_limit}s")
    print(f"  smote                : {'on (train-only)' if args.smote else 'off (organic)'}")
    print(f"  class-weight         : {args.class_weight}")
    print(f"  threshold-opt        : {'ON (F1-max on val/OOF)' if args.threshold_opt else 'OFF (default 0.5)'}")
    print(f"  seed                 : {args.seed}")
    print(f"  max-projects         : {args.max_projects or '(no downsample)'}")
    print(f"  cv-folds             : {args.cv_folds} {'(CV mode, no artifacts)' if args.cv_folds >= 2 else '(single split, save artifacts)'}")
    print(f"  stacking-base-bug    : {args.stacking_base_bug}")
    print(f"  stacking-base-smell  : {args.stacking_base_smell}")
    print(f"  stacking-automl      : {args.stacking_automl}")
    print(f"  log-level            : {args.log_level}")
    print("=" * 60)


# ── Veri yukleme ──────────────────────────────────────────────────

def _latest_filtered_parquet() -> Path | None:
    files = sorted(OUTPUT_DIR.glob("dataset_model_filtered_*.parquet"))
    return files[-1] if files else None


def load_training_frame(args: argparse.Namespace) -> pd.DataFrame:
    """
    --dataset verildiyse onu, yoksa en guncel filtered parquet'i yukle.

    Raises:
        FileNotFoundError: veri yoksa.
    """
    path = args.dataset or _latest_filtered_parquet()
    if path is None or not path.exists():
        raise FileNotFoundError(
            "Egitim verisi bulunamadi. Once scripts.collect + "
            "analysis/01_filter_categorize.py ile filtered parquet uretin."
        )
    logger.info("veri: %s", path)
    df = pd.read_parquet(path)
    logger.info("kayit sayisi: %d | proje: %d",
                len(df), df["project_name"].nunique() if "project_name" in df else 0)
    return df


def _resolve_label(task: str, args: argparse.Namespace) -> tuple[str, str]:
    """Task + args'a gore (label_col, variant_tag) dondur."""
    if task == "bug":
        if args.bug_label == "szz":
            return "bug_szz", "szz"
        return "bug_keyword", "keyword"
    if task == "smell":
        # Phase B (count) ileride regresyon olacak; V2'de binary egitilir.
        return "smell_binary", "p80"
    raise ValueError(f"Tanimsiz task: {task}")


def _features_for(task: str) -> tuple[str, ...]:
    if task == "smell":
        from pipeline.config import FEATURES_SMELL
        return FEATURES_SMELL
    return FEATURES_BUG


# ── Egitim ────────────────────────────────────────────────────────

def _prepare_splits(
    df: pd.DataFrame,
    features: tuple[str, ...],
    label_col: str,
    use_smote: bool,
) -> dict:
    """
    Project-based 70/15/15 two-stage split + scaler fit + opsiyonel SMOTE.

    F4 (Tantithamthavorn et al. 2017): train_dev bolumunde GroupKFold(5)
    cv iteratoru de dondurulur; ablation ve HP tuning icin kullanilabilir.
    """
    if label_col not in df.columns:
        raise KeyError(f"Etiket sutunu yok: {label_col}")
    if df[label_col].dropna().nunique() < 2:
        raise ValueError(f"Etiket tek sinifli, egitilemez: {label_col}")

    # F4 — two-stage split (Tantithamthavorn et al. 2017)
    split = two_stage_split(df, val_frac=0.15, test_frac=0.15, seed=RANDOM_STATE)
    train, val, test = split.train_dev, split.val, split.test

    Xtr, ytr = extract_xy(train, features, label_col)
    Xv,  yv  = extract_xy(val,   features, label_col)
    Xte, yte = extract_xy(test,  features, label_col)

    scaler = fit_scaler(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xv_s  = scaler.transform(Xv)
    Xte_s = scaler.transform(Xte)

    # GroupKFold cv iteratoru — train_dev icinde HP tuning / ablation (F4)
    # n_splits grup sayisindan fazla olamaz; cok az proje varsa devre disi.
    # ONEMLI: SMOTE'tan ONCE hesaplaniyor — aksi halde Xtr_s/ytr resampled
    # (sentetik veri eklenir) olur ama cv_groups orijinal proje listesinde
    # kalir → boyut uyumsuzlugu. Fold-icinde SMOTE isteyen caller orada uygular.
    n_cv_groups = train["project_name"].nunique() if "project_name" in train.columns else 0
    n_cv_splits = min(5, n_cv_groups)
    if n_cv_splits >= 2:
        cv        = GroupKFold(n_splits=n_cv_splits)
        cv_groups = train["project_name"].to_numpy()
        cv_iter   = list(cv.split(Xtr_s, ytr, groups=cv_groups))
    else:
        cv_iter = []

    smote_note = ""
    if use_smote:
        try:
            Xtr_s, ytr = apply_smote_train_only(Xtr_s, ytr, RANDOM_STATE)
            smote_note = f"SMOTE sonrasi train boyutu: {len(ytr)}"
        except ImportError:
            smote_note = "imblearn yok — SMOTE atlandi"
            logger.warning(smote_note)
        except ValueError as exc:
            smote_note = f"SMOTE atlandi: {exc}"
            logger.warning(smote_note)

    return {
        "scaler":   scaler,
        "Xtr":      Xtr_s,  "ytr":  ytr,
        "Xv":       Xv_s,   "yv":   yv,
        "Xte":      Xte_s,  "yte":  yte,
        "n_train":  len(ytr), "n_val": len(yv), "n_test": len(yte),
        "note":     smote_note,
        "cv_iter":  cv_iter,   # GroupKFold(5) splits — ablation icin
        "split":    split,     # TwoStageSplit — proje ID'leri dahil
    }


def _save_feature_names(
    models_dir: Path,
    tasks: list[str],
    existing: dict | None = None,
) -> None:
    """feature_names.json — seçilen task'larin sutun listelerini yaz."""
    names = dict(existing or {})
    for t in tasks:
        names[t] = list(_features_for(t))
    path = models_dir / "feature_names.json"
    path.write_text(json.dumps(names, indent=2), encoding="utf-8")
    logger.info("yazildi: %s", path)


def _args_stub(bug: str = "szz", smell: str = "binary") -> argparse.Namespace:
    """_resolve_label'in ihtiyaci olan alanlari mock eden kucuk namespace."""
    ns = argparse.Namespace()
    ns.bug_label = bug
    ns.smell_label = smell
    return ns


def _make_base_clf(model_name: str, n_neg: int = 1, n_pos: int = 1):
    """
    Stacking / standalone base ML model fabrikasi.

    model_name: "rf" | "lightgbm" | "xgboost"
    n_neg / n_pos: XGBoost scale_pos_weight hesabi icin.
    CLASS_WEIGHT global'i RF ve LR icin kullanilir; LightGBM
    is_unbalance=True ile kendi dengeleme mekanizmasini kullanir.
    """
    cw = CLASS_WEIGHT
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=400, n_jobs=-1,
            random_state=RANDOM_STATE, class_weight=cw,
        )
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(f"LightGBM kurulu degil: {exc}") from exc
        return LGBMClassifier(
            n_estimators=400, num_leaves=31, n_jobs=-1,
            is_unbalance=True, verbose=-1, random_state=RANDOM_STATE,
        )
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(f"XGBoost kurulu degil: {exc}") from exc
        scale_pos = float(n_neg) / max(n_pos, 1)
        return XGBClassifier(
            n_estimators=400, n_jobs=-1, scale_pos_weight=scale_pos,
            eval_metric="logloss", random_state=RANDOM_STATE,
            tree_method="hist",
        )
    raise ValueError(f"Bilinmeyen stacking_base: {model_name!r}")


def _automl_oof_fold(
    Xtr_fold: np.ndarray,
    ytr_fold: np.ndarray,
    Xva_fold: np.ndarray,
    feat_cols: list[str],
    fold_idx: int,
    tmp_dir: Path,
    time_limit: int,
) -> np.ndarray:
    """Tek fold icin AutoML OOF olasililari. STACKING_AUTOML global'ine gore dallanir."""
    if STACKING_AUTOML == "autogluon":
        from autogluon.tabular import TabularPredictor  # type: ignore[import-not-found]
        tr_df = pd.DataFrame(Xtr_fold, columns=feat_cols)
        tr_df["target"] = ytr_fold
        va_df = pd.DataFrame(Xva_fold, columns=feat_cols)
        ag_tmp = tmp_dir / f".ag_fold_{fold_idx}"
        p = TabularPredictor(
            label="target", path=str(ag_tmp),
            problem_type="binary", verbosity=0,
        ).fit(tr_df, time_limit=time_limit)
        proba = p.predict_proba(va_df).iloc[:, 1].to_numpy()
        return proba

    if STACKING_AUTOML == "h2o":
        try:
            import h2o  # type: ignore[import-not-found]
            from h2o.automl import H2OAutoML  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(f"H2O kurulu degil: {exc}") from exc
        tr_df = pd.DataFrame(Xtr_fold, columns=feat_cols)
        tr_df["target"] = ytr_fold.astype(str)
        va_df = pd.DataFrame(Xva_fold, columns=feat_cols)
        h_train = h2o.H2OFrame(tr_df)
        h_val   = h2o.H2OFrame(va_df)
        h_train["target"] = h_train["target"].asfactor()
        aml = H2OAutoML(
            max_runtime_secs=min(time_limit, 300), max_models=15,
            seed=RANDOM_STATE, verbosity="warn",
        )
        aml.train(x=feat_cols, y="target", training_frame=h_train)
        preds = aml.leader.predict(h_val).as_data_frame()
        return preds["p1"].values if "p1" in preds.columns else np.zeros(len(Xva_fold))

    if STACKING_AUTOML == "tpot":
        try:
            from tpot import TPOTClassifier  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(f"TPOT kurulu degil: {exc}") from exc
        tpot = TPOTClassifier(
            search_space="linear-light", scorers=["f1_weighted"],
            scorers_weights=[1], max_time_mins=min(10, time_limit // 60),
            max_eval_time_mins=1, n_jobs=1, verbose=0,
            random_state=RANDOM_STATE,
        )
        tpot.fit(Xtr_fold, ytr_fold)
        return tpot.predict_proba(Xva_fold)[:, 1]

    raise ValueError(f"Bilinmeyen stacking_automl: {STACKING_AUTOML!r}")


def _automl_refit_save(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    feat_cols: list[str],
    artifact_dir: Path,
    time_limit: int,
) -> object:
    """
    Tam train seti uzerinde AutoML refit ve kalici artifact yaz.

    Returns fitted model (predicting icin lazim).
    Artifact: autogluon → artifact_dir/ (dir), h2o → artifact_dir/ (dir), tpot → artifact_dir.pkl
    """
    if STACKING_AUTOML == "autogluon":
        from autogluon.tabular import TabularPredictor  # type: ignore[import-not-found]
        tr_df = pd.DataFrame(Xtr, columns=feat_cols)
        tr_df["target"] = ytr
        if artifact_dir.exists():
            import shutil
            shutil.rmtree(artifact_dir)
        ag = TabularPredictor(
            label="target", path=str(artifact_dir),
            problem_type="binary", verbosity=0,
        ).fit(tr_df, time_limit=time_limit)
        return ag

    if STACKING_AUTOML == "h2o":
        try:
            import h2o  # type: ignore[import-not-found]
            from h2o.automl import H2OAutoML  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(f"H2O kurulu degil: {exc}") from exc
        tr_df = pd.DataFrame(Xtr, columns=feat_cols)
        tr_df["target"] = ytr.astype(str)
        h_train = h2o.H2OFrame(tr_df)
        h_train["target"] = h_train["target"].asfactor()
        aml = H2OAutoML(
            max_runtime_secs=time_limit, max_models=20,
            seed=RANDOM_STATE, verbosity="warn",
        )
        aml.train(x=feat_cols, y="target", training_frame=h_train)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        h2o.save_model(aml.leader, path=str(artifact_dir), force=True)
        return aml

    if STACKING_AUTOML == "tpot":
        try:
            from tpot import TPOTClassifier  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(f"TPOT kurulu degil: {exc}") from exc
        tpot = TPOTClassifier(
            search_space="linear-light", scorers=["f1_weighted"],
            scorers_weights=[1], max_time_mins=min(10, time_limit // 60),
            max_eval_time_mins=1, n_jobs=1, verbose=0,
            random_state=RANDOM_STATE,
        )
        tpot.fit(Xtr, ytr)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(tpot.fitted_pipeline_, artifact_dir / "tpot_pipeline.pkl")
        return tpot

    raise ValueError(f"Bilinmeyen stacking_automl: {STACKING_AUTOML!r}")


def _automl_predict_proba(automl_model, Xte: np.ndarray, feat_cols: list[str]) -> np.ndarray:
    """Test seti icin AutoML olasiliklari. automl_model tipine gore dallanir."""
    if STACKING_AUTOML == "autogluon":
        te_df = pd.DataFrame(Xte, columns=feat_cols)
        pred = automl_model.predict_proba(te_df)
        if isinstance(pred, pd.DataFrame):
            return pred.iloc[:, -1].to_numpy()
        return np.array(pred)[:, 1]

    if STACKING_AUTOML == "h2o":
        import h2o  # type: ignore[import-not-found]
        te_df = pd.DataFrame(Xte, columns=feat_cols)
        h_te = h2o.H2OFrame(te_df)
        preds = automl_model.leader.predict(h_te).as_data_frame()
        return preds["p1"].values if "p1" in preds.columns else np.zeros(len(Xte))

    if STACKING_AUTOML == "tpot":
        return automl_model.predict_proba(Xte)[:, 1]

    raise ValueError(f"Bilinmeyen stacking_automl: {STACKING_AUTOML!r}")


def train_bug(
    df: pd.DataFrame,
    models_dir: Path,
    bug_label: str,
    use_smote: bool,
    autogluon_time_limit: int,
) -> dict:
    """T2 — Stacking (RF + AutoGluon -> LR meta, 3-fold OOF)."""
    label_col, variant = _resolve_label("bug", _args_stub(bug=bug_label))
    features = _features_for("bug")
    parts = _prepare_splits(df, features, label_col, use_smote)
    logger.info("T2 bug — %s | %s", label_col, parts["note"] or "smote uygulandi/yok")

    Xtr = parts["Xtr"]
    ytr = parts["ytr"]
    Xte = parts["Xte"]

    # AutoML bagimliligi kontrol — varsayilan autogluon
    if STACKING_AUTOML == "autogluon":
        try:
            from autogluon.tabular import TabularPredictor  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                f"AutoGluon kurulu degil ({exc}). T2 bug stacking icin gereklidir. "
                "`pip install autogluon.tabular` veya --stacking-automl h2o|tpot kullanin."
            )
    elif STACKING_AUTOML in ("h2o", "tpot"):
        pass  # _automl_* helper'lari ilgili import'u kendi icinde yapar

    n_neg_tr = int((ytr == 0).sum())
    n_pos_tr = int((ytr == 1).sum())
    feat_cols = list(features)
    kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    base_oof = np.zeros(len(Xtr))
    ag_oof   = np.zeros(len(Xtr))

    t0 = time.monotonic()

    # H2O bir kere baslatilir; her fold sonunda bellek temizlenir
    if STACKING_AUTOML == "h2o":
        try:
            import h2o  # type: ignore[import-not-found]
            h2o.init(nthreads=-1, max_mem_size="4G", verbose=False)
        except ImportError as exc:
            raise RuntimeError(f"H2O kurulu degil: {exc}") from exc

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(Xtr), 1):
        logger.info("T2 stacking fold %d/3 — OOF uretimi [base=%s, automl=%s]",
                    fold_idx, STACKING_BASE_BUG, STACKING_AUTOML)
        base = _make_base_clf(STACKING_BASE_BUG, n_neg_tr, n_pos_tr)
        base.fit(Xtr[tr_idx], ytr[tr_idx])
        base_oof[va_idx] = base.predict_proba(Xtr[va_idx])[:, 1]

        ag_oof[va_idx] = _automl_oof_fold(
            Xtr[tr_idx], ytr[tr_idx], Xtr[va_idx],
            feat_cols, fold_idx, models_dir, autogluon_time_limit,
        )
        if STACKING_AUTOML == "h2o":
            import h2o as _h2o  # type: ignore[import-not-found]
            _h2o.remove_all()

    # Meta-learner: OOF uzerinde kalibre LR (F5 — isotonic calibration)
    meta_base = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    meta = CalibratedClassifierCV(meta_base, method="isotonic", cv=3)
    meta.fit(np.c_[base_oof, ag_oof], ytr)

    # Base modelleri tum train uzerine refit
    logger.info("T2 base refit — %s + %s", STACKING_BASE_BUG, STACKING_AUTOML)
    base_full = _make_base_clf(STACKING_BASE_BUG, n_neg_tr, n_pos_tr)
    base_full.fit(Xtr, ytr)

    automl_artifact_dir = models_dir / f"bug_{STACKING_AUTOML}_base"
    automl_model = _automl_refit_save(
        Xtr, ytr, feat_cols, automl_artifact_dir, autogluon_time_limit,
    )

    # Test tahminleri — H2O cluster hala ayakta olmali; shutdown tahminden ONCE
    # cagrilirsa _automl_predict_proba'da h2o.H2OFrame() "Connection was closed"
    # hatasi atiyor (model artifact zaten _automl_refit_save sirasinda diske yazildi).
    base_test = base_full.predict_proba(Xte)[:, 1]
    ag_test   = _automl_predict_proba(automl_model, Xte, feat_cols)

    # H2O: tum tahminler bittikten SONRA JVM kapat
    if STACKING_AUTOML == "h2o":
        import h2o as _h2o  # type: ignore[import-not-found]
        _h2o.cluster().shutdown(prompt=False)

    y_proba   = meta.predict_proba(np.c_[base_test, ag_test])[:, 1]
    y_pred    = (y_proba >= 0.5).astype("int64")
    dur = time.monotonic() - t0

    metrics = classification_metrics(parts["yte"], y_pred, y_proba)
    logger.info("T2 metrics: %s (toplam fit %.1fs)", metrics, dur)

    # Kalici artifact'lar
    base_artifact = models_dir / f"bug_{STACKING_BASE_BUG}_base.joblib"
    joblib.dump(base_full,        base_artifact)
    joblib.dump(meta,             models_dir / "bug_meta_lr.joblib")
    joblib.dump(parts["scaler"],  models_dir / "scaler_bug.joblib")

    # Fold-temp'leri temizle (AutoGluon OOF temp dirs)
    import shutil
    for fold_idx in range(1, 4):
        tmp = models_dir / f".ag_fold_{fold_idx}"
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)

    return {"task": "bug", "label": label_col, "variant": variant,
            "metrics": metrics, "duration_secs": round(dur, 2),
            "n_train": parts["n_train"], "n_test": parts["n_test"],
            "stacking_base": STACKING_BASE_BUG, "stacking_automl": STACKING_AUTOML}


def train_smell(
    df: pd.DataFrame,
    models_dir: Path,
    smell_label: str,
    use_smote: bool,
) -> dict:
    """T3 — RandomForestClassifier (binary, Phase A)."""
    label_col, variant = _resolve_label("smell", _args_stub(smell=smell_label))
    features = _features_for("smell")
    parts = _prepare_splits(df, features, label_col, use_smote)
    logger.info("T3 smell — %s | %s", label_col, parts["note"] or "smote uygulandi/yok")

    n_neg_tr = int((parts["ytr"] == 0).sum())
    n_pos_tr = int((parts["ytr"] == 1).sum())
    clf = _make_base_clf(STACKING_BASE_SMELL, n_neg_tr, n_pos_tr)
    t0 = time.monotonic()
    clf.fit(parts["Xtr"], parts["ytr"])
    dur = time.monotonic() - t0

    y_pred  = clf.predict(parts["Xte"])
    y_proba = clf.predict_proba(parts["Xte"])[:, 1]
    metrics = classification_metrics(parts["yte"], y_pred, y_proba)
    logger.info("T3 metrics: %s (fit %.1fs)", metrics, dur)

    smell_artifact = models_dir / f"smell_{STACKING_BASE_SMELL}.joblib"
    joblib.dump(clf,             smell_artifact)
    joblib.dump(parts["scaler"], models_dir / "scaler_smell.joblib")
    return {"task": "smell", "label": label_col, "variant": variant,
            "metrics": metrics, "duration_secs": round(dur, 2),
            "n_train": parts["n_train"], "n_test": parts["n_test"],
            "base_model": STACKING_BASE_SMELL}


# ── CV Evaluation (V2.1) ─────────────────────────────────────────
# GroupKFold(N) modu — mean+/-std raporlar, artifact yazmaz.
# Tek-shot 70/15/15 split'in yuksek varyansi (16pp) goz onunde bulundurularak
# Tantithamthavorn 2017, Yatish 2019 standartlarina uyum icin eklenmistir.

def _find_optimal_threshold(
    y_true: np.ndarray, y_proba: np.ndarray,
) -> tuple[float, float]:
    """
    F1-maksimize eden threshold'u tara (0.05..0.95, 0.01 adim).

    Returns:
        (best_threshold, best_f1)
    """
    from sklearn.metrics import f1_score
    if len(y_true) == 0 or y_proba.size == 0:
        return 0.5, 0.0
    thresholds = np.linspace(0.05, 0.95, 91)
    f1s = np.array([
        f1_score(y_true, (y_proba >= t).astype("int64"), zero_division=0)
        for t in thresholds
    ])
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])


def _eval_bug_fold(
    Xtr, ytr, Xte, yte,
    features, autogluon_time_limit, tmp_dir,
) -> dict:
    """
    Tek-fold T2 bug stacking. Artifact yazmaz, sadece metrik.

    Eger THRESHOLD_OPT=True ise stacking icin OOF predictions uzerinde
    F1-maksimize eden threshold bulur (unbiased — meta-learner OOF).
    """
    feat_cols = list(features)
    kf_inner = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    n_neg = int((ytr == 0).sum())
    n_pos = int((ytr == 1).sum())
    base_oof = np.zeros(len(Xtr))
    ag_oof   = np.zeros(len(Xtr))

    if STACKING_AUTOML == "h2o":
        try:
            import h2o  # type: ignore[import-not-found]
            h2o.init(nthreads=-1, max_mem_size="4G", verbose=False)
        except ImportError as exc:
            raise RuntimeError(f"H2O kurulu degil: {exc}") from exc

    for fold_idx, (tr_idx, va_idx) in enumerate(kf_inner.split(Xtr), 1):
        base = _make_base_clf(STACKING_BASE_BUG, n_neg, n_pos)
        base.fit(Xtr[tr_idx], ytr[tr_idx])
        base_oof[va_idx] = base.predict_proba(Xtr[va_idx])[:, 1]

        ag_oof[va_idx] = _automl_oof_fold(
            Xtr[tr_idx], ytr[tr_idx], Xtr[va_idx],
            feat_cols, fold_idx, tmp_dir, autogluon_time_limit,
        )
        if STACKING_AUTOML == "h2o":
            import h2o as _h2o  # type: ignore[import-not-found]
            _h2o.remove_all()

    meta_base = LogisticRegression(
        max_iter=2000, random_state=RANDOM_STATE,
        class_weight=CLASS_WEIGHT,
    )
    meta = CalibratedClassifierCV(meta_base, method="isotonic", cv=3)
    meta.fit(np.c_[base_oof, ag_oof], ytr)

    # Threshold optimizasyonu: meta'nin OOF tahminleri uzerinde (unbiased)
    opt_threshold = 0.5
    if THRESHOLD_OPT:
        meta_oof_proba = meta.predict_proba(np.c_[base_oof, ag_oof])[:, 1]
        opt_threshold, _ = _find_optimal_threshold(ytr, meta_oof_proba)

    base_full = _make_base_clf(STACKING_BASE_BUG, n_neg, n_pos)
    base_full.fit(Xtr, ytr)

    ag_fold_dir = tmp_dir / ".ag_base"
    automl_fold = _automl_refit_save(
        Xtr, ytr, feat_cols, ag_fold_dir, autogluon_time_limit,
    )

    # Test tahminleri — H2O cluster hala ayakta olmali; shutdown tahminden ONCE
    # cagrilirsa _automl_predict_proba'da h2o.H2OFrame() "Connection was closed"
    # hatasi atiyordu (her fold bu yuzden basarisiz oluyordu).
    base_test = base_full.predict_proba(Xte)[:, 1]
    ag_test   = _automl_predict_proba(automl_fold, Xte, feat_cols)

    if STACKING_AUTOML == "h2o":
        import h2o as _h2o  # type: ignore[import-not-found]
        _h2o.cluster().shutdown(prompt=False)

    y_proba   = meta.predict_proba(np.c_[base_test, ag_test])[:, 1]

    # Hem default (0.5) hem optimal threshold ile metrik
    y_pred_default = (y_proba >= 0.5).astype("int64")
    metrics_default = classification_metrics(yte, y_pred_default, y_proba)
    metrics_default["_threshold"] = 0.5

    if THRESHOLD_OPT and abs(opt_threshold - 0.5) > 1e-6:
        y_pred_opt = (y_proba >= opt_threshold).astype("int64")
        metrics_opt = classification_metrics(yte, y_pred_opt, y_proba)
        metrics_opt["_threshold"] = opt_threshold
    else:
        metrics_opt = dict(metrics_default)

    # Cleanup
    import shutil
    for d in [ag_fold_dir,
              tmp_dir / ".ag_inner_1",
              tmp_dir / ".ag_inner_2",
              tmp_dir / ".ag_inner_3"]:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)

    return {"default": metrics_default, "optimal": metrics_opt}


def _eval_smell_fold(
    Xtr, ytr, Xte, yte,
    project_groups_tr: np.ndarray | None = None,
) -> dict:
    """
    Tek-fold T3 smell RF. Artifact yazmaz, sadece metrik.

    Threshold opt icin: outer-train'i project-based 90/10 split,
    inner-val uzerinde optimal threshold ara (unbiased).
    """
    opt_threshold = 0.5
    if THRESHOLD_OPT and project_groups_tr is not None:
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(
            n_splits=1, test_size=0.15, random_state=RANDOM_STATE,
        )
        try:
            tr_inner_idx, val_inner_idx = next(
                gss.split(Xtr, ytr, groups=project_groups_tr)
            )
            clf_inner = RandomForestClassifier(
                n_estimators=400, n_jobs=-1, random_state=RANDOM_STATE,
                class_weight=CLASS_WEIGHT,
            )
            clf_inner.fit(Xtr[tr_inner_idx], ytr[tr_inner_idx])
            val_proba = clf_inner.predict_proba(Xtr[val_inner_idx])[:, 1]
            opt_threshold, _ = _find_optimal_threshold(
                ytr[val_inner_idx], val_proba,
            )
        except (ValueError, StopIteration) as exc:
            logger.warning("Smell threshold opt atlandi: %s", exc)

    # Refit full outer-train, evaluate test
    clf = RandomForestClassifier(
        n_estimators=400, n_jobs=-1, random_state=RANDOM_STATE,
        class_weight=CLASS_WEIGHT,
    )
    clf.fit(Xtr, ytr)
    y_proba = clf.predict_proba(Xte)[:, 1]

    y_pred_default = (y_proba >= 0.5).astype("int64")
    metrics_default = classification_metrics(yte, y_pred_default, y_proba)
    metrics_default["_threshold"] = 0.5

    if THRESHOLD_OPT and abs(opt_threshold - 0.5) > 1e-6:
        y_pred_opt = (y_proba >= opt_threshold).astype("int64")
        metrics_opt = classification_metrics(yte, y_pred_opt, y_proba)
        metrics_opt["_threshold"] = opt_threshold
    else:
        metrics_opt = dict(metrics_default)

    return {"default": metrics_default, "optimal": metrics_opt}


def _summarize_folds(metrics_list: list[dict]) -> dict | None:
    """Per-fold metrik listesinden mean+/-std ozetler."""
    if not metrics_list:
        return None
    keys = [k for k in metrics_list[0].keys() if not k.startswith("_")]
    summary = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if k in m]
        if vals:
            summary[k] = {
                "mean": round(float(np.mean(vals)), 4),
                "std":  round(float(np.std(vals)), 4),
                "vals": [round(float(v), 4) for v in vals],
            }
    return summary


def run_cv(df: pd.DataFrame, args, tasks: list[str]) -> int:
    """
    GroupKFold(N) cross-validation modu.

    Her fold:
      - Projeleri N gruba bol, sirayla her birini test olarak kullan
      - Train fold'larinda model egit, test fold'da degerlendir
      - Artifact yazma — sadece metrik topla

    Sonunda her metrik icin mean+/-std raporlar.
    Cikis: cv_summary_<timestamp>.json (her fold metrigi + ozet).
    """
    from sklearn.model_selection import GroupKFold

    if "project_name" not in df.columns:
        logger.error("project_name sutunu yok, GroupKFold uygulanamiyor.")
        return 1

    n_groups = df["project_name"].nunique()
    if n_groups < args.cv_folds:
        logger.error(
            "Yeterli proje yok: %d proje vs %d fold istendi.",
            n_groups, args.cv_folds,
        )
        return 1

    groups = df["project_name"].to_numpy()
    gkf = GroupKFold(n_splits=args.cv_folds)
    fold_indices = list(gkf.split(df, groups=groups))

    bug_features = _features_for("bug")
    smell_features = _features_for("smell")
    bug_label_col, _ = _resolve_label("bug", args)
    smell_label_col, _ = _resolve_label("smell", args)

    bug_metrics: list[dict] = []
    smell_metrics: list[dict] = []

    tmp_dir = args.models_dir / ".cv_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    overall_t0 = time.monotonic()

    for fold_idx, (tr_indices, te_indices) in enumerate(fold_indices, 1):
        train_df = df.iloc[tr_indices].reset_index(drop=True)
        test_df  = df.iloc[te_indices].reset_index(drop=True)

        n_train_proj = train_df["project_name"].nunique()
        n_test_proj  = test_df["project_name"].nunique()
        logger.info(
            "=== Fold %d/%d — train: %d proj/%d files | test: %d proj/%d files ===",
            fold_idx, args.cv_folds,
            n_train_proj, len(train_df),
            n_test_proj,  len(test_df),
        )

        if "bug" in tasks:
            try:
                Xtr_raw, ytr = extract_xy(train_df, list(bug_features), bug_label_col)
                Xte_raw, yte = extract_xy(test_df,  list(bug_features), bug_label_col)
                scaler_b = fit_scaler(Xtr_raw)
                Xtr_s = scaler_b.transform(Xtr_raw)
                Xte_s = scaler_b.transform(Xte_raw)

                if args.smote:
                    try:
                        Xtr_s, ytr = apply_smote_train_only(Xtr_s, ytr, RANDOM_STATE)
                    except (ImportError, ValueError) as exc:
                        logger.warning("Fold %d SMOTE atlandi: %s", fold_idx, exc)

                t0 = time.monotonic()
                fold_result = _eval_bug_fold(
                    Xtr_s, ytr, Xte_s, yte,
                    bug_features, args.autogluon_time_limit, tmp_dir,
                )
                dur = time.monotonic() - t0
                # fold_result: {"default": metrics, "optimal": metrics}
                # Merge into a flat record for both
                merged = {
                    "_fold":     fold_idx,
                    "_n_train":  int(len(ytr)),
                    "_n_test":   int(len(yte)),
                    "_dur_secs": round(dur, 2),
                    "default":   fold_result["default"],
                    "optimal":   fold_result["optimal"],
                }
                bug_metrics.append(merged)
                m_def = fold_result["default"]
                m_opt = fold_result["optimal"]
                logger.info(
                    "Fold %d T2 bug: default f1=%.4f mcc=%.4f | opt(t=%.2f) f1=%.4f mcc=%.4f (fit %.1fs)",
                    fold_idx,
                    m_def["f1"], m_def["mcc"],
                    m_opt["_threshold"], m_opt["f1"], m_opt["mcc"], dur,
                )
            except RuntimeError as exc:
                logger.error("Fold %d T2 bug atlandi: %s", fold_idx, exc)
            except Exception as exc:  # noqa: BLE001
                logger.error("Fold %d T2 bug hatasi: %s", fold_idx, exc)

        if "smell" in tasks:
            Xtr_raw, ytr = extract_xy(train_df, list(smell_features), smell_label_col)
            Xte_raw, yte = extract_xy(test_df,  list(smell_features), smell_label_col)
            scaler_s = fit_scaler(Xtr_raw)
            Xtr_s = scaler_s.transform(Xtr_raw)
            Xte_s = scaler_s.transform(Xte_raw)

            # Threshold opt icin proje gruplari
            project_groups_tr = train_df["project_name"].to_numpy()

            if args.smote:
                try:
                    Xtr_s, ytr = apply_smote_train_only(Xtr_s, ytr, RANDOM_STATE)
                    # SMOTE sonrasi project_groups artik gecerli degil
                    project_groups_tr = None
                except (ImportError, ValueError) as exc:
                    logger.warning("Fold %d SMOTE atlandi: %s", fold_idx, exc)

            t0 = time.monotonic()
            fold_result = _eval_smell_fold(
                Xtr_s, ytr, Xte_s, yte,
                project_groups_tr=project_groups_tr,
            )
            dur = time.monotonic() - t0
            merged = {
                "_fold":     fold_idx,
                "_n_train":  int(len(ytr)),
                "_n_test":   int(len(yte)),
                "_dur_secs": round(dur, 2),
                "default":   fold_result["default"],
                "optimal":   fold_result["optimal"],
            }
            smell_metrics.append(merged)
            m_def = fold_result["default"]
            m_opt = fold_result["optimal"]
            logger.info(
                "Fold %d T3 smell: default f1=%.4f mcc=%.4f | opt(t=%.2f) f1=%.4f mcc=%.4f (fit %.1fs)",
                fold_idx,
                m_def["f1"], m_def["mcc"],
                m_opt["_threshold"], m_opt["f1"], m_opt["mcc"], dur,
            )

    # Cleanup
    import shutil
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    overall_dur = time.monotonic() - overall_t0

    # Aggregate raporlar
    logger.info("=" * 70)
    logger.info("== CV SUMMARY — %d-fold GroupKFold (toplam %.1fs) ==", args.cv_folds, overall_dur)
    logger.info("=" * 70)

    def _log_summary(metrics_list: list[dict], task_name: str) -> None:
        if not metrics_list:
            logger.info("%s: hicbir fold sonucu yok.", task_name)
            return
        logger.info("%s (%d fold):", task_name, len(metrics_list))
        keys = ["f1", "f1_weighted", "f1_macro", "accuracy", "mcc", "pr_auc"]

        # 'default' (threshold=0.5) ozeti
        logger.info("  --- DEFAULT threshold (0.5) ---")
        for k in keys:
            vals = [m["default"][k] for m in metrics_list if k in m.get("default", {})]
            if not vals:
                continue
            mn = float(np.mean(vals))
            sd = float(np.std(vals))
            vals_str = ", ".join(f"{v:.4f}" for v in vals)
            logger.info("    %-13s = %.4f +/- %.4f   [%s]", k, mn, sd, vals_str)

        # 'optimal' threshold ozeti (eger threshold_opt aktifse anlamli)
        if THRESHOLD_OPT:
            logger.info("  --- OPTIMAL threshold (F1-max on val/OOF) ---")
            thresholds = [m["optimal"].get("_threshold", 0.5) for m in metrics_list]
            logger.info("    %-13s = mean=%.3f, vals=[%s]", "threshold",
                        float(np.mean(thresholds)),
                        ", ".join(f"{t:.2f}" for t in thresholds))
            for k in keys:
                vals = [m["optimal"][k] for m in metrics_list if k in m.get("optimal", {})]
                if not vals:
                    continue
                mn = float(np.mean(vals))
                sd = float(np.std(vals))
                vals_str = ", ".join(f"{v:.4f}" for v in vals)
                logger.info("    %-13s = %.4f +/- %.4f   [%s]", k, mn, sd, vals_str)

    _log_summary(bug_metrics,   "T2 bug")
    _log_summary(smell_metrics, "T3 smell")

    # Default + optimal ayri ayri summarize
    def _flatten_to_default(metrics_list: list[dict]) -> list[dict]:
        return [m["default"] for m in metrics_list if "default" in m]

    def _flatten_to_optimal(metrics_list: list[dict]) -> list[dict]:
        return [m["optimal"] for m in metrics_list if "optimal" in m]

    # JSON dump
    summary = {
        "created_at":      datetime.now().isoformat(),
        "cv_folds":        args.cv_folds,
        "seed":            int(args.seed),
        "smote":           bool(args.smote),
        "class_weight":    str(args.class_weight),
        "threshold_opt":   bool(args.threshold_opt),
        "dataset_size":    int(len(df)),
        "n_projects":      int(df["project_name"].nunique()),
        "tasks":           tasks,
        # NOT: cv_summary'nin hangi etiket/config ile uretildigini kayda gecir.
        # Eksikligi gecmiste SZZ vs keyword kosumlarinin karistirilmasina yol acti.
        "bug_label":           bug_label_col,
        "smell_label":         smell_label_col,
        "stacking_base_bug":   STACKING_BASE_BUG,
        "stacking_base_smell": STACKING_BASE_SMELL,
        "stacking_automl":     STACKING_AUTOML,
        "bug_per_fold":    bug_metrics,
        "smell_per_fold":  smell_metrics,
        "bug_summary_default":   _summarize_folds(_flatten_to_default(bug_metrics)),
        "bug_summary_optimal":   _summarize_folds(_flatten_to_optimal(bug_metrics)),
        "smell_summary_default": _summarize_folds(_flatten_to_default(smell_metrics)),
        "smell_summary_optimal": _summarize_folds(_flatten_to_optimal(smell_metrics)),
        "total_secs":     round(overall_dur, 1),
    }
    out_path = args.models_dir / f"cv_summary_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("CV summary yazildi: %s", out_path)

    return 0


# ── Sanity test ──────────────────────────────────────────────────

def run_sanity(df: pd.DataFrame, trained_tasks: list[str]) -> bool:
    """
    predictor modulunu reload et, egittigimiz gorevler icin 1 satir tahmin uret.
    True = OK; False = warning (asla exception kaldirmaz).
    """
    try:
        import importlib
        from app import predictor as predictor_mod
        importlib.reload(predictor_mod)

        sample = df.head(1).copy()
        if "bug" in trained_tasks and predictor_mod.models_ready():
            pr, pb = predictor_mod.predict_bug(sample)
            logger.info("sanity T2 bug:    pred=%s proba=%.3f", pr[0], float(pb[0]))
        if "smell" in trained_tasks and predictor_mod.smell_available():
            pr, pb = predictor_mod.predict_smell(sample)
            logger.info("sanity T3 smell:  pred=%s proba=%.3f", pr[0], float(pb[0]))
        return True
    except Exception as exc:  # noqa: BLE001 — sanity hata olsa bile egitim cikmaz
        logger.warning("sanity test basarisiz: %s", exc)
        return False


# ── Entry point ───────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    tasks = _validate(args)

    if args.dry_run:
        _print_dry_run(args, tasks)
        return 0

    ensure_runtime_dirs()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    log_path = LOGS_DIR / f"train_final_{ts}.log"
    _setup_logging(args.log_level, log_path)

    logger.info("== F6 train_final baslangic — tasks=%s ==", tasks)
    logger.info("log: %s", log_path)

    # V2.1: variance analizi icin seed CLI'dan ayarlanabilir
    global RANDOM_STATE, CLASS_WEIGHT, THRESHOLD_OPT
    global STACKING_BASE_BUG, STACKING_BASE_SMELL, STACKING_AUTOML
    RANDOM_STATE        = int(args.seed)
    CLASS_WEIGHT        = None if args.class_weight == "none" else args.class_weight
    THRESHOLD_OPT       = bool(args.threshold_opt)
    STACKING_BASE_BUG   = args.stacking_base_bug
    STACKING_BASE_SMELL = args.stacking_base_smell
    STACKING_AUTOML     = args.stacking_automl
    logger.info(
        "seed: %d | class_weight: %s | threshold_opt: %s",
        RANDOM_STATE, CLASS_WEIGHT or "none", THRESHOLD_OPT,
    )
    logger.info(
        "stacking: base_bug=%s base_smell=%s automl=%s",
        STACKING_BASE_BUG, STACKING_BASE_SMELL, STACKING_AUTOML,
    )

    try:
        df = load_training_frame(args)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    # V2.1: --max-projects ile dataset'i sub-sample et (deterministic, seed-controlled)
    if args.max_projects:
        all_projects = df["project_name"].unique()
        if args.max_projects >= len(all_projects):
            logger.info(
                "--max-projects=%d >= mevcut %d, sub-sampling atlandi.",
                args.max_projects, len(all_projects),
            )
        else:
            rng = np.random.default_rng(RANDOM_STATE)
            chosen = rng.choice(all_projects, size=args.max_projects, replace=False)
            n_before = len(df)
            df = df[df["project_name"].isin(chosen)].reset_index(drop=True)
            logger.info(
                "max-projects: %d/%d proje secildi (seed=%d), satir: %d → %d",
                args.max_projects, len(all_projects), RANDOM_STATE,
                n_before, len(df),
            )

    # V2.1: CV modu — artifact yazmaz, mean+/-std raporlar.
    # cv_folds >= 2: GroupKFold(N) CV, default 1: tek-shot 70/15/15
    if args.cv_folds >= 2:
        args.models_dir.mkdir(parents=True, exist_ok=True)
        return run_cv(df, args, tasks)

    # Mevcut feature_names.json'u koru ve uzerine yaz
    existing_fn: dict = {}
    fn_path = args.models_dir / "feature_names.json"
    if fn_path.exists():
        try:
            existing_fn = json.loads(fn_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("feature_names.json bozuk; sifirdan yazilacak.")

    summaries: list[dict] = []
    try:
        if "bug" in tasks:
            try:
                summaries.append(train_bug(
                    df, args.models_dir,
                    bug_label=args.bug_label,
                    use_smote=args.smote,
                    autogluon_time_limit=args.autogluon_time_limit,
                ))
            except RuntimeError as exc:
                logger.error("T2 bug egitimi atlandi: %s", exc)
        if "smell" in tasks:
            summaries.append(train_smell(
                df, args.models_dir,
                smell_label=args.smell_label,
                use_smote=args.smote,
            ))
    except (KeyError, ValueError) as exc:
        logger.error("egitim sirasinda hata: %s", exc)
        return 1

    if not summaries:
        logger.error("Hicbir task basariyla egitilmedi.")
        return 1

    _save_feature_names(
        args.models_dir,
        tasks=[s["task"] for s in summaries],
        existing=existing_fn,
    )

    # model_config.json — predictor.py hangi artifact'lari yukleyecegini buradan okur
    model_config = {
        "stacking_base_bug":   STACKING_BASE_BUG,
        "stacking_base_smell": STACKING_BASE_SMELL,
        "stacking_automl":     STACKING_AUTOML,
        "trained_tasks":       [s["task"] for s in summaries],
        "created_at":          datetime.now().isoformat(),
    }
    mc_path = args.models_dir / "model_config.json"
    mc_path.write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    logger.info("model_config.json yazildi: %s", mc_path)

    # project_stats.json
    stats_path = args.models_dir / "project_stats.json"
    try:
        write_project_stats(df, stats_path)
        logger.info("yazildi: %s", stats_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("project_stats.json yazilamadi: %s", exc)

    # Ozet
    logger.info("== Egitim ozet ==")
    for s in summaries:
        m = s["metrics"]
        logger.info(
            "  %-5s (%-7s) f1=%.3f pr_auc=%.3f mcc=%.3f acc=%.3f  train=%d test=%d  %.1fs",
            s["task"], s["variant"], m["f1"], m.get("pr_auc", float("nan")),
            m["mcc"], m["accuracy"], s["n_train"], s["n_test"], s["duration_secs"],
        )

    # Sanity
    if not run_sanity(df, [s["task"] for s in summaries]):
        logger.warning("Sanity test basarisiz — artefaktlar yine de yazildi.")

    logger.info("== F6 train_final tamamlandi ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
