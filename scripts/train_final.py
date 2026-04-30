"""
scripts/train_final.py — Final model egitimi CLI (F6).

CLI contract (PLAN §12.2):

    python -m scripts.train_final [OPTIONS]

Isleyis (PLAN §15 F6 DoD):
  1. En guncel `output/dataset_model_filtered_*.parquet`'i yukle (veya --dataset)
  2. Her gorev icin:
       - commit → RandomForestClassifier           -> commit_rf.joblib
       - bug    → Stacking (RF + AutoGluon, LR meta) -> bug_rf_base + bug_ag_base + bug_meta_lr
       - smell  → RandomForestClassifier           -> smell_rf.joblib
     + scaler (StandardScaler) joblib'e yaz
  3. `feature_names.json` — 3 gorev icin feature isimleri
  4. `project_stats.json` — pipeline.project_stats.compute_project_stats
  5. Sanity: app.predictor ile tek satir tahmin — modellerin yuklendigi
     dogrulanir.

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from pipeline.config import (
    FEATURES_BUG,
    FEATURES_COMMIT,
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


RANDOM_STATE = 42
TASK_CHOICES = ("commit", "bug", "smell")
AUTOGLUON_TIME_LIMIT = 600  # PLAN §4.3 notu


# ── CLI ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scripts.train_final",
        description="MetricHunter V2 — Final model egitimi (F6).",
    )
    p.add_argument("--dataset", type=Path, default=None,
                   help="Filtered parquet (varsayilan: en guncel dataset_model_filtered_*)")
    p.add_argument("--tasks", type=str, default="commit,bug,smell",
                   help="Virgulle ayrilmis: commit,bug,smell")
    p.add_argument("--bug-label", choices=("keyword", "szz"), default="szz",
                   help="Bug etiket kaynagi")
    p.add_argument("--smell-label", choices=("binary", "count"), default="binary",
                   help="Smell etiket turu (V2'de count Phase B, su an binary)")
    p.add_argument("--models-dir", type=Path, default=MODELS_DIR,
                   help="Model artifact cikti dizini")
    p.add_argument("--autogluon-time-limit", type=int, default=AUTOGLUON_TIME_LIMIT,
                   help="AutoGluon fit time budget (saniye)")
    p.add_argument("--no-smote", action="store_true",
                   help="SMOTE'u atla (default: aktif, train-only)")
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
    print("  MetricHunter V2 — scripts.train_final  [--dry-run]")
    print("=" * 60)
    print(f"  dataset              : {args.dataset or '(son filtered)'}")
    print(f"  tasks                : {tasks}")
    print(f"  bug-label            : {args.bug_label}")
    print(f"  smell-label          : {args.smell_label}")
    print(f"  models-dir           : {args.models_dir}")
    print(f"  autogluon-time-limit : {args.autogluon_time_limit}s")
    print(f"  smote                : {'off' if args.no_smote else 'on (train-only)'}")
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
    if task == "commit":
        return "label_commit", "median"
    if task == "bug":
        if args.bug_label == "szz":
            return "bug_szz", "szz"
        return "bug_keyword", "keyword"
    if task == "smell":
        # Phase B (count) ileride regresyon olacak; V2'de binary egitilir.
        return "smell_binary", "p80"
    raise ValueError(f"Tanimsiz task: {task}")


def _features_for(task: str) -> tuple[str, ...]:
    return FEATURES_COMMIT if task == "commit" else FEATURES_BUG


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

    # GroupKFold cv iteratoru — train_dev icinde HP tuning / ablation (F4)
    # n_splits grup sayisindan fazla olamaz; cok az proje varsa devre disi
    n_cv_groups = train["project_name"].nunique() if "project_name" in train.columns else 0
    n_cv_splits = min(5, n_cv_groups)
    if n_cv_splits >= 2:
        cv        = GroupKFold(n_splits=n_cv_splits)
        cv_groups = train["project_name"].to_numpy()
        cv_iter   = list(cv.split(Xtr_s, ytr, groups=cv_groups))
    else:
        cv_iter = []

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


def train_commit(
    df: pd.DataFrame,
    models_dir: Path,
    use_smote: bool,
) -> dict:
    """T1 — RandomForestClassifier."""
    label_col, variant = _resolve_label("commit", _args_stub(bug="szz", smell="binary"))
    features = _features_for("commit")
    parts = _prepare_splits(df, features, label_col, use_smote)
    logger.info("T1 commit — %s | %s", label_col, parts["note"] or "smote uygulandi/yok")

    clf = RandomForestClassifier(
        n_estimators=400, n_jobs=-1, random_state=RANDOM_STATE,
    )
    t0 = time.monotonic()
    clf.fit(parts["Xtr"], parts["ytr"])
    dur = time.monotonic() - t0

    y_pred  = clf.predict(parts["Xte"])
    y_proba = clf.predict_proba(parts["Xte"])[:, 1]
    metrics = classification_metrics(parts["yte"], y_pred, y_proba)
    logger.info("T1 metrics: %s (fit %.1fs)", metrics, dur)

    joblib.dump(clf,            models_dir / "commit_rf.joblib")
    joblib.dump(parts["scaler"], models_dir / "scaler_commit.joblib")
    return {"task": "commit", "label": label_col, "variant": variant,
            "metrics": metrics, "duration_secs": round(dur, 2),
            "n_train": parts["n_train"], "n_test": parts["n_test"]}


def _args_stub(bug: str = "szz", smell: str = "binary") -> argparse.Namespace:
    """_resolve_label'in ihtiyaci olan alanlari mock eden kucuk namespace."""
    ns = argparse.Namespace()
    ns.bug_label = bug
    ns.smell_label = smell
    return ns


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

    # AutoGluon opsiyonel. Yoksa bu fazda T2 atlanir — Flask'ta commit+smell
    # hala calisir; caller logga uyari atar.
    try:
        from autogluon.tabular import TabularPredictor  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            f"AutoGluon kurulu degil ({exc}). T2 bug stacking icin gereklidir. "
            "`pip install autogluon.tabular` ile kurun veya --tasks icinde bug'i cikarin."
        )

    feat_cols = list(features)
    kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    rf_oof = np.zeros(len(Xtr))
    ag_oof = np.zeros(len(Xtr))

    t0 = time.monotonic()
    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(Xtr), 1):
        logger.info("T2 stacking fold %d/3 — OOF uretimi", fold_idx)
        rf = RandomForestClassifier(
            n_estimators=400, n_jobs=-1, random_state=RANDOM_STATE,
        )
        rf.fit(Xtr[tr_idx], ytr[tr_idx])
        rf_oof[va_idx] = rf.predict_proba(Xtr[va_idx])[:, 1]

        tr_df = pd.DataFrame(Xtr[tr_idx], columns=feat_cols)
        tr_df["target"] = ytr[tr_idx]
        va_df = pd.DataFrame(Xtr[va_idx], columns=feat_cols)

        ag_tmp = models_dir / f".ag_fold_{fold_idx}"
        p = TabularPredictor(
            label="target", path=str(ag_tmp),
            problem_type="binary", verbosity=0,
        ).fit(tr_df, time_limit=autogluon_time_limit)
        ag_oof[va_idx] = p.predict_proba(va_df).iloc[:, 1].to_numpy()

    # Meta-learner: OOF uzerinde LR
    meta = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    meta.fit(np.c_[rf_oof, ag_oof], ytr)

    # Base modelleri tum train uzerine refit
    logger.info("T2 base refit — RF + AutoGluon")
    rf_full = RandomForestClassifier(
        n_estimators=400, n_jobs=-1, random_state=RANDOM_STATE,
    )
    rf_full.fit(Xtr, ytr)

    tr_df_full = pd.DataFrame(Xtr, columns=feat_cols)
    tr_df_full["target"] = ytr
    ag_dir = models_dir / "bug_ag_base"
    # Eski artifact varsa temizle
    if ag_dir.exists():
        import shutil
        shutil.rmtree(ag_dir)
    ag_full = TabularPredictor(
        label="target", path=str(ag_dir),
        problem_type="binary", verbosity=0,
    ).fit(tr_df_full, time_limit=autogluon_time_limit)

    # Test tahminleri
    rf_test = rf_full.predict_proba(Xte)[:, 1]
    te_df   = pd.DataFrame(Xte, columns=feat_cols)
    ag_test = ag_full.predict_proba(te_df).iloc[:, 1].to_numpy()
    y_proba = meta.predict_proba(np.c_[rf_test, ag_test])[:, 1]
    y_pred  = (y_proba >= 0.5).astype("int64")
    dur = time.monotonic() - t0

    metrics = classification_metrics(parts["yte"], y_pred, y_proba)
    logger.info("T2 metrics: %s (toplam fit %.1fs)", metrics, dur)

    # Kalici artifact'lar
    joblib.dump(rf_full,         models_dir / "bug_rf_base.joblib")
    joblib.dump(meta,            models_dir / "bug_meta_lr.joblib")
    joblib.dump(parts["scaler"], models_dir / "scaler_bug.joblib")

    # Fold-temp'leri temizle
    import shutil
    for fold_idx in range(1, 4):
        tmp = models_dir / f".ag_fold_{fold_idx}"
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)

    return {"task": "bug", "label": label_col, "variant": variant,
            "metrics": metrics, "duration_secs": round(dur, 2),
            "n_train": parts["n_train"], "n_test": parts["n_test"]}


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

    clf = RandomForestClassifier(
        n_estimators=400, n_jobs=-1, random_state=RANDOM_STATE,
    )
    t0 = time.monotonic()
    clf.fit(parts["Xtr"], parts["ytr"])
    dur = time.monotonic() - t0

    y_pred  = clf.predict(parts["Xte"])
    y_proba = clf.predict_proba(parts["Xte"])[:, 1]
    metrics = classification_metrics(parts["yte"], y_pred, y_proba)
    logger.info("T3 metrics: %s (fit %.1fs)", metrics, dur)

    joblib.dump(clf,            models_dir / "smell_rf.joblib")
    joblib.dump(parts["scaler"], models_dir / "scaler_smell.joblib")
    return {"task": "smell", "label": label_col, "variant": variant,
            "metrics": metrics, "duration_secs": round(dur, 2),
            "n_train": parts["n_train"], "n_test": parts["n_test"]}


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
        if "commit" in trained_tasks:
            pr, pb = predictor_mod.predict_commit(sample)
            logger.info("sanity T1 commit: pred=%s proba=%.3f", pr[0], float(pb[0]))
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

    try:
        df = load_training_frame(args)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

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
        if "commit" in tasks:
            summaries.append(train_commit(
                df, args.models_dir, use_smote=not args.no_smote,
            ))
        if "bug" in tasks:
            try:
                summaries.append(train_bug(
                    df, args.models_dir,
                    bug_label=args.bug_label,
                    use_smote=not args.no_smote,
                    autogluon_time_limit=args.autogluon_time_limit,
                ))
            except RuntimeError as exc:
                logger.error("T2 bug egitimi atlandi: %s", exc)
        if "smell" in tasks:
            summaries.append(train_smell(
                df, args.models_dir,
                smell_label=args.smell_label,
                use_smote=not args.no_smote,
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
