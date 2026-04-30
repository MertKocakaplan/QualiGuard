# Changelog

V1 Flask uygulamasindan V2 hibrit AutoML boru hattina gecis.
Tarih formati: `YYYY-MM-DD`. Sadece kullanici-gorunur degisiklikler listelenir.

---

## [2.1.0] — F1-F6 enhancements (2026-04-29 → 2026-04-30)

V2.0 uzerine paper revize gereksinimleri icin 8 maddelik enhancement paketi.
Detay: `docs/v2_enhancement_plan.md`.

### Eklendi
- **AST + radon code smell tespiti** — `pipeline/code_smells.py`. 7 klasik
  smell (Long Method, Large Class, Long Parameter List, Deep Nesting, High CC,
  Low MI, God Function); Fowler (1999) + Lanza-Marinescu (2006).
  Prospector'in ~30 gunluk pylint type-inference yuku ~5 dk'ya indi.
- **Cognitive Complexity** — Campbell (2018) operasyonelize edildi.
  `cognitive_complexity_total` + `cognitive_complexity_max` ozellikleri
  `cognitive-complexity` paketiyle hesaplaniyor.
- **Bug keyword separation** — `BUG_KEYWORD_GROUPS` (fix/bug/error/defect/
  issue/anomaly) ayri sayim; `bug_kw_*_count` x6 yeni sutun.
- **Refactor ratio** — repo bazinda `refactor_commits / total_commits`.
- **Contribution Gini** — Mockus (2002) power-law katsayisi (`contribution_gini`).
- **Git-log proxy features** — `revert_count`, `inter_commit_time_cv`,
  `author_entropy`, `bug_fix_density`. GitHub PR/Issue API yerine.
- **Two-stage split protokolu** — `pipeline.model_utils.two_stage_split`.
  Project-based 70/15/15 holdout (Tantithamthavorn 2017) + 5-fold GroupKFold
  development pool icinde. Final test seti hyperparameter tuning'de kullanilmaz.
- **Calibrated risk score + 3-tier UI** — `app.health.risk_tier`,
  `app.predictor.predict_proba_calibrated`. Isotonic CalibratedClassifierCV
  (cv=3) ile meta-LR cikti kalibre. UI'de PASS / REVIEW / BLOCK badge'i.
- **Discovery merge logic** — yeni `--phase discovery` cagrilari onceki
  topics/description meta'sini silmez, birlestirir.
- **Sample validation scripti** — `scripts/validate_smell_sample.py`. AST +
  radon'u Prospector'a karsi 50-dosyalik orneklemde Cohen's kappa ile dogrular
  (Landis-Koch 1977). Prospector erisimi gerekli — koşulmasi bekleniyor.

### Degistirildi
- `FEATURES_COMMIT`: 29 → 35 (+6); `FEATURES_BUG = FEATURES_SMELL`: 36 → 48 (+12).
- `--skip-prospector` → `--skip-smells` (geriye uyumlu alias korundu).
- Test sayisi 137 → 216 (+79 yeni test).

### Sabitlenmis yeni limitler
- Smell esikleri: LONG_METHOD_LOC=50, LARGE_CLASS_LOC=500, LONG_PARAM_COUNT=5,
  NESTING_DEPTH=4, HIGH_CC=10, LOW_MI=20, GOD_FUNC_CC=15, GOD_FUNC_LOC=80.
- Risk tier: PASS (<P70) / REVIEW (P70–P90) / BLOCK (≥P90).
- `min_stars`: 5 → 50.

---

## [2.0.0] — bu surum

V1 (commit + bug tahmini) uzerine **code smell tespiti**, **1000 projelik veri
toplama pipeline'i** ve **hibrit AutoML** eklendi. Flask UI ayni temada kaldi;
arka taraf tamamen yeniden yazildi.

### Eklendi
- **T3 Smell tahmini** — Prospector smell_count + dinamik P80 esigi ile binary
  label; Random Forest model (PLAN §1).
- **SZZ bug etiketi** — `pydriller` ile bug-inducing commit'ler izleniyor;
  V1'deki keyword baseline yerine daha dogru `bug_szz` label'i uretiliyor
  (PLAN §3.6).
- **AutoGluon tabanli stacking** — T2 bug tahmini icin RF + AutoGluon base
  modellerin OOF tahminleri LR meta-ogrenici ile birlesiyor (PLAN §4.4).
- **1000 projelik veri toplama CLI'i** — `scripts/collect.py` (discovery,
  process, build fazlari; `--resume` guvenli, atomic checkpoint yazimi).
- **Flask V2 UI** — 3 tahmin, prospector opt-in, Project Health kartlari
  (defect density / refactor ratio / recent activity / bug-fix ratio),
  Smell Overview kartlari (total prospector smells, ML smell risk,
  Refactor Priority kesisimi), smell risk badge + prob bar + prospector
  sayac sutunu (PLAN §5, §17).
- **Ablation harness** — `analysis/02_model_training.py`, 3 gorev × 4 feature
  seti × 2 split × 10 model; smart pruning ile iterasyon 5-10x kisalir
  (PLAN §4.5).
- **Sonuc kesif scripti** — `analysis/03_results_exploration.py`; model
  ortalamasi bar, heatmap, confusion + PR, feature importance, yanlis
  siniflandirma CSV'si.
- **Filter & Categorize** — `analysis/01_filter_categorize.py`; sensitivity
  analizi (filtresiz vs `(10,100)` vs `(25,80)`) + kategori dagilimi
  (PLAN §4.2).
- **Project stats** — `models/project_stats.json` (global + kategori bazli);
  Flask panellerine veri saglar (PLAN §14.6).
- **Sanity test** — `scripts/train_final.py` sonunda predictor reload +
  mevcut dataset ilk satiri ile 3 tahmin dogrulanir.
- **Rate limit token paneli** — UI'de `ghp_...` token ekle/sil/durum
  gorsel kontrolu; rate limit 60 -> 5000 istek/saate cikiyor.
- **Prospector batch paralellestirmesi** — multiprocessing pool ile N worker,
  dosya basi timeout, kategori dagilimi cikisi (PLAN §3.7).

### Degistirildi
- Dizin yapisi parcalandi: agir kod `pipeline/` paketine, batch akisi
  `scripts/` altina, etkilesimli hucreler `analysis/*.py` (VS Code Jupyter
  `# %%` formati) olarak ayrildi. V1 monolitik `app/` artik sadece Flask
  sunumu (PLAN §2.1).
- Python 3.10 zorunlu — AutoGluon artifactlari 3.10.6'da egitildi.
- Label kaynaklari artik parquet'te paralel tutuluyor (`bug_keyword`,
  `bug_szz`, `smell_binary`, `smell_count`); secim egitim asamasinda
  ablation ile yapiliyor.
- Commit volume etiketi artik **proje genelinde global median** ile
  belirleniyor (V1: sabit esik = 5).
- Filter/threshold tercihi: filtresiz default, sensitivity raporu ile
  dogrulaniyor (PLAN §4.2).

### Kaldirildi
- V1 notebook'lari (eski referans olarak `archive/v1/` altinda saklandi).
- Sabit commit_count=5 esigi (global median'a tasindi).
- Monolitik `app/git_utils.py` — mantik `pipeline/git_metrics.py` icine
  dagildi ve genisletildi (ornek: `get_repo_commit_summary`).

### Sabitlenmis limitler
- Prospector default timeout: 60 sn/dosya, 4 worker.
- GitHub API token'siz: 60 ist/saat; token ile: 5000 ist/saat.
- Dataset hedef: 1000 proje, ~200-400K dosya kaydi.
- Sanity test: train sonunda predictor reload + 3 tahmin.

---

## [1.0.0] — eski surum (referans)

V1 Flask arayuzu (commit + bug tahmini, 16 radon metrigi, keyword-only bug
etiketi). Detaylar icin `archive/v1/`.
