# MetricHunter V2

Static metriklerden commit yogunlugu + bug + code smell tahmini yapan
hibrit AutoML boru hatti. V1 Flask arayuzunun uzerine, 1000 Python
projesinden olusan genis veri setiyle egitilen uc gorevli model.

Detayli tasarim: [`PLAN.md`](PLAN.md).
Degisiklikler: [`CHANGELOG.md`](CHANGELOG.md).
Makale iskeleti: [`docs/paper_outline.md`](docs/paper_outline.md).

## Dizin yapisi

```
v2/
  pipeline/        # Tum agir kod — config, checkpoint, rate_limit, discovery, metrics, ...
  scripts/         # Batch CLI (python -m scripts.collect ...)
  analysis/        # Interactive analiz (.py + # %% cells, VS Code Jupyter)
  app/             # Flask UI (predictor, analyzer, health, routes, templates)
  tests/           # pytest (136 passed, 2 skipped bu surumde)
  docs/            # Makale iskeleti ve ek belgeler
  output/          # Checkpoint/log/parquet/figures (gitignore)
  models/          # Model artifactlari (joblib + autogluon + scaler + stats)
  archive/v1/      # Eski notebook'lar (referans)
  run.py           # Flask giris noktasi
```

## Kurulum

```bash
python3.10 -m venv venv
venv\Scripts\activate              # Windows
pip install -r requirements-dev.txt
```

Python 3.10 zorunludur — AutoGluon artifactlari 3.10.6'da egitilmistir.

## Kullanim

### Flask (UI)

```bash
python run.py
# http://localhost:5000
```

`.env` dosyasina `GITHUB_TOKEN=ghp_...` ekleyerek rate limit'i artirin.

**V2 UI ozellikleri (F7):**

- Uc gorev icin tahmin — commit (mor), bug (kirmizi), smell (turuncu).
- Analiz formunda **Prospector opsiyonu** (opt-in); etkinse her dosya icin
  `smell_count` + kategori dagilimi UI'de gosterilir.
- **Project Health** kartlari — defect density (bug/KLOC), refactor oran,
  son 90 gun aktivitesi, bug-fix commit orani.
- **Smell Overview** kartlari — toplam prospector smell, ML tahmini smell
  sayaci, `bug_pred=1 AND smell_pred=1` kesisiminden hesaplanan
  **Refactor Priority** sayaci, top-3 kirli dosya.
- Tabloda smell risk badge + smell probability bar + prospector sayac sutunu
  (model / prospector varsa otomatik acilir).

### Veri toplama (CLI)

`scripts/collect.py` uc fazli bir CLI'dir:

| Faz | Is | Cikti |
|---|---|---|
| `discovery` | GitHub search + contributor filtresi | `output/checkpoints/discovery.json` |
| `process`   | Her proje icin clone + radon + git + SZZ + Prospector | `output/projects/<safe>.parquet` (per-project) |
| `build`     | Tum per-project parquet'leri birlestir + label'lar | `output/dataset_full_<ts>.parquet` |
| `all`       | Yukaridakileri sirayla calistirir | Hepsi |

```bash
# Config ozeti — hicbir sey yazmaz, plan raporu basar
python -m scripts.collect --dry-run

# 1) Discovery: 1000 proje bul
export GITHUB_TOKEN=ghp_...      # onerilir; token olmazsa rate limit cok dusuk
python -m scripts.collect --phase discovery --target 1000

# 2) Process: her projeyi isle — long running, checkpoint + resume guvenli
python -m scripts.collect --phase process --resume

# 3) Build: final parquet'i uret
python -m scripts.collect --phase build

# Hepsini sirayla (kisa testler icin)
python -m scripts.collect --target 20 --phase all
```

**Kaynak gereksinimleri (1000 proje, ortalama):**

- **Sure:** discovery ~10-30 dk (rate limit'e gore), process ~3-5 saat (Prospector dahil),
  build ~1-2 dk. Toplam ~4-6 saat.
- **Disk:** `repos/` ~25-40 GB (single-branch clone), `output/projects/` ~200 MB,
  `output/dataset_full_*.parquet` ~100-200 MB.
- **Ag:** Discovery GitHub API'sine bagli; `GITHUB_TOKEN` ile saatte 5000 istek
  (token'siz 60).

**Kesinti ve devam:**

- `Ctrl+C` (SIGINT) isleri duzenli keser; exit code `130` doner.
- `--resume` ile processed_projects.json'daki `status=ok` projeler atlanir;
  `failed` olanlar tekrar denenebilir.
- Checkpoint'ler atomic yazilir (temp + `os.replace`); yarim dosya kalmaz.

**Ekstra bayraklar:**

- `--skip-szz` — pydriller'i atla, sadece bug_keyword etiketi uret.
- `--skip-prospector` — Prospector'i atla, smell_count=None doldurulur.
- `--workers N` — Prospector paralel worker sayisi (varsayilan 4).
- `--log-level DEBUG` — API/Prospector cikti detaylari.

### Model egitimi (F6)

F5 ablation'dan secilen mimarileri kullanarak 3 gorev icin final artifact
uretir: `commit_rf.joblib`, `bug_rf_base.joblib` + `bug_ag_base/` +
`bug_meta_lr.joblib` (stacking), `smell_rf.joblib`, scaler'lar,
`feature_names.json`, `project_stats.json`.

```bash
# Config ozeti — yazmaz
python -m scripts.train_final --dry-run

# Tum gorevler (SZZ etiketi + binary smell)
python -m scripts.train_final

# Sadece commit + smell (AutoGluon olmadan calisir)
python -m scripts.train_final --tasks commit,smell

# Bug etiketi olarak keyword baseline
python -m scripts.train_final --bug-label keyword

# AutoGluon budget ayari (default 600s)
python -m scripts.train_final --autogluon-time-limit 1200

# SMOTE kapali
python -m scripts.train_final --no-smote
```

**Notlar:**

- AutoGluon opsiyonel; kurulu degilse T2 bug stacking atlanir ve log'a uyari
  yazilir. T1 commit + T3 smell bagimsiz olarak calisir.
- Her task sonunda test setinde `f1`, `pr_auc`, `mcc`, `accuracy` log'a yazilir.
- Sanity: egitim sonunda `app.predictor` reload edilir, mevcut dataset'in
  ilk satiri ile `predict_commit`/`predict_bug`/`predict_smell` cagrilir.
- Log dosyasi: `output/logs/train_final_<YYYYMMDD_HHMM>.log`

### Interactive analiz (VS Code)

`analysis/*.py` dosyalarini VS Code'da acin — `# %%` hucreleri Jupyter
extension ile hucre hucre calistirilir. Plotlar inline gorunur ve
`output/figures/` altina kaydedilir.

```bash
# Notebook'a cevirmek gerekirse (akademisyene gondermek icin):
jupytext --to ipynb analysis/02_model_training.py
```

#### F5 — Model training ablation (`analysis/02_model_training.py`)

`analysis/01_filter_categorize.py` cikisi (`output/dataset_model_filtered_*.parquet`)
uzerinde tam ablation calistirir: 3 task (commit, bug, smell) x feature set
(static/derived/process/all) x split (project-based, time-based) x model
(LR, RF, SVM, XGBoost, LightGBM, AutoGluon, MLP, CNN1D, LSTM, Stacking).

Agir bagimliliklar (xgboost, lightgbm, autogluon, tensorflow) opsiyoneldir —
import hatasi durumunda ilgili kombinasyon `status=skipped` ile atlanir,
`ablation_results_<ts>.csv`'e kaydedilir.

Hucreler:
1. Imports + `RANDOM_STATE`, `AUTOGLUON_TIME_LIMIT`
2. En guncel `dataset_model_filtered_*.parquet` yukle
3. `ABLATION` config: taskler, feature_setleri, split'ler, modeller
4. `MODEL_REGISTRY` — 10 model factory (heavy dep'ler try/except ile korumali)
5. `run_one_combo` — scaler fit+transform, SMOTE train-only, metric donus
6. Ana dongu — smart pruning ile alt kume
7. `ablation_results_<ts>.csv` yaz + top-10 F1
8. Skipped/failed raporu

**Smart pruning:** "static" feature set'te sadece bir alt kume model (`rf`,
`autogluon`, `stacking_rf_ag_meta_lr`) calistirilir; diger modeller sadece
`all` feature set'le kosar — iterasyonu 5-10x kisaltir.

#### F5 — Sonuc kesif (`analysis/03_results_exploration.py`)

Ablation CSV'sinden grafik ve ozet uretir:

1. En guncel `ablation_results_*.csv` yukle + `ok` olanlari filtrele
2. Model ortalamasi bar chart (`model_bars_<ts>.png`)
3. Task x feature_set F1 heatmap (`heatmap_<task>_<ts>.png`)
4. Task bazinda en iyi model + hiperparametre
5. Test setinde confusion matrix + PR curve (`confusion_pr_<task>_<model>.png`)
6. RF feature importance top-20 (task=commit, `feature_importance_commit.png`)
7. Misclassification analizi + per-class mean metrics (`misclassification_<ts>.csv`)

#### F4 — Filter & Categorize (`analysis/01_filter_categorize.py`)

`scripts.collect --phase all` cikisi (`output/dataset_full_*.parquet`)
uzerinde kategori atama + sensitivity analizi + filtered dataset uretimi
yapilir.

Hucreler:

1. Imports + klasor kurulumu (`output/figures/`)
2. En guncel `dataset_full_*.parquet` yukle
3. `output/checkpoints/discovery.json`'dan varsa topics/description oku
4. `add_project_categories` — her projeye kategori ata
5. Kategori dagilimi plot (`sensitivity_category_distribution.png`)
6. `add_dynamic_smell_binary` (P80) + `add_commit_label` (global median)
7. **Sensitivity:** filtresiz vs `(10, 100)` vs `(25, 80)` karsilastirma tablosu
8. Sensitivity plot (`sensitivity_commit_filters_<ts>.png`) + CSV export
9. Secilen filtre ile `apply_commit_filter`
10. `output/dataset_model_filtered_<ts>.parquet` yaz
11. Ornek proje -> kategori tablosu

**PLAN §4.2 tercihi:** filtresiz default; dramatik fark yoksa aynen birakin.
`CHOSEN_MIN`/`CHOSEN_MAX` degiskenlerini 9. hucrede elle degistirin.

> Not: V2 discovery henuz topics/description kaydetmiyor — kategorilendirme
> cogunlukla proje adina dusebilir, bu durumda `"Diger"` orani yukselir.
> Hucre 4 bu orani `%40` asarsa uyari yazdirir.

### Testler

```bash
python -m pytest                       # v2/ dizininden — tum testler
python -m pytest -v tests/test_checkpoint.py
python -m pytest tests/test_health.py  # F7 health helper'lari
```

Bu surumde **136 passed, 2 skipped** (imbalanced-learn + pydriller opsiyonel
bagimliliklarina bagli atlamalar).

## Dokumantasyon

- [`PLAN.md`](PLAN.md) — mimari kararlar, veri/model yaklasimi, faz checklist.
- [`CHANGELOG.md`](CHANGELOG.md) — V1 -> V2 tum kullanici-gorunur degisiklikler.
- [`docs/paper_outline.md`](docs/paper_outline.md) — makale taslak iskeleti;
  RQ'lar ve sonuc tablolari placeholder; gercek sayilari pipeline kosumundan
  sonra doldurun.

## Faz durumu

| Faz | Durum |
|---|---|
| F1 — Pipeline altyapisi | Tamam |
| F2 — SZZ + Prospector   | Tamam |
| F3 — Tam veri toplama   | Tamam |
| F4 — Filter & threshold | Tamam |
| F5 — Model training     | Tamam |
| F6 — Final training     | Tamam |
| F7 — Flask V2 UI        | Tamam |
| F8 — Paper + doc        | Aktif — bu surum |

## Lisans

Arastirma amacli, yayin oncesi surum.
