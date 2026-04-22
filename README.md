# MetricHunter V2

Static metriklerden commit yogunlugu + bug + code smell tahmini yapan
hibrit AutoML boru hatti. V1 Flask arayuzunun uzerine, 1000 Python
projesinden olusan genis veri setiyle egitilen uc gorevli model.

Detayli tasarim: [`PLAN.md`](PLAN.md).

## Dizin yapisi

```
v2/
  pipeline/        # Tum agir kod — config, checkpoint, rate_limit, discovery, metrics, ...
  scripts/         # Batch CLI (python -m scripts.collect ...)
  analysis/        # Interactive analiz (.py + # %% cells, VS Code Jupyter)
  app/             # Flask UI
  tests/           # pytest
  output/          # Checkpoint/log/parquet/figures (gitignore)
  models/          # Model artifactlari
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

### Veri toplama (CLI)

```bash
# Config ozeti (hiçbir şey yazmaz)
python -m scripts.collect --dry-run

# 10 projelik mini discovery
python -m scripts.collect --phase discovery --target 10

# Full (F3'te tam aktif)
python -m scripts.collect --target 1000
```

### Model egitimi (F6)

```bash
python -m scripts.train_final --dry-run
```

### Interactive analiz (VS Code)

`analysis/*.py` dosyalarini VS Code'da acin — `# %%` hucreleri Jupyter
extension ile hucre hucre calistirilir. Plotlar inline gorunur ve
`output/figures/` altina kaydedilir.

```bash
# Notebook'a cevirmek gerekirse (akademisyene gondermek icin):
jupytext --to ipynb analysis/02_model_training.py
```

### Testler

```bash
python -m pytest              # v2/ dizininden
python -m pytest -v tests/test_checkpoint.py
```

## Faz durumu

| Faz | Durum |
|---|---|
| F1 — Pipeline altyapisi | Aktif — bu surum |
| F2 — SZZ + Prospector   | Planli |
| F3 — Tam veri toplama   | Planli |
| F4 — Filter & threshold | Planli |
| F5 — Model training     | Planli |
| F6 — Final training     | Planli |
| F7 — Flask V2 UI        | Planli |
| F8 — Paper + doc        | Planli |

## Lisans

Arastirma amacli, yayin oncesi surum.
