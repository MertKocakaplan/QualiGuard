# QualiGuard

**An AI-driven adaptive quality gate for defect prediction and code smell detection in DevOps pipelines.**

QualiGuard mines public GitHub repositories of a target language and, for every source file, extracts static (radon), process (git-history), and cognitive-complexity metrics, derives defect labels with the SZZ algorithm, and detects seven classical code smells through abstract-syntax-tree analysis. It then trains models that predict, per file, the risk of defects and code smells, and exposes the calibrated risk as a three-tier quality gate — **PASS / REVIEW / BLOCK** — through a Flask web interface that analyses any GitHub repository or uploaded archive.

- **Defect prediction:** stacking hybrid — LightGBM + AutoGluon combined through an isotonic-calibrated logistic-regression meta-learner.
- **Code smell prediction:** threshold-optimised LightGBM.
- **Evaluation protocol:** cross-project 5-fold GroupKFold (leakage-free; files of one project never appear in more than one fold).

## Architecture

Layered design with one-directional dependencies — each layer uses only the layers beneath it:

```
Web / Presentation   app/  (routes, templates, static, run.py)
        v
Serving              app/  (predictor, analyzer, health)
        v
Modeling             pipeline/ + scripts/ + analysis/
        v
Acquisition /        pipeline/  (discovery, cloning, static/git/AST metrics,
Extraction                       SZZ labelling, CI-CD signals)
        v
Configuration        pipeline/config.py
```

## Requirements

- Python 3.10
- Git (used to clone the repositories under analysis)
- A GitHub personal access token (`public_repo` scope) — required for data collection and recommended for analysing large repositories

## Setup

```bash
git clone <repository-url>
cd <repository>
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
# source venv/bin/activate
pip install -r requirements-dev.txt     # full pipeline (collect + train + analysis + tests)
# pip install -r requirements.txt       # minimal web-serving subset only
```

Create a `.env` file in the project root:

```
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## Usage

This repository ships **source code only** — datasets and trained models are **not** included; they are produced from scratch by the pipeline below.

**1 — Collect data** (GitHub search -> clone -> metrics -> labels -> per-project Parquet -> merged dataset):

```bash
python -m scripts.collect --phase all --target 1000
```

**2 — Build the modelling dataset** (labelling, categories, within-project smell threshold, filtering -> `output/dataset_model_filtered_<ts>.parquet`):

```bash
python analysis/01_filter_categorize.py
```

**3 — (optional) Benchmark ten ML/DL/AutoML models** under cross-project CV:

```bash
python analysis/06_ml_baseline_cv.py --bug-label szz
```

**4 — Train the production models** (writes artifacts into `models/`):

```bash
python -m scripts.train_final --tasks bug,smell --threshold-opt \
    --stacking-base-bug lightgbm --stacking-base-smell lightgbm \
    --stacking-automl autogluon
```

**5 — Launch the web application:**

```bash
python run.py
# open http://127.0.0.1:5000
```

Paste a GitHub repository URL, or upload a ZIP archive that contains its `.git/` directory, to obtain per-file defect and code-smell predictions together with the PASS/REVIEW/BLOCK quality-gate tier and project-level health and DevOps indicators.

## Tests

```bash
python -m pytest -q
```

## Project structure

```
app/        Flask web application (model serving + user interface)
pipeline/   Data acquisition, metric extraction, labelling, and modelling utilities
scripts/    Command-line tools (collect, train_final, ...)
analysis/   Experiment scripts (filtering, feature selection, benchmark, sensitivity)
tests/      Automated test suite
run.py      Flask entry point
```

Produced at runtime and intentionally **not** version-controlled: `output/` (collected data and results), `models/` (trained artifacts), `repos/` (cloned repositories).
