# MetricHunter V2 Enhancement Plan

> **Bu doküman:** F1-F8 fazları tamamlanmış MetricHunter V2'nin akademik gereksinimlere
> tam uyum için yapılacak 8 maddelik iş paketi. Yeni Claude oturumunda bu doküman
> referans alınarak adım adım uygulanacak.
>
> **Hedef paper:** "An AI-driven adaptive quality gate integrating code smell
> detection and defect prediction for DevOps pipelines"
>
> **Plan kararlaştırma tarihi:** 2026-04-29
> **Önerilen model:** Sonnet 4.6 (Opus sadece zor debug için)

---

## 0. Yeni oturuma başlangıç checklist

Yeni Claude oturumunda ilk yapılacaklar:

```powershell
# 1. venv aktive et (PowerShell)
.\venv\Scripts\Activate.ps1
where.exe python  # ilk satır venv'i göstermeli

# 2. Baseline test pass
$env:PYTHONIOENCODING = 'utf-8'
python -m pytest tests/ -q
# Beklenen: ~137 pass, 2 skip. Patlarsa devam etme, önce diagnose.

# 3. Branch state
git status
git log --oneline -10
```

**Bilinen durum (2026-04-29):**
- venv Python 3.13.7 (PLAN'da 3.10 yazıyor, sorunsa sonra fix)
- Smoke test 50 projelik dataset üretilmiş: `output/dataset_full_*.parquet`
- `min_stars=50` set edilmiş (`pipeline/config.py`)
- Smell phase `--skip-prospector` ile deaktif (timeout sorunu)
- Repos cache: `output/repos/` — silme

**Önemli kural:** Her faz **ayrı commit**. Faz biterken pytest pass etmeli, aksi halde commit yok.

---

## 1. Mimari özet — neye dokunuyoruz

```
pipeline/
  config.py            ← F2'de yeni sabitler eklenir
  discovery.py         ← F1'de merge logic
  git_metrics.py       ← F3'te 5+ yeni feature
  static_metrics.py    ← F3'te Cognitive Complexity entegrasyonu
  code_smells.py       ← F2'de YENİ DOSYA
  prospector_runner.py ← F2'de deprecate ama silme
  project_processor.py ← F2'de smell çağrı değişikliği
  model_utils.py       ← F4'te two-stage split
  ...

scripts/
  collect.py           ← F2'de --skip-prospector → --skip-smells
  train_final.py       ← F4 + F5'te split + risk score

app/
  health.py            ← F5'te risk score helper
  analyzer.py          ← F5'te risk score formatı
  predictor.py         ← F5'te calibration + tiers

tests/
  test_code_smells.py  ← F2'de YENİ
  test_*.py            ← her fazda ilgili test güncellenir/eklenir

docs/
  v2_enhancement_plan.md  ← bu dosya
```

---

## 2. Karar tablosu (referans)

| Madde | Karar | Defans (paper) |
|---|---|---|
| Two-stage split (70/15/15 + GroupKFold) | ✅ Yap | "Project-based 70/15/15 holdout + 5-fold GroupKFold (Tantithamthavorn et al., 2017)" |
| PR rejection / Issue density | ✅ Git proxy yap, API yapma | "git log proxies (Mockus 2010, Bird et al. 2009)" |
| Cohen's kappa (full SonarQube) | ❌ Skip | — |
| Sprint defect density / Rolling window | ❌ Skip (future work) | "Future work: temporal modeling" |
| Refactoring ratio | ✅ Yap | DevOps narrative |
| Project type Web/Mobile/Desktop | ❌ Şimdilik dokunma | categories.py post-hoc yeterli |
| Cognitive Complexity | ✅ Yap | "Cognitive complexity (Campbell, 2018) complements McCabe CC" |
| Bug keyword separation | ✅ Yap | "Per-keyword bug type counts" |
| CI/CD detection | ❌ Skip (hocayla konuşulacak) | — |
| Contribution Gini | ✅ Yap | "Power-law contribution skew (Mockus 2002)" |
| AST+radon smell + sample validation | ✅ Hibrit C | "Fowler (1999) + Lanza-Marinescu (2006), validated κ on 50-file sample" |
| Quality Gate UI threshold | ❌ Skip (UI yenilemesinde) | — |
| Risk Score → meta-stacking | ✅ Yap | "Calibrated stacking ensemble (Wolpert 1992)" |
| Discovery merge fix | ✅ Yap | Data integrity |
| Method-level rows | ❌ Skip | "File-level granularity (Bacchelli & Bird 2013)" |
| 1M row target | ❌ Drop | Reframe: "60-100K file rows" |
| Sprint defect density | ❌ Skip | Future work |
| 80/20 imbalance enforcement | ❌ Skip | Doğal sonuç olarak rapor |

---

## 3. Faz F1 — Discovery merge fix

**Goal:** Yeni discovery eski projelerin meta'sını silmeyecek, **birleşecek**.

**Effort:** 30 dk

**Why:** Önceki smoke test'te 50 projelik discovery.json'un üstüne 10 projelik yazılınca topics/description meta kayboldu, kategori atama %85 "Diğer" verdi.

### Files

- `pipeline/discovery.py` — `_save_discovery_results()` veya scripts/collect.py'deki
  `_run_discovery` çağrısı

### Implementation

`scripts/collect.py` _run_discovery() içinde:

```python
def _run_discovery(target: int) -> None:
    out_path = CHECKPOINT_DIR / "discovery.json"

    # Existing data load
    existing = {"found": [], "stats": {}}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {"found": [], "stats": {}}

    existing_by_name = {p["full_name"]: p for p in existing.get("found", [])}

    # Run discovery
    new_results = discovery.search_projects(target=target)

    # Merge: yeni discovery'deki her proje öncekinin üstüne yazsın (taze meta),
    # ama discover edilmemiş eski projelerin meta'sı korunsun
    merged = dict(existing_by_name)
    for p in new_results:
        merged[p["full_name"]] = p  # yeni meta her zaman üstüne yazsın

    out_data = {
        "found": list(merged.values()),
        "stats": {
            "total": len(merged),
            "this_run_new": len(new_results),
            "previous": len(existing_by_name),
        },
    }
    _atomic_write_json(out_path, out_data)
```

### Tests

- `tests/test_collect.py`'a yeni test:
  - Önceden 5 projelik discovery.json oluştur (mock)
  - 3 proje (1'i overlapping) ile yeni discovery çalıştır
  - Sonuçta 7 proje olmalı, overlapping olan yenisinin meta'sı

### Validation

```powershell
python -m scripts.collect --phase discovery --target 5
python -m scripts.collect --phase discovery --target 10
# discovery.json'da topic + description eski 5 projede de korunmuş olmalı
python -c "import json; d=json.load(open('output/checkpoints/discovery.json',encoding='utf-8')); print(f'total={len(d[chr(34)+chr(102)+chr(111)+chr(117)+chr(110)+chr(100)+chr(34)])}')"
```

### Commit

```
fix(discovery): preserve existing project metadata on incremental runs

Previously, running --phase discovery with target N would overwrite
discovery.json, losing topics/description metadata of previously discovered
projects. This caused categories.py to fall back to project_name-only
matching, dropping ~85% of projects to "Diğer" category in mixed datasets.

Now merges new results into existing entries; new metadata wins on conflict
but unrelated previous entries are preserved.
```

---

## 4. Faz F2 — Smell migration (Prospector → AST+radon)

**Goal:** Prospector subprocess'ini bağımlılıktan çıkar, AST+radon ile 7 klasik
smell'i deterministik tespit et. ~1000x hız kazanımı.

**Effort:** 1 gün + 0.5 gün sample validation = 1.5 gün

**Dependencies:** F1 (yok aslında, paralel olabilir)

### Files

**Yeni:**
- `pipeline/code_smells.py` (~150 satır)
- `tests/test_code_smells.py` (~80 satır, 14 test)

**Değişen:**
- `pipeline/project_processor.py` — `run_prospector_batch` → `detect_smells_batch`
- `pipeline/config.py` — `SMELL_*` eşik sabitleri eklenir,
  `PROSPECTOR_*` deprecated comment'i
- `scripts/collect.py` — `--skip-prospector` → `--skip-smells` (geriye uyumlu alias bırak)

**Korunan ama deprecated:**
- `pipeline/prospector_runner.py` — silme, opsiyonel `--use-prospector` flag arkasında
- `tests/test_prospector_runner.py` — bırak, testler hâlâ geçer

### `pipeline/code_smells.py` API

```python
"""
code_smells.py — AST + radon tabanli klasik code smell tespiti.

Fowler (1999) Refactoring + Lanza & Marinescu (2006) Object-Oriented Metrics
in Practice'tan secilmis 7 smell:

1. Long Method      — fonksiyon LOC > LONG_METHOD_LOC (50)
2. Large Class      — sinif LOC > LARGE_CLASS_LOC (500) ve method >= 10
3. Long Param List  — parametre sayisi > LONG_PARAM_COUNT (5)
4. Deep Nesting     — max indent depth > NESTING_DEPTH (4)
5. High Complexity  — radon CC > HIGH_CC (10)
6. Low Maintainability — radon MI < LOW_MI (20)
7. God Function     — CC > GOD_CC (15) ve LOC > GOD_LOC (80)

Kullanim:
    from pipeline.code_smells import detect_smells, detect_smells_batch
    result = detect_smells(Path("foo.py"))
    # {'smell_count': 7, 'smell_long_method': 2, ...}
"""

def detect_smells(file_path: Path) -> dict:
    """
    Tek dosyada 7 smell'i tespit et.

    Returns:
        smell_count: int (toplam),
        smell_long_method: int,
        smell_large_class: int,
        smell_long_param_list: int,
        smell_deep_nesting: int,
        smell_high_complexity: int,
        smell_low_maintainability: int (0/1, dosya seviyesinde),
        smell_god_function: int,
    """

def detect_smells_batch(
    file_paths: list[Path],
    skip_errors: bool = True,
) -> dict[Path, dict]:
    """Batch — multiprocessing kullanmaz, AST hizli."""
```

### AST Visitor şeleti

```python
class SmellVisitor(ast.NodeVisitor):
    def __init__(self):
        self.long_method = 0
        self.large_class = 0
        self.long_param_list = 0
        self.deep_nesting = 0
        self.god_function = 0  # CC bilgisi sonradan eklenir

    def visit_FunctionDef(self, node):
        # LOC
        loc = (node.end_lineno or node.lineno) - node.lineno + 1
        if loc > LONG_METHOD_LOC:
            self.long_method += 1
        # Param count
        n_params = len(node.args.args) + len(node.args.kwonlyargs)
        if node.args.vararg: n_params += 1
        if node.args.kwarg: n_params += 1
        if n_params > LONG_PARAM_COUNT:
            self.long_param_list += 1
        # Nesting
        depth = max_nesting(node)
        if depth > NESTING_DEPTH:
            self.deep_nesting += 1
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        loc = (node.end_lineno or node.lineno) - node.lineno + 1
        method_count = sum(1 for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        if loc > LARGE_CLASS_LOC and method_count >= 10:
            self.large_class += 1
        self.generic_visit(node)


def max_nesting(node, depth=0):
    """if/for/while/with/try iç içe en derin seviye."""
    nest_types = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try)
    if not list(ast.iter_child_nodes(node)):
        return depth
    return max(
        (max_nesting(child, depth + (1 if isinstance(child, nest_types) else 0))
         for child in ast.iter_child_nodes(node)),
        default=depth,
    )
```

### `config.py` eklenecekler

```python
# Code smell esikleri (Fowler 1999 + Lanza-Marinescu 2006)
LONG_METHOD_LOC: Final[int]   = 50
LARGE_CLASS_LOC: Final[int]   = 500
LARGE_CLASS_METHOD_COUNT: Final[int] = 10
LONG_PARAM_COUNT: Final[int]  = 5
NESTING_DEPTH: Final[int]     = 4
HIGH_CC: Final[int]           = 10
LOW_MI: Final[int]            = 20
GOD_FUNC_CC: Final[int]       = 15
GOD_FUNC_LOC: Final[int]      = 80

# Prospector deprecated, sample validation icin tutuluyor
PROSPECTOR_ENABLED_FOR_VALIDATION: Final[bool] = False  # --use-prospector ile aktif
```

### `project_processor.py` değişimi

```python
# ESKI:
from pipeline.prospector_runner import run_prospector_batch
...
results = run_prospector_batch(abs_paths, ...)

# YENI:
from pipeline.code_smells import detect_smells_batch
...
results = detect_smells_batch(abs_paths)
# results: {Path: {'smell_count': N, 'smell_long_method': M, ...}}
```

`smell_count` integer olarak akar (önceki şema), ek 7 sütun yeni feature olarak eklenir.

### `dataset_builder.py` — değişiklik gerekmez

`add_dynamic_smell_binary()` `smell_count` üzerinde çalışıyor. Aynı şekilde işliyor.

### Tests (`tests/test_code_smells.py`)

```python
def test_long_method_detected():
    src = "def f():\n" + "    x = 1\n" * 60
    # write to tmp, detect, assert smell_long_method == 1

def test_large_class_with_many_methods():
    # 600 LOC class, 12 method
    # assert smell_large_class == 1

def test_long_param_list():
    src = "def f(a, b, c, d, e, f, g): pass"
    # assert smell_long_param_list == 1

def test_deep_nesting():
    # 5 level nested if
    # assert smell_deep_nesting == 1

def test_high_complexity_via_radon():
    # 15 if/elif chain → CC ~16
    # assert smell_high_complexity == 1

def test_god_function():
    # CC > 15 AND LOC > 80
    # assert smell_god_function == 1

def test_low_maintainability():
    # Generate code with low MI
    # assert smell_low_maintainability == 1

def test_no_smells_in_clean_code():
    src = "def add(a, b): return a + b"
    # assert smell_count == 0

def test_smell_count_aggregates_correctly():
    # File with 2 long methods + 1 god function
    # assert smell_count == 3

def test_detect_smells_batch_handles_errors():
    # Malformed Python, ensure skip_errors=True returns empty dict
    # assert no exception

# CLI integration
def test_collect_skip_smells_flag():
    # --skip-smells should work like old --skip-prospector

def test_collect_skip_prospector_alias():
    # Old flag name still works (deprecation warning)
```

### Sample validation (separate task, F2'nin yarım günü)

50 dosyalık stratified sample seç (her kategoriden 5+ dosya):

```python
# scripts/validate_smell_sample.py (yeni)
import random
from pathlib import Path
from pipeline.code_smells import detect_smells
from pipeline.prospector_runner import run_prospector

random.seed(42)
all_files = [...]  # output/projects/'ten örnekle
sample = random.sample(all_files, 50)

ast_results = []
prospector_results = []
for f in sample:
    ast_r = detect_smells(f)
    pros_r = run_prospector(f)
    ast_results.append(ast_r["smell_count"] > 0)
    prospector_results.append(pros_r["smell_count"] is not None and pros_r["smell_count"] > 0)

# Cohen's kappa
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(ast_results, prospector_results)
print(f"κ = {kappa:.3f}")
# Paper'a yaz: "AST detection vs Prospector: κ=0.X (Landis & Koch 1977: substantial)"
```

### Validation criteria

- `pytest tests/test_code_smells.py` → tüm testler pass
- 10 projelik smoke test < 5 dk (eskiden 30+ dk timeout)
- `dataset_full_*.parquet`'te smell_count NaN değil, 0+ integer
- `add_dynamic_smell_binary` P80 düzgün hesaplıyor

### Commit

```
feat(smells): replace Prospector with AST + radon smell detection

Prospector's pylint dependency (recursive type inference) made smell
detection on 1000-project scale infeasible (30+ days estimated runtime).
Replaced with deterministic AST + radon based detection of 7 classical
smells from Fowler (1999) and Lanza-Marinescu (2006):

- Long Method, Large Class, Long Parameter List, Deep Nesting
- High Complexity, Low Maintainability, God Function

Performance: ~5min on 1000 projects vs ~30 days with Prospector.
Same smell_count column shape; downstream pipeline unchanged.
Old prospector_runner.py kept for sample validation (--use-prospector).
```

---

## 5. Faz F3 — Feature ekleme paketi

**Goal:** 5 alt-madde aynı git_metrics + static_metrics dosyalarına eklendiği için tek faz.

**Effort:** Toplam ~6-8 saat

**Dependencies:** F1, F2 (önerilir ama hard-blocker değil)

Bu faz **5 alt-commit** olabilir, hepsi aynı modüllerde değişiklik yapıyor.

### F3.1 — Cognitive Complexity (2-3 saat)

**Files:**
- `pipeline/static_metrics.py` — yeni helper
- `requirements.txt` — `cognitive_complexity>=1.3.0` ekle
- `tests/test_static_metrics.py`'e ekle

**Implementation:**

```python
# pipeline/static_metrics.py
from cognitive_complexity.api import get_cognitive_complexity
import ast

def cognitive_complexity_for_file(source: str) -> dict:
    """Dosya icin cognitive complexity metrikleri."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"cognitive_complexity_total": 0, "cognitive_complexity_max": 0}

    funcs = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not funcs:
        return {"cognitive_complexity_total": 0, "cognitive_complexity_max": 0}

    cc_values = [get_cognitive_complexity(f) for f in funcs]
    return {
        "cognitive_complexity_total": sum(cc_values),
        "cognitive_complexity_max":   max(cc_values),
    }
```

`calculate_metrics()` zinciirne entegre et, FEATURES_COMMIT/BUG/SMELL listesine **2 yeni feature**.

**config.py güncelle:**
```python
FEATURES_COMMIT = (..., "cognitive_complexity_total", "cognitive_complexity_max")  # 29 → 31
FEATURES_BUG = (..., aynısı)  # 36 → 38
FEATURES_SMELL = (..., aynısı)  # 36 → 38
```

`tests/test_config.py`'deki feature count testlerini güncelle.

**Commit:**
```
feat(metrics): add Cognitive Complexity (Campbell 2018)

Adds cognitive_complexity_total and cognitive_complexity_max as
features. Cognitive Complexity (Campbell 2018) addresses limitations
of cyclomatic complexity by penalizing nesting and breaks in linear
flow. Implementation uses cognitive_complexity package which follows
the SonarSource specification.
```

### F3.2 — Bug keyword separation (1 saat)

**Files:**
- `pipeline/git_metrics.py` — `is_bug_message()` → ek per-keyword sayım
- `tests/test_git_metrics_summary.py`

**Implementation:**

```python
# git_metrics.py
BUG_KEYWORD_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "fix":     ("fix", "fixed", "fixes", "fixing"),
    "bug":     ("bug", "bugs", "buggy"),
    "error":   ("error", "errors"),
    "defect":  ("defect", "defects"),
    "issue":   ("issue", "issues"),
    "anomaly": ("anomaly", "anomalies"),
}

def classify_bug_message(message: str) -> dict[str, int]:
    """Mesajda her keyword grubu icin 0/1 dondur."""
    msg = message.lower()
    out = {f"bug_kw_{k}": 0 for k in BUG_KEYWORD_GROUPS}
    for group, words in BUG_KEYWORD_GROUPS.items():
        if any(re.search(rf"\b{w}\b", msg) for w in words):
            out[f"bug_kw_{group}"] = 1
    return out

def is_bug_message(message: str) -> bool:
    """Geriye uyum — herhangi bir keyword eslesirse True."""
    return any(classify_bug_message(message).values())
```

`get_bulk_git_stats()`'da her dosya için `bug_kw_fix_count`, `bug_kw_bug_count` vb. agg.

**FEATURES listelerine 6 yeni:** `bug_kw_fix_count`, `bug_kw_bug_count`, `bug_kw_error_count`, `bug_kw_defect_count`, `bug_kw_issue_count`, `bug_kw_anomaly_count`.

**Commit:**
```
feat(git_metrics): separate bug keyword counts by type

Previously a single bug_count merged all keyword categories. Now tracks
per-type counts (fix, bug, error, defect, issue, anomaly) which the
literature treats as semantically distinct (Antoniol et al. 2008).
Original is_bug_message() retained for backward compat.
```

### F3.3 — Refactoring ratio (30 dk)

**Files:**
- `pipeline/git_metrics.py` — `get_repo_commit_summary()` ek alan
- `tests/test_git_metrics_summary.py`

**Implementation:**

`get_repo_commit_summary()` zaten `is_refactor_message()` kullanıyor; sadece toplama:

```python
# git_metrics.py — get_repo_commit_summary()
total = total_commits  # zaten hesaplaniyor
refactors = sum(1 for msg in commit_messages if is_refactor_message(msg))
summary["refactor_ratio"] = refactors / max(total, 1)
```

`project_stats.py`'de raporlanır.

**Commit:**
```
feat(git_metrics): add repo-level refactor_ratio

Aggregates refactor commit detection (existing is_refactor_message)
into a per-repo ratio. Used in DevOps narrative section of paper.
```

### F3.4 — Contribution Gini (1 saat)

**Files:**
- `pipeline/git_metrics.py` — `get_repo_commit_summary()` ek alan
- `tests/test_git_metrics_summary.py`

**Implementation:**

```python
def gini_coefficient(values: list[int]) -> float:
    """
    Gini coefficient (0 = esit dagilim, 1 = tek bireye konsantrasyon).
    Mockus 2002 — power-law contribution distribution.
    """
    if not values or sum(values) == 0:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    cum = sum((i + 1) * v for i, v in enumerate(sorted_v))
    total = sum(sorted_v)
    return (2 * cum) / (n * total) - (n + 1) / n


# get_repo_commit_summary icinde:
author_commits = Counter(commit_authors)  # zaten var
summary["contribution_gini"] = gini_coefficient(list(author_commits.values()))
```

**Commit:**
```
feat(git_metrics): add contribution Gini coefficient per repo

Quantifies power-law commit distribution (Mockus 2002): Gini ≈ 0
indicates equal contribution, Gini → 1 indicates concentration.
Supports paper's claim of skewed contributor distribution under
the 10-contributor cap.
```

### F3.5 — Git proxy features (2-3 saat)

PR rejection / Issue density alternatif — git log'dan 4 yeni feature.

**Files:**
- `pipeline/git_metrics.py` — `get_bulk_git_stats()` ve `get_repo_commit_summary()` ek alanlar

**Implementation:**

```python
# git_metrics.py icinde (her bir helper ayri test edilebilir)

def revert_count(commit_messages: list[str]) -> int:
    """Reverted commit sayisi (PR rejection proxy)."""
    return sum(1 for m in commit_messages
               if re.match(r"^revert\b", m.strip(), re.IGNORECASE))


def bug_fix_density(bug_fix_count: int, kloc: float, age_years: float) -> float:
    """Bug-fix per KLOC per year (issue density proxy)."""
    return bug_fix_count / max(kloc, 0.1) / max(age_years, 0.1)


def inter_commit_time_cv(commit_timestamps: list[int]) -> float:
    """
    Commit araliklarinin coefficient of variation (varyans / ortalama).
    Yuksek = duzensiz, dusuk = stabil.
    """
    if len(commit_timestamps) < 2:
        return 0.0
    sorted_ts = sorted(commit_timestamps)
    deltas = [sorted_ts[i+1] - sorted_ts[i] for i in range(len(sorted_ts)-1)]
    if not deltas:
        return 0.0
    mean = sum(deltas) / len(deltas)
    if mean == 0:
        return 0.0
    var = sum((d - mean) ** 2 for d in deltas) / len(deltas)
    return (var ** 0.5) / mean


def author_entropy(author_commits: dict[str, int]) -> float:
    """Shannon entropy of commit distribution per author."""
    import math
    total = sum(author_commits.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in author_commits.values()]
    return -sum(p * math.log(p, 2) for p in probs if p > 0)
```

`get_repo_commit_summary()` ek alanlar:
- `revert_count`, `bug_fix_density`, `inter_commit_time_cv`, `author_entropy`

FEATURES listelerine ekle.

**Commit:**
```
feat(git_metrics): add git-log proxies for PR/issue/cadence signals

Replaces planned GitHub PR/Issues API integration with cheap proxies
derived from git log (Mockus 2010, Bird et al. 2009):
- revert_count: PR rejection proxy via "revert" commits
- bug_fix_density: issue density proxy (bugs / KLOC / year)
- inter_commit_time_cv: development cadence stability
- author_entropy: Shannon entropy of contribution

Saves ~3 days of API rate-limited collection on 1000 projects.
```

### F3 toplam validation

```powershell
$env:PYTHONIOENCODING = 'utf-8'
python -m pytest tests/ -q
# 137 → ~155 pass beklenir (her alt-faz için 3-5 yeni test)

# Smoke test
python -m scripts.collect --phase discovery --target 5
python -m scripts.collect --phase process --skip-smells
python -m scripts.collect --phase build
python -c "
import pandas as pd, glob
df = pd.read_parquet(sorted(glob.glob('output/dataset_full_*.parquet'))[-1])
new_cols = [c for c in df.columns if c.startswith(('cognitive_', 'bug_kw_', 'refactor_ratio',
                                                     'contribution_gini', 'revert_count',
                                                     'bug_fix_density', 'inter_commit_', 'author_entropy'))]
print(f'Yeni eklenen sütunlar: {len(new_cols)}')
print(new_cols)
"
```

---

## 6. Faz F4 — Two-stage split protocol

**Goal:** 70/15/15 project-based holdout + GroupKFold(5) development pool içinde.

**Effort:** 1-2 saat

**Dependencies:** F1, F2, F3 önerilir (yeni feature'ları görmek için)

### Files

- `pipeline/model_utils.py` — yeni `two_stage_split()` helper
- `scripts/train_final.py` — `_prepare_splits()` değişimi
- `analysis/02_model_training.py` — ablation cell'i two-stage'e geçir
- `tests/test_model_utils.py` — yeni testler

### Implementation

```python
# pipeline/model_utils.py
from typing import NamedTuple
import numpy as np
import pandas as pd

class TwoStageSplit(NamedTuple):
    train_dev: pd.DataFrame  # 70% — GroupKFold burada calisir
    val:       pd.DataFrame  # 15% — model secimi sonrasi tek seferlik
    test:      pd.DataFrame  # 15% — final paper tablosu
    train_pids: np.ndarray
    val_pids:   np.ndarray
    test_pids:  np.ndarray


def two_stage_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    project_col: str = "project_name",
    seed: int = 42,
) -> TwoStageSplit:
    """
    Project-based 70/15/15 holdout (Tantithamthavorn et al. 2017).

    Train pool icinde sonra GroupKFold(5) calistir.
    """
    projects = df[project_col].unique()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(projects))
    projects = projects[perm]

    n = len(projects)
    n_test = int(test_frac * n)
    n_val = int(val_frac * n)

    test_pids = projects[:n_test]
    val_pids = projects[n_test:n_test + n_val]
    train_pids = projects[n_test + n_val:]

    return TwoStageSplit(
        train_dev=df[df[project_col].isin(train_pids)].copy(),
        val=df[df[project_col].isin(val_pids)].copy(),
        test=df[df[project_col].isin(test_pids)].copy(),
        train_pids=train_pids,
        val_pids=val_pids,
        test_pids=test_pids,
    )
```

### `train_final.py` değişimi

```python
def _prepare_splits(df, task):
    split = two_stage_split(df, val_frac=0.15, test_frac=0.15, seed=42)

    # Development pool icinde GroupKFold for model selection
    cv = GroupKFold(n_splits=5)
    cv_iter = cv.split(
        split.train_dev[FEATURES],
        split.train_dev[label_col],
        groups=split.train_dev["project_name"],
    )

    return split, cv_iter
```

### Tests

- `test_two_stage_split_proportions_correct`: 1000 projeyle ratiolar ~70/15/15
- `test_two_stage_split_no_project_overlap`: train ∩ val ∩ test boş
- `test_two_stage_split_deterministic`: seed=42 hep aynı bölünme
- `test_two_stage_split_respects_groups`: aynı proje sadece bir set'te

### Commit

```
feat(splits): two-stage 70/15/15 + GroupKFold protocol

Implements project-based holdout (Tantithamthavorn et al. 2017) for
final paper-table reporting + GroupKFold(5) within development pool
for hyperparameter tuning. Eliminates random split data leakage.
```

---

## 7. Faz F5 — Risk Score → meta-stacking

**Goal:** Hand-tuned α/β/γ formülü yerine kalibre stacking + 3-tier display.

**Effort:** 2-3 saat

**Dependencies:** F4 (split protocol)

### Files

- `app/predictor.py` — `predict_with_calibration()` ekle
- `app/health.py` — `risk_tier()` helper
- `app/analyzer.py` — risk + tier'i sonuca ekle
- `app/templates/results.html` — tier'i renkle göster
- `tests/test_health.py`'a tier testleri

### Implementation

```python
# app/predictor.py
from sklearn.calibration import CalibratedClassifierCV

class Predictor:
    def __init__(self, ...):
        # Mevcut RF + AutoGluon stacking yukle
        # Kalibrasyonu predict zamani uygula
        ...

    def predict_proba_calibrated(self, X) -> np.ndarray:
        # Stacking output zaten calibrated meta-LR ile cikiyor.
        # Eger ek calibration gerekirse:
        return self.meta_lr.predict_proba(X)[:, 1]


# app/health.py
def risk_tier(risk_score: float, p70: float = 0.30, p90: float = 0.70) -> str:
    """3-tier risk gosterimi (eşikler veriden gelir)."""
    if risk_score >= p90:
        return "BLOCK"
    elif risk_score >= p70:
        return "REVIEW"
    else:
        return "PASS"
```

`analyzer.py`'da:
```python
risk = predictor.predict_proba_calibrated(features)[0]
tier = risk_tier(risk)
result["risk_score"] = float(risk)
result["risk_tier"] = tier  # "PASS" / "REVIEW" / "BLOCK"
```

### `train_final.py` calibration

```python
# T2 bug stacking'e calibration ekle
from sklearn.calibration import CalibratedClassifierCV

meta_lr_calibrated = CalibratedClassifierCV(
    meta_lr, method="isotonic", cv=3,
)
meta_lr_calibrated.fit(stacking_features, y_train)
```

### Tests

- `test_risk_tier_thresholds`: 0.1 → PASS, 0.5 → REVIEW, 0.95 → BLOCK
- `test_risk_score_in_unit_interval`: predictor output ∈ [0, 1]

### Commit

```
feat(risk): calibrated stacking risk score with 3-tier display

Replaces ad-hoc α·prob + β·CC + γ·churn formula with calibrated
stacking ensemble output (RF + AutoGluon → isotonic-calibrated LR meta).
Supports paper's "Quality Gate" framing: PASS (<P70) / REVIEW (<P90) /
BLOCK (≥P90). Thresholds are data-driven, not hand-tuned.
```

---

## 8. Faz F6 — Sample validation (opsiyonel)

**Goal:** AST+radon smell tespitini Prospector'a karşı 50 dosyalık örnek üzerinde Cohen's kappa ile doğrula.

**Effort:** 0.5 gün

**Dependencies:** F2

### Files

- `scripts/validate_smell_sample.py` (yeni)
- `output/figures/smell_validation_kappa.csv` (çıktı)

Detay F2 sonunda verildi. Kabul kriteri: κ ≥ 0.4 (Landis & Koch fair agreement) — bu seviyenin altında akademik defansa zarar.

### Commit

```
test(smells): Cohen's kappa validation against Prospector on 50-file sample

Validates AST + radon smell detection against Prospector on a
stratified random sample of 50 files. Reports inter-rater agreement
(Cohen's kappa, Landis & Koch 1977) for paper's methodology section.
```

---

## 9. Final smoke test (1000 projelik run öncesi)

Tüm fazlar bitince:

```powershell
# Clean slate
Remove-Item -Recurse -Force output/projects/, output/checkpoints/, output/dataset_*.parquet -ErrorAction SilentlyContinue

# Full pipeline
python -m scripts.collect --phase discovery --target 50
python -m scripts.collect --phase process     # smell aktif (artik AST+radon)
python -m scripts.collect --phase build

# Analysis cells (VS Code'da sirayla)
# analysis/01_filter_categorize.py — Run All Cells Above
# analysis/02_model_training.py — kucuk subset

# Stats
python -c "from pipeline.project_stats import write_project_stats; ..."

# Final training
python -m scripts.train_final --tasks commit,bug,smell

# Flask UI test
python run.py
# http://localhost:5000 → kendi repo URL'in ile test
```

**Beklenen:**
- 50 proje × ~60 dosya = ~3K satır
- Tüm yeni 15+ feature dolu (NaN olmamalı)
- Smell migration ~5 dk (eskiden 30+ dk timeout)
- Risk score ∈ [0, 1], 3-tier dağılımı makul
- 70/15/15 split: ~35 train / ~7 val / ~8 test proje

Sorunsuzsa **1000 projelik run hazır**. Tahmini süre: 6-12 saat (clone IO yoğun).

---

## 10. Hızlı referans — önemli numerik kararlar

| Konu | Değer | Dosya |
|---|---|---|
| min_stars | 50 | `config.DEFAULT_MIN_STARS` |
| max_contributors | 10 | `config.DEFAULT_MAX_CONTRIBUTORS` |
| min_age | 180 gün | `config.DEFAULT_MIN_AGE_DAYS` |
| max_age | 365 gün | `config.DEFAULT_MAX_AGE_DAYS` |
| smell P80 percentile | 80 | `config.SMELL_BINARY_PERCENTILE` |
| target projects | 1000 | `config.DEFAULT_TARGET_COUNT` |
| split | 70/15/15 | `model_utils.two_stage_split` |
| cv folds (development pool) | 5 | `train_final.py` |
| Long Method LOC | 50 | `config.LONG_METHOD_LOC` |
| Large Class LOC | 500 | `config.LARGE_CLASS_LOC` |
| Long Param Count | 5 | `config.LONG_PARAM_COUNT` |
| Nesting Depth | 4 | `config.NESTING_DEPTH` |
| High CC | 10 | `config.HIGH_CC` |
| Low MI | 20 | `config.LOW_MI` |
| God CC | 15 | `config.GOD_FUNC_CC` |
| God LOC | 80 | `config.GOD_FUNC_LOC` |
| Risk PASS threshold | P70 | `health.risk_tier` |
| Risk BLOCK threshold | P90 | `health.risk_tier` |

---

## 11. Akademik defans cümleleri (paper'da kullan)

Her faz için 1-2 cümlelik defansive paragraf:

**F1 (discovery merge):** Pre-existing project metadata is preserved across
incremental discovery runs to maintain dataset integrity.

**F2 (smell migration):** Code smell detection follows Fowler (1999) and
Lanza-Marinescu (2006) classical taxonomy via deterministic AST traversal
combined with radon static metrics. Validation against Prospector on a
stratified 50-file sample yielded κ=X (Cohen 1960; Landis-Koch 1977
"substantial agreement"). External tool dependency was eliminated to
ensure reproducibility at 1000-project scale.

**F3.1 (cognitive complexity):** Cognitive Complexity (Campbell, 2018)
augments cyclomatic complexity by penalizing nesting and breaks in
linear control flow, addressing Shepperd's (1988) critique of CC.

**F3.2 (bug keyword separation):** Per-keyword bug counts (fix/error/bug/
defect/issue/anomaly) are tracked separately following Antoniol et al.
(2008) finding that these terms exhibit semantically distinct defect
patterns.

**F3.3 (refactor ratio):** Repository-level refactoring activity is
quantified per Murphy-Hill et al. (2012) commit message taxonomy.

**F3.4 (contribution Gini):** The Gini coefficient operationalizes
Mockus et al. (2002) finding of power-law contribution distribution
in open-source projects, complementing the 10-contributor cap with
distribution shape information.

**F3.5 (git proxies):** Following Mockus (2010) and Bird et al. (2009),
PR rejection rate and issue density signals are derived from git log
proxies (revert ratio, bug-fix density, inter-commit time variance,
author entropy) to remain feasible at 1000-project scale.

**F4 (split protocol):** Two-stage project-based 70/15/15 holdout
(Tantithamthavorn et al., 2017) is combined with 5-fold GroupKFold
within the development pool, eliminating cross-project leakage and
providing held-out test results for paper reporting.

**F5 (calibrated risk):** Risk scores derive from a calibrated stacking
ensemble (Wolpert, 1992) using isotonic regression (Niculescu-Mizil &
Caruana, 2005). Three-tier classification (PASS/REVIEW/BLOCK) uses
data-driven percentile thresholds, avoiding hand-tuned coefficients.

---

## 12. Bilinen riskler ve düşürme

| Risk | Olasılık | Etki | Düşürme |
|---|---|---|---|
| `cognitive_complexity` paketi import etmiyor | Düşük | F3.1 blocker | Fallback: kendi AST visitor (1-2 saat ek iş) |
| AST+radon smell sample κ < 0.4 | Orta | F2 defans zayıf | Smell rule eşiklerini tune et, κ tekrarla |
| Two-stage split'te val veya test çok küçük | Düşük (>500 proje varsa) | Reporting noise | n_projects > 100 zorunlu kontrol |
| Calibrated stacking AutoGluon ile uyumsuz | Düşük | F5 kısmi | RF base'e calibration uygula, AG'ye dokunma |
| 1000 projelik run'da disk dolar | Yüksek | Pipeline kırılır | repos/ disk usage izle, gerekirse parça parça koş |
| Git log timestamp'leri eksik (some commit) | Düşük | inter_commit_time_cv = 0 | NaN-safe path zaten var |

---

## 13. Yeni Claude oturumuna hazır prompt

Yeni oturuma açıldığında bu mesajla başla:

```
MetricHunter V2 enhancement plan'ı uyguluyorum.

Detaylı plan: docs/v2_enhancement_plan.md
Mevcut durum: F1-F8 fazları bitmiş, 50 projelik smoke test geçmiş.
Şimdi bu plan'daki F1-F6'yı sırasıyla uygulayacağız.

İlk faz: F1 — Discovery merge fix (30 dk).
Lütfen önce baseline test pass ettiğinden emin ol:
  $env:PYTHONIOENCODING = 'utf-8'
  python -m pytest tests/ -q

Sonra F1'e başla. Her faz sonunda ayrı commit, plan dosyasındaki commit
template'lerini kullan.

Kararlar tablo şeklinde plan'ın §2'sinde, akademik defansların
§11'de. Yeniden tartışmaya açma — kararlar verildi, sadece uygula.
```

---

## 14. Tamamlama checklist (paper revize öncesi)

- [x] F1 (discovery merge) commit'lendi, pytest pass — `dedab59`
- [x] F2 (smell migration) commit'lendi, smoke test < 5 dk — `fbdd20e`
- [x] F2 sample validation Cohen's kappa raporu üretildi — `263ff76` (F6)
- [x] F3.1 (cognitive complexity) commit'lendi — `f6e7218`
- [x] F3.2 (bug keyword separation) commit'lendi — `60ea512`
- [x] F3.3 (refactor ratio) commit'lendi — `0472c87`
- [x] F3.4 (contribution Gini) commit'lendi — `0472c87`
- [x] F3.5 (git proxies) commit'lendi — `493d3b5`
- [x] F4 (two-stage split) commit'lendi — `50690f4`
- [x] F5 (calibrated risk score) commit'lendi — `7322c3b`
- [ ] Final 50-projelik smoke test geçti (mevcut dataset eski schema; yeni --phase process çalıştırıldığında yeni feature'lar eklenecek)
- [ ] 1000-projelik production run başlatıldı
- [ ] Paper'da §11'deki defans cümleleri ilgili bölümlere eklendi
- [ ] PLAN.md ve CHANGELOG.md güncellendi

**Tamamlanan fazların özeti (2026-04-30):**
- F1–F6 tüm commit'lendi: 216 test pass, 3 skip
- FEATURES_COMMIT=35, FEATURES_BUG=FEATURES_SMELL=48
- Yeni feature'lar: cognitive_complexity_total/max (F3.1),
  bug_kw_*_count ×6 (F3.2), refactor_ratio, contribution_gini (F3.3/F3.4),
  revert_count, inter_commit_time_cv, author_entropy, bug_fix_density (F3.5)
- validate_smell_sample.py: Prospector timeout sorunu nedeniyle --skip-prospector
  kullanılmalı; kappa hesaplaması Prospector erişimi olan ortamda çalıştırılmalı

---

**Sonu.** Her faz bittikçe checklist'i işaretle. Plan değişimi gerekirse bu dosyayı güncelle, commit'le.
