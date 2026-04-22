# MetricHunter V2 — Uygulama Planı

Son güncelleme: 2026-04-22

Bu dosya, akademisyen geri bildirimleri sonrası MetricHunter projesinin yeni
mimari yönünü tanımlar. Mevcut V1 (commit + bug tahmini) üzerine inşa edilir,
code smell tespiti ve geniş ölçekli veri üretimi ile genişletir.

---

## 0. Amaç

**AutoML temelli hibrit AI model** ile Python dosyaları üzerinde:

1. Commit yoğunluğu tahmini (mevcut — T1)
2. Modül bazlı yazılım hatası tespiti (mevcut iyileştirilir — T2)
3. Code smell / refactor ihtiyacı tespiti (**yeni** — T3)

Veri: 1000 Python projesi, yaklaşık 1M+ LOC, dosya bazlı kayıtlar.

---

## 1. Görev yapısı

Üç görev, hepsi dosya bazlı binary sınıflandırma.

| Görev | Label kaynağı | Feature seti | Özellik sayısı |
|---|---|---|---|
| T1 Commit volume | median(commit_count) per project | Static + derived + project | 29 |
| T2 Bug | SZZ (pydriller) + keyword baseline | Static + derived + project + process | 36 |
| T3 Smell | Prospector smell_count + dynamic threshold | Static + derived + project + process | 36 |

### Smell etiketi — fazlı yaklaşım

- **Phase A:** Binary — `has_smell = smell_count >= project_percentile_80`
- **Phase B:** Count — `smell_count` doğrudan regresyon/tahmin
- **Phase C (ileride):** Kategorik — Long Method, God Class, Duplicated Code vs.
  (proje tamamlandığında kalan süreye göre değerlendirilir)

### Etiket kaynakları (üç versiyon paralel üretilir)

Her dosya kaydında şu etiketler bulunur:

- `bug_keyword` — commit mesajı regex'i (mevcut baseline)
- `bug_szz` — SZZ algoritması (pydriller ile)
- `smell_binary` — Prospector + dynamic threshold
- `smell_count` — Prospector ham sayı

Model eğitiminde hangi etiketin kullanılacağı `model_training.ipynb`
içinde ablation ile belirlenir.

---

## 2. Mimari kararlar

### 2.1. Dosya yapısı değişikliği

```
Final/
  pipeline/                     # YENI - Python modulu (tum agir kod)
    __init__.py
    config.py                   # Sabitler, esikler, API ayarlari
    discovery.py                # GitHub search + filter
    cloning.py                  # Repo klonlama + temizleme
    static_metrics.py           # Radon (metrics.py'den tasinir)
    git_metrics.py              # git_utils'ten tasinir
    szz.py                      # YENI - pydriller SZZ implementasyonu
    prospector_runner.py        # YENI - Prospector calistirma + parse
    commits_before_bug.py       # YENI - deskriptif istatistik
    checkpoint.py               # YENI - faz/proje checkpoint yonetimi
    rate_limit.py               # YENI - GitHub API rate limit handler
    dataset_builder.py          # Tum parca dosyalari birlestirir
    model_utils.py              # YENI - split, scaler, metric helperlari
  scripts/                      # YENI - Batch CLI scriptler
    collect.py                  # pipeline_v6.ipynb yerine (tam veri toplama)
    train_final.py              # train_final.ipynb yerine (model artifactlari)
  analysis/                     # YENI - Interactive analiz (.py + # %% cells)
    01_filter_categorize.py     # filter_categorize.ipynb yerine
    02_model_training.py        # model_training.ipynb yerine
    03_results_exploration.py   # YENI - sonuc kesfi, plotlar
  app/                          # Flask app (genisletilir)
    # Flask, pipeline/ modulunden import eder (metrics.py/git_utils.py tasinir)
  output/
    checkpoints/                # YENI - faz checkpointleri
    projects/                   # YENI - per-project parquet dosyalari
    dataset_full_*.parquet      # Birlestirilmis final
    figures/                    # YENI - analiz scriptlerinin kaydettigi PNG/PDF
    ...
  models/
    commit_rf.joblib            # T1
    scaler_commit.joblib
    bug_rf_base.joblib          # T2 base
    bug_ag_base/                # T2 base
    bug_meta_lr.joblib          # T2 meta
    scaler_bug.joblib
    smell_*.joblib              # T3 (mimari model_training sonucuna gore)
    scaler_smell.joblib
    feature_names.json          # 3 gorevli
    project_stats.json          # YENI - global istatistikler (Flask icin)
  .vscode/
    settings.json               # YENI - Jupyter cell markers, interpreter ayari
```

### 2.1.1. Script vs analysis ayrımı

- `scripts/` — **Batch / long-running**. CLI argümanları alır, checkpoint yazar,
  terminal/screen/tmux'ta çalışır. Saf `.py`, `# %%` yok.
  Örnek kullanım: `python -m scripts.collect --target 1000 --resume`

- `analysis/` — **Interactive analiz**. VS Code'da `# %%` cell markers ile
  hücre hücre çalıştırılır (Jupyter extension). Plotlar inline görünür,
  `plt.savefig("output/figures/X.png")` ile dosyaya da kaydeder.
  Git diff temiz (Python kodu), kernel crash riski yok.
  Gerekirse `jupytext --to ipynb analysis/02_model_training.py` ile
  `.ipynb` üretilebilir (akademisyene göndermek için).

### 2.1.2. Eski notebook'lar

V1 notebook'ları silinmiyor — referans ve karşılaştırma için `archive/v1/`
klasörüne taşınır:

```
archive/v1/
  pipeline_v6.ipynb
  filter_categorize.ipynb
  model_training.ipynb
  train_final.ipynb
```

### 2.2. Format tercihleri

| Veri türü | Format | Sebep |
|---|---|---|
| Büyük dataset'ler (dataset_full, intermediate) | Parquet | Boyut, hız, tip koruma |
| Per-project intermediate | Parquet | Kesinti toleransı |
| Metadata (enriched_repos, categories) | JSON | Human-readable, küçük |
| Model artifacts | joblib + AutoGluon dir | Scikit-learn standard |
| Feature names, istatistikler | JSON | Human-readable |
| Akademik raporlar | CSV (yanında) | Excel uyumu |

### 2.3. Checkpoint & resume

Her faz kendi checkpoint dosyasını yazar:

- `output/checkpoints/discovery.json` — bulunan 1000 proje listesi
- `output/checkpoints/processed_projects.json` — işlenmiş proje set'i
- `output/checkpoints/phase_status.json` — genel faz durumu

Pipeline yeniden başlatıldığında:
1. processed_projects.json okunur
2. Bu listede olmayan projelere devam edilir
3. Her proje başarılı işlenince set'e eklenir ve diske flush edilir

### 2.4. Rate limit stratejisi

- Her GitHub API çağrısı `rate_limit.guarded_get()` üzerinden
- Header'dan `X-RateLimit-Remaining` takip
- Remaining < 10 → reset_at'a kadar bekle (uyku)
- 403 görünürse exponential backoff (5, 15, 45, 135 sn)
- Token yoksa uyarı + devam (çok yavaş ama çalışır)

---

## 3. Pipeline modülü (pipeline/)

### 3.1. config.py
- API endpoint'leri, timeout'lar
- Eşik sabitleri (proje yaş min/max, contributor max)
- Static tool path'leri
- Dataset sütun sıraları

### 3.2. discovery.py
- `search_projects(target_count=1000)` — search-time filtering
- Kriterler:
  - `language:python`
  - `created:<last_year>..<last_6mo>` (yaş 6-12 ay)
  - `stars:>=5` (tamamen amatör projeleri eler)
  - `is:public archived:false fork:false`
- Her sonuç için hızlı enrichment → contributor count ≤ 10 filtresi
- Hedef sayıya ulaşana kadar yıldız/yaş aralığı slide eder
- Checkpoint: her 50 bulunan projede flush

### 3.3. cloning.py
- Mevcut `clone_repo` logic'i
- `--depth=0` (full history — SZZ için gerekli)
- Timeout 10dk → büyük repo skip edilir, loglanır

### 3.4. static_metrics.py
- Mevcut `metrics.py`'den taşınır
- Değişmez, radon tabanlı 22 static + 4 derived

### 3.5. git_metrics.py
- Mevcut `git_utils.py`'den taşınır
- Ek olarak: commit zaman serileri saklanır (sprint analizi için hazırlık)
- Bug keyword yaklaşımı burada kalır (baseline için)

### 3.6. szz.py (YENI)
- `pydriller` wrapper
- Her proje için:
  1. Bug-fix commit'leri bul (keyword-based veya commit mesajından)
  2. Her fix commit için değişen satırları tespit
  3. `git blame` ile bu satırların hangi commit'te eklendiğini bul
  4. Bulunan commit'teki dosya → bug-introducing, işaretle
- Output: her dosya için `bug_szz` (0/1)
- Performans notu: pydriller tüm commit'leri walk eder, büyük repo'da yavaş
  olabilir. Timeout koy (10dk), aşılırsa keyword fallback.

### 3.7. prospector_runner.py (YENI)
- Subprocess: `prospector --output-format json --strictness medium <file>`
- JSON parse → message'ların sayısı = `smell_count`
- Message type'ları per-file saklanır (Phase C için hazır)
- Her dosya 1-3 sn sürer → 1000 proje × ortalama 50 dosya = ~1-3 saat
- Paralelleştirilebilir (multiprocessing) — 4 worker önerim

### 3.8. commits_before_bug.py (YENI)
- SZZ output'unu işler
- Her dosya için:
  - İlk bug-introducing commit'in kaçıncı commit olduğu
  - Ardışık bug'lar arası ortalama commit sayısı
- Proje seviyesinde ortalamalar — Flask health panel için
- Output: `project_stats.json` içine yazılır

### 3.9. checkpoint.py (YENI)
- `save_checkpoint(phase, data)`
- `load_checkpoint(phase) -> dict | None`
- `mark_project_done(project_name)`
- `is_project_done(project_name) -> bool`
- Atomic writes (temp file + rename) → kesintide corruption olmaz

### 3.10. rate_limit.py (YENI)
- `guarded_get(url, **kwargs)` — requests.get wrapper
- Header takibi, auto-sleep, backoff
- Kullanım sayısı istatistiği

### 3.11. dataset_builder.py
- Tüm `output/projects/*.parquet` dosyalarını birleştirir
- 3 etiket versiyonunu sütun olarak ekler
- Dynamic threshold ile `smell_binary` üretir (her projenin kendi percentile'ı)
- Output: `dataset_full_<timestamp>.parquet`

---

## 4. Script ve analiz dosyaları

### 4.1. scripts/collect.py (batch — pipeline_v6 yerine)
- `pipeline.*` modülünü import eder
- CLI argümanları: `--target 1000`, `--resume`, `--skip-szz`, `--skip-prospector`,
  `--phase discovery|process|build`
- Progress: tqdm + log dosyası (`output/logs/collect_<timestamp>.log`)
- Özet istatistikler: toplam LOC, proje sayısı, başarı oranı
- Terminal/screen/tmux'ta çalıştırılır, kernel riski yok
- Checkpoint: her proje sonrası atomic flush

### 4.2. analysis/01_filter_categorize.py (interactive — `# %%` cells)
- Kategori mantığı kalır
- **Agresif sensitivity filtresi kaldırılır** — 25/80 eşikleri kaldırılır
- Dynamic threshold burada uygulanır (smell_binary üretimi)
- Sensitivity analysis yeniden yapılır (yeni veri setinde)
  - Filtresiz vs min=10/max=100 vs min=25/max=80 karşılaştırılır
  - Dramatik fark yoksa filtresiz tercih edilir
- Plotlar: inline (VS Code) + `output/figures/sensitivity_*.png` kaydeder
- Output: `dataset_model_filtered_<timestamp>.parquet`

### 4.3. analysis/02_model_training.py (interactive — ablation matrisi)
Boyutlar:

| Boyut | Değerler |
|---|---|
| Görev | T1, T2, T3 |
| Label (T2) | keyword / SZZ |
| Feature seti | Static / +Derived / +Process / All |
| Model | LR, RF, SVM, XGB, LGBM, AutoGluon, MLP, CNN, LSTM |
| Split | Project-based / Time-based |
| Stacking | Tekil model / Hibrit (RF+AG meta LR) |

Smart pruning: tam matris çok pahalı → feature seti "All" ve en iyi 3 model
(RF, AutoGluon, Stacking) üzerinde full ablation. Diğer modeller baseline.

Split detayı:
- **Primary: project-based 70/15/15**
  - 1000 proje → 700 train / 150 val / 150 test
  - Train içinde hyperparam tuning için GroupKFold(5)
- **Secondary: time-based** (robustness raporu için)
  - Repo içi median commit tarihinden böl

Dengesizlik:
- Doğal skewed dağılım korunur (median split yok)
- SMOTE sadece train fold'unda
- Metrics: F1, PR-AUC, MCC (primary), accuracy ikincil

Hücre hücre çalıştırma: model eğitim hücrelerini tek tek deneyebilir, ara
sonuçları DataFrame olarak inspect edebilirsin. Uzun süren eğitim hücresi
(AutoGluon time_limit=600s) VS Code cell olarak çalışır, sonuç pickle'lanır
bir sonraki çalıştırmada tekrar eğitmeye gerek yok.

### 4.4. analysis/03_results_exploration.py (interactive)
- Model karşılaştırma plotları (bar charts, confusion matrices, PR curves)
- Feature importance grafikleri
- Ablation sonuç tabloları (CSV export → makaleye)
- Hata analizi: yanlış sınıflandırılan dosyalara bak

### 4.5. scripts/train_final.py (batch — train_final.ipynb yerine)
- 3 görev için en iyi mimariyi (model_training'den) eğitir
- Tüm artifact'ları `models/` altına yazar
- Quick sanity test (tek satır tahmini)
- Log dosyasına özet yazar

---

## 5. Flask app değişiklikleri

### 5.1. predictor.py
- 3 tahmin fonksiyonu: `predict_commit`, `predict_bug`, `predict_smell`
- Singleton model loader aynı pattern (lazy + double-checked locking)
- `project_stats.json` de load edilir (Flask UI için)

### 5.2. analyzer.py
Akış:
1. GitHub project info (mevcut)
2. Repo klonlama (mevcut)
3. Python dosyası listesi (mevcut)
4. Radon metrikleri (mevcut)
5. Bulk git stats (mevcut)
6. **YENI: Prospector her dosyada çalıştır** → her dosya için kesin smell sayısı
   - Kullanıcı isterse kapatılabilir (query param `?prospector=false`)
   - Performans şikayeti gelirse default kapalı olur
7. 3 ML tahmini (commit + bug + smell)
8. Proje seviyesinde istatistik hesapla (commits-before-bug, refactor oranı vb.)

### 5.3. Results sayfası

Mevcut dosya tablosuna eklenecek sütunlar:
- **Smell Risk** (ML tahmini, Low/High badge)
- **Smell Prob.** (progress bar)
- **Prospector Count** (kesin tool sonucu, tıklanınca detay)

**Yeni paneller (sidebar veya üst satır kartlar):**

- **Proje sağlığı kartı:**
  - Ortalama ilk-bug'a-kadar commit sayısı
  - Son 6 ay sprint defect density yaklaşık
  - Refactoring commit oranı (keyword yaklaşık: refactor|cleanup|rename)
  - Bug trend grafiği (zaman serisi, küçük spark line)

- **Smell breakdown kartı:**
  - Toplam smell sayısı projede
  - Smell density (smell / KLOC)
  - Top 10 en smelly dosya (mini liste)
  - Phase C'de: smell türleri pie chart

- **Öneri kartı:**
  - "N dosya hem bug risk hem smell risk → öncelikli refactor"
  - "Commit volume yüksek + smell yüksek → kararsız modül"

### 5.4. UI teknik detay
- Mevcut neumorphism tema korunur
- Yeni kartlar aynı stil (card-raised)
- Smell için turuncu accent (bug kırmızı, commit mor ile çakışmaz)
- DataTables mevcut, yeni sütunlar orderable ekle

---

## 6. Fazlı ilerleme

| Faz | Süre | Çıktı |
|---|---|---|
| F1 — Pipeline altyapısı | 1 hafta | `pipeline/` modülü, discovery + checkpoint + resume, 50 projelik pilot çalışır |
| F2 — SZZ + Prospector | 1 hafta | 3 etiket üreten pipeline, 50 projede doğrulanmış |
| F3 — Tam veri toplama | 3-5 gün çalışma | 1000 projelik dataset, 1M+ LOC |
| F4 — Filter & threshold | 2-3 gün | `dataset_model_filtered` üretilir, sensitivity rapor |
| F5 — Model training | 1 hafta | Ablation matrisi, karşılaştırma raporu, en iyi mimariler seçilmiş |
| F6 — Final training | 2-3 gün | 3 görev için model artifactları |
| F7 — Flask V2 | 1 hafta | 3 tahmin + dashboard panelleri + Prospector entegrasyon |
| F8 — Paper + doc | 3-5 gün | Yeni RQ'lar, güncellenmiş bulgular, PLAN.md final |

**Toplam:** ~5-7 hafta (veri toplama süresi dışı ~4-6 hafta aktif iş)

---

## 7. Scope dışı (WON'T — şimdilik)

- **Temporal / sprint-level dataset** — her commit snapshot'ı (10-100x veri)
- **SonarQube entegrasyonu** — Java + Docker + ağır setup
- **Commit-level risk tahmini** — ayrı veri şeması, ayrı görev
- **Refactoring classifier** — fragile, keyword yaklaşık yeter
- **CodeBERT fine-tuning** — mevcut pipeline'da deneme var, training'e girmedi,
  bu sürümde de kapsam dışı
- **Smell kategorileri (Phase C)** — proje sonunda süreye göre değerlendirilir

---

## 8. Açık notlar / ileriye yönelik

- Akademisyene **"1M LOC mu 1M satır kayıt mı"** netleştirmesi: pipeline ikisini
  de raporlar, yanlış anlaşılma kalmaz
- Time-based split sonuçları project-based'le paralel olmalı — değilse akademik
  tartışma konusu, sonuçlar rapora yansır
- SZZ performans sorun olursa sadece bug-fix yoğun projelerde çalıştırılabilir
- Prospector `strictness` seviyesi ayarlanabilir (`veryhigh`, `high`, `medium`,
  `low`, `verylow`) — medium default, veri toplama sonrası tune edilebilir

---

## 10. Geliştirme konvansiyonları

### 10.1. Python sürümü
- **3.10.x** kullanılır. Sebep: mevcut `bug_ag_base/metadata.json` AutoGluon
  1.5.0'ı Python 3.10.6'da eğitmiş. Flask `require_py_version_match=False`
  workaround'u kullanıyor ama yeni pipeline tamamen 3.10'da kalmalı.
- Yeni venv: `python3.10 -m venv venv && venv\Scripts\activate` (Windows)

### 10.2. Kod stili
- **Type hints zorunlu** public fonksiyonlarda
- **Docstring** public fonksiyonlarda — Google style, parametreler + dönüş
- **Emoji yasak** — dosya içeriklerinde, commit mesajlarında, log'larda, UI'da
  (kullanıcı mevcut projede de kullanmadı, devam)
- **Yorum dili: Türkçe** (mevcut projeyle tutarlı)
- **Identifier dili: İngilizce** (standart)
- **Satır uzunluğu: 100** karakter
- **Formatlayıcı: `black`** + **`isort`**
- **Linter: `ruff`** (opsiyonel, CI yok)
- **`from __future__ import annotations`** her modül başında
- **Absolute imports**: `from pipeline.config import X` (relative yok)

### 10.3. Logging
- Python stdlib `logging` (print kullanma, sadece CLI user-facing çıktıda OK)
- Logger: `logger = logging.getLogger(__name__)`
- Format: `%(asctime)s %(levelname)s [%(name)s] %(message)s`
- Seviye politikası:
  - **DEBUG** — API response'ları, per-file detay
  - **INFO** — faz başlangıç/bitiş, her 50 projede progress
  - **WARNING** — skip edilen proje, timeout, missing data
  - **ERROR** — kurtarılamayan hata, traceback ile
- Output: stdout + `output/logs/<script>_<YYYYMMDD_HHMM>.log`
- Rotating handler yok (tek iş, tek log)

### 10.4. Hata yönetimi
- **Her proje bağımsız** — bir proje patlarsa pipeline devam eder
- Hata yakalama ve checkpoint'e "failed" işareti, detay log'a
- Retry politikası:
  - Network 5xx: 3 retry, linear backoff (2/5/10 sn)
  - 403 rate limit: reset_at'a kadar sleep
  - 429 throttle: 60sn bekle, 3 retry
  - Git clone timeout: tek deneme, skip
  - Prospector timeout: tek deneme, o dosya smell_count=None
  - SZZ timeout: tek deneme, bug_szz fallback to bug_keyword
- Kritik hata (disk full, OOM): fail fast, log + exit
- `try/except Exception`'dan kaçın — spesifik exception yakala

### 10.5. Test yaklaşımı
- **pytest** framework, `tests/` altında
- Her yeni modül için minimum 2 unit test
- Ağ çağrıları `responses` library ile mock
- Prospector/git subprocess çağrıları `unittest.mock` ile
- Integration test: 5 projelik mini pipeline end-to-end
- Coverage hedefi yok, akademik proje
- Flask smoke test: `client.get("/")` 200 döndüğünü kontrol

---

## 11. Bağımlılıklar

### 11.1. Yeni paketler (requirements-dev.txt)
```
pydriller>=2.5            # SZZ
prospector>=1.10          # Code smell
pyarrow>=14               # Parquet (autogluon zaten indiriyor)
jupytext>=1.16            # .py <-> .ipynb çevrimi
pytest>=7
pytest-mock>=3
responses>=0.24           # HTTP mocking
black>=23
isort>=5
ruff>=0.1
tqdm                      # Zaten var
```

### 11.2. Mevcut kritik paketler — dokunma
- `autogluon==1.5.0` (Python 3.10.6 ile eğitilmiş artefaktlar var)
- `scikit-learn==1.7.2`
- `pandas==2.3.3`
- `radon==6.0.1`
- `flask==3.1.0`
- `joblib==1.3.2`

### 11.3. requirements dosyaları
- `requirements.txt` — Flask runtime için minimum set (production)
- `requirements-dev.txt` — pipeline + analiz + test (full)
- Sürümler pin (`==`) edilir reproducibility için

---

## 12. CLI contract'ları

### 12.1. `scripts/collect.py`

```
python -m scripts.collect [OPTIONS]

OPTIONS:
  --target INT              Hedef proje sayısı                 [default: 1000]
  --min-age-days INT        Minimum proje yaşı                  [default: 180]
  --max-age-days INT        Maksimum proje yaşı                 [default: 365]
  --max-contributors INT    Maksimum contributor sayısı         [default: 10]
  --min-stars INT           Minimum yıldız                      [default: 5]
  --phase {discovery,process,build,all}                         [default: all]
  --resume                  Checkpoint'ten devam et
  --skip-szz                SZZ adımını atla
  --skip-prospector         Prospector'u atla
  --workers INT             Prospector paralel worker           [default: 4]
  --output-dir PATH                                             [default: output/]
  --log-level {DEBUG,INFO,WARNING,ERROR}                        [default: INFO]
  --dry-run                 Hiçbir şey yazma, sadece rapor
  --help
```

Exit codes: `0` başarılı, `1` genel hata, `2` config hatası, `130` user interrupt.

### 12.2. `scripts/train_final.py`

```
python -m scripts.train_final [OPTIONS]

OPTIONS:
  --dataset PATH            Input filtered parquet      [default: son filtered]
  --tasks STR               Virgülle: commit,bug,smell  [default: commit,bug,smell]
  --bug-label {keyword,szz}                             [default: szz]
  --smell-label {binary,count}                          [default: binary]
  --models-dir PATH                                     [default: models/]
  --log-level {DEBUG,INFO,WARNING,ERROR}                [default: INFO]
  --help
```

---

## 13. Modül API contract'ları

Implementing agent bu imzalara sadık kalmalı. Değişiklik gerekirse PLAN.md
güncellenmeli.

### 13.1. `pipeline/checkpoint.py`

```python
def save_checkpoint(phase: str, data: dict) -> None:
    """Atomic write. output/checkpoints/{phase}.json. Temp + os.replace."""

def load_checkpoint(phase: str) -> dict | None:
    """Dosya yoksa None."""

def mark_project_done(project_name: str, result: dict) -> None:
    """processed_projects.json'a ekle, atomic flush."""

def is_project_done(project_name: str) -> bool: ...

def get_processed_set() -> set[str]:
    """Sadece status=ok olanlar döner."""
```

### 13.2. `pipeline/rate_limit.py`

```python
def guarded_get(url: str, **kwargs) -> requests.Response:
    """
    requests.get wrapper:
    - Remaining < 10 → reset_at'a kadar sleep
    - 403 → exponential backoff (5/15/45/135 sn, max 3 retry)
    - 429 → 60 sn sleep, max 3 retry
    - 5xx → 3 retry, linear (2/5/10 sn)
    - Diğer → response döner (caller handle eder)
    """

def current_quota() -> dict:
    """{'remaining': int, 'limit': int, 'reset_at': datetime | None}"""
```

### 13.3. `pipeline/discovery.py`

```python
def search_projects(
    target_count: int = 1000,
    min_age_days: int = 180,
    max_age_days: int = 365,
    max_contributors: int = 10,
    min_stars: int = 5,
) -> list[dict]:
    """
    Her item:
    {
        'full_name': 'user/repo',
        'clone_url': 'https://github.com/user/repo.git',
        'stars': int,
        'created_at': 'iso8601',
        'project_age_days': int,
        'contributor_count': int,  # enrichment sonrası
        'default_branch': str,
    }

    Strateji: query slider (yaş aralığı + yıldız aralığı) → search API
    (max 1000 result per query) → enrichment → contributor_count filter →
    hedef sayıya ulaşana kadar slider kaydır.
    Checkpoint: output/checkpoints/discovery.json
    """
```

### 13.4. `pipeline/szz.py`

```python
def compute_szz_labels(
    repo_path: Path,
    head_files: list[str],
    bug_fix_commits: list[str],
    timeout_seconds: int = 600,
) -> dict[str, int]:
    """
    pydriller Git + ModificationsWithNames ile bug-introducing commit'leri tespit
    ve head_files üzerinde label üret.

    Döner: {file_path: 0|1}. Timeout aşılırsa boş dict; caller fallback yapar.
    """
```

### 13.5. `pipeline/prospector_runner.py`

```python
def run_prospector(
    file_path: Path,
    strictness: str = "medium",
    timeout_seconds: int = 30,
) -> dict:
    """
    Subprocess: prospector --output-format json --strictness {strictness} <file>
    Döner:
    {
        'smell_count': int | None,          # None = hata/timeout
        'categories': dict[str, int],       # {'pylint': 5, 'mccabe': 2}
        'messages': list[dict],             # ham msg list (Phase C)
    }
    """

def run_prospector_batch(
    file_paths: list[Path],
    workers: int = 4,
    strictness: str = "medium",
) -> dict[Path, dict]:
    """multiprocessing.Pool ile paralel."""
```

### 13.6. `pipeline/commits_before_bug.py`

```python
def compute_stats(
    commits_df: pd.DataFrame,     # file_path, commit_idx, is_bug_intro
) -> dict:
    """
    Döner:
    {
        'mean_commits_to_first_bug': float,
        'median_commits_to_first_bug': float,
        'mean_commits_between_bugs': float,
        'by_file': {file_path: int},  # ilk bug commit indeksi
    }
    """
```

---

## 14. Data şemaları

### 14.1. `output/projects/<safe_repo_name>.parquet`

| Sütun | Tip | Açıklama |
|---|---|---|
| file_path | string | Repo root'a göre göreceli |
| project_name | string | GitHub user/repo |
| stars | int32 | |
| contributor_count | int32 | |
| project_age_days | int32 | |
| commit_count | int32 | |
| bug_count | int32 | Bug-keyword commit sayısı |
| bug_keyword | int8 | 0/1 baseline etiket |
| bug_szz | int8 | 0/1 SZZ etiket (None olabilir) |
| commits_to_first_bug | int32 | -1 = bug yok |
| n_authors | int32 | |
| file_age_days | float32 | |
| churn_total | int32 | |
| avg_churn_per_commit | float32 | |
| max_single_churn | int32 | |
| recent_commits_90d | int32 | |
| loc, lloc, sloc | int32 | Radon raw |
| comments, multi, blank, single_comments | int32 | |
| cc_mean, cc_max, cc_total | float32 | |
| num_functions | int32 | |
| h_vocabulary ... h_calculated_length | float32 | 8 Halstead |
| maintainability_index | float32 | |
| comment_ratio, doc_ratio | float32 | |
| complexity_density, comment_per_function, avg_function_length, effort_per_line | float32 | Derived |
| smell_count | int32 (nullable) | Prospector |
| smell_categories | string | JSON encoded dict |

### 14.2. `output/dataset_full_<timestamp>.parquet`
Yukarıdakine ek olarak:
- `category_primary` string
- `categories_all` string (virgülle)
- `label_commit` int8 — `commit_count >= global_median`
- `smell_binary` int8 — `smell_count >= project_percentile_80`

### 14.3. `output/checkpoints/discovery.json`
```json
{
  "started_at": "2026-04-22T10:00:00Z",
  "completed_at": null,
  "criteria": {
    "min_age_days": 180,
    "max_age_days": 365,
    "max_contributors": 10,
    "min_stars": 5
  },
  "target_count": 1000,
  "found_count": 743,
  "found": [
    {
      "full_name": "user/repo",
      "clone_url": "https://...",
      "stars": 42,
      "created_at": "2025-09-14T...",
      "project_age_days": 220,
      "contributor_count": 6,
      "default_branch": "main"
    }
  ]
}
```

### 14.4. `output/checkpoints/processed_projects.json`
```json
{
  "processed": {
    "user/repo": {
      "status": "ok",
      "files": 42,
      "total_loc": 5230,
      "bugs_szz": 8,
      "smells_total": 67,
      "completed_at": "2026-04-22T10:45:00Z"
    },
    "user/repo2": {
      "status": "failed",
      "error": "git clone timeout after 600s",
      "failed_at": "2026-04-22T11:02:00Z"
    }
  }
}
```

### 14.5. `models/feature_names.json` (genişletilmiş)
```json
{
  "commit": ["loc", "lloc", ..., "project_age_days"],
  "bug":    ["loc", ..., "project_age_days"],
  "smell":  ["loc", ..., "project_age_days"]
}
```

### 14.6. `models/project_stats.json` (yeni)
```json
{
  "global": {
    "n_projects": 1000,
    "n_files": 452318,
    "total_loc": 1203948,
    "avg_commits_to_first_bug": 8.3,
    "median_smell_density": 0.12,
    "bug_rate_keyword": 0.31,
    "bug_rate_szz": 0.18
  },
  "by_category": {
    "AI/ML": {"n_projects": 640, "bug_rate_szz": 0.19, ...},
    "Web":   {...}
  }
}
```

---

## 15. Faz Definition of Done (DoD)

### F1 — Pipeline altyapısı
- [ ] `pipeline/` modülü `__init__.py` ile importable
- [ ] `pipeline/config.py` tüm sabitleri export eder
- [ ] `pipeline/checkpoint.py` implement + unit test geçer
- [ ] `pipeline/rate_limit.py` implement + mock test geçer
- [ ] `pipeline/discovery.py` 10 projelik mini çalıştırma başarılı
- [ ] `pipeline/cloning.py`, `pipeline/static_metrics.py`, `pipeline/git_metrics.py`
      → eski `app/metrics.py` ve `app/git_utils.py`'den taşındı
- [ ] Flask app hâlâ çalışıyor (smoke test: `python run.py` + `GET /`)
- [ ] `scripts/collect.py` CLI parse eder, `--dry-run` çıktı verir
- [ ] `.vscode/settings.json` jupyter markers + interpreter ayarlı
- [ ] `archive/v1/` oluşturuldu, 4 eski notebook taşındı
- [ ] `tests/` dizini + minimum 3 unit test
- [ ] `requirements-dev.txt` güncel
- [ ] `.gitignore` yazıldı (`output/`, `repos/`, `models/*.joblib`, `bug_ag_base/`, `__pycache__`, `*.parquet`, `.env`)
- [ ] `git init` yapıldı, ilk commit atıldı

### F2 — SZZ + Prospector
- [ ] `pipeline/szz.py` pydriller ile çalışır
- [ ] `pipeline/prospector_runner.py` subprocess + JSON parse + timeout
- [ ] 5 projelik pilot test: her iki etiket üretiliyor
- [ ] SZZ timeout'u fallback yapıyor
- [ ] Prospector 4 worker paralel doğrulandı

### F3 — Tam veri toplama
- [ ] 1000 matching proje bulundu
- [ ] Per-project parquet dosyaları yazıldı
- [ ] `dataset_full_*.parquet` birleştirildi
- [ ] Toplam LOC, başarı oranı, ortalama süre rapor edildi
- [ ] Resume test edildi (yarıda Ctrl+C, devam çalıştı)
- [ ] Log dosyası düzgün, uyarı/hata sayısı makul (<%10 failed)

### F4 — Filter & threshold
- [ ] `analysis/01_filter_categorize.py` cell'leri VS Code'da çalışır
- [ ] Sensitivity analysis filtreli vs filtresiz karşılaştırma
- [ ] `smell_binary` dynamic threshold üretildi
- [ ] `label_commit` üretildi
- [ ] `dataset_model_filtered_*.parquet` yazıldı
- [ ] Figures `output/figures/` altına kaydedildi

### F5 — Model training
- [ ] `analysis/02_model_training.py` ablation matris çalıştı
- [ ] T1, T2 (keyword+szz), T3 için sonuç tablosu üretildi
- [ ] Project-based vs time-based karşılaştırma raporu
- [ ] En iyi mimariler seçildi (PLAN.md'ye not düşüldü)
- [ ] `analysis/03_results_exploration.py` plot'lar üretti
- [ ] Sonuç CSV'leri `output/` altında

### F6 — Final training
- [ ] `scripts/train_final.py` 3 görev için çalıştı
- [ ] `models/` altında tüm artifact'lar mevcut
- [ ] `project_stats.json` yazıldı
- [ ] Sanity test: `predictor.load_all()` + tek satır tahmin

### F7 — Flask V2
- [ ] `app/predictor.py` 3 fonksiyon export ediyor
- [ ] `app/analyzer.py` Prospector entegre (opsiyonel bayrak)
- [ ] Results sayfasında 3 tahmin + yeni paneller render ediliyor
- [ ] Canlı test: bilinen repo üzerinde analiz çalışır
- [ ] UI responsive hâlâ çalışıyor (mobil layout bozulmadı)

### F8 — Paper + doc
- [ ] `README.md` güncellendi (setup, usage, yapı)
- [ ] PLAN.md'ye final sonuçlar eklendi
- [ ] Makale taslağı güncellendi (yeni RQ'lar, sonuçlar)
- [ ] CHANGELOG.md yazıldı (V1 → V2)

---

## 16. Git workflow

Proje şu an git repository değil (CLAUDE.md env: `Is a git repository: false`).

F1 başlangıcında:
1. `git init`
2. `.gitignore` yaz:
   ```
   output/
   repos/
   models/*.joblib
   models/bug_ag_base/
   __pycache__/
   *.pyc
   *.parquet
   .env
   venv/
   .vscode/launch.json
   *.log
   ```
3. İlk commit: "initial: V1 snapshot before V2 refactor"
4. V1 dosyalarını `archive/v1/` altına `git mv` ile taşı (history korunur)
5. Her faz sonunda commit: `feat(F1): pipeline scaffolding` gibi
6. Branch gerekmiyor (tek developer) — `main` üzerinde çalış
7. Remote ekleme opsiyonel (GitHub'a push edilecekse manual)

---

## 17. Flask V2 UI — detaylı spec

### 17.1. Results sayfası layout

**Üst satır (mevcut, değişmez):**
- Stat cards: Stars, Contributors, Project Age, Analyzed Files

**Yeni satır — Proje Sağlığı (3 kart):**

Kart 1: "First Bug Appearance"
- Ana sayı: `avg_commits_to_first_bug` (örn. "12")
- Alt yazı: "average commits before first bug"
- Yanında küçük: `median: 8, range: 2-34`

Kart 2: "Sprint Defect Density"
- Ana sayı: `bugs / KLOC / sprint` son 6 ay (örn. "0.8")
- Alt yazı: "bugs per KLOC per 2-week sprint"
- Trend ikonu (↑ kırmızı / → nötr / ↓ yeşil)

Kart 3: "Refactor Ratio"
- Ana yüzde: refactor commit oranı (örn. "12%")
- Alt yazı: "commits tagged refactor/cleanup/rename"
- (keyword tabanlı yaklaşık hesap)

**Yeni satır — Smell Özet (2 kart):**

Kart 1: "Total Smells"
- Ana sayı: Prospector toplam
- Alt yazı: `smell density: {count/KLOC}`

Kart 2: "Refactor Priority"
- Ana sayı: "N files" (hem bug_pred=1 hem smell_pred=1)
- Alt yazı: "Files with both bug + smell risk"
- Badge: "Start here"

**Ana tablo (mevcut + 3 yeni sütun):**

| File | LOC | CC | MI | Commit Risk | Commit Prob | Bug Risk | Bug Prob | **Smell Risk** | **Smell Prob** | **Prospector Count** |

Sıralama default: `smell_prob` azalan.

**Dosya detay modalı (yeni, satıra tıklayınca açılır):**
- Üst: file path, LOC, badges (commit/bug/smell risk)
- Tab 1: Metrics — tüm radon + git metrikleri tablosu
- Tab 2: Prospector messages — ham liste (line, severity, rule, message)
- Tab 3: History — dosyanın commit sayısı zaman ekseni + bug-fix commit işaretleri

### 17.2. Renk paleti (mevcut korunarak genişletme)

- Commit = `--accent: #4f46e5` (mor, mevcut)
- Bug = `--danger: #e11d48` (kırmızı, mevcut)
- **Smell = `--smell: #ea580c` (turuncu, yeni)** — bug'dan ayırt edilebilir, warn'dan (#f59e0b) koyu

### 17.3. Responsive
- Mobilde yeni kartlar stack olur (mevcut media query'ler aynı pattern)
- Tabloda smell sütunları overflow scrollable kalır

---

## 18. Implementing agent için notlar

Bu bölüm planı başka bir sohbette uygulayacak agent için özel yönergeler.

- **Her faza başlamadan önce** PLAN.md'yi oku, özellikle ilgili DoD bölümünü
  (§15)
- **Faz içinde tek seferde bitirmeye çalışma** — her modülü yazıp test et, bir
  sonrakine geç
- **Soru işareti olan noktada** → bu sohbete (PM oturumu) geri dön, ona danış
- **Acceptance kriterleri checklist** — faz sonunda DoD'daki her maddeyi tek tek
  doğrula
- **Yeni dosya oluştururken** → PLAN.md'deki dosya yapısına (§2.1) sadık kal
- **Function imzalarını** (§13) değiştirmek gerekirse PLAN.md de güncellenmeli
- **Commit sıklığı:** her alt modül tamamlanınca commit (F1.1 checkpoint, F1.2
  rate_limit ...)
- **Log dosyası boyutu büyürse** → ilgili projede WARNING'lar azalt, DEBUG'a düş
- **Pilot test** her fazda 5-10 proje üzerinde, full 1000'e geçmeden
- **Bu PLAN.md'yi güncel tut** — kararlar değişirse §9 karar geçmişine not ekle
- **PM oturumuna geri dönüş** gerektiğinde: "F2 başlarken pydriller timeout
  davranışı belirsiz, PM'e sor" gibi açık not düş

---

## 9. Karar geçmişi (özet)

- 2026-04-22: V2 hedefleri belirlendi (3 görev, 1000 proje, Prospector, SZZ)
- 2026-04-22: Parquet formatı, modüler pipeline yapısı, project-based split onaylandı
- 2026-04-22: Flask Prospector inference-time açık, performans şikayeti olursa kapatılabilir
- 2026-04-22: Phase A binary smell → Phase B count, Phase C kategorik sona kalırsa
- 2026-04-22: **Notebook formatı terkedildi.** `.ipynb` → `.py` + `# %%` cells
  (interactive analiz için) ve saf `.py` scripts (batch için). Sebep: VS Code
  native destek, temiz git diff, kernel crash riski yok, tek format. Eski
  notebook'lar `archive/v1/` altına taşınır.
