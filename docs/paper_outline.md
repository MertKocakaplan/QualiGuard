# MetricHunter V2 — Makale Taslak Ceketi

Akademisyen geri bildirimi sonrasi yeniden yapilandirilmis V2 sonuclarini
raporlamak icin iskelet. Bu dosya **placeholder tablolar + RQ stub'lari**
icerir; gercek sayilar kullanici pipeline'i kosturduktan sonra elle
doldurulmalidir.

Tavsiye: bu dosyayi dogrudan Overleaf/Word'e tasimayin; baslik ve tablolari
ana metin dosyanizda iskele olarak kullanin.

---

## 1. Basak soru

Statik kod metrikleri (radon + git + project) tek basina; commit
yogunlugu, dosya bazli hata riski ve code smell riski icin ne kadar iyi
**tahmin sinyali** saglar? Hibrit AutoML mimarisi, tek-model baseline'a
kiyasla anlamli fark yaratir mi?

---

## 2. Arastirma sorulari

### RQ1 — Feature set katkisi
Static-only vs static+derived vs +project vs +all feature setleri her
uc gorevde F1/PR-AUC/MCC acisindan nasil ayrisir?

- Kaynak: `analysis/02_model_training.py` -> `ablation_results_<ts>.csv`
  (ok statuslu satirlar)
- Hipotez: +process (git churn + recent_commits_90d) eklenmesi T2 bug
  gorevinde en buyuk sicramayi yaratir; T1 commit ve T3 smell daha az
  fayda gorur.

**Tablo RQ1** _(doldurulacak)_:

| Gorev | Feature set | Best F1 | Best model | Delta vs static |
|---|---|---|---|---|
| T1 commit | static     | — | — | baseline |
| T1 commit | derived    | — | — | — |
| T1 commit | all        | — | — | — |
| T2 bug    | static     | — | — | baseline |
| T2 bug    | process    | — | — | — |
| T2 bug    | all        | — | — | — |
| T3 smell  | static     | — | — | baseline |
| T3 smell  | all        | — | — | — |

### RQ2 — Split stratejisi
Project-based 70/15/15 split, time-based split'e kiyasla model
performansini nasil etkiler? (Data leakage riski acisindan tartisma)

- Kaynak: `analysis/02_model_training.py`; her gorev icin iki split
  ayri satir.
- Hipotez: time-based split F1'i `-0.05..-0.15` dusurur; leakage
  kontrolu icin raporlanmali.

**Tablo RQ2** _(doldurulacak)_:

| Gorev | Split | F1 | PR-AUC | MCC |
|---|---|---|---|---|
| T1 | project | — | — | — |
| T1 | time    | — | — | — |
| T2 | project | — | — | — |
| T2 | time    | — | — | — |
| T3 | project | — | — | — |
| T3 | time    | — | — | — |

### RQ3 — Model secimi
Klasik ML (LR, RF, SVM), modern boosting (XGBoost, LightGBM), AutoGluon
ve DL (MLP, CNN1D, LSTM) arasinda her gorev icin en iyi model hangisidir?

- Kaynak: ablation CSV, `model` sutunu.
- Hipotez: T2'de stacking (RF + AutoGluon -> LR) en iyi, T1/T3'de saf
  RF yeterli.

**Tablo RQ3** _(doldurulacak — her gorev icin top 3)_:

| Gorev | Rank | Model | F1 | Konfig |
|---|---|---|---|---|
| T1 | 1 | — | — | — |
| T1 | 2 | — | — | — |
| T1 | 3 | — | — | — |
| T2 | 1 | — | — | — |
| T2 | 2 | — | — | — |
| T2 | 3 | — | — | — |
| T3 | 1 | — | — | — |
| T3 | 2 | — | — | — |
| T3 | 3 | — | — | — |

### RQ4 — SZZ vs keyword etiketi
Bug etiketinin SZZ algoritmasi (pydriller) ile keyword-only baseline
arasindaki farki model performansina nasil yansir?

- Kaynak: `scripts/train_final.py --bug-label szz` vs `--bug-label keyword`
- Hipotez: SZZ daha kaliteli etiket sagladigi icin F1'i `+0.05..+0.10`
  arttirir; ancak smaller positive class oraniyla PR-AUC dusebilir.

**Tablo RQ4** _(doldurulacak)_:

| Etiket | Pozitif oran | Best F1 | PR-AUC | MCC |
|---|---|---|---|---|
| bug_keyword | — | — | — | — |
| bug_szz     | — | — | — | — |

### RQ5 — Smell etiketi duyarliligi
P80 smell_count esigi yerine farkli yuzdelikler (P70, P85, P90) secilirse
pozitif oran ve T3 modelinin kalitesi nasil degisir?

- Kaynak: `analysis/01_filter_categorize.py` (sensitivity tablolari)
  genislettilebilir.
- Hipotez: P80 pozitif oran %18-22 arasinda tutarak dengeli; P90 seyrek
  sinif (<10%) yaratir, RF performansi duser.

**Tablo RQ5** _(doldurulacak)_:

| Esik (yuzdelik) | Pozitif oran | Best F1 (T3) | PR-AUC |
|---|---|---|---|
| P70 | — | — | — |
| P80 | — | — | — |
| P85 | — | — | — |
| P90 | — | — | — |

---

## 3. Yontem ozeti

- **Veri:** GitHub search ile toplanan 1000 Python projesi; her proje icin
  tek-branch clone + radon static metric + git log + SZZ + Prospector
  smell.
- **Dataset:** ~200-400K dosya kaydi (proje bazinda 150-400 arasi, test
  filtresi sonrasi). `output/dataset_full_<ts>.parquet`.
- **Filtre:** Default filtresiz; `add_commit_label` global median, P80
  smell esigi.
- **Split:** Project-based 70/15/15 (train/val/test); time-based da
  raporlanir (RQ2).
- **On-isleme:** StandardScaler (train uzerinde fit); SMOTE yalniz
  train'de (eger `--smote`).
- **Metrikler:** F1, PR-AUC, MCC, accuracy. Rapor: test setinde.

---

## 4. Ana bulgular (taslak)

_(kullanici tarafindan doldurulacak — Results bolumuinun ozeti)_

- T1 commit: `—` (en iyi model, F1)
- T2 bug:    `—` (en iyi stacking veya saf model)
- T3 smell:  `—` (en iyi model; P80 esik dogrulamasi)
- Hibrit stacking T2'de tek modele karsi avantaj sagladi mi? Evet/Hayir.
- Feature set delta'lari RQ1'deki hipotezi destekliyor mu?

---

## 5. Tehditler

- **Iç tehlike:** Project-based split leakage'i engeller, ancak ayni
  kurulusun birden fazla projesi hala korelasyon yaratabilir.
- **Dis tehlike:** Python + GitHub odakli; kurumsal closed-source kod
  tabanlarina genellenemez.
- **Yapi:** Prospector ve radon'un metrik tanimi sabit kabul edilir;
  baska aracin uretecegi metric dagilimi farkli sonuclanabilir.
- **Zaman:** Dataset 2024-Q4 GitHub snapshot'i; trend kaymasina maruz.

---

## 6. Yeniden uretilebilirlik

- Tum kod: `v2/pipeline/`, `v2/scripts/`, `v2/analysis/`
- CLI'lar:
  - `python -m scripts.collect --phase all --target 1000`
  - `analysis/01_filter_categorize.py` (interaktif)
  - `analysis/02_model_training.py` (interaktif)
  - `python -m scripts.train_final`
- Artifactlar: `models/*.joblib`, `models/bug_ag_base/`, `feature_names.json`,
  `project_stats.json`
- Random seed: `RANDOM_STATE = 42` (tum script ve notebook'larda sabit)
- Test: `python -m pytest` (136 passed, 2 skipped)

---

## 7. Ekler

- `analysis/03_results_exploration.py` cikisi: `output/figures/`
- Ablation raw: `output/ablation_results_<ts>.csv`
- Project-level ozet: `models/project_stats.json`
- Kategori dagilim plotu: `output/figures/sensitivity_category_distribution.png`
