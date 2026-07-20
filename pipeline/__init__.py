"""
pipeline/ — QualiGuard V2 veri toplama ve etiketleme modulleri.

Alt modul ozeti:
    config              : Sabitler ve esikler
    checkpoint          : Faz/proje checkpoint yonetimi (atomic I/O)
    rate_limit          : GitHub API rate limit guard + token yardimcilari
    discovery           : GitHub arama + enrichment
    cloning             : Git clone sarmalayicisi
    static_metrics      : Radon tabanli statik metrikler (V1'den tasindi)
    git_metrics         : Git log/churn/bug-keyword metrikleri (V1'den tasindi)
    szz                 : SZZ etiketleme (F2)
    prospector_runner   : Prospector subprocess calistirici (F2)
    commits_before_bug  : Bug-oncesi commit istatistikleri (F2)
    dataset_builder     : Per-project parquet'leri birlestirme
    model_utils         : Split / scaler / metric yardimcilari
"""
from __future__ import annotations

__version__ = "2.0.0-f1"
