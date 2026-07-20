"""
run.py — QualiGuard V2 Flask giris noktasi.

Kullanim:
    python run.py

Not: .env yuklemesi pipeline.config modulunde modul-seviyesinde yapilir;
ayrica burada tekrar yuklemeye gerek yok. app paketi import edildiginde
dolayli olarak pipeline.config tetiklenir.
"""
from __future__ import annotations

import os

# pipeline.config import'u modul-seviyesinde .env'i ortama yukler.
# Bu yuzden app/create_app'tan ONCE import etmek yeterli.
from pipeline.config import PROJECT_ROOT  # noqa: F401 — side-effect ile .env yukler

from app import create_app

app = create_app()


def _print_startup_info() -> None:
    token_ok = bool(os.environ.get("GITHUB_TOKEN", "").strip())
    token_status = "Tanimli" if token_ok else "Tanimli DEGIL (rate limit: 60 istek/saat)"

    print("=" * 55)
    print("  QualiGuard V2 — Flask Sunucusu")
    print("=" * 55)
    print(f"  Adres       : http://localhost:5000")
    print(f"  GitHub Token: {token_status}")
    if not token_ok:
        print()
        print("  Token eklemek icin proje kokunde .env dosyasi olusturun:")
        print("  GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print("=" * 55)


if __name__ == "__main__":
    _print_startup_info()
    app.run(debug=False, host="0.0.0.0", port=5000)
