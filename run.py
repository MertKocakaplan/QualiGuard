"""
run.py — MetricHunter V2 Flask giris noktasi.

Kullanim:
    python run.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _load_env(env_path: Path) -> None:
    """.env dosyasini Flask baslamadan ortama yukle."""
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key   = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_env_file = Path(__file__).parent / ".env"
_load_env(_env_file)

# Env yuklendikten sonra import et — pipeline.config zamaninda dogru degeri okur
from app import create_app  # noqa: E402

app = create_app()


def _print_startup_info() -> None:
    token_ok = bool(os.environ.get("GITHUB_TOKEN", "").strip())
    token_status = "Tanimli" if token_ok else "Tanimli DEGIL (rate limit: 60 istek/saat)"

    print("=" * 55)
    print("  MetricHunter V2 — Flask Sunucusu")
    print("=" * 55)
    print(f"  Adres       : http://localhost:5000")
    print(f"  GitHub Token: {token_status}")
    if not token_ok:
        print()
        print("  Token eklemek icin v2/.env dosyasi olusturun:")
        print("  GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print("=" * 55)


if __name__ == "__main__":
    _print_startup_info()
    app.run(debug=False, host="0.0.0.0", port=5000)
