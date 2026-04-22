"""
app/__init__.py — Flask uygulama factory.

V2'de bu paket artik iskelet cekirdegi tutar; agir kod `pipeline`
paketine tasindi (PLAN §2.1). Analyzer ve route'lar `pipeline.*`
modullerini import eder.
"""
from __future__ import annotations

from flask import Flask


def create_app() -> Flask:
    """Flask app factory. Blueprint + jinja filter kaydi yapar."""
    app = Flask(__name__)
    app.secret_key = "metrihunter-secret-2026"

    from .routes import bp
    app.register_blueprint(bp)

    @app.template_filter("format_number")
    def format_number(value):
        try:
            v = int(value)
            if v >= 1_000_000:
                return f"{v / 1_000_000:.1f}M"
            if v >= 1_000:
                return f"{v / 1_000:.1f}K"
            return str(v)
        except (TypeError, ValueError):
            return str(value)

    @app.template_filter("truncate_path")
    def truncate_path(path, max_len=55):
        if len(path) <= max_len:
            return path
        parts = path.split("/")
        if len(parts) <= 2:
            return "..." + path[-(max_len - 3):]
        tail = "/".join(parts[-2:])
        if len(tail) + 4 <= max_len:
            return ".../" + tail
        return "..." + tail[-(max_len - 3):]

    return app
