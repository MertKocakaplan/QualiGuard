"""
routes.py — Flask route tanimlari.

V2'de token/rate-limit yardimcilari `pipeline.rate_limit` paketinden
gelir; Flask dosyalari yalniz HTTP handler'lar.
"""
from __future__ import annotations

import logging
import os
import random
import re
import shutil
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import requests
from flask import Blueprint, abort, jsonify, render_template, request
from werkzeug.utils import secure_filename

from pipeline.config import GITHUB_RATELIMIT_URL, GITHUB_TOKEN_ENV
from pipeline.rate_limit import github_headers, github_token_configured

from . import predictor
from .analyzer import analyze_repo, analyze_zip

logger = logging.getLogger(__name__)

bp = Blueprint("main", __name__)

_ENV_FILE = Path(__file__).parent.parent / ".env"

# ── Gorev deposu ────────────────────────────────────────────────
_tasks: dict = {}
_tasks_lock  = threading.Lock()
_MAX_TASKS   = 200
_TASK_TTL    = 7200


def _is_github_url(url: str) -> bool:
    return bool(re.match(
        r"https?://github\.com/[\w.\-]+/[\w.\-]+(\.git)?/?$",
        url.strip(),
    ))


def _set_task(task_id: str, **kwargs):
    with _tasks_lock:
        if task_id not in _tasks:
            _tasks[task_id] = {"created_at": time.time()}
        _tasks[task_id].update(kwargs)


def _get_task(task_id: str) -> dict | None:
    with _tasks_lock:
        return _tasks.get(task_id)


def _cleanup_tasks():
    now = time.time()
    with _tasks_lock:
        expired = [tid for tid, t in _tasks.items()
                   if now - t.get("created_at", now) > _TASK_TTL]
        for tid in expired:
            del _tasks[tid]
        if len(_tasks) > _MAX_TASKS:
            ordered = sorted(_tasks, key=lambda t: _tasks[t].get("created_at", 0))
            for tid in ordered[: len(_tasks) - _MAX_TASKS]:
                del _tasks[tid]


def _mask_token(token: str) -> str:
    if not token:
        return ""
    if len(token) <= 8:
        return "****"
    return token[:4] + "..." + token[-4:]


def _write_env(token: str) -> None:
    lines: list[str] = []
    if _ENV_FILE.exists():
        lines = _ENV_FILE.read_text(encoding="utf-8").splitlines()

    new_line = f"{GITHUB_TOKEN_ENV}={token}"
    updated  = False
    for i, line in enumerate(lines):
        if line.startswith(f"{GITHUB_TOKEN_ENV}="):
            lines[i] = new_line
            updated  = True
            break
    if not updated:
        lines.append(new_line)
    _ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _remove_env_token() -> None:
    if not _ENV_FILE.exists():
        return
    lines = [l for l in _ENV_FILE.read_text(encoding="utf-8").splitlines()
             if not l.startswith(f"{GITHUB_TOKEN_ENV}=")]
    _ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Token API ────────────────────────────────────────────────────

@bp.route("/api/token/status")
def api_token_status():
    token = os.environ.get(GITHUB_TOKEN_ENV, "").strip()

    if not token:
        return jsonify({
            "active":     False,
            "masked":     None,
            "remaining":  None,
            "limit":      60,
            "reset_at":   None,
            "login":      None,
            "message":    "No token configured.",
        })

    try:
        rl_resp = requests.get(
            GITHUB_RATELIMIT_URL,
            headers=github_headers(),
            timeout=8,
        )

        if rl_resp.status_code == 401:
            return jsonify({
                "active":    False,
                "masked":    _mask_token(token),
                "remaining": None,
                "limit":     None,
                "reset_at":  None,
                "login":     None,
                "message":   "Token is invalid or expired.",
            })

        rl_data   = rl_resp.json()
        core      = rl_data.get("resources", {}).get("core", {})
        remaining = core.get("remaining")
        limit     = core.get("limit")
        reset_ts  = core.get("reset")
        reset_str = (datetime.fromtimestamp(reset_ts).strftime("%H:%M")
                     if reset_ts else None)

        login = None
        try:
            user_resp = requests.get(
                "https://api.github.com/user",
                headers=github_headers(),
                timeout=8,
            )
            if user_resp.status_code == 200:
                login = user_resp.json().get("login")
        except requests.RequestException:
            pass

        return jsonify({
            "active":    True,
            "masked":    _mask_token(token),
            "remaining": remaining,
            "limit":     limit,
            "reset_at":  reset_str,
            "login":     login,
            "message":   "Token is active.",
        })

    except requests.RequestException as exc:
        return jsonify({
            "active":    False,
            "masked":    _mask_token(token),
            "remaining": None,
            "limit":     None,
            "reset_at":  None,
            "login":     None,
            "message":   f"Could not connect to GitHub: {str(exc)[:120]}",
        })


@bp.route("/api/token", methods=["POST"])
def api_token_set():
    data  = request.get_json(silent=True) or {}
    token = (data.get("token") or "").strip()

    if not token:
        return jsonify({"ok": False, "message": "Token cannot be empty."}), 400

    if not re.match(r"^(ghp_|github_pat_|gho_|ghu_|ghs_)\w+$", token):
        return jsonify({
            "ok": False,
            "message": "Invalid token format.",
        }), 400

    try:
        resp = requests.get(
            GITHUB_RATELIMIT_URL,
            headers={
                "Authorization": f"token {token}",
                "Accept":        "application/vnd.github+json",
            },
            timeout=8,
        )
        if resp.status_code == 401:
            return jsonify({"ok": False, "message": "Invalid token (401)."}), 400
        if resp.status_code not in (200, 304):
            return jsonify({"ok": False, "message": f"GitHub did not respond: HTTP {resp.status_code}."}), 400
    except requests.RequestException as exc:
        return jsonify({"ok": False, "message": f"Could not connect to GitHub: {str(exc)[:120]}"}), 502

    os.environ[GITHUB_TOKEN_ENV] = token
    try:
        _write_env(token)
    except OSError:
        pass

    return jsonify({"ok": True, "message": "Token saved successfully."})


@bp.route("/api/token", methods=["DELETE"])
def api_token_delete():
    os.environ.pop(GITHUB_TOKEN_ENV, None)
    try:
        _remove_env_token()
    except OSError:
        pass
    return jsonify({"ok": True, "message": "Token cleared."})


# ── Ana route'lar ────────────────────────────────────────────────

@bp.route("/")
def index():
    return render_template(
        "index.html",
        models_ready=predictor.models_ready(),
        github_token_ok=github_token_configured(),
    )


def _parse_bool_field(val: str | None) -> bool:
    if val is None:
        return False
    return val.strip().lower() in ("1", "true", "on", "yes", "evet")


@bp.route("/analyze", methods=["POST"])
def analyze():
    url = (request.form.get("url") or "").strip()

    if not url:
        return jsonify({"error": "URL cannot be empty."}), 400
    if not _is_github_url(url):
        return jsonify({"error": "Enter a valid GitHub repository URL."}), 400
    if not predictor.models_ready():
        return jsonify({"error": "Models are not ready. Run scripts/train_final.py first."}), 503

    if random.random() < 0.1:
        threading.Thread(target=_cleanup_tasks, daemon=True).start()

    task_id = str(uuid.uuid4())
    _set_task(task_id, status="running", percent=0, message="Starting...", result=None)

    def run():
        def cb(pct, msg):
            _set_task(task_id, percent=pct, message=msg)

        result = analyze_repo(
            url,
            progress_callback=cb,
        )

        if result.get("error"):
            short_err = result["error"].split("\n")[0][:300]
            _set_task(task_id, status="error", percent=100, message=short_err, result=result)
        else:
            _set_task(task_id, status="done", percent=100, message="Completed!", result=result)

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"task_id": task_id})


# ── F7 — ZIP upload endpoint ──────────────────────────────────────

@bp.route("/analyze_upload", methods=["POST"])
def analyze_upload():
    """
    Yuklenmis ZIP dosyasini extract + analiz et.

    Beklentiler:
      - Multipart form, dosya alani adi "zipfile"
      - Uzanti: .zip
      - Boyut <= 100 MB (Flask MAX_CONTENT_LENGTH; asilirsa 413 doner)
      - Icerikte .git/ dizini ZORUNLU (analyze_zip dogrular)

    Asenkron: arka plan thread'inde analyze_zip; task_id ile polling.
    """
    file = request.files.get("zipfile")
    if file is None or not file.filename:
        return jsonify({"error": "No ZIP file selected."}), 400

    fname = file.filename.strip()
    if not fname.lower().endswith(".zip"):
        return jsonify({"error": "Only .zip files are accepted."}), 400

    if not predictor.models_ready():
        return jsonify({"error":
            "Models are not ready. Run scripts/train_final.py first."}), 503

    # ZIP'i guvenli isimle gecici dizine kaydet
    upload_dir = Path(tempfile.mkdtemp(prefix="mh_upload_"))
    safe_name  = secure_filename(fname) or "upload.zip"
    zip_path   = upload_dir / safe_name
    try:
        file.save(str(zip_path))
    except OSError as exc:
        shutil.rmtree(upload_dir, ignore_errors=True)
        logger.exception("Failed to save ZIP: %s", exc)
        return jsonify({"error": f"Could not save the ZIP on the server: {exc}"}), 500

    if random.random() < 0.1:
        threading.Thread(target=_cleanup_tasks, daemon=True).start()

    task_id = str(uuid.uuid4())
    _set_task(task_id, status="running", percent=0,
              message="ZIP uploaded, starting analysis...", result=None)

    def run():
        def cb(pct, msg):
            _set_task(task_id, percent=pct, message=msg)
        try:
            result = analyze_zip(zip_path, progress_callback=cb)
            if result.get("error"):
                short_err = result["error"].split("\n")[0][:300]
                _set_task(task_id, status="error", percent=100,
                          message=short_err, result=result)
            else:
                _set_task(task_id, status="done", percent=100,
                          message="Completed!", result=result)
        finally:
            # Yuklenmis ZIP + upload tmp dir'i temizle (analiz tmp'i analyzer ici)
            shutil.rmtree(upload_dir, ignore_errors=True)

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"task_id": task_id})


# Flask MAX_CONTENT_LENGTH'i asarsa 413 ile JSON donsun (HTML hata yerine).
@bp.app_errorhandler(413)
def _too_large(_e):
    return jsonify({
        "error": "The uploaded file is too large; the ZIP exceeds the 100 MB limit.",
    }), 413


@bp.route("/api/status/<task_id>")
def api_status(task_id):
    try:
        uuid.UUID(task_id)
    except ValueError:
        return jsonify({"error": "Invalid task_id."}), 400

    task = _get_task(task_id)
    if task is None:
        return jsonify({
            "status": "not_found",
            "percent": 0,
            "message": "Task not found; the session may have expired.",
        }), 404

    return jsonify({
        "status":  task.get("status", "running"),
        "percent": task.get("percent", 0),
        "message": task.get("message", ""),
    })


@bp.route("/results/<task_id>")
def results(task_id):
    try:
        uuid.UUID(task_id)
    except ValueError:
        abort(400)

    task = _get_task(task_id)
    if task is None:
        return render_template(
            "index.html",
            models_ready=predictor.models_ready(),
            github_token_ok=github_token_configured(),
            flash_message="The result has expired or was not found.",
        )

    if task.get("status") == "running":
        return render_template(
            "index.html",
            models_ready=True,
            pending_task=task_id,
            github_token_ok=github_token_configured(),
        )

    result = task.get("result") or {}
    project_stats = predictor.get_project_stats() or {}
    # UI flags + visible_count: scripts block (DataTables init) bunlari template-set
    # araciligiyla goremiyor (Jinja2 block-scope kirilgan); route context'inden gec
    # ki hem content hem scripts block'larinda gorunsun.
    smell_ok = predictor.smell_available()
    prosp_on = bool(result.get("prospector_enabled"))
    smell_sum = result.get("smell_summary") or {}
    show_prosp = bool(prosp_on and smell_sum.get("prospector_enabled"))
    visible_count = 7 + (2 if smell_ok else 0) + (1 if show_prosp else 0)
    return render_template(
        "results.html",
        result=result,
        task_id=task_id,
        project_stats=project_stats,
        smell_model_available=smell_ok,
        show_smell=smell_ok,
        show_prospector=show_prosp,
        visible_count=visible_count,
    )
