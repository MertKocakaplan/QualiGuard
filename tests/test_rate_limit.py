"""
test_rate_limit.py — guarded_get ve header/quota mantigi.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
import requests

from pipeline import rate_limit


# ── github_headers ───────────────────────────────────────────────

def test_headers_without_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    headers = rate_limit.github_headers()
    assert "Authorization" not in headers
    assert headers["Accept"].startswith("application/vnd.github")


def test_headers_with_token(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_xyz")
    headers = rate_limit.github_headers()
    assert headers["Authorization"] == "token ghp_xyz"


def test_github_token_configured(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert rate_limit.github_token_configured() is False
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_xyz")
    assert rate_limit.github_token_configured() is True


# ── guarded_get davranis ─────────────────────────────────────────

def _mock_resp(status_code: int, headers: dict | None = None, body: dict | None = None):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.json.return_value = body if body is not None else {}
    return resp


def test_guarded_get_passes_through_200(monkeypatch):
    resp = _mock_resp(200, headers={"X-RateLimit-Remaining": "5000"})
    with patch("pipeline.rate_limit.requests.get", return_value=resp) as rg:
        out = rate_limit.guarded_get("https://api.github.com/rate_limit")
    assert out is resp
    assert rg.call_count == 1


def test_guarded_get_retries_on_5xx_then_succeeds(monkeypatch):
    fail = _mock_resp(503)
    ok   = _mock_resp(200, headers={"X-RateLimit-Remaining": "4999"})
    call_log = [fail, fail, ok]

    with patch("pipeline.rate_limit.requests.get", side_effect=call_log), \
         patch("pipeline.rate_limit.time.sleep", return_value=None):
        out = rate_limit.guarded_get("https://x")
    assert out is ok


def test_guarded_get_sleeps_on_403_rate_limit(monkeypatch):
    """403 + remaining=0 -> reset'e kadar sleep (proaktif davranis)."""
    reset_ts = str(int(datetime.now(timezone.utc).timestamp()) + 1)
    blocked = _mock_resp(403, headers={
        "X-RateLimit-Remaining": "0",
        "X-RateLimit-Reset":     reset_ts,
    }, body={"message": "API rate limit exceeded"})
    ok = _mock_resp(200, headers={"X-RateLimit-Remaining": "10"})

    with patch("pipeline.rate_limit.requests.get", side_effect=[blocked, ok]), \
         patch("pipeline.rate_limit.time.sleep", return_value=None) as sleep:
        out = rate_limit.guarded_get("https://x")
    assert out is ok
    assert sleep.called


def test_guarded_get_quota_updated_from_headers():
    headers = {
        "X-RateLimit-Remaining": "4500",
        "X-RateLimit-Limit":     "5000",
        "X-RateLimit-Reset":     str(int(datetime.now(timezone.utc).timestamp()) + 3600),
    }
    resp = _mock_resp(200, headers=headers)
    with patch("pipeline.rate_limit.requests.get", return_value=resp):
        rate_limit.guarded_get("https://x")
    q = rate_limit.current_quota()
    assert q["remaining"] == 4500
    assert q["limit"]     == 5000
    assert q["reset_at"]  is not None


def test_guarded_get_429_retries_then_returns(monkeypatch):
    throttled = _mock_resp(429)
    ok        = _mock_resp(200)
    with patch("pipeline.rate_limit.requests.get", side_effect=[throttled, ok]), \
         patch("pipeline.rate_limit.time.sleep", return_value=None):
        out = rate_limit.guarded_get("https://x")
    assert out is ok


def test_guarded_get_raises_after_repeated_network_errors(monkeypatch):
    with patch("pipeline.rate_limit.requests.get",
               side_effect=requests.ConnectionError("down")), \
         patch("pipeline.rate_limit.time.sleep", return_value=None):
        with pytest.raises(requests.RequestException):
            rate_limit.guarded_get("https://x")
