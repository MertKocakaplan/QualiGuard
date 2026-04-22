/* token.js — GitHub Token panel management (active on all pages) */
(function () {
  "use strict";

  var navBtn       = document.getElementById("tokenNavBtn");
  var navDot       = document.getElementById("tokenNavDot");
  var panel        = document.getElementById("tokenPanel");
  var overlay      = document.getElementById("tokenOverlay");
  var closeBtn     = document.getElementById("tokenPanelClose");
  var statusBox    = document.getElementById("tokenStatusBox");
  var tokenInput   = document.getElementById("tokenInput");
  var toggleBtn    = document.getElementById("tokenToggleBtn");
  var toggleIcon   = document.getElementById("tokenToggleIcon");
  var inputError   = document.getElementById("tokenInputError");
  var saveBtn      = document.getElementById("tokenSaveBtn");
  var clearBtn     = document.getElementById("tokenClearBtn");

  if (!navBtn) return;

  var _statusCache    = null;
  var _panelOpen      = false;
  var _tokenVisible   = false;

  /* ── Update dot on page load ────────────────────────── */
  fetchStatus(function (data) {
    _statusCache = data;
    updateDot(data.active);
  });

  /* ── Navbar button ──────────────────────────────────── */
  navBtn.addEventListener("click", function () {
    if (_panelOpen) {
      closePanel();
    } else {
      openPanel();
    }
  });

  /* ── Close panel ────────────────────────────────────── */
  closeBtn.addEventListener("click", closePanel);
  overlay.addEventListener("click",  closePanel);

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && _panelOpen) closePanel();
  });

  /* ── Show / hide token ──────────────────────────────── */
  toggleBtn.addEventListener("click", function () {
    _tokenVisible = !_tokenVisible;
    tokenInput.type      = _tokenVisible ? "text"     : "password";
    toggleIcon.innerHTML = _tokenVisible ? "&#9675;"  : "&#9679;";
    toggleBtn.title      = _tokenVisible ? "Hide"     : "Show";
  });

  /* ── Save token ─────────────────────────────────────── */
  saveBtn.addEventListener("click", function () {
    var token = tokenInput.value.trim();
    clearInputError();

    if (!token) {
      showInputError("Token cannot be empty.");
      tokenInput.focus();
      return;
    }
    if (!/^(ghp_|github_pat_|gho_|ghu_|ghs_)\w+$/.test(token)) {
      showInputError("Invalid format. Token must start with 'ghp_'.");
      tokenInput.focus();
      return;
    }

    setSaving(true);

    fetch("/api/token", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ token: token }),
    })
    .then(function (r) { return r.json(); })
    .then(function (data) {
      setSaving(false);
      if (data.ok) {
        tokenInput.value = "";
        _tokenVisible    = false;
        tokenInput.type  = "password";
        toggleIcon.innerHTML = "&#9679;";
        refreshStatus();
      } else {
        showInputError(data.message || "Failed to save token.");
      }
    })
    .catch(function () {
      setSaving(false);
      showInputError("Could not connect to server.");
    });
  });

  /* ── Clear token ────────────────────────────────────── */
  clearBtn.addEventListener("click", function () {
    if (!confirm("Are you sure you want to clear the token?")) return;

    fetch("/api/token", { method: "DELETE" })
    .then(function (r) { return r.json(); })
    .then(function () { refreshStatus(); })
    .catch(function () { refreshStatus(); });
  });

  /* ── Refresh status ─────────────────────────────────── */
  function refreshStatus() {
    renderStatusLoading();
    fetchStatus(function (data) {
      _statusCache = data;
      updateDot(data.active);
      renderStatusData(data);
    });
  }

  /* ── Open panel ─────────────────────────────────────── */
  function openPanel() {
    _panelOpen = true;
    panel.classList.remove("hidden");
    overlay.classList.remove("hidden");
    navBtn.setAttribute("aria-expanded", "true");
    panel.removeAttribute("aria-hidden");
    refreshStatus();
    setTimeout(function () { tokenInput.focus(); }, 120);
  }

  /* ── Close panel ────────────────────────────────────── */
  function closePanel() {
    _panelOpen = false;
    panel.classList.add("hidden");
    overlay.classList.add("hidden");
    navBtn.setAttribute("aria-expanded", "false");
    panel.setAttribute("aria-hidden", "true");
    clearInputError();
    navBtn.focus();
  }

  /* ── API ────────────────────────────────────────────── */
  function fetchStatus(cb) {
    fetch("/api/token/status")
    .then(function (r) { return r.json(); })
    .then(cb)
    .catch(function () {
      cb({ active: false, masked: null, remaining: null,
           limit: null, reset_at: null, login: null,
           message: "Could not retrieve status." });
    });
  }

  /* ── Dot ────────────────────────────────────────────── */
  function updateDot(active) {
    navDot.className = "token-nav-dot " + (active ? "dot-active" : "dot-inactive");
    navBtn.title     = active ? "Token active" : "Token not configured";
  }

  /* ── Status box render ──────────────────────────────── */
  function renderStatusLoading() {
    statusBox.innerHTML =
      '<div class="token-status-loading">Checking...</div>';
  }

  function renderStatusData(d) {
    var cls   = d.active ? "status-active" : "status-inactive";
    var icon  = d.active ? "&#9679;" : "&#9675;";
    var label = d.active ? "Active" : "Not Active";

    var html = '<div class="token-status-row">'
      + '<span class="status-dot ' + cls + '" aria-hidden="true">' + icon + '</span>'
      + '<span class="status-label ' + cls + '">' + label + '</span>';

    if (d.masked) {
      html += '<code class="status-masked">' + esc(d.masked) + '</code>';
    }
    html += '</div>';

    if (d.login) {
      html += '<div class="status-meta">Account: <strong>@' + esc(d.login) + '</strong></div>';
    }

    if (d.remaining !== null && d.limit !== null) {
      var pct    = Math.round((d.remaining / d.limit) * 100);
      var barCls = pct > 40 ? "bar-ok" : pct > 10 ? "bar-warn" : "bar-low";
      html += '<div class="status-meta">'
        + 'API Quota: <strong>' + d.remaining + ' / ' + d.limit + '</strong>'
        + (d.reset_at ? ' &mdash; Resets: <strong>' + esc(d.reset_at) + '</strong>' : '')
        + '</div>'
        + '<div class="rate-bar-track"><div class="rate-bar ' + barCls + '" style="width:' + pct + '%"></div></div>';
    }

    if (d.message && !d.active) {
      html += '<div class="status-message">' + esc(d.message) + '</div>';
    }

    statusBox.innerHTML = html;
  }

  /* ── UI helpers ─────────────────────────────────────── */
  function setSaving(on) {
    saveBtn.disabled  = on;
    clearBtn.disabled = on;
    saveBtn.textContent = on ? "Verifying..." : "Save & Verify";
  }

  function showInputError(msg) {
    inputError.textContent = msg;
    inputError.classList.remove("hidden");
    tokenInput.setAttribute("aria-invalid", "true");
  }

  function clearInputError() {
    inputError.textContent = "";
    inputError.classList.add("hidden");
    tokenInput.removeAttribute("aria-invalid");
  }

  function esc(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

}());
