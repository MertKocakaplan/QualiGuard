/* app.js — Form submit, URL validation, progress polling */
(function () {
  "use strict";

  /* ── DOM references ──────────────────────────────────── */
  var form        = document.getElementById("analyzeForm");
  if (!form) return;  /* do nothing on results page */

  var submitBtn   = document.getElementById("submitBtn");
  var urlInput    = document.getElementById("repoUrl");
  var progressSec = document.getElementById("progressSection");
  var progressBar = document.getElementById("progressBar");
  var progressMsg = document.getElementById("progressMsg");
  var progressPct = document.getElementById("progressPct");
  var progressTrk = document.querySelector(".progress-track[role='progressbar']");
  var errorBox    = document.getElementById("errorBox");
  var urlError    = document.getElementById("urlError");

  var POLL_MS = 1500;

  /* ── Resume pending task on page load ───────────────── */
  if (window._pendingTask) {
    showProgress();
    disableForm();
    pollStatus(window._pendingTask);
  }

  /* ── Show flash message if present ──────────────────── */
  if (window._flashMessage) {
    showError(window._flashMessage);
  }

  /* ── Clear inline error on input change ─────────────── */
  urlInput.addEventListener("input", function () {
    clearInlineError();
    clearError();
  });

  /* ── Form submit ─────────────────────────────────────── */
  form.addEventListener("submit", function (e) {
    e.preventDefault();

    var url = urlInput.value.trim();
    clearError();
    clearInlineError();

    if (!url) {
      showInlineError("URL cannot be empty.");
      urlInput.focus();
      return;
    }

    if (!isGithubUrl(url)) {
      showInlineError(
        "Please enter a valid GitHub repository URL. " +
        "Example: https://github.com/user/repo"
      );
      urlInput.focus();
      return;
    }

    showProgress();
    setPercent(0, "Starting...");
    disableForm();

    fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: "url=" + encodeURIComponent(url)
    })
    .then(function (r) {
      var status = r.status;
      return r.json().then(function (d) {
        return { status: status, data: d };
      });
    })
    .then(function (res) {
      if (res.status === 503) {
        showError(res.data.error || "Server not ready.");
        hideProgress();
        enableForm();
        return;
      }
      if (res.data.error) {
        showError(res.data.error);
        hideProgress();
        enableForm();
        return;
      }
      pollStatus(res.data.task_id);
    })
    .catch(function (err) {
      showError(
        "Could not connect to the server. Check your internet connection. " +
        "(" + (err.message || "Network error") + ")"
      );
      hideProgress();
      enableForm();
    });
  });

  /* ── Polling ─────────────────────────────────────────── */
  function pollStatus(taskId) {
    var failCount = 0;
    var MAX_FAIL  = 5;

    var interval = setInterval(function () {
      fetch("/api/status/" + taskId)
      .then(function (r) {
        var httpStatus = r.status;
        return r.json().then(function (d) {
          return { httpStatus: httpStatus, data: d };
        });
      })
      .then(function (res) {
        failCount = 0;

        var data = res.data;

        if (res.httpStatus === 404 || data.status === "not_found") {
          clearInterval(interval);
          showError(
            "Analysis task not found. The server may have restarted. " +
            "Please try again."
          );
          hideProgress();
          enableForm();
          return;
        }

        setPercent(data.percent || 0, data.message || "");

        if (data.status === "done") {
          clearInterval(interval);
          setPercent(100, "Redirecting...");
          setTimeout(function () {
            window.location.href = "/results/" + taskId;
          }, 350);

        } else if (data.status === "error") {
          clearInterval(interval);
          showError(data.message || "An error occurred during analysis.");
          hideProgress();
          enableForm();
        }
      })
      .catch(function (err) {
        failCount++;
        if (failCount >= MAX_FAIL) {
          clearInterval(interval);
          showError(
            "Connection to server lost " + MAX_FAIL + " times. " +
            "Refresh the page and try again."
          );
          hideProgress();
          enableForm();
        }
      });
    }, POLL_MS);
  }

  /* ── Helper functions ────────────────────────────────── */

  function setPercent(pct, msg) {
    var clamped = Math.min(100, Math.max(0, Math.round(pct)));
    progressBar.style.width = clamped + "%";
    if (progressTrk) progressTrk.setAttribute("aria-valuenow", clamped);
    progressPct.textContent = clamped + "%";
    progressMsg.textContent = msg || "";
  }

  function showProgress()  { progressSec.classList.remove("hidden"); }
  function hideProgress()  { progressSec.classList.add("hidden"); }

  function disableForm() {
    submitBtn.disabled = true;
    urlInput.disabled  = true;
  }

  function enableForm() {
    submitBtn.disabled = false;
    urlInput.disabled  = false;
    /* F7 — Upload form'u da varsa etkinlestir */
    if (uploadBtn) uploadBtn.disabled = false;
    if (zipInput)  zipInput.disabled  = false;
    /* Focus aktif sekmeye */
    var activePanel = document.querySelector(".source-panel.is-active");
    if (activePanel && activePanel.dataset.tab === "upload" && zipInput) {
      zipInput.focus();
    } else {
      urlInput.focus();
    }
  }

  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.classList.remove("hidden");
    errorBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function clearError() {
    errorBox.textContent = "";
    errorBox.classList.add("hidden");
  }

  function showInlineError(msg) {
    urlError.textContent = msg;
    urlError.classList.remove("hidden");
    urlInput.setAttribute("aria-invalid", "true");
  }

  function clearInlineError() {
    urlError.textContent = "";
    urlError.classList.add("hidden");
    urlInput.removeAttribute("aria-invalid");
  }

  function isGithubUrl(url) {
    return /^https?:\/\/github\.com\/[\w.\-]+\/[\w.\-]+(\.git)?\/?$/.test(url);
  }

  /* ── F7 — Tab toggle (GitHub URL <-> Upload ZIP) ────── */
  var tabs   = document.querySelectorAll(".source-tab");
  var panels = document.querySelectorAll(".source-panel");
  tabs.forEach(function (btn) {
    btn.addEventListener("click", function () {
      var name = btn.dataset.tab;
      tabs.forEach(function (b) {
        var on = (b === btn);
        b.classList.toggle("is-active", on);
        b.setAttribute("aria-selected", on ? "true" : "false");
      });
      panels.forEach(function (p) {
        p.classList.toggle("is-active", p.dataset.tab === name);
      });
      clearError();
    });
  });

  /* ── F7 — Upload form submit ────────────────────────── */
  var uploadForm = document.getElementById("uploadForm");
  var zipInput   = document.getElementById("zipInput");
  var zipError   = document.getElementById("zipError");
  var uploadBtn  = document.getElementById("uploadBtn");

  if (uploadForm) {
    zipInput.addEventListener("change", function () {
      clearZipError();
      clearError();
    });

    uploadForm.addEventListener("submit", function (e) {
      e.preventDefault();
      clearError();
      clearZipError();

      var file = zipInput.files && zipInput.files[0];
      if (!file) {
        showZipError("Please select a ZIP file.");
        zipInput.focus();
        return;
      }
      if (!/\.zip$/i.test(file.name)) {
        showZipError("Only .zip files are accepted.");
        zipInput.focus();
        return;
      }
      /* Boyut on-kontrol — Flask zaten 413 ile reddeder, ama erken UX uyari */
      var maxBytes = 100 * 1024 * 1024;
      if (file.size > maxBytes) {
        showZipError(
          "ZIP file is too large (" +
          (file.size / 1024 / 1024).toFixed(1) +
          " MB). Limit: 100 MB."
        );
        return;
      }

      showProgress();
      setPercent(0, "Uploading ZIP file...");
      disableUploadForm();

      var fd = new FormData();
      fd.append("zipfile", file);

      fetch("/analyze_upload", {
        method: "POST",
        body:   fd
      })
      .then(function (r) {
        var status = r.status;
        return r.json().then(function (d) {
          return { status: status, data: d };
        });
      })
      .then(function (res) {
        if (res.status === 413) {
          showError(res.data.error || "ZIP file too large (max 100 MB).");
          hideProgress();
          enableUploadForm();
          return;
        }
        if (res.status === 503) {
          showError(res.data.error || "Server not ready.");
          hideProgress();
          enableUploadForm();
          return;
        }
        if (res.data.error) {
          showError(res.data.error);
          hideProgress();
          enableUploadForm();
          return;
        }
        /* Polling — analyze_repo ile ayni task_id mekanizmasi */
        pollStatus(res.data.task_id);
      })
      .catch(function (err) {
        showError(
          "Could not connect to the server. (" +
          (err.message || "Network error") + ")"
        );
        hideProgress();
        enableUploadForm();
      });
    });
  }

  function disableUploadForm() {
    if (uploadBtn) uploadBtn.disabled = true;
    if (zipInput)  zipInput.disabled  = true;
  }

  function enableUploadForm() {
    if (uploadBtn) uploadBtn.disabled = false;
    if (zipInput)  zipInput.disabled  = false;
    if (zipInput)  zipInput.focus();
  }

  function showZipError(msg) {
    if (!zipError) return;
    zipError.textContent = msg;
    zipError.classList.remove("hidden");
    if (zipInput) zipInput.setAttribute("aria-invalid", "true");
  }

  function clearZipError() {
    if (!zipError) return;
    zipError.textContent = "";
    zipError.classList.add("hidden");
    if (zipInput) zipInput.removeAttribute("aria-invalid");
  }

}());
