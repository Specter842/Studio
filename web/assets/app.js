/* video_pipeline — studio frontend. No framework, no build step: this talks
   directly to the orchestrator's own API (POST /render, GET /jobs/{id}), the
   same one n8n calls. */

(() => {
  "use strict";

  const API = window.location.origin;
  const $ = (id) => document.getElementById(id);

  // ---------------------------------------------------------------------
  // Smooth-scroll nav
  // ---------------------------------------------------------------------

  document.querySelectorAll("[data-scroll]").forEach((el) => {
    el.addEventListener("click", () => {
      const target = document.querySelector(el.dataset.scroll);
      if (target) target.scrollIntoView({ behavior: "smooth" });
    });
  });

  $("year").textContent = new Date().getFullYear();

  // ---------------------------------------------------------------------
  // Copy-to-clipboard micro-interaction
  // ---------------------------------------------------------------------

  function flashCopied(button) {
    const original = button.textContent;
    button.classList.add("is-copied");
    button.textContent = "Copied";
    setTimeout(() => {
      button.classList.remove("is-copied");
      button.textContent = original;
    }, 1400);
  }

  async function copy(text, button) {
    try {
      await navigator.clipboard.writeText(text);
      flashCopied(button);
    } catch {
      // Clipboard API can be denied; the button just won't flash. Not fatal.
    }
  }

  $("copy-endpoint").addEventListener("click", (e) => {
    copy(`${API}/render`, e.currentTarget);
  });

  $("copy-curl").addEventListener("click", (e) => {
    const snippet = `curl -X POST ${API}/render -H "Content-Type: application/json" -d '{"audio":"./inputs/track.mp3"}'`;
    copy(snippet, e.currentTarget);
  });

  // ---------------------------------------------------------------------
  // Health check
  // ---------------------------------------------------------------------

  async function checkHealth() {
    const dot = $("health-dot");
    const label = $("health-label");
    try {
      const res = await fetch(`${API}/health`, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const body = await res.json();
      dot.className = "dot dot--ok";
      label.textContent = body.auth_required
        ? "orchestrator online · token required"
        : "orchestrator online";
    } catch {
      dot.className = "dot dot--bad";
      label.textContent = "orchestrator unreachable";
    }
  }

  checkHealth();
  setInterval(checkHealth, 15000);

  // ---------------------------------------------------------------------
  // Format preset (Stage 02)
  // ---------------------------------------------------------------------

  const formatSelect = $("f-format");
  const customFormatRow = $("row-custom-format");

  formatSelect.addEventListener("change", () => {
    customFormatRow.hidden = formatSelect.value !== "custom";
  });

  function currentFormat() {
    if (formatSelect.value === "custom") {
      return {
        width: numberOrNull($("f-width").value),
        height: numberOrNull($("f-height").value),
        fps: numberOrNull($("f-fps").value),
      };
    }
    const [width, height, fps] = formatSelect.value.split("x").map(Number);
    return { width, height, fps };
  }

  // ---------------------------------------------------------------------
  // Collecting the render request
  // ---------------------------------------------------------------------

  function numberOrNull(raw) {
    if (raw === "" || raw === null || raw === undefined) return null;
    const n = Number(raw);
    return Number.isFinite(n) ? n : null;
  }

  function stringOrNull(raw) {
    const trimmed = (raw || "").trim();
    return trimmed === "" ? null : trimmed;
  }

  function buildRequest() {
    const format = currentFormat();
    const body = {
      audio: stringOrNull($("f-audio").value),
      clips: stringOrNull($("f-clips").value),
      brief: $("f-brief").value.trim(),
      duration: numberOrNull($("f-duration").value),
      seed: numberOrNull($("f-seed").value),
      width: format.width,
      height: format.height,
      fps: format.fps,
      transition: $("f-transition").value,
      title: $("f-title").value.trim(),
      title_style: $("f-title-style").value,
      title_position: $("f-title-position").value,
      title_at: numberOrNull($("f-title-at").value),
      title_seconds: numberOrNull($("f-title-seconds").value),
      verify: $("f-verify").checked,
      no_local: $("f-no-local").checked,
      generator: stringOrNull($("f-generator").value),
      stock_per_query: numberOrNull($("f-stock-per-query").value),
      generate: numberOrNull($("f-generate").value),
      max_spend_usd: numberOrNull($("f-max-spend").value),
      wait: false,
    };
    return body;
  }

  // ---------------------------------------------------------------------
  // Render + poll
  // ---------------------------------------------------------------------

  const renderBtn = $("render-btn");
  const renderError = $("render-error");
  const statusPill = $("status-pill");
  const terminalLog = $("terminal-log");
  const terminalCost = $("terminal-cost");
  const preview = $("preview");
  const previewVideo = $("preview-video");
  const previewDownload = $("preview-download");

  let polling = null;
  let consecutiveFailures = 0;
  // A render can run for minutes; one dropped poll tick must not stop it.
  // Only give up after several in a row (~7s of no contact at 1.2s/tick) —
  // long enough to ride out a blip, short enough to still notice a server
  // that is actually gone.
  const MAX_CONSECUTIVE_FAILURES = 6;

  function setStatus(status) {
    statusPill.textContent = status;
    statusPill.className = "status-pill";
    if (status === "running" || status === "queued") {
      statusPill.classList.add("status-pill--running");
    } else if (status === "succeeded") {
      statusPill.classList.add("status-pill--succeeded");
    } else if (status === "failed") {
      statusPill.classList.add("status-pill--failed");
    }
  }

  function setBusy(busy) {
    renderBtn.disabled = busy;
    renderBtn.classList.toggle("is-busy", busy);
    renderBtn.querySelector(".render-btn__label").textContent = busy
      ? "RENDERING…"
      : "RENDER";
  }

  function showError(message) {
    renderError.textContent = message;
    renderError.hidden = !message;
  }

  function extractCost(logText) {
    const match = logText.match(/Estimated run cost: \$[\d.]+[^\n]*/);
    return match ? match[0] : "";
  }

  async function pollJob(jobId) {
    try {
      const res = await fetch(`${API}/jobs/${jobId}`, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const job = await res.json();
      consecutiveFailures = 0;

      setStatus(job.status);
      terminalLog.textContent = job.log_tail || "(no output yet)";
      terminalLog.scrollTop = terminalLog.scrollHeight;

      const cost = extractCost(job.log_tail || "");
      if (cost) terminalCost.textContent = cost;

      if (job.status === "succeeded" || job.status === "failed") {
        clearInterval(polling);
        polling = null;
        setBusy(false);

        if (job.status === "succeeded" && job.out) {
          const filename = job.out.split(/[\\/]/).pop();
          const src = `${API}/media/${encodeURIComponent(filename)}`;
          previewVideo.src = src;
          previewDownload.href = src;
          previewDownload.setAttribute("download", filename);
          preview.hidden = false;
        } else if (job.status === "failed") {
          showError(job.error || "The render failed. See the log above.");
        }
      }
    } catch (err) {
      consecutiveFailures += 1;
      if (consecutiveFailures < MAX_CONSECUTIVE_FAILURES) return; // try again next tick

      clearInterval(polling);
      polling = null;
      setBusy(false);
      showError(`Lost contact with the orchestrator: ${err.message}`);
    }
  }

  async function startRender() {
    showError("");
    preview.hidden = true;
    consecutiveFailures = 0;
    if (polling) clearInterval(polling);

    const body = buildRequest();
    if (!body.audio) {
      showError("Audio track is required — Stage 01.");
      document.querySelector("#stage-source").scrollIntoView({ behavior: "smooth" });
      $("f-audio").focus();
      return;
    }

    setBusy(true);
    setStatus("queued");
    terminalLog.textContent = "submitting…";
    terminalCost.textContent = "";

    try {
      const res = await fetch(`${API}/render`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (res.status === 422) {
        const detail = await res.json();
        throw new Error(
          Array.isArray(detail.detail)
            ? detail.detail.map((d) => d.msg).join("; ")
            : detail.detail || "Invalid request."
        );
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const job = await res.json();
      setStatus(job.status);
      terminalLog.textContent = job.log_tail || "started…";

      if (job.status === "succeeded" || job.status === "failed") {
        // wait:false still returns instantly with "queued" in practice, but
        // handle the synchronous case too rather than assume.
        setBusy(false);
        if (job.status === "failed") showError(job.error || "The render failed.");
        return;
      }

      polling = setInterval(() => pollJob(job.job_id), 1200);
    } catch (err) {
      setBusy(false);
      setStatus("failed");
      showError(`Could not start the render: ${err.message}`);
    }
  }

  renderBtn.addEventListener("click", startRender);
})();
