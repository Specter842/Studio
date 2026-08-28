"""Optional paid adapter: fal.ai's gateway to Seedance, Kling, Veo3, FLUX and
Nano Banana. **Disabled by default and never invoked unless enabled.**

This exists for the specific shot where local generation genuinely is not good
enough or not fast enough, and somebody has decided to pay for it. It is not a
fallback: if `paid_adapters_enabled` is false, constructing this adapter raises
rather than quietly costing money.

Everything goes through one endpoint shape and one auth header, which is the
whole reason to use a gateway rather than five separate SDKs. Swapping models
is `fal.model` in settings.yaml.

Three things protect the bill, and all three matter:

  * The adapter cannot be constructed while paid adapters are disabled.
  * Budget is reserved *before* every attempt, retries included, so a retry
    loop hits the ceiling instead of the credit card.
  * Only server errors are retried. A rejected request is a bug, and retrying
    it just buys the same rejection again.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import config
from ffmpeg_tools import ffmpeg_bin, run
from generators.base_adapter import (
    GenerationError,
    GenerationRequest,
    GeneratorAdapter,
    register,
)

log = logging.getLogger(__name__)

QUEUE_URL = "https://queue.fal.run"

# Retried because they are transient. 4xx is not in this list on purpose.
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".avif"}


class FalError(GenerationError):
    """fal.ai could not fulfil the request."""


@register
class FalGatewayAdapter(GeneratorAdapter):
    """One adapter for every model fal.ai fronts."""

    name = "fal_gateway"
    is_paid = True
    # Conservative class-level figure used before a model is known. The real
    # per-call estimate is resolved per model in __init__ and shadows this.
    estimated_cost_per_call_usd = 0.50

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        settings = self.settings
        self.model = str(settings.get("fal.model", "")).strip()
        if not self.model:
            raise FalError(
                "fal.model is not set in config/settings.yaml. Name the model "
                "explicitly — there is no sensible default for something that "
                "costs money per call."
            )

        self.timeout_seconds = float(settings.get("fal.timeout_seconds", 900))
        self.poll_seconds = float(settings.get("fal.poll_seconds", 3.0))
        self.max_retries = int(settings.get("fal.max_retries", 3))
        self.backoff_seconds = float(settings.get("fal.backoff_seconds", 2.0))

        costs = settings.get("fal.costs", {}) or {}
        default_cost = float(settings.get("fal.default_cost_usd", 0.50))
        # Instance attribute shadows the ClassVar, so reserve_budget() charges
        # what this model actually costs.
        self.estimated_cost_per_call_usd = float(costs.get(self.model, default_cost))
        if self.model not in costs:
            log.warning(
                "No declared cost for %r; assuming $%.4f per call. Add it under "
                "fal.costs in settings.yaml so the budget guard is accurate.",
                self.model, self.estimated_cost_per_call_usd,
            )

        self.api_key = config.secret("FAL_KEY")
        if not self.api_key:
            raise FalError(
                "FAL_KEY is not set in .env. Get a key at https://fal.ai/ and "
                "put it there — never in code or settings.yaml."
            )

    # -- public API -------------------------------------------------------

    def generate(self, prompt: str, **kwargs) -> Path:
        request = self.build_request(prompt, **kwargs)
        destination = self.cached_path(request)
        if destination.exists():
            log.info("fal cache hit (no charge): %s", destination.name)
            return destination

        payload = self._submit_with_retries(request)
        asset_url = _first_asset_url(payload)
        if not asset_url:
            raise FalError(
                f"{self.model} returned no downloadable asset. Response keys: "
                f"{sorted(payload)}"
            )

        downloaded = self._download(asset_url, request)
        return self._normalise(downloaded, destination, request)

    # -- request ----------------------------------------------------------

    def _arguments(self, request: GenerationRequest) -> dict[str, Any]:
        """Model arguments. fal ignores what a given model does not use."""
        arguments: dict[str, Any] = {
            "prompt": request.prompt,
            "image_size": {"width": request.width, "height": request.height},
            "duration": round(request.seconds),
            "num_frames": request.frames,
            "fps": request.fps,
        }
        if request.negative_prompt:
            arguments["negative_prompt"] = request.negative_prompt
        if request.seed is not None:
            arguments["seed"] = request.seed
        arguments.update(self.settings.get("fal.arguments", {}) or {})
        arguments.update(request.extra)
        return arguments

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }

    def _submit_with_retries(self, request: GenerationRequest) -> dict[str, Any]:
        import httpx

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            # Charged before the call, once per attempt: a retry is a second
            # billable request, and the ceiling has to see it coming.
            self.reserve_budget(
                note=f"{self.model} attempt {attempt}: {request.prompt[:40]}"
            )
            try:
                return self._submit_once(request)
            except FalError as exc:
                status = getattr(exc, "status_code", None)
                if status is not None and status not in _RETRYABLE_STATUS:
                    # A rejected request will be rejected identically next time.
                    raise
                last_error = exc
            except httpx.HTTPError as exc:
                last_error = exc

            if attempt < self.max_retries:
                wait = self.backoff_seconds * (2 ** (attempt - 1))
                log.warning(
                    "fal attempt %d/%d failed (%s); retrying in %.0fs",
                    attempt, self.max_retries, last_error, wait,
                )
                time.sleep(wait)

        raise FalError(
            f"{self.model} failed after {self.max_retries} attempt(s): {last_error}"
        )

    def _submit_once(self, request: GenerationRequest) -> dict[str, Any]:
        import httpx

        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            response = client.post(
                f"{QUEUE_URL}/{self.model}",
                json=self._arguments(request),
                headers=self._headers(),
            )
            if response.status_code >= 400:
                error = FalError(
                    f"fal.ai returned HTTP {response.status_code} for "
                    f"{self.model}: {response.text[:500]}"
                )
                error.status_code = response.status_code  # type: ignore[attr-defined]
                raise error

            queued = response.json()
            status_url = queued.get("status_url")
            response_url = queued.get("response_url")
            if not status_url or not response_url:
                raise FalError(f"Unexpected queue response: {queued}")

            log.info("fal queued %s (%s)", self.model, queued.get("request_id", "?"))
            self._await_completion(client, status_url)

            final = client.get(response_url, headers=self._headers(), timeout=120.0)
            final.raise_for_status()
            return final.json()

    def _await_completion(self, client, status_url: str) -> None:
        deadline = time.monotonic() + self.timeout_seconds
        started = time.monotonic()
        announced = 0.0

        while time.monotonic() < deadline:
            status_response = client.get(
                status_url, headers=self._headers(), timeout=30.0
            )
            status_response.raise_for_status()
            payload = status_response.json()
            status = str(payload.get("status", "")).upper()

            if status == "COMPLETED":
                log.info("fal finished in %.1fs", time.monotonic() - started)
                return
            if status in ("FAILED", "ERROR", "CANCELLED"):
                raise FalError(f"fal.ai job {status}: {payload}")

            elapsed = time.monotonic() - started
            if elapsed - announced >= 30.0:
                announced = elapsed
                log.info(
                    "  fal %s (%.0fs elapsed, queue position %s)",
                    status or "working", elapsed, payload.get("queue_position", "?"),
                )
            time.sleep(self.poll_seconds)

        raise FalError(
            f"{self.model} did not finish within {self.timeout_seconds:.0f}s. "
            f"The request may still be running and billable — check fal.ai."
        )

    # -- asset handling ---------------------------------------------------

    def _download(self, url: str, request: GenerationRequest) -> Path:
        import httpx

        suffix = Path(url.split("?", 1)[0]).suffix or ".mp4"
        raw_path = (
            self.cache_dir / f"{self.name}_raw_{request.cache_key(self.name)}{suffix}"
        )
        with httpx.stream("GET", url, timeout=300.0, follow_redirects=True) as response:
            response.raise_for_status()
            partial = raw_path.with_suffix(raw_path.suffix + ".part")
            with partial.open("wb") as handle:
                for chunk in response.iter_bytes(chunk_size=1 << 16):
                    handle.write(chunk)
            partial.replace(raw_path)

        log.info("Downloaded %s", raw_path.name)
        return raw_path

    def _normalise(
        self, source: Path, destination: Path, request: GenerationRequest
    ) -> Path:
        """Guarantee a video, whatever the model produced.

        Image models (FLUX, Nano Banana) are reachable through the same gateway
        and are useful for stills, but the timeline cuts video. A still becomes
        a clip with a slow pan rather than being rejected — a locked-off frozen
        frame in the middle of a beat-cut edit reads as a rendering fault.
        """
        if source.suffix.lower() not in _IMAGE_SUFFIXES:
            source.replace(destination)
            return destination

        log.info("Converting still %s into a %.1fs pan", source.name, request.seconds)
        width, height = request.width, request.height
        # Oversize, then track across the excess. Simpler and more predictable
        # than zoompan, whose duration semantics differ between ffmpeg builds.
        overscan = 1.18
        run([
            ffmpeg_bin(), "-hide_banner", "-nostdin", "-y", "-v", "error",
            "-loop", "1", "-t", f"{request.seconds:.3f}", "-i", str(source),
            "-vf",
            f"scale={int(width * overscan)}:{int(height * overscan)}"
            f":force_original_aspect_ratio=increase,"
            f"crop={width}:{height}"
            f":x='(in_w-out_w)*t/{max(request.seconds, 0.001):.3f}'"
            f":y='(in_h-out_h)/2',"
            f"fps={request.fps:g},format=yuv420p",
            "-c:v", "libx264", "-crf", "18", "-movflags", "+faststart",
            str(destination),
        ])
        source.unlink(missing_ok=True)
        return destination


def _first_asset_url(payload: dict[str, Any]) -> str | None:
    """Find the asset URL in a fal response.

    Response shapes differ per model — `video.url`, `images[0].url`, `output`,
    a bare `url` — so this walks the payload rather than hard-coding one model
    family's schema.
    """
    for key in ("video", "image", "audio", "output", "file"):
        value = payload.get(key)
        if isinstance(value, dict) and value.get("url"):
            return str(value["url"])
        if isinstance(value, str) and value.startswith("http"):
            return value

    for key in ("videos", "images", "files", "outputs"):
        value = payload.get(key)
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, dict) and first.get("url"):
                return str(first["url"])
            if isinstance(first, str) and first.startswith("http"):
                return first

    if isinstance(payload.get("url"), str):
        return str(payload["url"])
    return None
