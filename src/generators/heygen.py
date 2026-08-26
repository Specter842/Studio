"""Optional paid adapter: HeyGen talking-head avatars. **Off by default.**

Wire this in only when a project specifically needs a presenter or spokesperson
shot. Unlike the local ComfyUI path there is no free tier at all — HeyGen bills
per minute of generated video — so it is gated exactly like `fal_gateway`:

  * cannot be constructed while `paid_adapters_enabled` is false
  * budget reserved before every attempt, retries included
  * only server errors retried; a rejected request is a bug, not bad luck

It is never part of a default run. `generator_default` stays `local_comfyui`,
and reaching this adapter takes a deliberate `--generator heygen`.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import config
from generators.base_adapter import (
    GenerationError,
    GenerationRequest,
    GeneratorAdapter,
    register,
)

log = logging.getLogger(__name__)

GENERATE_URL = "https://api.heygen.com/v2/video/generate"
STATUS_URL = "https://api.heygen.com/v1/video_status.get"

_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class HeyGenError(GenerationError):
    """HeyGen could not fulfil the request."""


@register
class HeyGenAdapter(GeneratorAdapter):
    """Generates a talking-head clip from a script line."""

    name = "heygen"
    is_paid = True
    # HeyGen prices per minute of output. This is the per-call default used by
    # the budget guard until `heygen.cost_per_call_usd` is set from their
    # current pricing; deliberately pessimistic, since overestimating stops a
    # run early and underestimating does not stop it at all.
    estimated_cost_per_call_usd = 1.00

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        settings = self.settings

        self.avatar_id = str(settings.get("heygen.avatar_id", "")).strip()
        self.voice_id = str(settings.get("heygen.voice_id", "")).strip()
        if not self.avatar_id or not self.voice_id:
            raise HeyGenError(
                "heygen.avatar_id and heygen.voice_id must both be set in "
                "config/settings.yaml. List yours with:\n"
                "  curl -H 'X-Api-Key: $HEYGEN_API_KEY' "
                "https://api.heygen.com/v2/avatars"
            )

        self.avatar_style = str(settings.get("heygen.avatar_style", "normal"))
        self.background = str(settings.get("heygen.background_color", "#000000"))
        self.timeout_seconds = float(settings.get("heygen.timeout_seconds", 1800))
        self.poll_seconds = float(settings.get("heygen.poll_seconds", 5.0))
        self.max_retries = int(settings.get("heygen.max_retries", 3))
        self.backoff_seconds = float(settings.get("heygen.backoff_seconds", 2.0))

        self.estimated_cost_per_call_usd = float(
            settings.get(
                "heygen.cost_per_call_usd", type(self).estimated_cost_per_call_usd
            )
        )

        self.api_key = config.secret("HEYGEN_API_KEY")
        if not self.api_key:
            raise HeyGenError(
                "HEYGEN_API_KEY is not set in .env. Get a key at "
                "https://app.heygen.com/settings — never put it in code or "
                "settings.yaml."
            )

    # -- public API -------------------------------------------------------

    def generate(self, prompt: str, **kwargs) -> Path:
        """`prompt` is the line the avatar speaks."""
        request = self.build_request(prompt, **kwargs)
        destination = self.cached_path(request)
        if destination.exists():
            log.info("HeyGen cache hit (no charge): %s", destination.name)
            return destination

        video_id = self._submit_with_retries(request)
        url = self._await_video(video_id)
        return self._download(url, destination)

    # -- request ----------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {"X-Api-Key": self.api_key, "Content-Type": "application/json"}

    def _payload(self, request: GenerationRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "video_inputs": [{
                "character": {
                    "type": "avatar",
                    "avatar_id": self.avatar_id,
                    "avatar_style": self.avatar_style,
                },
                "voice": {
                    "type": "text",
                    "input_text": request.prompt,
                    "voice_id": self.voice_id,
                },
                "background": {"type": "color", "value": self.background},
            }],
            "dimension": {"width": request.width, "height": request.height},
        }
        payload.update(self.settings.get("heygen.arguments", {}) or {})
        payload.update(request.extra)
        return payload

    def _submit_with_retries(self, request: GenerationRequest) -> str:
        import httpx

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self.reserve_budget(
                note=f"heygen attempt {attempt}: {request.prompt[:40]}"
            )
            try:
                return self._submit_once(request)
            except HeyGenError as exc:
                status = getattr(exc, "status_code", None)
                if status is not None and status not in _RETRYABLE_STATUS:
                    raise
                last_error = exc
            except httpx.HTTPError as exc:
                last_error = exc

            if attempt < self.max_retries:
                wait = self.backoff_seconds * (2 ** (attempt - 1))
                log.warning(
                    "HeyGen attempt %d/%d failed (%s); retrying in %.0fs",
                    attempt, self.max_retries, last_error, wait,
                )
                time.sleep(wait)

        raise HeyGenError(
            f"HeyGen failed after {self.max_retries} attempt(s): {last_error}"
        )

    def _submit_once(self, request: GenerationRequest) -> str:
        import httpx

        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            response = client.post(
                GENERATE_URL, json=self._payload(request), headers=self._headers()
            )
            if response.status_code >= 400:
                error = HeyGenError(
                    f"HeyGen returned HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )
                error.status_code = response.status_code  # type: ignore[attr-defined]
                raise error

            body = response.json()
            # HeyGen reports some failures with a 200 and an error field.
            if body.get("error"):
                raise HeyGenError(f"HeyGen rejected the request: {body['error']}")

            video_id = (body.get("data") or {}).get("video_id")
            if not video_id:
                raise HeyGenError(f"HeyGen returned no video_id: {body}")
            log.info("HeyGen queued video %s", video_id)
            return str(video_id)

    def _await_video(self, video_id: str) -> str:
        import httpx

        deadline = time.monotonic() + self.timeout_seconds
        started = time.monotonic()
        announced = 0.0

        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            while time.monotonic() < deadline:
                response = client.get(
                    STATUS_URL, params={"video_id": video_id},
                    headers=self._headers(),
                )
                response.raise_for_status()
                data = response.json().get("data") or {}
                status = str(data.get("status", "")).lower()

                if status == "completed":
                    url = data.get("video_url")
                    if not url:
                        raise HeyGenError(
                            f"HeyGen completed {video_id} but returned no URL."
                        )
                    log.info("HeyGen finished in %.1fs", time.monotonic() - started)
                    return str(url)

                if status in ("failed", "error"):
                    raise HeyGenError(
                        f"HeyGen failed on {video_id}: "
                        f"{data.get('error') or data}"
                    )

                elapsed = time.monotonic() - started
                if elapsed - announced >= 30.0:
                    announced = elapsed
                    log.info("  HeyGen %s (%.0fs elapsed)", status or "working", elapsed)
                time.sleep(self.poll_seconds)

        raise HeyGenError(
            f"HeyGen did not finish {video_id} within {self.timeout_seconds:.0f}s. "
            f"The video may still be rendering and billable — check your account."
        )

    def _download(self, url: str, destination: Path) -> Path:
        import httpx

        with httpx.stream("GET", url, timeout=600.0, follow_redirects=True) as response:
            response.raise_for_status()
            partial = destination.with_suffix(destination.suffix + ".part")
            with partial.open("wb") as handle:
                for chunk in response.iter_bytes(chunk_size=1 << 16):
                    handle.write(chunk)
            partial.replace(destination)

        log.info("Downloaded %s", destination.name)
        return destination
