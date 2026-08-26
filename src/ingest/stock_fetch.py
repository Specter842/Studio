"""Licensed stock footage from the Pexels and Pixabay APIs.

This is the sanctioned version of "search the web for clips". Both services
publish a real API, both are free, and both licence their footage for
commercial reuse. Neither requires attribution, but the credits of everything
downloaded are recorded next to the cache anyway — it costs nothing and it is
the difference between being able to credit contributors and not.

Deliberately not supported: pulling video off YouTube, TikTok, Instagram or
arbitrary sites. That is a copyright and terms-of-service problem, not a
technical one, and no amount of care in the code fixes it.
"""

from __future__ import annotations

import abc
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import config

log = logging.getLogger(__name__)

PEXELS_SEARCH_URL = "https://api.pexels.com/videos/search"
PIXABAY_SEARCH_URL = "https://pixabay.com/api/videos/"

CREDITS_JSON = "CREDITS.json"
CREDITS_MARKDOWN = "CREDITS.md"

# Pixabay allows 100 requests/minute, Pexels 200/hour. Neither is close to
# binding for a handful of searches, so one polite retry pass is enough.
MAX_RETRIES = 3
BACKOFF_SECONDS = 2.0


class StockError(RuntimeError):
    """A stock source could not be queried."""


class MissingApiKey(StockError):
    """No API key configured for a requested source."""


@dataclass(frozen=True)
class StockClip:
    """One downloadable result, normalised across both services."""

    source: str
    id: str
    download_url: str
    width: int
    height: int
    duration: float
    page_url: str = ""
    credit_name: str = ""
    credit_url: str = ""

    @property
    def filename(self) -> str:
        return f"{self.source}_{self.id}_{self.width}x{self.height}.mp4"

    def __str__(self) -> str:
        return (
            f"{self.source}#{self.id} {self.width}x{self.height} "
            f"{self.duration:.1f}s"
        )


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------

class StockSource(abc.ABC):
    name: str = ""
    key_env_var: str = ""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    @classmethod
    def from_env(cls) -> "StockSource":
        key = config.secret(cls.key_env_var)
        if not key:
            raise MissingApiKey(
                f"{cls.key_env_var} is not set in .env, so {cls.name} cannot be "
                f"searched. Keys are free at {cls.signup_url()}."
            )
        return cls(key)

    @classmethod
    @abc.abstractmethod
    def signup_url(cls) -> str: ...

    @abc.abstractmethod
    def search(
        self,
        query: str,
        *,
        count: int,
        min_duration: float,
        min_width: int,
        orientation: str,
    ) -> list[StockClip]: ...

    def _get(self, url: str, *, params: dict, headers: dict | None = None) -> dict:
        """GET with a short retry pass, backing off on rate limits."""
        import httpx

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = httpx.get(
                    url, params=params, headers=headers or {}, timeout=30.0
                )
                if response.status_code == 429:
                    wait = BACKOFF_SECONDS * (2 ** attempt)
                    log.warning(
                        "%s rate-limited; waiting %.0fs", self.name, wait
                    )
                    time.sleep(wait)
                    continue
                if response.status_code in (401, 403):
                    raise MissingApiKey(
                        f"{self.name} rejected the API key in .env "
                        f"({self.key_env_var}, HTTP {response.status_code}). "
                        f"Check it at {self.signup_url()}."
                    )
                response.raise_for_status()
                return response.json()
            except MissingApiKey:
                raise
            except Exception as exc:  # noqa: BLE001 - retried below
                last_error = exc
                if attempt < MAX_RETRIES - 1:
                    time.sleep(BACKOFF_SECONDS * (2 ** attempt))

        raise StockError(f"{self.name} search failed: {last_error}") from last_error


class PexelsSource(StockSource):
    name = "pexels"
    key_env_var = "PEXELS_API_KEY"

    @classmethod
    def signup_url(cls) -> str:
        return "https://www.pexels.com/api/"

    def search(
        self,
        query: str,
        *,
        count: int,
        min_duration: float,
        min_width: int,
        orientation: str,
    ) -> list[StockClip]:
        payload = self._get(
            PEXELS_SEARCH_URL,
            params={
                "query": query,
                # Over-fetch: results are filtered on duration and width below.
                "per_page": min(80, max(count * 3, 15)),
                "orientation": orientation,
                "size": "medium",
            },
            headers={"Authorization": self.api_key},
        )

        clips: list[StockClip] = []
        for video in payload.get("videos", []):
            best = _best_pexels_file(video.get("video_files", []), min_width)
            if not best:
                continue
            duration = float(video.get("duration") or 0)
            if duration < min_duration:
                continue
            user = video.get("user") or {}
            clips.append(StockClip(
                source=self.name,
                id=str(video.get("id")),
                download_url=best["link"],
                width=int(best.get("width") or 0),
                height=int(best.get("height") or 0),
                duration=duration,
                page_url=str(video.get("url") or ""),
                credit_name=str(user.get("name") or ""),
                credit_url=str(user.get("url") or ""),
            ))
        return clips


class PixabaySource(StockSource):
    name = "pixabay"
    key_env_var = "PIXABAY_API_KEY"

    @classmethod
    def signup_url(cls) -> str:
        return "https://pixabay.com/api/docs/"

    def search(
        self,
        query: str,
        *,
        count: int,
        min_duration: float,
        min_width: int,
        orientation: str,
    ) -> list[StockClip]:
        payload = self._get(
            PIXABAY_SEARCH_URL,
            params={
                "key": self.api_key,
                "q": query,
                "per_page": min(200, max(count * 3, 15)),
                "video_type": "film",
                "safesearch": "true",
            },
        )

        clips: list[StockClip] = []
        for hit in payload.get("hits", []):
            best = _best_pixabay_file(hit.get("videos", {}), min_width)
            if not best:
                continue
            duration = float(hit.get("duration") or 0)
            if duration < min_duration:
                continue
            clips.append(StockClip(
                source=self.name,
                id=str(hit.get("id")),
                download_url=best["url"],
                width=int(best.get("width") or 0),
                height=int(best.get("height") or 0),
                duration=duration,
                page_url=str(hit.get("pageURL") or ""),
                credit_name=str(hit.get("user") or ""),
                credit_url=(
                    f"https://pixabay.com/users/{hit.get('user')}-{hit.get('user_id')}/"
                    if hit.get("user_id") else ""
                ),
            ))
        return clips


SOURCES: dict[str, type[StockSource]] = {
    PexelsSource.name: PexelsSource,
    PixabaySource.name: PixabaySource,
}


def _best_pexels_file(files: Sequence[dict], min_width: int) -> dict | None:
    """Smallest MP4 that still meets the width requirement.

    Not the largest available: a 4K download of a clip that will be scaled to
    1080p wastes bandwidth, disk and decode time for no visible gain.
    """
    candidates = [
        entry for entry in files
        if str(entry.get("file_type", "")).endswith("mp4") and entry.get("link")
    ]
    if not candidates:
        return None
    big_enough = [
        entry for entry in candidates if int(entry.get("width") or 0) >= min_width
    ]
    pool = big_enough or candidates
    return min(pool, key=lambda entry: int(entry.get("width") or 0) or 10**9)


def _best_pixabay_file(videos: dict, min_width: int) -> dict | None:
    candidates = [
        entry for entry in videos.values()
        if isinstance(entry, dict) and entry.get("url")
    ]
    if not candidates:
        return None
    big_enough = [
        entry for entry in candidates if int(entry.get("width") or 0) >= min_width
    ]
    pool = big_enough or candidates
    return min(pool, key=lambda entry: int(entry.get("width") or 0) or 10**9)


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------

class StockFetcher:
    """Searches the configured sources and downloads results into the cache."""

    def __init__(
        self,
        cache_dir: Path | str,
        *,
        sources: Iterable[str] = ("pexels", "pixabay"),
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.requested_sources = list(sources)
        self._ledger = CreditLedger(self.cache_dir)

    def available_sources(self) -> list[StockSource]:
        """Sources that actually have a key, with a clear note about the rest."""
        ready: list[StockSource] = []
        for name in self.requested_sources:
            source_class = SOURCES.get(name)
            if not source_class:
                log.warning("Unknown stock source %r; skipping.", name)
                continue
            try:
                ready.append(source_class.from_env())
            except MissingApiKey as exc:
                log.warning("%s", exc)
        return ready

    def search(
        self,
        query: str,
        *,
        count: int = 4,
        min_duration: float = 2.0,
        min_width: int = 1280,
        orientation: str = "landscape",
    ) -> list[StockClip]:
        """Search every available source and interleave the results.

        Interleaved rather than concatenated so a single source cannot supply
        the whole edit when both are configured.
        """
        per_source: list[list[StockClip]] = []
        for source in self.available_sources():
            try:
                found = source.search(
                    query,
                    count=count,
                    min_duration=min_duration,
                    min_width=min_width,
                    orientation=orientation,
                )
            except StockError as exc:
                log.warning("%s", exc)
                continue
            log.info("%s: %d result(s) for %r", source.name, len(found), query)
            per_source.append(found)

        merged: list[StockClip] = []
        for index in range(max((len(group) for group in per_source), default=0)):
            for group in per_source:
                if index < len(group):
                    merged.append(group[index])
        return merged[:count]

    def download(self, clip: StockClip) -> Path:
        """Fetch one clip into the cache, skipping anything already there."""
        import httpx

        destination = self.cache_dir / clip.filename
        if destination.exists() and destination.stat().st_size > 0:
            log.debug("Cached: %s", destination.name)
            self._ledger.record(clip)
            return destination

        log.info("Downloading %s", clip)
        with httpx.stream(
            "GET", clip.download_url, timeout=120.0, follow_redirects=True
        ) as response:
            response.raise_for_status()
            # Written to a partial file first, so an interrupted download can
            # never be mistaken for a cache hit on the next run.
            partial = destination.with_suffix(".part")
            with partial.open("wb") as handle:
                for chunk in response.iter_bytes(chunk_size=1 << 16):
                    handle.write(chunk)
            partial.replace(destination)

        self._ledger.record(clip)
        return destination

    def fetch(
        self,
        queries: Sequence[str],
        *,
        per_query: int = 2,
        min_duration: float = 2.0,
        min_width: int = 1280,
        orientation: str = "landscape",
    ) -> list[Path]:
        """Search and download for each query; returns local paths."""
        paths: list[Path] = []
        seen: set[str] = set()

        for query in queries:
            for clip in self.search(
                query,
                count=per_query,
                min_duration=min_duration,
                min_width=min_width,
                orientation=orientation,
            ):
                identity = f"{clip.source}:{clip.id}"
                if identity in seen:
                    continue
                seen.add(identity)
                try:
                    paths.append(self.download(clip))
                except Exception as exc:  # noqa: BLE001 - one bad URL is not fatal
                    log.warning("Could not download %s: %s", clip, exc)

        self._ledger.save()
        if paths:
            log.info("Stock: %d clip(s) in %s", len(paths), self.cache_dir)
        return paths


class CreditLedger:
    """Keeps a record of who made each downloaded clip."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.path = self.cache_dir / CREDITS_JSON
        self.entries: dict[str, dict[str, Any]] = {}
        if self.path.is_file():
            try:
                self.entries = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log.warning("%s is unreadable; starting a fresh credit list.", self.path)

    def record(self, clip: StockClip) -> None:
        self.entries[clip.filename] = asdict(clip)

    def save(self) -> None:
        if not self.entries:
            return
        self.path.write_text(
            json.dumps(self.entries, indent=2, sort_keys=True), encoding="utf-8"
        )
        (self.cache_dir / CREDITS_MARKDOWN).write_text(
            self.as_markdown(), encoding="utf-8"
        )

    def as_markdown(self) -> str:
        lines = [
            "# Stock footage credits",
            "",
            "Neither Pexels nor Pixabay requires attribution, but this is who "
            "made the clips used here.",
            "",
        ]
        for filename in sorted(self.entries):
            entry = self.entries[filename]
            author = entry.get("credit_name") or "unknown"
            page = entry.get("page_url") or ""
            lines.append(
                f"- `{filename}` — {author} via {entry.get('source', '?')}"
                + (f" ({page})" if page else "")
            )
        return "\n".join(lines) + "\n"


def queries_from_brief(brief: str, *, limit: int = 6) -> list[str]:
    """Turn a free-text brief into search queries.

    Semicolons and newlines are explicit separators, so `--brief "city at
    night; neon signs"` searches twice. Without them the brief is one query —
    stock search engines handle a short phrase better than a bag of keywords,
    and inventing extra queries by splitting on spaces mostly returns noise.
    """
    parts = [part.strip() for part in re.split(r"[;\n]+", brief) if part.strip()]
    return (parts or [brief.strip()])[:limit]
