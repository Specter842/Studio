"""Collects clips from every source into one list for the assembler.

Phase 1 had one source: a folder. Phase 2 adds licensed stock footage and AI
generation, and the whole point is that the assembler cannot tell the
difference — everything arrives as a `ClipInfo` with an `origin` label, and the
timeline is built the same way regardless.

Sourcing is deliberately forgiving. A missing API key or an unreachable ComfyUI
degrades to a loud warning and the clips that *did* arrive, because losing a
render because one of three sources was unavailable is worse than a shorter
pool. It only fails when nothing at all was found.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from budget import Budget, PaidAdapterDisabled
from editing.transitions import VideoFormat
from ingest.local_clips import ClipInfo, NoClipsFound, probe_clip, scan_folder
from ingest.stock_fetch import StockFetcher, queries_from_brief

log = logging.getLogger(__name__)


@dataclass
class SourcingResult:
    clips: list[ClipInfo] = field(default_factory=list)
    #: Human-readable notes about anything that did not work.
    warnings: list[str] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for clip in self.clips:
            totals[clip.origin] = totals.get(clip.origin, 0) + 1
        return totals

    def summary(self) -> str:
        if not self.clips:
            return "no clips"
        parts = ", ".join(
            f"{count} {origin}" for origin, count in sorted(self.counts.items())
        )
        return f"{len(self.clips)} clip(s): {parts}"


def gather_clips(
    *,
    local_dir: Path | None,
    brief: str = "",
    settings,
    budget: Budget,
    cache_dir: Path,
    video_format: VideoFormat,
    stock_per_query: int | None = None,
    generate_count: int | None = None,
    generator_name: str | None = None,
    recursive: bool = False,
    seed: int | None = None,
) -> SourcingResult:
    """Assemble the clip pool from local files, stock footage and generation."""
    result = SourcingResult()

    if local_dir is not None:
        _add_local(result, local_dir, recursive)

    if brief:
        queries = queries_from_brief(brief)
        _add_stock(
            result,
            queries=queries,
            settings=settings,
            cache_dir=cache_dir,
            video_format=video_format,
            per_query=(
                stock_per_query
                if stock_per_query is not None
                else int(settings.get("sourcing.stock_clips_per_query", 2))
            ),
        )
        _add_generated(
            result,
            queries=queries,
            settings=settings,
            budget=budget,
            cache_dir=cache_dir,
            video_format=video_format,
            count=(
                generate_count
                if generate_count is not None
                else int(settings.get("sourcing.generate_clips", 0))
            ),
            generator_name=generator_name or settings.generator_default,
            seed=seed,
        )
    elif stock_per_query or generate_count:
        result.warnings.append(
            "--stock-per-query/--generate need --brief to say what to look for."
        )

    if not result.clips:
        raise NoClipsFound(
            "No clips from any source.\n"
            + "\n".join(f"  - {warning}" for warning in result.warnings)
        )

    log.info("Sourced %s", result.summary())
    return result


def _add_local(result: SourcingResult, local_dir: Path, recursive: bool) -> None:
    try:
        result.clips.extend(scan_folder(local_dir, recursive=recursive))
    except (NoClipsFound, NotADirectoryError) as exc:
        # Not fatal on its own: a brief-only run is a legitimate way to work.
        result.warnings.append(f"local clips: {exc}")
        log.warning("No local clips: %s", exc)


def _add_stock(
    result: SourcingResult,
    *,
    queries: list[str],
    settings,
    cache_dir: Path,
    video_format: VideoFormat,
    per_query: int,
) -> None:
    if per_query <= 0:
        return

    fetcher = StockFetcher(
        cache_dir / "stock",
        sources=settings.get("sourcing.stock_sources", ["pexels", "pixabay"]),
    )
    if not fetcher.available_sources():
        message = (
            "No stock API keys in .env (PEXELS_API_KEY / PIXABAY_API_KEY). "
            "Both are free; see https://www.pexels.com/api/ and "
            "https://pixabay.com/api/docs/."
        )
        result.warnings.append(message)
        log.warning("%s", message)
        return

    try:
        paths = fetcher.fetch(
            queries,
            per_query=per_query,
            min_duration=float(settings.get("sourcing.min_stock_duration", 2.0)),
            min_width=video_format.width,
            orientation=_orientation(video_format),
        )
    except Exception as exc:  # noqa: BLE001 - one source failing is not fatal
        result.warnings.append(f"stock fetch: {exc}")
        log.warning("Stock fetch failed: %s", exc)
        return

    _probe_into(result, paths, origin="stock")


def _add_generated(
    result: SourcingResult,
    *,
    queries: list[str],
    settings,
    budget: Budget,
    cache_dir: Path,
    video_format: VideoFormat,
    count: int,
    generator_name: str,
    seed: int | None,
) -> None:
    if count <= 0:
        return

    # Imported here so a run that generates nothing never touches the adapters.
    from generators import GenerationError, UnknownAdapter, get_adapter

    try:
        adapter = get_adapter(
            generator_name,
            settings=settings,
            budget=budget,
            cache_dir=cache_dir / "generated",
        )
    except (PaidAdapterDisabled, UnknownAdapter, GenerationError) as exc:
        result.warnings.append(f"generator {generator_name}: {exc}")
        log.warning("Generator unavailable: %s", exc)
        return

    log.info("Generating %d clip(s) with %s", count, generator_name)
    seconds = float(settings.get("sourcing.generated_seconds", 4.0))
    negative = str(settings.get("sourcing.negative_prompt", ""))

    paths: list[Path] = []
    for index in range(count):
        prompt = queries[index % len(queries)]
        try:
            paths.append(adapter.generate(
                prompt,
                seconds=seconds,
                width=video_format.width,
                height=video_format.height,
                fps=video_format.fps,
                negative_prompt=negative,
                # Varied per clip so repeated prompts do not return the same
                # shot, and reproducible when --seed is given.
                seed=None if seed is None else seed + index,
            ))
        except Exception as exc:  # noqa: BLE001 - keep whatever did generate
            result.warnings.append(f"generation {index + 1}/{count}: {exc}")
            log.warning("Generation %d/%d failed: %s", index + 1, count, exc)

    _probe_into(result, paths, origin=adapter.name)


def _probe_into(result: SourcingResult, paths: list[Path], *, origin: str) -> None:
    for path in paths:
        try:
            clip = probe_clip(path, origin=origin)
        except Exception as exc:  # noqa: BLE001 - a bad download is not fatal
            result.warnings.append(f"{path.name}: {exc}")
            log.warning("Unusable %s clip %s: %s", origin, path.name, exc)
            continue
        if clip.is_usable:
            result.clips.append(clip)
        else:
            log.warning("Skipping %s: zero duration or resolution.", path.name)


def _orientation(video_format: VideoFormat) -> str:
    if video_format.width > video_format.height:
        return "landscape"
    if video_format.height > video_format.width:
        return "portrait"
    return "square"
