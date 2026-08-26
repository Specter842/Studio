"""Scan a folder of local video files and probe each one with ffprobe.

This is the Phase 1 clip source. In Phase 2, stock downloads and ComfyUI
generations produce the same ClipInfo objects, so the assembler continues to
work without knowing where a clip came from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ffmpeg_tools import parse_fraction, probe_json

log = logging.getLogger(__name__)

# Containers ffmpeg reads happily. Deliberately conservative — an unreadable
# file surfaces as a clear skip rather than a render failure 200 segments in.
VIDEO_EXTENSIONS: tuple[str, ...] = (
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv",
)


@dataclass(frozen=True)
class ClipInfo:
    """Everything the assembler needs to know about one source clip."""

    path: Path
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    codec: str = ""
    # "local" in Phase 1; "stock" / "comfyui" / "fal" once Phase 2 lands.
    origin: str = "local"

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0

    @property
    def is_usable(self) -> bool:
        return self.duration > 0 and self.width > 0 and self.height > 0

    def __str__(self) -> str:
        return (
            f"{self.path.name} "
            f"({self.width}x{self.height} @{self.fps:.2f}fps, {self.duration:.2f}s)"
        )


class NoClipsFound(RuntimeError):
    """The clips folder yielded nothing usable."""


def probe_clip(path: Path | str, *, origin: str = "local") -> ClipInfo:
    """Read duration / resolution / fps for one video file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Clip not found: {path}")

    data = probe_json(path)
    streams = data.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    if not video_streams:
        raise ValueError(f"{path.name} has no video stream.")

    stream = video_streams[0]
    fmt = data.get("format", {})

    # Stream duration is missing on some containers (notably WebM); the format
    # level duration is the reliable fallback.
    duration = float(stream.get("duration") or fmt.get("duration") or 0.0)

    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)

    # Phone footage is stored landscape with a rotation flag. Report the
    # displayed dimensions so aspect-ratio decisions downstream are correct.
    if _rotation_degrees(stream) in (90, 270):
        width, height = height, width

    # avg_frame_rate is the honest average; r_frame_rate is the container's
    # nominal rate and is wrong (often 1000/1) for variable frame rate files.
    fps = parse_fraction(stream.get("avg_frame_rate"))
    if fps <= 0:
        fps = parse_fraction(stream.get("r_frame_rate"))

    return ClipInfo(
        path=path,
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        codec=str(stream.get("codec_name") or ""),
        origin=origin,
    )


def _rotation_degrees(stream: dict) -> int:
    """Pull rotation out of either the tag or the display matrix side data."""
    tag = stream.get("tags", {}).get("rotate")
    if tag is not None:
        try:
            return int(float(tag)) % 360
        except (TypeError, ValueError):
            pass
    for side_data in stream.get("side_data_list", []) or []:
        if "rotation" in side_data:
            try:
                # The display matrix reports the inverse of the applied rotation.
                return int(-float(side_data["rotation"])) % 360
            except (TypeError, ValueError):
                pass
    return 0


def scan_folder(
    folder: Path | str,
    *,
    extensions: Sequence[str] = VIDEO_EXTENSIONS,
    min_duration: float = 0.0,
    recursive: bool = False,
) -> list[ClipInfo]:
    """Probe every video file in `folder`, sorted by filename for determinism.

    Files that fail to probe are logged and skipped rather than aborting the
    scan — one corrupt download shouldn't cost you the whole render.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Clips folder not found: {folder}")

    suffixes = {ext.lower() for ext in extensions}
    pattern = "**/*" if recursive else "*"
    candidates = sorted(
        p for p in folder.glob(pattern)
        if p.is_file() and p.suffix.lower() in suffixes
    )

    clips: list[ClipInfo] = []
    for candidate in candidates:
        try:
            clip = probe_clip(candidate)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the scan
            log.warning("Skipping %s: %s", candidate.name, exc)
            continue

        if not clip.is_usable:
            log.warning("Skipping %s: zero duration or resolution.", candidate.name)
            continue
        if clip.duration < min_duration:
            log.warning(
                "Skipping %s: %.2fs is under the %.2fs minimum.",
                candidate.name, clip.duration, min_duration,
            )
            continue

        clips.append(clip)
        log.debug("Ingested %s", clip)

    if not clips:
        raise NoClipsFound(
            f"No usable video files in {folder} "
            f"(looked for: {', '.join(sorted(suffixes))})."
        )

    log.info("Ingested %d clip(s) from %s", len(clips), folder)
    return clips


def total_duration(clips: Iterable[ClipInfo]) -> float:
    return sum(clip.duration for clip in clips)
