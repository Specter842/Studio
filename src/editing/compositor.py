"""Lays 3D elements over the finished edit.

A separate ffmpeg pass rather than part of the main filter graph, on purpose.
The main render is capped into passes and stitched with the concat demuxer, and
an overlay that has to span pass boundaries would fight that. Compositing after
the fact costs one more video encode and keeps both halves simple.

The audio is stream-copied through, so the beat alignment established by the
assembler cannot be disturbed by anything here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ffmpeg_tools import ffmpeg_bin, run_with_progress

log = logging.getLogger(__name__)

POSITIONS = {
    # y expression for the overlay filter; x is always centred.
    "center": "(H-h)/2",
    "top": "H*0.12",
    "bottom": "H*0.88-h",
}


@dataclass(frozen=True)
class Overlay:
    """One element to lay over the video."""

    #: ffmpeg input path. An image-sequence pattern (frame_%04d.png) or a file.
    input_path: str
    fps: float
    start: float
    duration: float
    fade: float = 0.3
    position: str = "center"
    opacity: float = 1.0
    #: Width as a fraction of the output width.
    scale: float = 1.0

    @property
    def end(self) -> float:
        return self.start + self.duration


def composite(
    video_path: Path | str,
    overlays: list[Overlay],
    out_path: Path | str,
    *,
    encode: dict | None = None,
    timeout: float | None = None,
) -> Path:
    """Burn `overlays` onto `video_path`, writing `out_path`."""
    video_path = Path(video_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encode = encode or {}

    if not overlays:
        raise ValueError("composite() needs at least one overlay.")

    command: list[str] = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-y",
                          "-i", str(video_path)]
    for overlay in overlays:
        # -framerate must precede the input: for an image sequence it is the
        # only thing that tells ffmpeg how fast the frames run.
        command += ["-framerate", f"{overlay.fps:g}", "-i", overlay.input_path]

    command += ["-filter_complex", _build_graph(overlays)]
    command += [
        "-map", "[v]",
        # No audio stream to map if the source had none (an element-only test).
        "-map", "0:a?",
        "-c:v", str(encode.get("video_codec", "libx264")),
        "-crf", str(encode.get("crf", 20)),
        "-preset", str(encode.get("preset", "medium")),
        "-pix_fmt", str(encode.get("pixel_format", "yuv420p")),
        # Copied, never re-encoded: the audio is what the cuts were aligned to.
        "-c:a", "copy",
    ]
    if encode.get("faststart", True):
        command += ["-movflags", "+faststart"]
    command += [str(out_path)]

    log.info("Compositing %d element(s) onto %s", len(overlays), video_path.name)
    run_with_progress(command, timeout=timeout)
    return out_path


def _build_graph(overlays: list[Overlay]) -> str:
    """Prepare each element, then chain them onto the base video."""
    lines: list[str] = []
    for index, overlay in enumerate(overlays, start=1):
        lines.append(_prepare_overlay(index, overlay))

    current = "0:v"
    for index, overlay in enumerate(overlays, start=1):
        label = "v" if index == len(overlays) else f"b{index}"
        y = POSITIONS.get(overlay.position, POSITIONS["center"])
        lines.append(
            f"[{current}][ov{index}]overlay=x=(W-w)/2:y={y}"
            # eof_action=pass lets the video continue once the element ends;
            # repeatlast=0 stops the last element frame being frozen on screen
            # for the rest of the video, which is the default and never wanted.
            f":eof_action=pass:repeatlast=0[{label}]"
        )
        current = label

    return ";\n".join(lines)


def _prepare_overlay(index: int, overlay: Overlay) -> str:
    filters = ["format=rgba"]

    if overlay.scale != 1.0:
        # -2 keeps the height even, which yuv420p requires.
        filters.append(f"scale=iw*{overlay.scale:g}:-2")

    if overlay.opacity < 1.0:
        filters.append(f"colorchannelmixer=aa={overlay.opacity:g}")

    fade = min(overlay.fade, overlay.duration / 2.0)
    if fade > 0:
        # alpha=1 fades the alpha channel rather than toward black, which is
        # the difference between a title dissolving and a black box dissolving.
        filters.append(f"fade=t=in:st=0:d={fade:.3f}:alpha=1")
        filters.append(
            f"fade=t=out:st={max(0.0, overlay.duration - fade):.3f}"
            f":d={fade:.3f}:alpha=1"
        )

    # Shift the element to its timeline position. Without this every overlay
    # would start at zero regardless of when it was asked for.
    filters.append("setpts=PTS-STARTPTS")
    if overlay.start > 0:
        filters.append(f"setpts=PTS+{overlay.start:.6f}/TB")

    return f"[{index}:v]" + ",".join(filters) + f"[ov{index}]"
