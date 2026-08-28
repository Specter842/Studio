"""The transition library: hard cuts, and every dissolve ffmpeg's xfade knows.

`available_transitions()` renders a real, tiny clip through every family below
and checks the pixels actually moved — a wrong `xfade=transition=` value is a
silent no-op in some ffmpeg builds (falls back to a plain fade) rather than an
error, so "it didn't crash" is not proof it did the right transition.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from conftest import requires_ffmpeg
from editing import transitions
from editing.transitions import (
    CROSSFADE,
    CUT,
    XFADE_TRANSITIONS,
    InputSpec,
    VideoFormat,
    available_transitions,
    build_video_graph,
)
from ffmpeg_tools import ffmpeg_bin, ffprobe_bin

FORMAT = VideoFormat(160, 90, 24.0)


def specs(count: int, frames: int = 48) -> list[InputSpec]:
    return [InputSpec(input_index=i, frames=frames) for i in range(count)]


# --- the library itself -----------------------------------------------------

def test_every_xfade_name_is_unique() -> None:
    assert len(XFADE_TRANSITIONS) == len(set(XFADE_TRANSITIONS))


def test_the_library_has_every_transition_the_installed_ffmpeg_ships() -> None:
    """Regression: this list was hand-copied from `ffmpeg -h filter=xfade` on
    the 8.1.2 build. If a name here doesn't exist on the machine running the
    tests, building a graph with it must fail loudly, not silently degrade."""
    known = set(available_transitions())
    for name in XFADE_TRANSITIONS:
        assert name in known, f"{name!r} is not a transition this ffmpeg knows"


def test_cut_and_crossfade_are_not_xfade_names() -> None:
    assert CUT not in XFADE_TRANSITIONS
    assert CROSSFADE not in XFADE_TRANSITIONS


# --- graph construction ------------------------------------------------------

def test_crossfade_is_an_alias_for_the_plain_fade_transition() -> None:
    graph = build_video_graph(specs(2), FORMAT, transition=CROSSFADE, crossfade_seconds=0.5)
    assert "xfade=transition=fade:" in graph


@pytest.mark.parametrize("name", ["wipeleft", "pixelize", "circleopen", "dissolve", "zoomin"])
def test_any_named_transition_is_wired_into_the_graph(name: str) -> None:
    graph = build_video_graph(specs(2), FORMAT, transition=name, crossfade_seconds=0.5)
    assert f"xfade=transition={name}:" in graph


def test_an_unknown_transition_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown transition"):
        build_video_graph(specs(2), FORMAT, transition="teleport")


def test_multi_segment_chains_use_the_same_transition_at_every_join() -> None:
    graph = build_video_graph(specs(4), FORMAT, transition="wipeup", crossfade_seconds=0.3)
    assert graph.count("xfade=transition=wipeup:") == 3


# --- rendered proof -----------------------------------------------------------

def _render_two_colours(transition: str, tmp_path: Path) -> Path:
    """Cross-fade a solid red clip into a solid blue one and return the file."""
    red = tmp_path / "red.mp4"
    blue = tmp_path / "blue.mp4"
    for path, colour in ((red, "red"), (blue, "blue")):
        subprocess.run([
            ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"color=c={colour}:s=160x90:r=24:d=1.5",
            "-pix_fmt", "yuv420p", str(path),
        ], check=True, capture_output=True)

    graph = build_video_graph(
        [InputSpec(0, 36), InputSpec(1, 36)], FORMAT,
        transition=transition, crossfade_seconds=0.4,
    )
    graph_path = tmp_path / "graph.txt"
    graph_path.write_text(graph, encoding="utf-8")

    out_path = tmp_path / f"{transition}.mp4"
    subprocess.run([
        ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
        "-i", str(red), "-i", str(blue),
        "-filter_complex_script", str(graph_path),
        "-map", f"[{transitions.VIDEO_OUT}]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path),
    ], check=True, capture_output=True)
    return out_path


def _sample_luma_track(path: Path, count: int = 30) -> list[float]:
    """Mean luma of `count` evenly spaced frames across the whole file.

    Scanning rather than pinning one hand-picked timestamp sidesteps every
    seek/keyframe/rounding edge case around exactly where a join lands —
    what matters here is the *shape* of the luma curve across the file, not
    hitting one exact millisecond.
    """
    duration_raw = subprocess.run([
        ffprobe_bin(), "-v", "error",
        "-i", str(path), "-show_entries", "format=duration",
        "-of", "csv=p=0",
    ], capture_output=True, text=True)
    duration = float(duration_raw.stdout.strip() or 0.0)
    assert duration > 0, f"could not read duration of {path}"

    raw = subprocess.run([
        ffmpeg_bin(), "-hide_banner", "-v", "error", "-i", str(path),
        "-vf", f"fps={count / duration:.6f},scale=1:1:flags=area",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ], check=True, capture_output=True).stdout
    return [float(b) for b in raw]


@requires_ffmpeg
@pytest.mark.parametrize("transition", [
    "fade", "wipeleft", "circleopen", "pixelize", "dissolve", "hblur", "zoomin",
])
def test_the_named_transition_actually_changes_the_picture(
    transition: str, tmp_path: Path
) -> None:
    """Not just 'ffmpeg didn't error' — somewhere across the join the picture
    must pass through a value clearly between pure red and pure blue, proving
    a real dissolve ran there rather than an ffmpeg build silently
    substituting a hard cut for an unrecognised transition name."""
    out_path = _render_two_colours(transition, tmp_path)
    samples = _sample_luma_track(out_path)

    # A fixed luma margin rather than a percentage of the range: geometric
    # wipes (circleopen, pixelize...) only spend a frame or two actually
    # mid-transition — a percentage-of-range band is tight enough to miss
    # them by chance depending on exactly which instant got sampled. x264
    # compression noise near a solid colour is a couple of luma units at
    # most, so anything more than that off either endpoint is a real
    # in-between frame, not encoder noise.
    low, high = min(samples), max(samples)
    band_low, band_high = low + 4, high - 4
    blended = [s for s in samples if band_low < s < band_high]

    assert blended, (
        f"{transition}: no frame fell between the two colours "
        f"(range {low:.0f}-{high:.0f}, samples={samples}) — looks like a hard "
        f"cut, not a dissolve"
    )


@requires_ffmpeg
def test_a_hard_cut_has_no_dissolve_frame_at_all(tmp_path: Path) -> None:
    """Contrast case for the test above: a plain cut must NOT blend."""
    red = tmp_path / "red.mp4"
    blue = tmp_path / "blue.mp4"
    for path, colour in ((red, "red"), (blue, "blue")):
        subprocess.run([
            ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"color=c={colour}:s=160x90:r=24:d=1.0",
            "-pix_fmt", "yuv420p", str(path),
        ], check=True, capture_output=True)

    graph = build_video_graph(
        [InputSpec(0, 24), InputSpec(1, 24)], FORMAT, transition=CUT,
    )
    graph_path = tmp_path / "graph.txt"
    graph_path.write_text(graph, encoding="utf-8")
    out_path = tmp_path / "cut.mp4"
    subprocess.run([
        ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
        "-i", str(red), "-i", str(blue),
        "-filter_complex_script", str(graph_path),
        "-map", f"[{transitions.VIDEO_OUT}]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path),
    ], check=True, capture_output=True)

    samples = _sample_luma_track(out_path)
    # A fixed luma margin rather than a percentage of the range: geometric
    # wipes (circleopen, pixelize...) only spend a frame or two actually
    # mid-transition — a percentage-of-range band is tight enough to miss
    # them by chance depending on exactly which instant got sampled. x264
    # compression noise near a solid colour is a couple of luma units at
    # most, so anything more than that off either endpoint is a real
    # in-between frame, not encoder noise.
    low, high = min(samples), max(samples)
    band_low, band_high = low + 4, high - 4
    blended = [s for s in samples if band_low < s < band_high]

    # Red and blue are far apart in luma; a dissolve would land between them.
    # A hard cut must never produce an in-between frame anywhere in the file.
    assert not blended, f"cut looks blended somewhere: {samples}"
