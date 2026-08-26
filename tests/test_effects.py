"""Cut-timed post effects: punch-zoom and glitch, plus the chain that combines
them with a colour grade.

Both effects are verified by rendering a real test pattern designed so the
effect is unmistakable in the pixels — a bordered frame for zoom (the border
must vanish only in the zoom window), a hard colour split for glitch (the
channels must misalign only in the glitch window) — and reading every frame
back in one pass (see `conftest.frame_metric_track`). Two independently
plausible implementations (`scale`/`crop` driven by `t`, and `zoompan` driven
by `on`) were tried and silently did nothing on the installed ffmpeg before
landing on the one used here; these tests are what would have caught that.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from conftest import frame_metric_track, requires_ffmpeg
from editing import effects
from editing.transitions import VideoFormat
from ffmpeg_tools import ffmpeg_bin

FMT = VideoFormat(200, 200, 25.0)


# --- filter-string construction (no rendering) ------------------------------

def test_no_cuts_means_no_glitch_filter() -> None:
    assert effects.glitch_filters((), shift_px=6) is None


def test_zero_shift_means_no_glitch_filter() -> None:
    assert effects.glitch_filters((1.0, 2.0), shift_px=0) is None


def test_glitch_windows_cover_every_cut() -> None:
    chain = effects.glitch_filters((1.0, 2.5, 4.0), seconds=0.1, shift_px=6)
    assert chain.count("between(") == 3
    assert "rh=6" in chain and "bh=-6" in chain


def test_build_post_chain_is_none_when_everything_is_disabled() -> None:
    assert effects.build_post_chain((1.0, 2.0), FMT) is None


def test_build_post_chain_with_only_a_look_is_a_simple_linear_stage() -> None:
    chain = effects.build_post_chain((1.0,), FMT, look="punchy")
    assert chain.count(";") == 0
    assert "eq=" in chain


def test_build_post_chain_stages_are_wired_in_order() -> None:
    """look -> zoom -> glitch, each stage consuming the previous one's label."""
    chain = effects.build_post_chain(
        (1.0, 2.0), FMT,
        look="blackout", punch_zoom_amount=0.2, glitch_shift_px=5,
    )
    assert chain.index("eq=") < chain.index("overlay=") < chain.index("rgbashift=")
    assert "[outv]" in chain.splitlines()[-1]


# --- rendered proof: punch-zoom ---------------------------------------------

@pytest.fixture
def bordered(tmp_path: Path) -> Path:
    """A white 6px frame border on black — zoom must crop it out of view."""
    path = tmp_path / "bordered.mp4"
    subprocess.run([
        ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=200x200:r=25:d=4",
        "-vf", "drawbox=x=0:y=0:w=iw:h=ih:color=white:t=6",
        "-pix_fmt", "yuv420p", str(path),
    ], check=True, capture_output=True)
    return path


def _render_chain(source: Path, chain: str, tmp_path: Path, name: str) -> Path:
    """`chain` is a full `build_post_chain()` result: already self-contained,
    starting with its own `[input_label]` and ending in `[outv]`."""
    graph_path = tmp_path / f"{name}_graph.txt"
    graph_path.write_text(chain, encoding="utf-8")
    out_path = tmp_path / f"{name}.mp4"
    subprocess.run([
        ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
        "-i", str(source), "-filter_complex_script", str(graph_path),
        "-map", "[outv]", "-pix_fmt", "yuv420p", str(out_path),
    ], check=True, capture_output=True)
    return out_path


@requires_ffmpeg
def test_punch_zoom_hides_the_border_only_inside_its_window(
    bordered: Path, tmp_path: Path
) -> None:
    chain = effects.build_post_chain(
        (1.0, 2.0, 3.0), FMT,
        punch_zoom_amount=0.25, punch_zoom_seconds=0.3,
        input_label="0:v",  # feed the source directly, no transitions stage
    )
    out = _render_chain(bordered, chain, tmp_path, "zoom")

    track = frame_metric_track(out, metrics=("YAVG",), crop="crop=4:4:0:0")["YAVG"]
    fps = FMT.fps

    def at(seconds: float) -> float:
        return track[round(seconds * fps)]

    # Away from any cut: the white border fills the sampled corner.
    assert at(0.5) > 200
    assert at(1.5) > 200
    assert at(3.5) > 200
    # Inside every zoom window: the border has been cropped out of frame,
    # leaving the black interior.
    assert at(1.0) < 50
    assert at(2.0) < 50
    assert at(3.0) < 50


@requires_ffmpeg
def test_punch_zoom_every_nth_only_fires_on_selected_cuts() -> None:
    """every_nth is handled by the caller slicing cut_times before this
    module sees them — confirmed here at the chain-construction level."""
    chain_all = effects.build_post_chain(
        (1.0, 2.0, 3.0), FMT, punch_zoom_amount=0.2,
    )
    chain_every_other = effects.build_post_chain(
        (1.0, 2.0, 3.0), FMT, punch_zoom_amount=0.2, punch_zoom_every_nth=2,
    )
    assert chain_all.count("between(") == 3
    assert chain_every_other.count("between(") == 2


# --- rendered proof: glitch --------------------------------------------------

@pytest.fixture
def split_frame(tmp_path: Path) -> Path:
    """Left half black, right half white — a channel split is unmistakable."""
    path = tmp_path / "split.mp4"
    subprocess.run([
        ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=200x200:r=25:d=4",
        "-vf", "drawbox=x=100:y=0:w=100:h=200:color=white:t=fill",
        "-pix_fmt", "yuv420p", str(path),
    ], check=True, capture_output=True)
    return path


@requires_ffmpeg
def test_glitch_splits_channels_only_inside_its_window(
    split_frame: Path, tmp_path: Path
) -> None:
    chain = effects.build_post_chain(
        (1.0, 2.0, 3.0), FMT,
        glitch_shift_px=10, glitch_seconds=0.2,
        input_label="0:v",
    )
    out = _render_chain(split_frame, chain, tmp_path, "glitch")

    # A thin strip just left of the black/white boundary. With channels
    # aligned this reads a stable ~16 (h264 block noise off the hard edge,
    # not true black). `bh=-10` pulls the blue channel in from 10px further
    # right — across the boundary — only inside a glitch window, lifting luma
    # to a clearly separated ~41. Calibrated against a real render rather
    # than assumed, since the compression floor turned out to be nonzero.
    track = frame_metric_track(out, metrics=("YAVG",), crop="crop=2:2:94:99")["YAVG"]
    fps = FMT.fps

    def at(seconds: float) -> float:
        return track[round(seconds * fps)]

    assert at(0.5) < 25     # baseline: no glitch nearby
    assert at(1.0) > 30     # glitch window: channel split visible
    assert at(1.5) < 25
    assert at(2.0) > 30
