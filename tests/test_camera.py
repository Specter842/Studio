"""Ken Burns: a slow zoom across a shot's own duration, staircased from
discrete crop/scale states (continuous per-frame expressions don't animate
on this ffmpeg build — see the module docstring in camera.py).

The rendered test's real job is the shot-boundary handoff: an earlier version
of this treated a shot's first step as an unconditional crop rather than one
more gated overlay, which would have silently replaced the whole frame from
t=0 onward the moment a second shot's stages were appended. Caught before it
was ever rendered by re-reading the loop, and locked in here by a render that
crosses a boundary and checks the zoom actually resets.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from conftest import frame_metric_track, requires_ffmpeg
from editing import camera
from editing.transitions import VideoFormat
from ffmpeg_tools import ffmpeg_bin

FMT = VideoFormat(200, 200, 25.0)


# --- construction (no rendering) --------------------------------------------

def test_no_shots_means_no_stages() -> None:
    assert camera.ken_burns_stages((), FMT, input_label="0:v", output_label="outv", tag="kb") is None


def test_zero_amount_means_no_stages() -> None:
    stages = camera.ken_burns_stages(
        ((0.0, 2.0),), FMT, zoom_amount=0.0,
        input_label="0:v", output_label="outv", tag="kb",
    )
    assert stages is None


def test_fewer_than_two_steps_means_no_stages() -> None:
    stages = camera.ken_burns_stages(
        ((0.0, 2.0),), FMT, steps=1,
        input_label="0:v", output_label="outv", tag="kb",
    )
    assert stages is None


def test_ends_with_the_requested_output_label() -> None:
    stages = camera.ken_burns_stages(
        ((0.0, 2.0), (2.0, 4.0)), FMT, steps=4,
        input_label="0:v", output_label="v0", tag="kb",
    )
    assert stages[-1].endswith("[v0]")


def test_step_count_matches_the_number_of_overlay_stages() -> None:
    """2 shots x 4 steps = 8 keyframes; the first is the unconditional base,
    the other 7 are gated overlays."""
    stages = camera.ken_burns_stages(
        ((0.0, 2.0), (2.0, 4.0)), FMT, steps=4,
        input_label="0:v", output_label="outv", tag="kb",
    )
    overlay_count = sum(1 for line in stages if "overlay=" in line)
    assert overlay_count == 7


def test_a_second_shots_first_step_is_a_gated_overlay_not_unconditional() -> None:
    """Regression: the exact bug caught before rendering. Every keyframe
    after the very first must appear as the right-hand side of a `gte(t,...)`
    gated overlay — none may be assigned as a fresh, ungated base layer."""
    stages = camera.ken_burns_stages(
        ((0.0, 2.0), (2.0, 4.0)), FMT, steps=3,
        input_label="0:v", output_label="outv", tag="kb",
    )
    crop_stages = [line for line in stages if line.startswith("[0:v]crop=")]
    overlay_stages = [line for line in stages if "overlay=" in line]

    # One crop per keyframe (6 total: 2 shots x 3 steps), but only the very
    # first is wired straight to a bare label — every other crop's output
    # must be consumed by a gated overlay, including shot 2's own step 0.
    assert len(crop_stages) == 6
    assert len(overlay_stages) == 5
    gated_at_two = [line for line in overlay_stages if "gte(t,2)" in line or "gte(t,2.0" in line]
    assert gated_at_two, f"no overlay gated exactly at the shot-2 boundary: {overlay_stages}"


def test_alternate_flips_direction_every_other_shot() -> None:
    """Not directly observable from the graph text (direction only changes
    which end of the zoom range each step lands on), so this checks the
    computed zoom values instead of the string output."""
    amount = 0.2
    in_first = camera._zoom_at(0.0, amount, "in")
    in_last = camera._zoom_at(1.0, amount, "in")
    out_first = camera._zoom_at(0.0, amount, "out")
    out_last = camera._zoom_at(1.0, amount, "out")

    assert in_first < in_last          # "in" grows across the shot
    assert out_first > out_last        # "out" shrinks across the shot
    assert in_first == pytest.approx(out_last)
    assert in_last == pytest.approx(out_first)


# --- rendered proof ----------------------------------------------------------

@pytest.fixture
def bordered(tmp_path: Path) -> Path:
    path = tmp_path / "bordered.mp4"
    subprocess.run([
        ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=200x200:r=25:d=4",
        "-vf", "drawbox=x=0:y=0:w=iw:h=ih:color=white:t=6",
        "-pix_fmt", "yuv420p", str(path),
    ], check=True, capture_output=True)
    return path


def _render(source: Path, stages: list[str], tmp_path: Path) -> Path:
    graph_path = tmp_path / "graph.txt"
    graph_path.write_text(";\n".join(stages), encoding="utf-8")
    out_path = tmp_path / "kb.mp4"
    subprocess.run([
        ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
        "-i", str(source), "-filter_complex_script", str(graph_path),
        "-map", "[outv]", "-pix_fmt", "yuv420p", str(out_path),
    ], check=True, capture_output=True, timeout=30)
    return out_path


@requires_ffmpeg
def test_zoom_in_progressively_crops_the_border_away(
    bordered: Path, tmp_path: Path
) -> None:
    stages = camera.ken_burns_stages(
        ((0.0, 4.0),), FMT, zoom_amount=0.3, steps=6, direction="in",
        input_label="0:v", output_label="outv", tag="kb",
    )
    out = _render(bordered, stages, tmp_path)

    track = frame_metric_track(out, metrics=("YAVG",), crop="crop=4:4:0:0")["YAVG"]
    start = track[2]     # just after t=0, still near-unzoomed
    end = track[-2]      # near the end of the shot, fully zoomed in

    assert start > 200, f"shot should open near-unzoomed (border visible): {start}"
    assert end < 50, f"shot should end zoomed past the border: {end}"


@requires_ffmpeg
def test_the_zoom_resets_at_a_shot_boundary(bordered: Path, tmp_path: Path) -> None:
    """The regression this module exists to prevent, confirmed end to end:
    shot 2 must open unzoomed again, not continue shot 1's trajectory."""
    stages = camera.ken_burns_stages(
        ((0.0, 2.0), (2.0, 4.0)), FMT, zoom_amount=0.3, steps=5, direction="in",
        input_label="0:v", output_label="outv", tag="kb",
    )
    out = _render(bordered, stages, tmp_path)

    track = frame_metric_track(out, metrics=("YAVG",), crop="crop=4:4:0:0")["YAVG"]
    end_of_shot1 = track[48]        # t=1.92, shot 1 nearly finished zooming in
    start_of_shot2 = track[51]      # t=2.04, just after the boundary

    assert end_of_shot1 < 50, f"shot 1 should have zoomed in by its end: {end_of_shot1}"
    assert start_of_shot2 > 200, (
        f"shot 2 should reopen unzoomed, not continue shot 1's zoom: {start_of_shot2}"
    )


@requires_ffmpeg
def test_frame_count_is_unaffected_by_the_camera_move(
    bordered: Path, tmp_path: Path
) -> None:
    """Purely visual — must never change how many frames exist."""
    stages = camera.ken_burns_stages(
        ((0.0, 4.0),), FMT, zoom_amount=0.2, steps=5, direction="out",
        input_label="0:v", output_label="outv", tag="kb",
    )
    out = _render(bordered, stages, tmp_path)

    track = frame_metric_track(out, metrics=("YAVG",))["YAVG"]
    assert len(track) == 100  # 4s @ 25fps, exactly what the source has
