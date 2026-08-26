"""Speed ramp: a shot that opens slow and whips up to speed into the cut.

The rendered test encodes the *source* frame index as luma (`geq=lum=mod(N,256)`)
so reading the *output*'s luma back tells us exactly which source frame is on
screen at every output frame — proof of the actual playback rate, not just
that the graph parsed. This caught a real bug during development: without a
`fps=` re-snap immediately after the ramp's `concat`, the segment came out at
62 frames instead of the planned 60, because `setpts=PTS/factor` leaves
timestamps off the exact frame grid and the final render's own `-r fps` pads
extra frames to reconcile them — silently reintroducing the exact
"a fraction of a frame drifts at every cut" failure this codebase's whole
frame-exact design exists to prevent.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from conftest import frame_metric_track, requires_ffmpeg
from editing.speed_ramp import SpeedRamp, ramp_chain, source_seconds_needed, stage_frames
from ffmpeg_tools import ffmpeg_bin

FPS = 30.0


# --- pure math ---------------------------------------------------------------

def test_rejects_a_fraction_outside_zero_one() -> None:
    with pytest.raises(ValueError, match="fraction"):
        SpeedRamp(fraction=0.0)
    with pytest.raises(ValueError, match="fraction"):
        SpeedRamp(fraction=1.0)


def test_rejects_a_non_positive_speed_factor() -> None:
    with pytest.raises(ValueError, match="positive"):
        SpeedRamp(slow_factor=0.0)
    with pytest.raises(ValueError, match="positive"):
        SpeedRamp(fast_factor=-1.0)


def test_stage_frames_split_by_the_configured_fraction() -> None:
    n1, n2 = stage_frames(100, SpeedRamp(fraction=0.4))
    assert (n1, n2) == (40, 60)


def test_stage_frames_always_leave_at_least_one_frame_each_side() -> None:
    """A very short shot must not collapse one stage to zero frames."""
    n1, n2 = stage_frames(2, SpeedRamp(fraction=0.05))
    assert n1 >= 1 and n2 >= 1 and n1 + n2 == 2

    n1, n2 = stage_frames(2, SpeedRamp(fraction=0.95))
    assert n1 >= 1 and n2 >= 1 and n1 + n2 == 2


def test_source_seconds_needed_is_less_than_output_when_ramp_nets_slower() -> None:
    """fraction=0.45, slow=0.5, fast=1.3: weighted average speed < 1, so the
    ramp consumes less source than a flat, unramped shot of the same length."""
    ramp = SpeedRamp(fraction=0.45, slow_factor=0.5, fast_factor=1.3)
    needed = source_seconds_needed(90, FPS, ramp)
    assert needed < 90 / FPS


def test_source_seconds_needed_matches_the_weighted_sum() -> None:
    ramp = SpeedRamp(fraction=0.5, slow_factor=0.5, fast_factor=1.5)
    n1, n2 = stage_frames(60, ramp)
    expected = (n1 / FPS) * 0.5 + (n2 / FPS) * 1.5
    assert source_seconds_needed(60, FPS, ramp) == pytest.approx(expected)


def test_ramp_chain_ends_with_the_requested_output_label() -> None:
    lines = ramp_chain(60, FPS, SpeedRamp(), input_label="pre", output_label="v0")
    assert lines[-1].endswith("[v0]")
    assert lines[0].startswith("[pre]")


def test_ramp_chain_resnaps_to_the_frame_grid() -> None:
    """Regression: the fps= stage that fixed the 62-vs-60-frame bug."""
    lines = ramp_chain(60, FPS, SpeedRamp(), input_label="pre", output_label="v0")
    assert any(line.strip().startswith(f"[") and "fps=30" in line for line in lines)


# --- rendered proof ------------------------------------------------------

@pytest.fixture(scope="module")
def counter_source(tmp_path_factory) -> Path:
    """Luma = source frame index (mod 256): the output's luma track IS the
    apparent source-frame track."""
    path = tmp_path_factory.mktemp("ramp") / "counter.mp4"
    subprocess.run([
        ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s=64x64:r={FPS:g}:d=6",
        "-vf", "geq=lum='mod(N,256)':cb=128:cr=128",
        "-pix_fmt", "yuv420p", str(path),
    ], check=True, capture_output=True)
    return path


def _render_ramp(source: Path, frames: int, ramp: SpeedRamp, tmp_path: Path) -> Path:
    lines = [f"[0:v]setpts=PTS-STARTPTS[pre]"]
    lines += ramp_chain(frames, FPS, ramp, input_label="pre", output_label="ramped")
    lines.append(f"[ramped]trim=end_frame={frames},setpts=PTS-STARTPTS,format=yuv420p[outv]")
    graph_path = tmp_path / "graph.txt"
    graph_path.write_text(";\n".join(lines), encoding="utf-8")

    out_path = tmp_path / "ramped.mp4"
    subprocess.run([
        ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
        "-i", str(source), "-filter_complex_script", str(graph_path),
        "-map", "[outv]", "-r", f"{FPS:g}", "-pix_fmt", "yuv420p", str(out_path),
    ], check=True, capture_output=True)
    return out_path


@requires_ffmpeg
def test_ramped_segment_produces_exactly_the_planned_frame_count(
    counter_source: Path, tmp_path: Path
) -> None:
    frames = 60
    out = _render_ramp(counter_source, frames, SpeedRamp(), tmp_path)

    track = frame_metric_track(out, metrics=("YAVG",))["YAVG"]

    assert len(track) == frames


@requires_ffmpeg
def test_the_slow_stage_advances_source_frames_slower_than_the_fast_stage(
    counter_source: Path, tmp_path: Path
) -> None:
    ramp = SpeedRamp(fraction=0.5, slow_factor=0.5, fast_factor=1.5)
    frames = 60
    n1, _ = stage_frames(frames, ramp)
    out = _render_ramp(counter_source, frames, ramp, tmp_path)

    track = frame_metric_track(out, metrics=("YAVG",))["YAVG"]

    slow_advance = (track[n1 - 1] - track[0]) / (n1 - 1)
    fast_advance = (track[-1] - track[n1]) / (len(track) - 1 - n1)

    # Not pinned to the exact factor (frame-level quantisation adds noise),
    # but the ordering and rough magnitude must hold: slow stage advances
    # source frames at roughly half speed, fast stage at roughly 1.5x, and
    # critically, fast is unambiguously faster than slow.
    assert slow_advance < 0.75
    assert fast_advance > 1.2
    assert fast_advance > slow_advance * 2


@requires_ffmpeg
def test_a_flat_unramped_segment_advances_one_source_frame_per_output_frame(
    counter_source: Path, tmp_path: Path
) -> None:
    """Sanity anchor: SpeedRamp(1x, 1x) must look like no ramp at all."""
    ramp = SpeedRamp(fraction=0.5, slow_factor=1.0, fast_factor=1.0)
    frames = 40
    out = _render_ramp(counter_source, frames, ramp, tmp_path)

    track = frame_metric_track(out, metrics=("YAVG",))["YAVG"]

    advance = (track[-1] - track[0]) / (len(track) - 1)
    assert advance == pytest.approx(1.0, abs=0.15)
