"""Grain, light leaks, temporal blend (motion blur/ghosting), chroma key.

Two real hangs were found building this module, both the same shape: a
generated `color=` source with no `d=` runs forever by design, and both
`overlay` and `blend` default `shortest` to false regardless of which input
is actually finite — so a graph that looks obviously bounded (one real,
finite input; one infinite generated one) hung ffmpeg indefinitely until
`shortest=1` was added explicitly. Every render call below runs under a hard
subprocess timeout so a regression fails the test instead of hanging CI.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from conftest import frame_metric_track, requires_ffmpeg
from editing import compositing
from editing.transitions import VideoFormat
from ffmpeg_tools import ffmpeg_bin

FMT = VideoFormat(100, 100, 10.0)
RENDER_TIMEOUT = 20.0


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, capture_output=True, timeout=RENDER_TIMEOUT)


# --- grain -------------------------------------------------------------------

def test_grain_zero_strength_is_a_true_no_op() -> None:
    assert compositing.grain_filter(0) == ""


def test_grain_names_the_strength() -> None:
    assert "alls=15" in compositing.grain_filter(15)


def test_grain_strength_is_clamped_to_the_valid_range() -> None:
    assert "alls=100" in compositing.grain_filter(500)
    assert "alls=0" not in compositing.grain_filter(500)


@requires_ffmpeg
def test_grain_measurably_adds_noise(tmp_path: Path) -> None:
    src = tmp_path / "flat.mp4"
    _run([
        ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=gray:s=64x64:r=10:d=1",
        "-pix_fmt", "yuv420p", str(src),
    ])
    out = tmp_path / "grainy.mp4"
    _run([
        ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
        "-i", str(src), "-vf", compositing.grain_filter(30),
        "-pix_fmt", "yuv420p", str(out),
    ])

    # A flat grey field has no per-frame variation on its own; grain's
    # `allf=t+u` (temporal + per-pixel random) must make frames differ from
    # each other even though the source never changes.
    track = frame_metric_track(out, metrics=("YAVG",))["YAVG"]
    assert len(set(round(v, 1) for v in track)) > 1, "grain produced identical frames"


# --- temporal blend (motion blur / ghosting) ---------------------------------

def test_temporal_blend_of_one_frame_is_a_true_no_op() -> None:
    assert compositing.temporal_blend_filter(1) == ""


def test_temporal_blend_names_the_window_and_equal_weights() -> None:
    chain = compositing.temporal_blend_filter(5)
    assert "frames=5" in chain
    assert chain.count("1") == 5  # five equal weights


@requires_ffmpeg
def test_temporal_blend_matches_the_arithmetic_mean_of_the_window(
    tmp_path: Path,
) -> None:
    """counter.mp4: luma = source frame index. A 5-frame tmix's output at
    frame i must equal the mean of raw frames [i-4..i] exactly."""
    src = tmp_path / "counter.mp4"
    _run([
        ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=64x64:r=25:d=1",
        "-vf", "geq=lum='mod(N,256)':cb=128:cr=128",
        "-pix_fmt", "yuv420p", str(src),
    ])
    out = tmp_path / "blended.mp4"
    _run([
        ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
        "-i", str(src), "-vf", compositing.temporal_blend_filter(5),
        "-pix_fmt", "yuv420p", str(out),
    ])

    raw = frame_metric_track(src, metrics=("YAVG",))["YAVG"]
    mixed = frame_metric_track(out, metrics=("YAVG",))["YAVG"]

    for i in (4, 10, 15, 20):
        expected = sum(raw[i - 4:i + 1]) / 5
        assert mixed[i] == pytest.approx(expected, abs=0.1)


# --- light leak ----------------------------------------------------------

def test_light_leak_stages_end_with_the_requested_output_label() -> None:
    lines = compositing.light_leak_stages(
        FMT, input_label="pre", output_label="v0", tag="lk"
    )
    assert lines[-1].endswith("[v0]")


def test_light_leak_geq_uses_uppercase_frame_dimension_variables() -> None:
    """Regression: geq rejects lowercase w/h ('Undefined constant'), unlike
    scale/crop which accept lowercase. Uppercase W/H only."""
    lines = compositing.light_leak_stages(
        FMT, input_label="pre", output_label="v0", tag="lk"
    )
    geq_line = next(line for line in lines if "geq=" in line)
    assert "W*" in geq_line or "(W" in geq_line
    assert "w*" not in geq_line and "(w" not in geq_line


def test_light_leak_blend_is_shortest_gated() -> None:
    """Regression: the generated leak source has no duration and runs
    forever by design; without shortest=1 on blend, rendering never ends."""
    lines = compositing.light_leak_stages(
        FMT, input_label="pre", output_label="v0", tag="lk"
    )
    blend_line = next(line for line in lines if "blend=" in line)
    assert "shortest=1" in blend_line


@requires_ffmpeg
def test_light_leak_is_brightest_near_its_declared_center(tmp_path: Path) -> None:
    lines = compositing.light_leak_stages(
        FMT, input_label="0:v", output_label="outv", tag="lk",
        center=(0.85, 0.15), radius_frac=0.5, opacity=0.6,
    )
    graph_path = tmp_path / "graph.txt"
    graph_path.write_text(";\n".join(lines), encoding="utf-8")
    out_path = tmp_path / "leak.mp4"
    _run([
        ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=gray:s=100x100:r=10:d=1",
        "-filter_complex_script", str(graph_path),
        "-map", "[outv]", "-pix_fmt", "yuv420p", str(out_path),
    ])

    near = frame_metric_track(out_path, metrics=("YAVG",), crop="crop=2:2:85:15")["YAVG"][0]
    far = frame_metric_track(out_path, metrics=("YAVG",), crop="crop=2:2:5:85")["YAVG"][0]
    assert near > far


# --- chroma key ------------------------------------------------------------

@pytest.fixture
def green_box(tmp_path: Path) -> Path:
    path = tmp_path / "greenbox.mp4"
    _run([
        ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=100x100:r=10:d=1",
        "-vf", "drawbox=x=25:y=25:w=50:h=50:color=0x00FF00:t=fill",
        "-pix_fmt", "yuv420p", str(path),
    ])
    return path


def test_chroma_key_with_a_background_ends_opaque_at_the_requested_label() -> None:
    lines = compositing.chroma_key_stages(
        FMT, input_label="pre", output_label="v0", tag="ck", background="0xFF0000",
    )
    assert lines[-1].endswith("[v0]")
    assert any("overlay=" in line for line in lines)


def test_chroma_key_without_a_background_stays_a_single_stage() -> None:
    lines = compositing.chroma_key_stages(
        FMT, input_label="pre", output_label="v0", tag="ck", background=None,
    )
    assert len(lines) == 1
    assert lines[0].endswith("[v0]")
    assert "overlay=" not in lines[0]


def test_chroma_key_overlay_is_shortest_gated() -> None:
    """Regression: same infinite-`color=`-background hang as the light leak."""
    lines = compositing.chroma_key_stages(
        FMT, input_label="pre", output_label="v0", tag="ck", background="0xFF0000",
    )
    overlay_line = next(line for line in lines if "overlay=" in line)
    assert "shortest=1" in overlay_line


@requires_ffmpeg
def test_chroma_keyed_region_reveals_the_background_colour(
    green_box: Path, tmp_path: Path
) -> None:
    lines = ["[0:v]null[pre]"] + compositing.chroma_key_stages(
        FMT, input_label="pre", output_label="outv", tag="ck",
        key_color="0x00FF00", background="0xFF0000",
    )
    graph_path = tmp_path / "graph.txt"
    graph_path.write_text(";\n".join(lines), encoding="utf-8")
    out_path = tmp_path / "keyed.mp4"
    _run([
        ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
        "-i", str(green_box), "-filter_complex_script", str(graph_path),
        "-map", "[outv]", "-pix_fmt", "yuv420p", str(out_path),
    ])

    corner = frame_metric_track(out_path, metrics=("YAVG",), crop="crop=2:2:0:0")["YAVG"][0]
    center = frame_metric_track(out_path, metrics=("YAVG",), crop="crop=2:2:49:49")["YAVG"][0]

    # Corner was black (unkeyed, opaque) and stays dark; centre was the keyed
    # green box, now showing red background through — a clearly different,
    # higher luma than the still-black corner.
    assert center > corner + 20
