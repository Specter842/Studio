"""Colour grade presets: rendered and measured, not just parsed.

Each preset is checked against a real render of a textured test pattern —
`signalstats` gives real brightness/saturation numbers, so a preset that
claims to darken and desaturate has to actually do that, not just produce a
filter graph that parses.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from conftest import frame_metric_track, requires_ffmpeg
from editing.looks import LOOKS, NONE, custom_grade, look_filters, lut_filter
from ffmpeg_tools import ffmpeg_bin


def test_none_is_a_true_no_op() -> None:
    assert look_filters(NONE) == ""


def test_every_other_look_produces_a_nonempty_chain() -> None:
    for name in LOOKS:
        if name == NONE:
            continue
        assert look_filters(name), f"{name} produced an empty filter chain"


def test_an_unknown_look_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown look"):
        look_filters("cyberpunk")


@pytest.fixture(scope="module")
def bars(tmp_path_factory) -> Path:
    """SMPTE colour bars: a spread of hues and luma so grading is visible."""
    path = tmp_path_factory.mktemp("looks") / "bars.mp4"
    subprocess.run([
        ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
        "-f", "lavfi", "-i", "smptebars=s=320x180:r=24:d=1",
        "-pix_fmt", "yuv420p", str(path),
    ], check=True, capture_output=True)
    return path


def _render_look(name: str, source: Path, tmp_path: Path) -> Path:
    graph = f"[0:v]{look_filters(name)}[outv]"
    graph_path = tmp_path / f"graph_{name}.txt"
    graph_path.write_text(graph, encoding="utf-8")
    out_path = tmp_path / f"{name}.mp4"
    subprocess.run([
        ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
        "-i", str(source), "-filter_complex_script", str(graph_path),
        "-map", "[outv]", "-pix_fmt", "yuv420p", str(out_path),
    ], check=True, capture_output=True)
    return out_path


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


@requires_ffmpeg
def test_blackout_is_darker_and_less_saturated(bars: Path, tmp_path: Path) -> None:
    base = frame_metric_track(bars, metrics=("YAVG", "SATAVG"))
    out = _render_look("blackout", bars, tmp_path)
    graded = frame_metric_track(out, metrics=("YAVG", "SATAVG"))

    assert _mean(graded["YAVG"]) < _mean(base["YAVG"]) - 10
    assert _mean(graded["SATAVG"]) < _mean(base["SATAVG"]) - 10


@requires_ffmpeg
def test_punchy_is_brighter_and_more_saturated(bars: Path, tmp_path: Path) -> None:
    base = frame_metric_track(bars, metrics=("YAVG", "SATAVG"))
    out = _render_look("punchy", bars, tmp_path)
    graded = frame_metric_track(out, metrics=("YAVG", "SATAVG"))

    assert _mean(graded["YAVG"]) > _mean(base["YAVG"])
    assert _mean(graded["SATAVG"]) > _mean(base["SATAVG"])


@requires_ffmpeg
def test_bleach_is_strongly_desaturated(bars: Path, tmp_path: Path) -> None:
    base = frame_metric_track(bars, metrics=("SATAVG",))
    out = _render_look("bleach", bars, tmp_path)
    graded = frame_metric_track(out, metrics=("SATAVG",))

    assert _mean(graded["SATAVG"]) < _mean(base["SATAVG"]) - 20


@requires_ffmpeg
def test_teal_orange_raises_saturation_without_crushing_brightness(
    bars: Path, tmp_path: Path
) -> None:
    base = frame_metric_track(bars, metrics=("YAVG", "SATAVG"))
    out = _render_look("teal_orange", bars, tmp_path)
    graded = frame_metric_track(out, metrics=("YAVG", "SATAVG"))

    assert _mean(graded["SATAVG"]) > _mean(base["SATAVG"])
    # The grade should be visible without being a different exposure.
    assert abs(_mean(graded["YAVG"]) - _mean(base["YAVG"])) < 15


@requires_ffmpeg
def test_looks_differ_from_each_other(bars: Path, tmp_path: Path) -> None:
    """Two different presets should not collapse to the same numbers."""
    measured = {}
    for name in LOOKS:
        if name == NONE:
            continue
        out = _render_look(name, bars, tmp_path)
        track = frame_metric_track(out, metrics=("YAVG", "SATAVG"))
        measured[name] = (_mean(track["YAVG"]), _mean(track["SATAVG"]))

    values = list(measured.values())
    assert len(set(values)) == len(values), f"two looks produced identical stats: {measured}"


# --- LUTs -------------------------------------------------------------------

INVERT_CUBE = """LUT_3D_SIZE 2
1.0 1.0 1.0
0.0 1.0 1.0
1.0 0.0 1.0
0.0 0.0 1.0
1.0 1.0 0.0
0.0 1.0 0.0
1.0 0.0 0.0
0.0 0.0 0.0
"""


def test_lut_filter_names_the_file_and_a_default_interp() -> None:
    chain = lut_filter("look.cube")
    assert "lut3d=file=" in chain
    assert "interp=tetrahedral" in chain


def test_lut_filter_escapes_a_windows_drive_letter_path() -> None:
    """A raw `C:\\...` path collides with ffmpeg's own `:` option separator —
    an unescaped colon makes the drive letter parse as a bogus filter option,
    not the LUT file. Regression-worthy: it fails to *build*, not to look
    right, so nothing short of actually parsing the graph catches it."""
    chain = lut_filter(r"C:\Users\me\looks\invert.cube")
    assert r"C\:" in chain
    assert "\\\\" in chain  # backslashes escaped too


@requires_ffmpeg
def test_a_real_lut_measurably_changes_the_image(tmp_path: Path) -> None:
    """An invert LUT on white input should read back near-black."""
    cube_path = tmp_path / "invert.cube"
    cube_path.write_text(INVERT_CUBE, encoding="utf-8")

    src = tmp_path / "white.mp4"
    subprocess.run([
        ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=white:s=64x64:r=10:d=1",
        "-pix_fmt", "yuv420p", str(src),
    ], check=True, capture_output=True)

    out = tmp_path / "inverted.mp4"
    subprocess.run([
        ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
        "-i", str(src), "-vf", lut_filter(str(cube_path)),
        "-pix_fmt", "yuv420p", str(out),
    ], check=True, capture_output=True)

    track = frame_metric_track(out, metrics=("YAVG",))["YAVG"]
    assert track[0] < 30, f"white was not inverted to near-black: {track[0]}"


# --- custom_grade (general 3-way corrector) ---------------------------------

def test_custom_grade_with_all_defaults_is_a_true_no_op() -> None:
    assert custom_grade() == ""


def test_custom_grade_only_emits_colorbalance_when_a_wheel_moved() -> None:
    chain = custom_grade(gain=(0.2, 0.0, 0.0))
    assert "colorbalance=" in chain
    assert "rh=0.2" in chain
    assert "eq=" not in chain  # saturation/contrast/brightness untouched


def test_custom_grade_only_emits_eq_when_tone_moved() -> None:
    chain = custom_grade(saturation=1.5)
    assert "eq=" in chain
    assert "colorbalance=" not in chain


@requires_ffmpeg
def test_custom_grade_lift_darkens_shadows_measurably(bars: Path, tmp_path: Path) -> None:
    base = frame_metric_track(bars, metrics=("YAVG",))
    chain = custom_grade(lift=(-0.3, -0.3, -0.3))
    graph_path = tmp_path / "custom_graph.txt"
    graph_path.write_text(f"[0:v]{chain}[outv]", encoding="utf-8")
    out_path = tmp_path / "custom.mp4"
    subprocess.run([
        ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
        "-i", str(bars), "-filter_complex_script", str(graph_path),
        "-map", "[outv]", "-pix_fmt", "yuv420p", str(out_path),
    ], check=True, capture_output=True)

    graded = frame_metric_track(out_path, metrics=("YAVG",))
    assert _mean(graded["YAVG"]) < _mean(base["YAVG"])
