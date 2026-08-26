"""Vignette, sharpen/soften, denoise — the Resolve-finishing-page category.

All three are constant-parameter filters (no per-frame expression involved),
which is the one class of thing repeatedly confirmed unreliable elsewhere in
this codebase — so these get one rendered proof each rather than the heavier
scrutiny given to punch-zoom/speed-ramp/light-leak, plus the cheap zero-value
no-op checks every effect module in this package carries.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from conftest import frame_metric_track, requires_ffmpeg
from editing import finishing
from ffmpeg_tools import ffmpeg_bin


def test_vignette_zero_is_a_true_no_op() -> None:
    assert finishing.vignette_filter(0) == ""


def test_sharpen_zero_is_a_true_no_op() -> None:
    assert finishing.sharpen_filter(0) == ""


def test_denoise_zero_is_a_true_no_op() -> None:
    assert finishing.denoise_filter(0) == ""


def test_sharpen_accepts_negative_amounts_to_soften() -> None:
    assert "luma_amount=-1" in finishing.sharpen_filter(-1)


def test_denoise_chroma_strength_is_half_luma() -> None:
    chain = finishing.denoise_filter(6)
    assert chain == "hqdn3d=6:3:6:3"


@pytest.fixture
def flat_white(tmp_path: Path) -> Path:
    path = tmp_path / "flat.mp4"
    subprocess.run([
        ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=white:s=200x200:r=10:d=1",
        "-pix_fmt", "yuv420p", str(path),
    ], check=True, capture_output=True)
    return path


def _render(source: Path, vf: str, tmp_path: Path, name: str) -> Path:
    out = tmp_path / f"{name}.mp4"
    subprocess.run([
        ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
        "-i", str(source), "-vf", vf, "-pix_fmt", "yuv420p", str(out),
    ], check=True, capture_output=True, timeout=20)
    return out


@requires_ffmpeg
def test_vignette_darkens_the_corners_relative_to_the_centre(
    flat_white: Path, tmp_path: Path
) -> None:
    out = _render(flat_white, finishing.vignette_filter(0.9), tmp_path, "vig")

    center = frame_metric_track(out, metrics=("YAVG",), crop="crop=4:4:98:98")["YAVG"][0]
    corner = frame_metric_track(out, metrics=("YAVG",), crop="crop=4:4:0:0")["YAVG"][0]

    assert corner < center - 30


@requires_ffmpeg
def test_a_stronger_vignette_darkens_the_corner_more(
    flat_white: Path, tmp_path: Path
) -> None:
    mild = _render(flat_white, finishing.vignette_filter(0.2), tmp_path, "mild")
    heavy = _render(flat_white, finishing.vignette_filter(0.95), tmp_path, "heavy")

    mild_corner = frame_metric_track(mild, metrics=("YAVG",), crop="crop=4:4:0:0")["YAVG"][0]
    heavy_corner = frame_metric_track(heavy, metrics=("YAVG",), crop="crop=4:4:0:0")["YAVG"][0]

    assert heavy_corner < mild_corner


@requires_ffmpeg
def test_sharpen_and_denoise_render_without_error(
    flat_white: Path, tmp_path: Path
) -> None:
    """Lighter proof than most: these are well-worn, constant-parameter
    ffmpeg filters, not the surprising-per-frame-expression territory found
    elsewhere. What matters is that this module's own string construction
    produces syntax ffmpeg actually accepts."""
    _render(flat_white, finishing.sharpen_filter(2.0), tmp_path, "sharp")
    _render(flat_white, finishing.sharpen_filter(-1.0), tmp_path, "soft")
    _render(flat_white, finishing.denoise_filter(4.0), tmp_path, "denoise")
