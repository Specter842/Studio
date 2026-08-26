"""Headless Blender and the compositor that lays its output over the edit.

The Blender tests render for real when Blender is installed and skip cleanly
when it is not — 3D elements are an optional extra, and the suite must stay
green on a machine without them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from animation3d import blender
from animation3d.blender import BlenderNotFound, Element
from conftest import requires_ffmpeg
from editing import compositor
from editing.compositor import Overlay
from ffmpeg_tools import probe_json

requires_blender = pytest.mark.skipif(
    not blender.have_blender(),
    reason="Blender not installed (optional; set BLENDER_BIN in .env to point at it)",
)


# --- discovery ------------------------------------------------------------

@requires_blender
def test_blender_is_found_and_reports_a_version() -> None:
    assert Path(blender.blender_bin()).is_file()
    assert "blender" in blender.blender_version().lower()


def test_an_explicit_bad_path_is_rejected(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLENDER_BIN", str(tmp_path / "not-here"))

    with pytest.raises(BlenderNotFound, match="not a file"):
        blender.blender_bin()


def test_a_missing_blender_explains_that_it_is_optional(monkeypatch) -> None:
    monkeypatch.delenv("BLENDER_BIN", raising=False)
    monkeypatch.setattr(blender.shutil, "which", lambda _name: None)
    monkeypatch.setattr(blender, "_SEARCH_DIRS", ())

    with pytest.raises(BlenderNotFound, match="optional"):
        blender.blender_bin()


@pytest.mark.parametrize("name,expected", [
    ("Blender 4.5", (4, 5)),
    ("Blender 3.6 LTS", (3, 6)),
    ("blender", (0,)),
])
def test_install_directories_sort_by_version(name, expected) -> None:
    """So an old 2.x left behind cannot shadow a current install."""
    assert blender._version_key(name) == expected


# --- rendering ------------------------------------------------------------

@requires_blender
def test_render_produces_an_rgba_sequence(tmp_path: Path) -> None:
    element = blender.render_text_reveal(
        "HELLO", cache_dir=tmp_path, width=320, height=180, fps=12.0, seconds=0.5
    )

    assert element.frames == 6
    assert element.duration == pytest.approx(0.5)
    assert len(list(element.directory.glob("frame_*.png"))) == 6
    assert "%04d" in element.input_path


@requires_blender
@requires_ffmpeg
def test_the_background_is_actually_transparent(tmp_path: Path) -> None:
    """Without alpha the element would paint a black box over the video."""
    element = blender.render_text_reveal(
        "HI", cache_dir=tmp_path, width=160, height=90, fps=8.0, seconds=0.25
    )
    frame = sorted(element.directory.glob("frame_*.png"))[-1]

    probe = probe_json(frame)["streams"][0]

    assert "a" in probe["pix_fmt"], f"no alpha channel: {probe['pix_fmt']}"


@requires_blender
def test_an_identical_element_is_not_rendered_twice(tmp_path: Path) -> None:
    """A Blender render costs real seconds; the same title must not pay twice."""
    calls: list[str] = []
    original = blender.run_script

    def counting(script, arguments, **kwargs):
        calls.append(str(script))
        return original(script, arguments, **kwargs)

    blender.run_script, previous = counting, blender.run_script
    try:
        first = blender.render_text_reveal(
            "CACHED", cache_dir=tmp_path, width=160, height=90, fps=8.0, seconds=0.25
        )
        second = blender.render_text_reveal(
            "CACHED", cache_dir=tmp_path, width=160, height=90, fps=8.0, seconds=0.25
        )
    finally:
        blender.run_script = previous

    assert first.directory == second.directory
    assert len(calls) == 1


@requires_blender
def test_changing_the_script_version_invalidates_the_cache(
    tmp_path: Path, monkeypatch
) -> None:
    first = blender.render_text_reveal(
        "V", cache_dir=tmp_path, width=160, height=90, fps=8.0, seconds=0.25
    )
    monkeypatch.setattr(blender, "TEXT_REVEAL_VERSION", 999)
    second = blender.render_text_reveal(
        "V", cache_dir=tmp_path, width=160, height=90, fps=8.0, seconds=0.25
    )

    assert first.directory != second.directory


@requires_blender
def test_output_paths_handed_to_blender_are_absolute(tmp_path: Path, monkeypatch):
    """Blender resolves relative render paths against a .blend that does not
    exist here, and silently writes nothing."""
    captured: dict = {}
    monkeypatch.setattr(
        blender, "run_script",
        lambda script, arguments, **kw: captured.update(arguments) or "",
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(blender.BlenderError):  # no frames, since nothing ran
        blender.render_text_reveal("X", cache_dir=Path("relative/cache"), seconds=0.1)

    assert Path(captured["output_prefix"]).is_absolute()


# --- compositing ----------------------------------------------------------

def make_overlay(**overrides) -> Overlay:
    defaults = dict(
        input_path="frames/frame_%04d.png", fps=30.0, start=0.0, duration=2.0
    )
    return Overlay(**{**defaults, **overrides})


def test_the_graph_shifts_an_overlay_to_its_start_time() -> None:
    graph = compositor._build_graph([make_overlay(start=4.5)])

    assert "setpts=PTS+4.500000/TB" in graph


def test_an_overlay_at_zero_needs_no_shift() -> None:
    graph = compositor._build_graph([make_overlay(start=0.0)])

    assert "PTS+" not in graph


def test_the_element_stops_instead_of_freezing_on_screen() -> None:
    """overlay's default repeatlast would leave the last frame up forever."""
    graph = compositor._build_graph([make_overlay()])

    assert "repeatlast=0" in graph
    assert "eof_action=pass" in graph


def test_fades_act_on_alpha_not_toward_black() -> None:
    graph = compositor._build_graph([make_overlay(fade=0.4)])

    assert "fade=t=in:st=0:d=0.400:alpha=1" in graph
    assert "fade=t=out:st=1.600:d=0.400:alpha=1" in graph


def test_a_fade_cannot_exceed_half_the_element() -> None:
    graph = compositor._build_graph([make_overlay(duration=0.5, fade=10.0)])

    assert "d=0.250:alpha=1" in graph


def test_multiple_overlays_chain_onto_one_output() -> None:
    graph = compositor._build_graph([make_overlay(), make_overlay(start=5.0)])

    assert graph.count("overlay=") == 2
    assert "[b1]" in graph
    assert graph.rstrip().endswith("[v]")


@pytest.mark.parametrize("position", ["center", "top", "bottom"])
def test_every_position_produces_a_y_expression(position) -> None:
    graph = compositor._build_graph([make_overlay(position=position)])

    assert f"y={compositor.POSITIONS[position]}" in graph


def test_composite_requires_something_to_composite(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one overlay"):
        compositor.composite(tmp_path / "a.mp4", [], tmp_path / "b.mp4")


@requires_blender
@requires_ffmpeg
def test_an_element_ends_up_in_the_finished_video(
    sample_clips: list[Path], tmp_path: Path
) -> None:
    """The Phase 3 claim: 3D elements appear in the output when requested.

    Checked by pixels, not by exit code — the title is rendered bright over a
    dark base, so the frame it covers must get measurably brighter while the
    element is on screen and return to normal after it ends.
    """
    import subprocess

    from ffmpeg_tools import ffmpeg_bin

    base = tmp_path / "base.mp4"
    subprocess.run([
        ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=320x180:r=24:d=4",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(base),
    ], check=True, capture_output=True)

    element = blender.render_text_reveal(
        "TITLE", cache_dir=tmp_path / "elements",
        width=320, height=180, fps=24.0, seconds=1.0,
    )
    out_path = tmp_path / "titled.mp4"
    compositor.composite(
        base,
        [Overlay(element.input_path, element.fps, start=0.5,
                 duration=element.duration, fade=0.1)],
        out_path,
        encode={"preset": "ultrafast"},
    )

    assert out_path.is_file()

    def brightness(at: float) -> float:
        raw = subprocess.run([
            ffmpeg_bin(), "-hide_banner", "-v", "error",
            "-ss", f"{at}", "-i", str(out_path), "-frames:v", "1",
            "-vf", "scale=1:1:flags=area", "-f", "rawvideo",
            "-pix_fmt", "gray", "-",
        ], check=True, capture_output=True).stdout
        return raw[0] if raw else 0.0

    during = brightness(1.0)
    after = brightness(3.5)

    assert during > after + 2, (
        f"title not visible: brightness during={during}, after={after}"
    )
    assert after <= 2, f"element did not clear after it ended (brightness {after})"


def test_element_duration_comes_from_frames_and_fps() -> None:
    element = Element(Path("."), fps=25.0, frames=50, width=1920, height=1080)

    assert element.duration == pytest.approx(2.0)
