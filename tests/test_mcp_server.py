"""The MCP tool surface — tested as plain functions, the same way
orchestrator_cli's build_argv is tested directly rather than through a full
HTTP round trip. Every tool in mcp_server.py is module-level and importable
specifically so this works without standing up an MCP session.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import mcp_server
from conftest import requires_ffmpeg


# --- list_clips --------------------------------------------------------------

@requires_ffmpeg
def test_list_clips_reports_real_metadata(sample_clips: list[Path]) -> None:
    result = mcp_server.list_clips(str(sample_clips[0].parent))

    assert result["count"] == len(sample_clips)
    assert result["total_duration_seconds"] > 0
    names = {Path(c["path"]).name for c in result["clips"]}
    assert names == {p.name for p in sample_clips}
    first = result["clips"][0]
    assert first["width"] > 0 and first["height"] > 0 and first["fps"] > 0


def test_list_clips_reports_an_error_not_an_exception(tmp_path: Path) -> None:
    result = mcp_server.list_clips(str(tmp_path / "nowhere"))

    assert "error" in result
    assert result["clips"] == []


# --- analyze_audio -------------------------------------------------------

@requires_ffmpeg
def test_analyze_audio_matches_the_known_click_track(click_track: Path) -> None:
    result = mcp_server.analyze_audio(str(click_track))

    assert result["bpm"] == pytest.approx(120.0, abs=2.0)
    assert result["beat_count"] > 0
    assert len(result["beats"]) == result["beat_count"]
    assert result["sections"]
    assert "summary" in result


def test_analyze_audio_reports_an_error_for_a_missing_file(tmp_path: Path) -> None:
    result = mcp_server.analyze_audio(str(tmp_path / "nope.wav"))

    assert "error" in result


# --- list_looks / list_transitions ----------------------------------------

def test_list_looks_includes_the_known_presets() -> None:
    looks = mcp_server.list_looks()

    assert "none" in looks
    assert "punchy" in looks
    assert "blackout" in looks


def test_list_transitions_includes_cut_and_xfade_types() -> None:
    transitions = mcp_server.list_transitions()

    assert "cut" in transitions
    assert "crossfade" in transitions
    assert len(transitions) > 20  # cut/crossfade plus ~58 xfade names


# --- plan_edit -------------------------------------------------------------

@requires_ffmpeg
def test_plan_edit_returns_a_full_segment_timeline(
    sample_clips: list[Path], click_track: Path
) -> None:
    result = mcp_server.plan_edit(
        clips=str(sample_clips[0].parent), audio=str(click_track),
        duration=8.0, seed=1, width=320, height=180, fps=30.0,
    )

    assert "error" not in result
    assert result["segments"]
    assert result["cut_count"] == len(result["segments"]) - 1
    first = result["segments"][0]
    assert first["timeline_start"] == pytest.approx(0.0)
    assert Path(first["clip"]).name  # a real clip path, not a placeholder
    # No overlaps or gaps: each segment starts where the previous one ends.
    for prev, cur in zip(result["segments"], result["segments"][1:]):
        assert cur["timeline_start"] == pytest.approx(
            prev["timeline_start"] + prev["duration"], abs=1e-3
        )


@requires_ffmpeg
def test_plan_edit_respects_beats_per_cut_overrides(
    sample_clips: list[Path], click_track: Path
) -> None:
    """A denser cut request should produce more, shorter segments."""
    sparse = mcp_server.plan_edit(
        clips=str(sample_clips[0].parent), audio=str(click_track),
        duration=8.0, seed=1,
        beats_per_cut_low=16, beats_per_cut_medium=16, beats_per_cut_high=16,
    )
    dense = mcp_server.plan_edit(
        clips=str(sample_clips[0].parent), audio=str(click_track),
        duration=8.0, seed=1,
        beats_per_cut_low=1, beats_per_cut_medium=1, beats_per_cut_high=1,
    )

    assert len(dense["segments"]) > len(sparse["segments"])


def test_plan_edit_reports_an_error_for_a_missing_clips_folder(
    tmp_path: Path, click_track: Path
) -> None:
    result = mcp_server.plan_edit(clips=str(tmp_path / "nowhere"), audio=str(click_track))

    assert "error" in result


# --- render_edit -----------------------------------------------------------

@requires_ffmpeg
def test_render_edit_produces_a_real_file(
    sample_clips: list[Path], click_track: Path, tmp_path: Path
) -> None:
    out_path = tmp_path / "out" / "mcp_render.mp4"
    result = mcp_server.render_edit(
        audio=str(click_track), out=str(out_path),
        clips=str(sample_clips[0].parent),
        duration=4.0, width=320, height=180, fps=30, seed=2,
    )

    assert result["status"] == "succeeded", result.get("log_tail")
    assert Path(result["out"]).is_file()
    assert "$0.00" in result["log_tail"]


def test_render_edit_requires_audio(tmp_path: Path) -> None:
    result = mcp_server.render_edit(audio="", out=str(tmp_path / "x.mp4"))

    assert result["status"] == "failed"
    assert "audio" in result["error"]


# --- server construction ----------------------------------------------------

def test_create_server_registers_every_tool() -> None:
    server = mcp_server.create_server()

    registered = {tool.name for tool in server._tool_manager.list_tools()}
    expected = {t.__name__ for t in mcp_server.TOOLS}
    assert registered == expected
