"""Combining local, stock and generated clips into one pool.

The behaviour under test is mostly about degradation: a missing key or an
absent ComfyUI must cost you those clips and nothing else. Losing a whole
render because one of three sources was unavailable is the failure mode worth
designing against.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import sourcing
from budget import Budget
from conftest import make_settings, requires_ffmpeg
from editing.transitions import VideoFormat
from ingest.local_clips import NoClipsFound
from test_stock_fetch import stock_server  # noqa: F401 - reused fixture

FORMAT = VideoFormat(1280, 720, 30.0)


def gather(tmp_path: Path, *, local_dir=None, brief="", budget=None, **kwargs):
    return sourcing.gather_clips(
        local_dir=local_dir,
        brief=brief,
        settings=make_settings(**kwargs.pop("settings", {})),
        budget=budget or Budget(),
        cache_dir=tmp_path / "cache",
        video_format=FORMAT,
        **kwargs,
    )


@requires_ffmpeg
def test_local_only_is_still_a_valid_run(sample_clips: list[Path], tmp_path: Path):
    result = gather(tmp_path, local_dir=sample_clips[0].parent)

    assert len(result.clips) == len(sample_clips)
    assert result.counts == {"local": len(sample_clips)}
    assert result.warnings == []


@requires_ffmpeg
def test_stock_clips_join_local_ones(
    stock_server, sample_clips: list[Path], tmp_path: Path  # noqa: F811
):
    result = gather(
        tmp_path,
        local_dir=sample_clips[0].parent,
        brief="a rainy city",
        stock_per_query=2,
    )

    assert result.counts["stock"] == 2
    assert result.counts.get("local", 0) >= 1
    assert all(clip.is_usable for clip in result.clips)


@requires_ffmpeg
def test_a_brief_alone_can_supply_the_whole_pool(
    stock_server, tmp_path: Path  # noqa: F811
):
    result = gather(tmp_path, local_dir=None, brief="a rainy city", stock_per_query=2)

    assert result.counts == {"stock": 2}


@requires_ffmpeg
def test_multiple_queries_are_searched_separately(
    stock_server, tmp_path: Path  # noqa: F811
):
    gather(
        tmp_path, local_dir=None,
        brief="city at night; neon signs", stock_per_query=1,
    )

    searches = [r for r in stock_server.requests if r.path == "/pexels/search"]
    assert {r.param("query") for r in searches} == {"city at night", "neon signs"}


@requires_ffmpeg
def test_missing_stock_keys_warn_loudly_and_keep_the_local_clips(
    sample_clips: list[Path], tmp_path: Path, monkeypatch
):
    # setenv("") not delenv: an internal config.load_env() reload would
    # silently repopulate a merely-*unset* var from a real .env (see the
    # longer note in test_pipeline_e2e.py). Empty stays empty.
    monkeypatch.setenv("PEXELS_API_KEY", "")
    monkeypatch.setenv("PIXABAY_API_KEY", "")

    result = gather(
        tmp_path, local_dir=sample_clips[0].parent,
        brief="a rainy city", stock_per_query=2,
    )

    assert "stock" not in result.counts
    assert result.counts.get("local", 0) >= 1
    assert any("PEXELS_API_KEY" in warning for warning in result.warnings)


def test_no_clips_from_anywhere_is_an_error(tmp_path: Path, monkeypatch):
    # setenv("") not delenv: an internal config.load_env() reload would
    # silently repopulate a merely-*unset* var from a real .env (see the
    # longer note in test_pipeline_e2e.py). Empty stays empty.
    monkeypatch.setenv("PEXELS_API_KEY", "")
    monkeypatch.setenv("PIXABAY_API_KEY", "")

    with pytest.raises(NoClipsFound) as excinfo:
        gather(tmp_path, local_dir=tmp_path / "empty", brief="anything")

    # The error has to say what went wrong with each source, not just "none".
    assert "PEXELS_API_KEY" in str(excinfo.value)


@requires_ffmpeg
def test_generation_is_off_unless_asked_for(
    stock_server, sample_clips: list[Path], tmp_path: Path  # noqa: F811
):
    """A brief must not silently start a GPU job that takes minutes."""
    result = gather(
        tmp_path, local_dir=sample_clips[0].parent,
        brief="a rainy city", stock_per_query=1,
    )

    assert "local_comfyui" not in result.counts


@requires_ffmpeg
def test_an_unreachable_comfyui_costs_only_the_generated_clips(
    stock_server, sample_clips: list[Path], tmp_path: Path  # noqa: F811
):
    result = gather(
        tmp_path,
        local_dir=sample_clips[0].parent,
        brief="a rainy city",
        stock_per_query=1,
        generate_count=2,
        settings={"comfyui": {"host": "http://127.0.0.1:1", "workflow": "missing.json"}},
    )

    assert result.counts.get("local", 0) >= 1
    assert result.counts["stock"] == 1
    assert "local_comfyui" not in result.counts
    assert any("generat" in warning.lower() for warning in result.warnings)


@requires_ffmpeg
def test_requesting_a_paid_generator_while_disabled_costs_nothing(
    sample_clips: list[Path], tmp_path: Path
):
    budget = Budget(max_spend_usd=0.0, paid_adapters_enabled=False)

    result = gather(
        tmp_path,
        local_dir=sample_clips[0].parent,
        brief="a rainy city",
        stock_per_query=0,
        generate_count=3,
        generator_name="fal_gateway",
        budget=budget,
    )

    assert budget.spent_usd == 0.0
    assert budget.entries == []
    assert any("paid_adapters_enabled" in warning for warning in result.warnings)
    # What matters is that nothing was generated and nothing was charged. The
    # exact local count is incidental here and is covered by the ingest tests.
    assert "fal_gateway" not in result.counts
    assert result.counts.get("local", 0) >= 1


@requires_ffmpeg
def test_sourcing_without_a_brief_says_what_is_missing(
    sample_clips: list[Path], tmp_path: Path
):
    result = gather(
        tmp_path, local_dir=sample_clips[0].parent, stock_per_query=4
    )

    assert any("--brief" in warning for warning in result.warnings)


def test_the_summary_names_every_origin(tmp_path: Path):
    from ingest.local_clips import ClipInfo

    result = sourcing.SourcingResult(clips=[
        ClipInfo(Path("a.mp4"), 1.0, 16, 9, 30.0, False, origin="local"),
        ClipInfo(Path("b.mp4"), 1.0, 16, 9, 30.0, False, origin="stock"),
        ClipInfo(Path("c.mp4"), 1.0, 16, 9, 30.0, False, origin="stock"),
    ])

    assert result.summary() == "3 clip(s): 1 local, 2 stock"


@pytest.mark.parametrize("width,height,expected", [
    (1920, 1080, "landscape"),
    (1080, 1920, "portrait"),
    (1080, 1080, "square"),
])
def test_orientation_follows_the_output_format(width, height, expected):
    assert sourcing._orientation(VideoFormat(width, height, 30.0)) == expected
