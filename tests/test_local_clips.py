"""ffprobe-backed ingest."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import CLIP_FPS, CLIP_HEIGHT, CLIP_SECONDS, CLIP_WIDTH, requires_ffmpeg
from ingest.local_clips import (
    NoClipsFound,
    ClipInfo,
    probe_clip,
    scan_folder,
    total_duration,
)


@requires_ffmpeg
def test_probe_reads_duration_resolution_and_fps(sample_clips: list[Path]) -> None:
    clip = probe_clip(sample_clips[0])

    assert clip.width == CLIP_WIDTH
    assert clip.height == CLIP_HEIGHT
    assert clip.fps == pytest.approx(CLIP_FPS, abs=0.01)
    assert clip.duration == pytest.approx(CLIP_SECONDS, abs=0.1)
    assert clip.origin == "local"
    assert clip.is_usable


@requires_ffmpeg
def test_scan_folder_finds_every_clip_in_a_stable_order(
    sample_clips: list[Path],
) -> None:
    clips = scan_folder(sample_clips[0].parent)

    assert len(clips) == len(sample_clips)
    assert [clip.path.name for clip in clips] == sorted(
        path.name for path in sample_clips
    )
    assert total_duration(clips) == pytest.approx(
        CLIP_SECONDS * len(sample_clips), abs=0.5
    )


@requires_ffmpeg
def test_scan_folder_skips_unreadable_files(
    sample_clips: list[Path], tmp_path: Path
) -> None:
    for path in sample_clips:
        (tmp_path / path.name).write_bytes(path.read_bytes())
    (tmp_path / "truncated.mp4").write_bytes(b"not actually an mp4")

    clips = scan_folder(tmp_path)

    assert len(clips) == len(sample_clips)
    assert all(clip.path.name != "truncated.mp4" for clip in clips)


@requires_ffmpeg
def test_min_duration_filters_short_clips(sample_clips: list[Path]) -> None:
    with pytest.raises(NoClipsFound):
        scan_folder(sample_clips[0].parent, min_duration=CLIP_SECONDS * 10)


def test_empty_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(NoClipsFound):
        scan_folder(tmp_path)


def test_missing_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        scan_folder(tmp_path / "does-not-exist")


def test_aspect_ratio_is_derived_from_display_dimensions() -> None:
    clip = ClipInfo(
        path=Path("x.mp4"), duration=1.0, width=1920, height=1080,
        fps=30.0, has_audio=False,
    )
    assert clip.aspect_ratio == pytest.approx(16 / 9)
