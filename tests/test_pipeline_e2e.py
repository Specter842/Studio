"""End to end: a folder of clips plus one audio track becomes one MP4.

This is the file that decides whether Phase 1 is done. It renders once, with
outbound network calls blocked at the socket layer, and then asserts against
the actual file: it plays, it has both streams, it is the right length, and its
cuts land on the beat.

The whole module runs in a few seconds — the fixtures are four-second clips and
a sixteen-second click track, deliberately, so this can be run on every change
rather than saved for the end.
"""

from __future__ import annotations

import socket
import subprocess
from pathlib import Path

import pytest

import pipeline
from conftest import requires_ffmpeg
from editing.verify import detect_scene_cuts, verify_cut_accuracy
from ffmpeg_tools import ffmpeg_bin, probe_json
from test_stock_fetch import stock_server  # noqa: F401 - reused fixture

TOLERANCE_SECONDS = 0.100
OUT_WIDTH, OUT_HEIGHT, OUT_FPS = 320, 180, 30
BEATS_PER_CUT = {"low": 8, "medium": 4, "high": 2}

# The fixture clips are flat primaries, so any cut moves a channel by ~250.
# Within a shot only the moving white box varies the average, by at most ~7.
COLOUR_CHANGE = 60


def _blocked(*args, **kwargs):
    raise AssertionError(
        "Phase 1 made a network call. The local path must be entirely offline."
    )


def _frame_colours(video_path: Path) -> list[tuple[int, int, int]]:
    """Average RGB of every frame, in one decode pass.

    `scale=1:1:flags=area` box-averages the whole frame down to a single pixel,
    so the entire video comes back as three bytes per frame.
    """
    raw = subprocess.run(
        [
            ffmpeg_bin(), "-hide_banner", "-v", "error",
            "-i", str(video_path), "-an",
            "-vf", "scale=1:1:flags=area",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        check=True, capture_output=True,
    ).stdout
    return [
        (raw[index], raw[index + 1], raw[index + 2])
        for index in range(0, len(raw) - 2, 3)
    ]


def _colour_distance(
    first: tuple[int, int, int], second: tuple[int, int, int]
) -> int:
    return max(abs(a - b) for a, b in zip(first, second))


@pytest.fixture(scope="session")
def rendered(sample_clips: list[Path], click_track: Path, tmp_path_factory) -> Path:
    """Run the real CLI once, offline, and hand back the output path."""
    out_path = tmp_path_factory.mktemp("render") / "final.mp4"

    # Blocking connect() rather than socket() itself: plenty of libraries build
    # socket objects during import without ever dialling out, and failing those
    # would prove nothing.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(socket.socket, "connect", _blocked)
        patch.setattr(socket.socket, "connect_ex", _blocked)
        patch.setattr(socket, "create_connection", _blocked)

        exit_code = pipeline.main([
            "--clips", str(sample_clips[0].parent),
            "--audio", str(click_track),
            "--out", str(out_path),
            "--width", str(OUT_WIDTH),
            "--height", str(OUT_HEIGHT),
            "--fps", str(OUT_FPS),
            "--seed", "1",
        ])

    assert exit_code == 0, "pipeline exited non-zero"
    return out_path


@pytest.fixture(scope="session")
def expected_plan(sample_clips: list[Path], click_track: Path):
    """The plan the `rendered` fixture should have produced.

    Rebuilt from the same inputs and seed rather than captured from the run, so
    that comparing it against the finished file also proves the edit is
    reproducible — the same inputs and seed give the same cuts.

    The values here mirror config/settings.yaml; anything that only affects
    encoding rather than cut positions is left at its default.
    """
    from audio.beat_detect import detect_beats
    from editing.assembler import build_plan
    from editing.transitions import VideoFormat
    from ingest.local_clips import scan_folder

    return build_plan(
        scan_folder(sample_clips[0].parent),
        detect_beats(click_track, start_bpm=120.0),
        video_format=VideoFormat(OUT_WIDTH, OUT_HEIGHT, OUT_FPS),
        beats_per_cut=BEATS_PER_CUT,
        seed=1,
    )


@requires_ffmpeg
def test_output_exists_and_is_a_readable_mp4(rendered: Path) -> None:
    assert rendered.is_file()
    assert rendered.stat().st_size > 10_000

    probe = probe_json(rendered)
    codecs = {stream["codec_type"] for stream in probe["streams"]}
    assert codecs == {"video", "audio"}


@requires_ffmpeg
def test_output_matches_the_requested_format(rendered: Path) -> None:
    video = next(
        stream for stream in probe_json(rendered)["streams"]
        if stream["codec_type"] == "video"
    )

    assert video["width"] == OUT_WIDTH
    assert video["height"] == OUT_HEIGHT
    assert video["codec_name"] == "h264"


@requires_ffmpeg
def test_output_length_matches_the_music(rendered: Path, beat_grid) -> None:
    duration = float(probe_json(rendered)["format"]["duration"])

    # The edit runs to the end of the analysed audio, within a frame.
    assert duration == pytest.approx(beat_grid.duration, abs=0.2)


@requires_ffmpeg
def test_cuts_land_on_the_beat(rendered: Path, beat_grid) -> None:
    """Phase 1's definition of done, measured rather than asserted by eye."""
    cuts = detect_scene_cuts(rendered)

    assert cuts, "scene detection found no cuts at all"

    errors = {
        cut: abs(cut - beat_grid.nearest_beat(cut))
        for cut in cuts
    }
    worst = max(errors.values())
    offenders = {
        cut: error for cut, error in errors.items() if error > TOLERANCE_SECONDS
    }

    assert not offenders, (
        f"{len(offenders)} of {len(cuts)} cuts missed their beat by more than "
        f"{TOLERANCE_SECONDS * 1000:.0f}ms: "
        + ", ".join(f"{cut:.3f}s off by {error * 1000:.0f}ms"
                    for cut, error in sorted(offenders.items()))
    )
    assert worst <= TOLERANCE_SECONDS


@requires_ffmpeg
def test_scene_detection_confirms_the_planned_cuts(
    rendered: Path, expected_plan
) -> None:
    """The same check `--verify` performs, run against a known-good plan."""
    report = verify_cut_accuracy(
        rendered, list(expected_plan.cut_times), tolerance=TOLERANCE_SECONDS
    )

    assert report.detected
    assert report.within_tolerance, report.summary()


@requires_ffmpeg
def test_every_planned_cut_lands_on_exactly_its_frame(
    rendered: Path, expected_plan
) -> None:
    """The strict version of the accuracy claim.

    Scene detection can only confirm a subset of cuts (it scores each change
    relative to the previous one, so a steady montage suppresses its own
    detections). The fixture clips are flat colours, so averaging each frame to
    a single pixel gives an exact answer instead: a colour change happens at
    precisely the planned frames, and at no others.

    That makes this a check on three things at once — no segment was dropped,
    none was duplicated, and none is a frame early or late.
    """
    colours = _frame_colours(rendered)
    changes = {
        index for index in range(1, len(colours))
        if _colour_distance(colours[index], colours[index - 1]) > COLOUR_CHANGE
    }
    planned = {round(cut * OUT_FPS) for cut in expected_plan.cut_times}

    assert changes == planned, (
        f"early/late or spurious: {sorted(changes - planned)}; "
        f"missing entirely: {sorted(planned - changes)}"
    )


@requires_ffmpeg
def test_chunked_rendering_produces_an_identical_timeline(
    sample_clips: list[Path], click_track: Path, tmp_path: Path, expected_plan
) -> None:
    """Long edits render in capped passes; the joins must be invisible.

    Rendering three segments at a time forces several pass boundaries, then
    checks the finished file frame by frame. Any drift, dropped frame or
    duplicated frame at a join would move every later cut and show up here.
    """
    out_path = tmp_path / "chunked.mp4"

    exit_code = pipeline.main([
        "--clips", str(sample_clips[0].parent),
        "--audio", str(click_track),
        "--out", str(out_path),
        "--width", str(OUT_WIDTH), "--height", str(OUT_HEIGHT),
        "--fps", str(OUT_FPS),
        "--seed", "1",
        "--max-inputs-per-pass", "3",
    ])

    assert exit_code == 0
    colours = _frame_colours(out_path)
    changes = {
        index for index in range(1, len(colours))
        if _colour_distance(colours[index], colours[index - 1]) > COLOUR_CHANGE
    }
    planned = {round(cut * OUT_FPS) for cut in expected_plan.cut_times}

    assert changes == planned, (
        f"pass joins shifted the timeline; "
        f"unexpected {sorted(changes - planned)}, missing {sorted(planned - changes)}"
    )


@requires_ffmpeg
def test_the_run_made_no_network_calls(rendered: Path) -> None:
    """Documented as its own test; enforced by the fixture that renders."""
    assert rendered.is_file()


@requires_ffmpeg
def test_dry_run_plans_without_writing_anything(
    sample_clips: list[Path], click_track: Path, tmp_path: Path, capsys
) -> None:
    out_path = tmp_path / "should_not_exist.mp4"

    exit_code = pipeline.main([
        "--clips", str(sample_clips[0].parent),
        "--audio", str(click_track),
        "--out", str(out_path),
        "--dry-run",
    ])

    assert exit_code == 0
    assert not out_path.exists()
    assert "Estimated run cost: $0.00" in capsys.readouterr().out


@requires_ffmpeg
def test_verify_flag_reports_accuracy_and_succeeds(
    sample_clips: list[Path], click_track: Path, tmp_path: Path, capsys
) -> None:
    out_path = tmp_path / "verified.mp4"

    exit_code = pipeline.main([
        "--clips", str(sample_clips[0].parent),
        "--audio", str(click_track),
        "--out", str(out_path),
        "--width", str(OUT_WIDTH), "--height", str(OUT_HEIGHT),
        "--fps", str(OUT_FPS),
        "--duration", "6",
        "--verify",
    ])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "PASS" in output
    assert "Estimated run cost: $0.00 (no paid calls made)" in output


@requires_ffmpeg
def test_crossfade_renders_and_keeps_the_timeline_length(
    sample_clips: list[Path], click_track: Path, tmp_path: Path, beat_grid
) -> None:
    out_path = tmp_path / "crossfade.mp4"

    exit_code = pipeline.main([
        "--clips", str(sample_clips[0].parent),
        "--audio", str(click_track),
        "--out", str(out_path),
        "--width", str(OUT_WIDTH), "--height", str(OUT_HEIGHT),
        "--fps", str(OUT_FPS),
        "--transition", "crossfade",
        "--duration", "8",
    ])

    assert exit_code == 0
    duration = float(probe_json(out_path)["format"]["duration"])
    # Dissolves are centred on the beat, so they must not shorten the edit.
    assert duration == pytest.approx(8.0, abs=0.2)


@requires_ffmpeg
def test_brief_puts_a_stock_clip_on_the_timeline(
    stock_server, sample_clips: list[Path], click_track: Path,  # noqa: F811
    tmp_path: Path, capsys,
) -> None:
    """Phase 2's definition of done.

    Two assertions, deliberately separate: that a stock clip reaches the
    *timeline* (not merely the cache), and that the run still reports $0.

    The full track is used rather than a short window. The clip picker cycles a
    shuffled rotation, so every clip is drawn once before any is drawn twice —
    which guarantees the two stock clips appear only once the edit has at least
    as many shots as there are clips. A 6-second window gives three shots out
    of six clips, and whether a stock one is among them comes down to the seed.
    """
    out_path = tmp_path / "brief.mp4"
    common = [
        "--clips", str(sample_clips[0].parent),
        "--audio", str(click_track),
        "--out", str(out_path),
        "--brief", "a rainy city at night",
        "--stock-per-query", "2",
        "--cache-dir", str(tmp_path / "cache"),
        "--width", str(OUT_WIDTH), "--height", str(OUT_HEIGHT),
        "--fps", str(OUT_FPS), "--seed", "5",
    ]

    assert pipeline.main([*common, "--dry-run"]) == 0
    planned = capsys.readouterr().out
    assert "[stock]" in planned, "no stock clip made it onto the timeline"
    assert "[local]" in planned, "local clips were dropped"

    assert pipeline.main(common) == 0
    rendered_output = capsys.readouterr().out

    assert out_path.is_file()
    assert "stock" in rendered_output
    assert "Estimated run cost: $0.00 (no paid calls made)" in rendered_output


@requires_ffmpeg
def test_a_brief_without_keys_still_renders_and_says_why(
    sample_clips: list[Path], click_track: Path, tmp_path: Path, capsys, monkeypatch
) -> None:
    """Degraded runs must be obvious, not silent."""
    # setenv("") rather than delenv: pipeline.main() calls config.load_env()
    # internally, which reloads a *currently-unset* var from a real .env with
    # override=False — delenv leaves the var unset and gets silently undone
    # by that reload once real keys exist in .env. An empty string is still
    # "set", so the reload leaves it alone, and secret() already treats an
    # empty value as missing.
    monkeypatch.setenv("PEXELS_API_KEY", "")
    monkeypatch.setenv("PIXABAY_API_KEY", "")

    exit_code = pipeline.main([
        "--clips", str(sample_clips[0].parent),
        "--audio", str(click_track),
        "--out", str(tmp_path / "degraded.mp4"),
        "--brief", "a rainy city",
        "--cache-dir", str(tmp_path / "cache"),
        "--width", str(OUT_WIDTH), "--height", str(OUT_HEIGHT),
        "--fps", str(OUT_FPS), "--duration", "4",
    ])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert (tmp_path / "degraded.mp4").is_file()
    assert "Warnings:" in output
    assert "PEXELS_API_KEY" in output


@requires_ffmpeg
def test_a_paid_generator_is_refused_and_the_run_still_costs_nothing(
    sample_clips: list[Path], click_track: Path, tmp_path: Path, capsys
) -> None:
    exit_code = pipeline.main([
        "--clips", str(sample_clips[0].parent),
        "--audio", str(click_track),
        "--out", str(tmp_path / "nopaid.mp4"),
        "--brief", "a rainy city",
        "--generate", "3",
        "--generator", "fal_gateway",
        "--cache-dir", str(tmp_path / "cache"),
        "--width", str(OUT_WIDTH), "--height", str(OUT_HEIGHT),
        "--fps", str(OUT_FPS), "--duration", "4",
    ])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "paid_adapters_enabled" in output
    assert "Estimated run cost: $0.00 (no paid calls made)" in output


def test_missing_clips_folder_is_a_clean_error(click_track: Path, tmp_path: Path) -> None:
    exit_code = pipeline.main([
        "--clips", str(tmp_path / "nope"),
        "--audio", str(click_track),
        "--out", str(tmp_path / "out.mp4"),
    ])

    assert exit_code == 1
