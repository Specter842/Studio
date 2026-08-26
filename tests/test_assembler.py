"""Timeline planning: which beats become cuts, and what fills the shots."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from audio.beat_detect import BeatGrid, Section
from editing.assembler import (
    ClipPicker,
    Effects,
    PlanningError,
    build_filter_graph,
    build_plan,
    choose_cut_times,
)
from editing.transitions import CROSSFADE, CUT, VideoFormat
from ingest.local_clips import ClipInfo

FPS = 30.0
FORMAT = VideoFormat(width=320, height=180, fps=FPS)
BEATS_PER_CUT = {"low": 8, "medium": 4, "high": 2}


def make_grid(
    *, beat_seconds: float = 0.5, count: int = 32, label: str = "high"
) -> BeatGrid:
    """A perfectly regular grid, so expected cut positions are arithmetic."""
    beats = tuple(index * beat_seconds for index in range(count))
    duration = count * beat_seconds
    return BeatGrid(
        audio_path=Path("synthetic.wav"),
        bpm=60.0 / beat_seconds,
        duration=duration,
        beats=beats,
        beat_strength=tuple(1.0 if i % 4 == 0 else 0.4 for i in range(count)),
        downbeat_phase=0,
        beats_per_bar=4,
        sections=(Section(0.0, duration, label, 0.9),),
    )


def make_clips(count: int = 4, duration: float = 4.0) -> list[ClipInfo]:
    return [
        ClipInfo(
            path=Path(f"clip_{index}.mp4"),
            duration=duration,
            width=320, height=180, fps=FPS, has_audio=False,
        )
        for index in range(count)
    ]


# --- choosing cut points --------------------------------------------------

def test_cuts_land_on_beats_at_the_section_density() -> None:
    grid = make_grid(label="high")  # high -> every 2 beats -> every 1.0s

    cuts = choose_cut_times(grid, beats_per_cut=BEATS_PER_CUT)

    assert cuts[0] == 0.0
    assert cuts[-1] == pytest.approx(grid.duration)
    for cut in cuts[1:-1]:
        assert cut in grid.beats
    interior = cuts[1:-1]
    assert all(
        later - earlier == pytest.approx(1.0)
        for earlier, later in zip(interior, interior[1:])
    )


def test_low_energy_sections_hold_longer_than_high_energy_ones() -> None:
    fast = choose_cut_times(make_grid(label="high"), beats_per_cut=BEATS_PER_CUT)
    slow = choose_cut_times(make_grid(label="low"), beats_per_cut=BEATS_PER_CUT)

    assert len(fast) > len(slow)


def test_max_shot_seconds_pulls_long_holds_back_to_an_earlier_beat() -> None:
    grid = make_grid(label="low")  # would otherwise hold for 8 beats = 4.0s

    cuts = choose_cut_times(
        grid, beats_per_cut=BEATS_PER_CUT, max_shot_seconds=1.5
    )

    shots = [later - earlier for earlier, later in zip(cuts, cuts[1:])]
    assert max(shots) <= 1.5 + 1e-6
    for cut in cuts[1:-1]:
        assert cut in grid.beats  # still on the grid, just an earlier beat


def test_min_shot_seconds_skips_beats_that_are_too_close_together() -> None:
    grid = make_grid(beat_seconds=0.1, count=60, label="high")

    cuts = choose_cut_times(
        grid, beats_per_cut={"high": 1}, min_shot_seconds=0.5
    )

    shots = [later - earlier for earlier, later in zip(cuts, cuts[1:])]
    assert min(shots) >= 0.5 - 1e-6


def test_snapping_moves_a_cut_onto_the_nearest_bar_line() -> None:
    grid = make_grid(label="medium")  # 4 beats per cut, bars are 4 beats
    snapped = choose_cut_times(
        grid, beats_per_cut={"medium": 3}, snap_to_downbeats=True
    )
    unsnapped = choose_cut_times(
        grid, beats_per_cut={"medium": 3}, snap_to_downbeats=False
    )

    downbeats = set(grid.downbeats)
    snapped_hits = sum(1 for cut in snapped[1:-1] if cut in downbeats)
    unsnapped_hits = sum(1 for cut in unsnapped[1:-1] if cut in downbeats)

    assert snapped_hits > unsnapped_hits


def test_empty_edit_window_raises() -> None:
    with pytest.raises(PlanningError):
        choose_cut_times(make_grid(), beats_per_cut=BEATS_PER_CUT, start=5.0, end=5.0)


# --- clip selection -------------------------------------------------------

def test_picker_never_repeats_a_clip_back_to_back() -> None:
    picker = ClipPicker(make_clips(4), seed=1)

    picks = [picker.pick(0.5)[0].path for _ in range(40)]

    assert all(
        earlier != later for earlier, later in zip(picks, picks[1:])
    )


def test_picker_advances_through_each_clip_instead_of_replaying_the_start() -> None:
    picker = ClipPicker(make_clips(2, duration=8.0), seed=1)

    starts: dict[Path, list[float]] = {}
    for _ in range(6):
        clip, source_in = picker.pick(1.0)
        starts.setdefault(clip.path, []).append(source_in)

    assert any(len(set(values)) > 1 for values in starts.values())


def test_picker_falls_back_to_the_longest_clip_when_none_is_long_enough() -> None:
    clips = [
        ClipInfo(Path("short.mp4"), 1.0, 320, 180, FPS, False),
        ClipInfo(Path("longer.mp4"), 2.0, 320, 180, FPS, False),
    ]
    picker = ClipPicker(clips, seed=0)

    clip, source_in = picker.pick(5.0)

    assert clip.path.name == "longer.mp4"
    assert source_in == 0.0


def test_picker_with_no_clips_raises() -> None:
    with pytest.raises(PlanningError):
        ClipPicker([], seed=0)


# --- full plan ------------------------------------------------------------

def test_plan_segments_tile_the_timeline_without_gaps_or_overlap() -> None:
    plan = build_plan(
        make_clips(), make_grid(), video_format=FORMAT,
        beats_per_cut=BEATS_PER_CUT, seed=7,
    )

    assert plan.segments[0].timeline_start == 0.0
    for earlier, later in zip(plan.segments, plan.segments[1:]):
        assert earlier.timeline_end == pytest.approx(later.timeline_start, abs=1e-9)
    assert plan.segments[-1].timeline_end == pytest.approx(
        plan.total_duration, abs=1e-9
    )


def test_cut_positions_never_drift_from_their_beats() -> None:
    """The whole point of quantising against absolute positions.

    Every cut must sit within half a frame of its beat, including the last one
    — that is what summing per-segment durations would fail to guarantee.
    """
    plan = build_plan(
        make_clips(), make_grid(count=64), video_format=FORMAT,
        beats_per_cut=BEATS_PER_CUT, seed=3,
    )

    half_frame = 0.5 / FPS
    for segment in plan.segments:
        assert abs(segment.timeline_start - segment.beat_time) <= half_frame


def test_every_segment_is_a_whole_number_of_frames() -> None:
    plan = build_plan(
        make_clips(), make_grid(), video_format=FORMAT,
        beats_per_cut=BEATS_PER_CUT, seed=3,
    )

    for segment in plan.segments:
        assert segment.frames >= 1
        assert segment.timeline_start * FPS == pytest.approx(
            round(segment.timeline_start * FPS), abs=1e-6
        )


def test_short_clips_are_padded_rather_than_shortening_the_shot() -> None:
    clips = [ClipInfo(Path(f"tiny_{i}.mp4"), 0.4, 320, 180, FPS, False) for i in range(3)]
    grid = make_grid(beat_seconds=1.0, count=8, label="high")  # 2s shots

    plan = build_plan(
        clips, grid, video_format=FORMAT, beats_per_cut=BEATS_PER_CUT, seed=0
    )

    assert any(segment.pad_seconds > 0 for segment in plan.segments)
    # The timeline is still exactly as long as the music asked for.
    assert plan.total_duration == pytest.approx(grid.duration, abs=1.0 / FPS)


def test_seed_makes_the_edit_reproducible() -> None:
    first = build_plan(
        make_clips(), make_grid(), video_format=FORMAT,
        beats_per_cut=BEATS_PER_CUT, seed=42,
    )
    second = build_plan(
        make_clips(), make_grid(), video_format=FORMAT,
        beats_per_cut=BEATS_PER_CUT, seed=42,
    )

    assert [s.clip.path for s in first.segments] == [s.clip.path for s in second.segments]


def test_start_and_end_window_the_edit() -> None:
    plan = build_plan(
        make_clips(), make_grid(count=64), video_format=FORMAT,
        beats_per_cut=BEATS_PER_CUT, seed=0, start=4.0, end=10.0,
    )

    assert plan.audio_start == 4.0
    assert plan.total_duration == pytest.approx(6.0, abs=1.0 / FPS)
    assert plan.segments[0].beat_time == pytest.approx(4.0)


# --- filter graph ---------------------------------------------------------

def test_cut_graph_concatenates_and_clamps_every_segment() -> None:
    plan = build_plan(
        make_clips(), make_grid(), video_format=FORMAT,
        beats_per_cut=BEATS_PER_CUT, seed=0, transition=CUT,
    )

    graph = build_filter_graph(plan)

    assert f"concat=n={len(plan.segments)}:v=1:a=0[outv]" in graph
    assert graph.count("trim=end_frame=") == len(plan.segments)
    assert "[outa]" in graph
    # ffmpeg's parser rejects scientific notation; no value may reach it.
    assert "e-0" not in graph


def test_crossfade_graph_chains_one_xfade_per_join() -> None:
    plan = build_plan(
        make_clips(), make_grid(), video_format=FORMAT,
        beats_per_cut=BEATS_PER_CUT, seed=0,
        transition=CROSSFADE, crossfade_seconds=0.2,
    )

    graph = build_filter_graph(plan)

    assert graph.count("xfade=") == len(plan.segments) - 1
    assert "concat=" not in graph


def test_crossfade_longer_than_a_shot_is_rejected_with_an_explanation() -> None:
    plan = build_plan(
        make_clips(), make_grid(), video_format=FORMAT,
        beats_per_cut=BEATS_PER_CUT, seed=0,
        transition=CROSSFADE, crossfade_seconds=5.0,
    )

    with pytest.raises(ValueError, match="shorter than"):
        build_filter_graph(plan)


def test_crop_and_pad_produce_different_scaling() -> None:
    padded = build_filter_graph(
        build_plan(
            make_clips(), make_grid(count=8),
            video_format=VideoFormat(320, 180, FPS, fit="pad"),
            beats_per_cut=BEATS_PER_CUT, seed=0,
        )
    )
    cropped = build_filter_graph(
        build_plan(
            make_clips(), make_grid(count=8),
            video_format=VideoFormat(320, 180, FPS, fit="crop"),
            beats_per_cut=BEATS_PER_CUT, seed=0,
        )
    )

    assert "pad=320:180" in padded and "crop=320:180" not in padded
    assert "crop=320:180" in cropped and "pad=320:180" not in cropped


# --- post-processing effects ------------------------------------------------

def test_disabled_effects_produce_the_exact_same_graph_as_before() -> None:
    """A plan built without naming any effect must render byte-identical to
    how it always did — no `pre_out`/post stage inserted for nothing."""
    plan = build_plan(
        make_clips(), make_grid(), video_format=FORMAT,
        beats_per_cut=BEATS_PER_CUT, seed=0,
    )

    graph = build_filter_graph(plan)

    assert "pre_out" not in graph
    assert f"concat=n={len(plan.segments)}:v=1:a=0[outv]" in graph


def test_an_active_look_routes_through_pre_out_then_grades() -> None:
    plan = build_plan(
        make_clips(), make_grid(), video_format=FORMAT,
        beats_per_cut=BEATS_PER_CUT, seed=0, effects=Effects(look="punchy"),
    )

    graph = build_filter_graph(plan)

    assert f"concat=n={len(plan.segments)}:v=1:a=0[pre_out]" in graph
    assert "[pre_out]eq=" in graph
    assert graph.count("[outv]") == 1


def test_an_unknown_look_is_rejected_when_the_graph_is_built() -> None:
    plan = build_plan(
        make_clips(), make_grid(), video_format=FORMAT,
        beats_per_cut=BEATS_PER_CUT, seed=0, effects=Effects(look="nope"),
    )

    with pytest.raises(ValueError, match="Unknown look"):
        build_filter_graph(plan)


def test_punch_zoom_and_glitch_both_key_off_every_cut_by_default() -> None:
    plan = build_plan(
        make_clips(), make_grid(count=16), video_format=FORMAT,
        beats_per_cut=BEATS_PER_CUT, seed=0,
        effects=Effects(punch_zoom_amount=0.2, glitch_shift_px=6),
    )

    graph = build_filter_graph(plan)
    expected_cuts = len(plan.cut_times)

    assert graph.count("overlay=x=0:y=0:enable=") == 1
    assert graph.count("rgbashift=") == 1
    # One between() term per cut in each of the two enable expressions.
    assert graph.count("between(") == expected_cuts * 2


def test_local_cut_times_matches_cut_times_for_the_whole_plan() -> None:
    """The whole edit always starts its own video stream at local t=0."""
    plan = build_plan(
        make_clips(), make_grid(), video_format=FORMAT,
        beats_per_cut=BEATS_PER_CUT, seed=0,
    )

    assert plan.local_cut_times == plan.cut_times


def test_local_cut_times_is_relative_when_a_chunk_starts_partway_through() -> None:
    """What chunked rendering actually needs: a slice of a longer edit whose
    segments keep their original absolute `timeline_start`, but whose own
    rendered pass restarts its PTS at 0."""
    plan = build_plan(
        make_clips(), make_grid(count=24), video_format=FORMAT,
        beats_per_cut=BEATS_PER_CUT, seed=0,
    )
    # Simulate what _render_chunked does: take a slice starting partway in.
    chunk = replace(plan, segments=plan.segments[3:7])

    origin = chunk.segments[0].timeline_start
    assert origin > 0, "test needs a chunk that doesn't start at the top"
    expected = tuple(
        segment.timeline_start - origin for segment in chunk.segments[1:]
    )
    assert chunk.local_cut_times == expected
    assert chunk.local_cut_times[0] < chunk.segments[0].timeline_start, (
        "local cut times must not leak the plan's absolute positions"
    )


def test_ramp_is_off_by_default() -> None:
    plan = build_plan(
        make_clips(), make_grid(), video_format=FORMAT,
        beats_per_cut=BEATS_PER_CUT, seed=0,
    )

    assert all(segment.speed_ramp is None for segment in plan.segments)


def test_ramp_applies_when_enabled_and_footage_allows_it() -> None:
    plan = build_plan(
        make_clips(count=4, duration=30.0), make_grid(count=16), video_format=FORMAT,
        beats_per_cut=BEATS_PER_CUT, seed=0,
        effects=Effects(speed_ramp_enabled=True),
    )

    assert any(segment.speed_ramp is not None for segment in plan.segments)


def test_ramp_never_changes_output_timing() -> None:
    """The whole point: a ramp changes how much source a shot spends, never
    how many output frames it produces or where it sits on the beat grid."""
    kwargs = dict(
        video_format=FORMAT, beats_per_cut=BEATS_PER_CUT, seed=3,
    )
    plain = build_plan(make_clips(count=4, duration=30.0), make_grid(count=16), **kwargs)
    ramped = build_plan(
        make_clips(count=4, duration=30.0), make_grid(count=16), **kwargs,
        effects=Effects(speed_ramp_enabled=True),
    )

    assert [s.frames for s in plain.segments] == [s.frames for s in ramped.segments]
    assert [s.timeline_start for s in plain.segments] == [
        s.timeline_start for s in ramped.segments
    ]
    assert [s.beat_time for s in plain.segments] == [s.beat_time for s in ramped.segments]
    assert plain.total_duration == ramped.total_duration
    # And it must actually have been applied somewhere, or the comparison above
    # would be vacuously true.
    assert any(s.speed_ramp is not None for s in ramped.segments)


def test_ramp_falls_back_to_unramped_when_the_clip_is_too_short() -> None:
    """A clip barely longer than a shot's own duration has no room to also
    read the extra footage a net-slower ramp preset would ask for."""
    tight_clips = make_clips(count=4, duration=0.6)
    plan = build_plan(
        tight_clips, make_grid(beat_seconds=0.5, count=16), video_format=FORMAT,
        beats_per_cut=BEATS_PER_CUT, seed=0,
        effects=Effects(speed_ramp_enabled=True),
    )

    assert all(segment.speed_ramp is None for segment in plan.segments)


def test_ramp_is_never_combined_with_padding() -> None:
    tight_clips = make_clips(count=4, duration=0.6)
    plan = build_plan(
        tight_clips, make_grid(beat_seconds=0.5, count=16), video_format=FORMAT,
        beats_per_cut=BEATS_PER_CUT, seed=0,
        effects=Effects(speed_ramp_enabled=True),
    )

    for segment in plan.segments:
        if segment.pad_seconds > 0:
            assert segment.speed_ramp is None


def test_ramp_graph_uses_split_trim_setpts_concat() -> None:
    plan = build_plan(
        make_clips(count=4, duration=30.0), make_grid(count=16), video_format=FORMAT,
        beats_per_cut=BEATS_PER_CUT, seed=0,
        effects=Effects(speed_ramp_enabled=True),
    )

    graph = build_filter_graph(plan)

    assert "split=2" in graph
    assert "concat=n=2:v=1:a=0" in graph
    assert graph.count("setpts=(PTS-STARTPTS)/") >= 2


def test_effects_is_active_reflects_any_single_effect() -> None:
    assert not Effects().is_active
    assert Effects(look="blackout").is_active
    assert Effects(punch_zoom_amount=0.1).is_active
    assert Effects(glitch_shift_px=1).is_active
    assert Effects(speed_ramp_enabled=True).is_active
    assert not Effects(look="none", punch_zoom_amount=0.0, glitch_shift_px=0).is_active
