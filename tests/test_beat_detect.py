"""Beat, downbeat and section analysis against a click track of known tempo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from audio.beat_detect import BeatDetectionError, detect_beats
from conftest import TEST_BEATS, TEST_BPM


def test_detects_the_known_tempo(beat_grid) -> None:
    assert beat_grid.bpm == pytest.approx(TEST_BPM, abs=2.0)


def test_beats_are_evenly_spaced_at_the_known_interval(beat_grid) -> None:
    intervals = np.diff(np.array(beat_grid.beats))
    expected = 60.0 / TEST_BPM

    assert float(np.median(intervals)) == pytest.approx(expected, abs=0.02)
    # No dropped or doubled beats anywhere in the grid.
    assert float(np.max(np.abs(intervals - expected))) < 0.05


def test_finds_most_of_the_beats(beat_grid) -> None:
    """`trim=True` discards beats at the edges where onset strength ramps up.

    The fixture starts abruptly on beat one with no lead-in, which is the worst
    case for that: a couple of beats at each end are expected to go. What must
    not happen is losing a meaningful chunk of the middle.
    """
    assert len(beat_grid.beats) >= TEST_BEATS * 0.85
    assert len(beat_grid.beats) <= TEST_BEATS + 1


def test_downbeat_phase_lands_on_the_accented_beat(beat_grid) -> None:
    """The fixture accents beat 1 of every bar, so the phase must select it."""
    downbeats = beat_grid.downbeats

    assert beat_grid.beats_per_bar == 4
    assert len(downbeats) >= TEST_BEATS // 4 - 1
    # Every downbeat should be a whole number of bars from the first one.
    bar_seconds = 4 * 60.0 / TEST_BPM
    offsets = (np.array(downbeats) - downbeats[0]) / bar_seconds
    assert np.allclose(offsets, np.round(offsets), atol=0.05)


def test_downbeats_are_the_strongest_beats(beat_grid) -> None:
    strength = np.array(beat_grid.beat_strength)
    phase, per_bar = beat_grid.downbeat_phase, beat_grid.beats_per_bar

    on_bar = strength[phase::per_bar]
    off_bar = np.delete(strength, np.arange(phase, len(strength), per_bar))

    assert float(np.mean(on_bar)) > float(np.mean(off_bar))


def test_sections_tile_the_whole_track_without_gaps(beat_grid) -> None:
    sections = beat_grid.sections

    assert sections
    assert sections[0].start == pytest.approx(0.0, abs=0.01)
    assert sections[-1].end == pytest.approx(beat_grid.duration, abs=0.05)
    for earlier, later in zip(sections, sections[1:]):
        assert earlier.end == pytest.approx(later.start, abs=1e-6)
    assert all(section.label in ("low", "medium", "high") for section in sections)


def test_label_at_covers_every_point_in_the_track(beat_grid) -> None:
    for moment in np.linspace(0.0, beat_grid.duration, 25):
        assert beat_grid.label_at(float(moment)) in ("low", "medium", "high")


def test_nearest_beat_picks_the_closer_side(beat_grid) -> None:
    beats = beat_grid.beats
    midpoint = (beats[3] + beats[4]) / 2.0

    assert beat_grid.nearest_beat(beats[3] + 0.01) == pytest.approx(beats[3])
    assert beat_grid.nearest_beat(beats[4] - 0.01) == pytest.approx(beats[4])
    # Exactly halfway is allowed to go either way, but must pick a neighbour.
    assert beat_grid.nearest_beat(midpoint) in (beats[3], beats[4])


# --- energy and structure -------------------------------------------------
#
# These exist because an earlier version had this exactly backwards. It sampled
# RMS at the beat frame itself, and where the beat grid slipped off the
# transients it reported the loudest passage of a track as its quietest — so
# the edit cut fast through the intro and held through the drop. Every test
# below fails against that version.

def test_quiet_and_loud_passages_are_labelled_the_right_way_round(
    structured_grid,
) -> None:
    """The fixture is soft for 4 bars, busy for 8, soft for 4."""
    label_at = structured_grid.label_at

    assert label_at(2.0) == "low"      # intro
    assert label_at(8.0) == "high"     # busy middle
    assert label_at(14.0) == "low"     # outro


def test_measured_energy_ranks_the_sections_correctly(structured_grid) -> None:
    quiet = [s for s in structured_grid.sections if s.label == "low"]
    loud = [s for s in structured_grid.sections if s.label == "high"]

    assert quiet and loud
    assert min(s.energy for s in loud) > max(s.energy for s in quiet)


def test_energy_survives_a_beat_grid_that_is_off_the_transients(
    structured_track: Path,
) -> None:
    """The direct regression.

    Beat tracking can slip half a beat where a track changes texture and never
    recover. Energy measurement has to stay correct when it does, so this feeds
    in a deliberately offset grid and asserts the loud section still reads as
    loud.
    """
    import librosa
    from audio.beat_detect import _beat_energy

    hop = 512
    samples, sr = librosa.load(structured_track, sr=22050, mono=True)
    beat_seconds = 60.0 / TEST_BPM

    # A perfect grid, and the same grid shoved half a beat off the kicks.
    aligned = librosa.time_to_frames(
        np.arange(TEST_BEATS) * beat_seconds, sr=sr, hop_length=hop
    )
    offset = librosa.time_to_frames(
        np.arange(TEST_BEATS) * beat_seconds + beat_seconds / 2, sr=sr, hop_length=hop
    )

    for grid in (aligned, offset):
        energy = _beat_energy(librosa, samples, sr, hop, np.clip(grid, 0, None))
        assert float(np.mean(energy[10:22])) > float(np.mean(energy[0:7])), (
            "loud middle measured as quieter than the soft intro"
        )


def test_cuts_are_denser_where_the_music_is(structured_grid) -> None:
    """The behaviour all of the above exists to protect."""
    from editing.assembler import choose_cut_times

    cuts = choose_cut_times(
        structured_grid, beats_per_cut={"low": 8, "medium": 4, "high": 2}
    )
    shots = [
        (start, end - start) for start, end in zip(cuts, cuts[1:])
    ]
    intro = [length for start, length in shots if start < 4.0]
    middle = [length for start, length in shots if 4.5 <= start < 11.5]

    assert intro and middle
    assert min(intro) > max(middle), (
        f"intro shots {intro} should be longer than middle shots {middle}"
    )


def test_a_flat_track_gets_no_invented_structure(beat_grid) -> None:
    """Every beat of the plain click track is identical; say so."""
    assert {section.label for section in beat_grid.sections} == {"medium"}


def test_silence_raises_a_clear_error(tmp_path: Path) -> None:
    import soundfile

    path = tmp_path / "silence.wav"
    soundfile.write(path, np.zeros(22050 * 3), 22050)

    with pytest.raises(BeatDetectionError):
        detect_beats(path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        detect_beats(tmp_path / "nope.wav")
