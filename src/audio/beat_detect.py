"""Beat, downbeat and song-structure analysis.

`librosa.beat.beat_track` does the primary work — it is the well-tested default
and handles the overwhelming majority of 4/4 music correctly.

On top of the raw beat grid we derive two things the editor actually needs,
following the ideas in the open-source BeatSync Engine
(https://github.com/Merserk/BeatSync-Engine):

  * downbeats — librosa reports beats but not where the bar starts. Assuming a
    fixed time signature, the bar phase is the beat offset whose onsets are
    consistently strongest. Cutting on bar lines reads as musical; cutting on
    arbitrary beats reads as mechanical.
  * sections — a track's energy is not uniform. Splitting it into segments and
    labelling each low/medium/high lets the assembler cut fast through a drop
    and hold long through an intro, which is most of what separates an edit
    that feels arranged from one that feels like a metronome.

This module only analyses. Deciding which beats become cuts is an editorial
decision and lives in editing/assembler.py.
"""

from __future__ import annotations

import logging
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# 22.05kHz is plenty for onset/tempo work and roughly halves analysis time
# versus loading at 44.1kHz. A 512-sample hop at 22050Hz is ~23ms of
# resolution, comfortably inside the 100ms cut accuracy the pipeline targets.
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_HOP_LENGTH = 512

ENERGY_LABELS = ("low", "medium", "high")

# How many rows the per-beat energy contributes to the segmentation features,
# against chroma's 12 and MFCC's 13. Enough to be heard, not enough to drown
# out an arrangement change that keeps the same loudness.
ENERGY_FEATURE_ROWS = 8


class BeatDetectionError(RuntimeError):
    """The audio produced no usable beat grid."""


@dataclass(frozen=True)
class Section:
    """A stretch of the track with roughly consistent energy."""

    start: float
    end: float
    label: str          # one of ENERGY_LABELS
    energy: float       # 0..1, relative to this track's own dynamic range

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class BeatGrid:
    """The full musical analysis of one audio file."""

    audio_path: Path
    bpm: float
    duration: float
    beats: tuple[float, ...]
    # Onset strength at each beat, normalised to 0..1. Same length as `beats`.
    beat_strength: tuple[float, ...]
    # Index of the first downbeat; every `beats_per_bar`-th beat from there.
    downbeat_phase: int
    beats_per_bar: int
    sections: tuple[Section, ...]

    @property
    def downbeats(self) -> tuple[float, ...]:
        return tuple(self.beats[self.downbeat_phase::self.beats_per_bar])

    def is_downbeat_index(self, index: int) -> bool:
        return (index - self.downbeat_phase) % self.beats_per_bar == 0

    def section_at(self, time: float) -> Section | None:
        for section in self.sections:
            if section.start <= time < section.end:
                return section
        return self.sections[-1] if self.sections else None

    def label_at(self, time: float) -> str:
        section = self.section_at(time)
        return section.label if section else "medium"

    def nearest_beat(self, time: float) -> float:
        """Closest beat timestamp to `time`. Used to verify cut accuracy."""
        if not self.beats:
            raise BeatDetectionError("Beat grid is empty.")
        position = bisect_left(self.beats, time)
        candidates = []
        if position > 0:
            candidates.append(self.beats[position - 1])
        if position < len(self.beats):
            candidates.append(self.beats[position])
        return min(candidates, key=lambda beat: abs(beat - time))

    def summary(self) -> str:
        parts = [
            f"{self.bpm:.1f} BPM",
            f"{len(self.beats)} beats",
            f"{len(self.downbeats)} downbeats",
            f"{self.duration:.2f}s",
        ]
        if self.sections:
            shape = "/".join(section.label for section in self.sections)
            parts.append(f"sections: {shape}")
        return ", ".join(parts)


def detect_beats(
    audio_path: Path | str,
    *,
    start_bpm: float | None = 120.0,
    tightness: float = 100.0,
    beats_per_bar: int = 4,
    trim_silence: bool = False,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> BeatGrid:
    """Analyse `audio_path` and return its beat grid.

    `start_bpm` is a prior, not a constraint: pass None to let librosa choose
    freely, or set it when a track keeps getting tracked at half/double tempo.
    """
    # Imported lazily: librosa pulls in numba and costs ~2s to import, which is
    # wasted on any run that never touches audio (e.g. `--help`).
    import librosa

    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio track not found: {audio_path}")

    log.info("Analysing %s ...", audio_path.name)
    samples, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    if samples.size == 0:
        raise BeatDetectionError(f"{audio_path.name} decoded to zero samples.")

    # Checked explicitly rather than left to the tracker: on silence librosa
    # will happily return an evenly spaced grid at the tempo prior, which is a
    # confidently wrong answer rather than an error.
    peak = float(np.max(np.abs(samples)))
    if peak < 1e-4:
        raise BeatDetectionError(
            f"{audio_path.name} is silent (peak amplitude {peak:.2e}); "
            f"there is no beat to track."
        )

    duration = float(librosa.get_duration(y=samples, sr=sr))
    onset_envelope = librosa.onset.onset_strength(
        y=samples, sr=sr, hop_length=hop_length
    )

    beat_kwargs = {
        "onset_envelope": onset_envelope,
        "sr": sr,
        "hop_length": hop_length,
        "tightness": tightness,
        "trim": trim_silence,
        "units": "frames",
    }
    if start_bpm is not None:
        beat_kwargs["start_bpm"] = float(start_bpm)

    tempo, beat_frames = librosa.beat.beat_track(**beat_kwargs)
    beat_frames = np.asarray(beat_frames, dtype=int)

    if beat_frames.size < 2:
        raise BeatDetectionError(
            f"Only {beat_frames.size} beat(s) found in {audio_path.name}. "
            f"The track may be silent, too short, or non-rhythmic."
        )

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    bpm = _tempo_from_beats(beat_times, fallback=tempo)

    strength = _beat_strength(onset_envelope, beat_frames)
    downbeat_phase = _estimate_downbeat_phase(strength, beats_per_bar)
    sections = _detect_sections(
        librosa, samples, sr, hop_length, beat_frames, beat_times, duration
    )

    grid = BeatGrid(
        audio_path=audio_path,
        bpm=bpm,
        duration=duration,
        beats=tuple(float(t) for t in beat_times),
        beat_strength=tuple(float(s) for s in strength),
        downbeat_phase=downbeat_phase,
        beats_per_bar=beats_per_bar,
        sections=sections,
    )
    log.info("Beat analysis: %s", grid.summary())
    return grid


def _tempo_from_beats(beat_times: np.ndarray, *, fallback) -> float:
    """Tempo implied by the tracked beats, rather than the tracker's estimate.

    librosa's static estimate is picked off a log-spaced tempo grid and is only
    good to a couple of BPM on short excerpts — it reports 117.5 for a metronome
    running at exactly 120. The tracked beats are more precise, and they are
    what the edit actually cuts to, so the interval between them is the honest
    number to report. The estimate still does its real job as the tracker's own
    prior; it is only the reported value that changes here.

    Mean of the inliers, not the median. Beat times are quantised to the
    analysis hop (~23ms), so at 120 BPM the true 0.500s interval can only be
    stored as 0.488s or 0.511s — the distribution is bimodal and the median
    snaps to whichever side is more common, reporting 117.5. Averaging recovers
    the real interval. Intervals more than half a beat from the median are
    dropped first, so a missed beat cannot drag the average.
    """
    intervals = np.diff(np.asarray(beat_times, dtype=float))
    if intervals.size == 0:
        # librosa >= 0.10 returns tempo as an array even for a static estimate.
        return float(np.atleast_1d(fallback)[0])

    median_interval = float(np.median(intervals))
    inliers = intervals[np.abs(intervals - median_interval) < 0.5 * median_interval]
    interval = float(np.mean(inliers)) if inliers.size else median_interval

    if interval <= 0:
        return float(np.atleast_1d(fallback)[0])
    return 60.0 / interval


def _beat_strength(onset_envelope: np.ndarray, beat_frames: np.ndarray) -> np.ndarray:
    """Onset strength sampled at each beat, normalised to 0..1.

    Scaled by the 95th percentile rather than the max so one loud crash doesn't
    flatten every other beat to near zero. Taken as the peak over a one-frame
    window either side, so a beat that lands a frame off its transient is not
    scored as a weak beat.
    """
    frames = np.clip(np.asarray(beat_frames, dtype=int), 0, len(onset_envelope) - 1)
    values = np.array([
        float(np.max(onset_envelope[max(0, frame - 1):frame + 2]))
        for frame in frames
    ])
    ceiling = float(np.percentile(values, 95)) if values.size else 0.0
    if ceiling <= 0:
        return np.zeros_like(values, dtype=float)
    return np.clip(values / ceiling, 0.0, 1.0)


def _estimate_downbeat_phase(strength: np.ndarray, beats_per_bar: int) -> int:
    """Pick the beat offset whose onsets are strongest on average.

    In 4/4 the kick on beat one is almost always the loudest event in the bar,
    so averaging onset strength over each candidate phase and taking the
    maximum recovers the bar line without a full downbeat-tracking model.
    """
    if beats_per_bar < 2 or strength.size < beats_per_bar:
        return 0
    phase_scores = [
        float(np.mean(strength[phase::beats_per_bar]))
        for phase in range(beats_per_bar)
    ]
    return int(np.argmax(phase_scores))


def _detect_sections(
    librosa,
    samples: np.ndarray,
    sr: int,
    hop_length: int,
    beat_frames: np.ndarray,
    beat_times: np.ndarray,
    duration: float,
) -> tuple[Section, ...]:
    """Split the track into structural segments and label each by energy.

    Boundaries come from agglomerative clustering over beat-synchronous timbre
    (MFCC) and harmony (chroma) features — the same features that separate a
    verse from a chorus. Labels come from RMS energy within each segment,
    measured relative to this track's own range so a quiet acoustic song still
    gets a usable low/medium/high split.
    """
    beat_count = len(beat_frames)
    energy = _beat_energy(librosa, samples, sr, hop_length, beat_frames)

    # Below ~2 bars there is no structure to find; one section is the honest
    # answer and keeps short test fixtures working.
    if beat_count < 8:
        return (Section(0.0, duration, "medium", float(np.mean(energy))),)

    # One candidate section per ~8 beats (2 bars). Deliberately finer than the
    # structure actually wanted: a 4-bar granularity cannot represent a
    # quiet/loud/quiet arrangement on a short track at all, and over-segmenting
    # is recoverable — the merge pass below folds anything too short back in.
    section_count = int(np.clip(beat_count // 8, 2, 8))

    try:
        # tuning=0 skips librosa's tuning estimation, which warns on material
        # with no clear pitch content (percussion, sound design). Structural
        # segmentation compares chroma frames against each other, so a global
        # tuning offset would cancel out anyway.
        chroma = librosa.feature.chroma_stft(
            y=samples, sr=sr, hop_length=hop_length, tuning=0.0
        )
        mfcc = librosa.feature.mfcc(y=samples, sr=sr, hop_length=hop_length, n_mfcc=13)
        columns = len(beat_frames)
        features = np.vstack([
            _fit_columns(
                librosa.util.normalize(librosa.util.sync(chroma, beat_frames), axis=0),
                columns,
            ),
            _fit_columns(
                librosa.util.normalize(librosa.util.sync(mfcc, beat_frames), axis=0),
                columns,
            ),
            # Energy has to be in here, and it is easy to leave out. Chroma and
            # MFCC are both normalised per frame, so they describe harmony and
            # timbre with loudness deliberately divided out — segmenting on them
            # alone puts boundaries where the chords change and walks straight
            # past the point where the drums come in. Since the entire purpose
            # of these sections is to label them by energy, energy belongs in
            # the features that decide where they start.
            #
            # Repeated because a single row would be outvoted 25-to-1 by
            # features that know nothing about it.
            np.tile(energy, (ENERGY_FEATURE_ROWS, 1)),
        ])
        features = np.nan_to_num(features)
        boundaries = librosa.segment.agglomerative(features, section_count)
    except Exception as exc:  # noqa: BLE001 - fall back to even splits, don't fail the run
        log.warning("Structure detection failed (%s); using even sections.", exc)
        boundaries = np.linspace(0, beat_count, section_count, endpoint=False)

    boundary_beats = sorted({int(b) for b in boundaries} | {0})
    boundary_beats = [b for b in boundary_beats if 0 <= b < beat_count]

    spans: list[tuple[float, float, float]] = []
    for position, start_beat in enumerate(boundary_beats):
        end_beat = (
            boundary_beats[position + 1]
            if position + 1 < len(boundary_beats)
            else beat_count
        )
        start_time = float(beat_times[start_beat]) if position else 0.0
        end_time = (
            float(beat_times[end_beat]) if end_beat < beat_count else duration
        )
        spans.append((start_time, end_time, float(np.mean(energy[start_beat:end_beat]))))

    # Four beats is the shortest span that can carry a sense of energy; below
    # that it is a clustering artefact, not a part of the arrangement.
    beat_interval = float(np.median(np.diff(beat_times))) if beat_count > 1 else 0.5
    return _label_sections(
        _merge_short_spans(spans, max(1.0, 4.0 * beat_interval))
    )


def _fit_columns(matrix: np.ndarray, columns: int) -> np.ndarray:
    """Force a beat-synchronous feature block to exactly `columns` wide.

    `librosa.util.sync` treats the beat frames as segment boundaries, so when
    the first beat is not at frame 0 it emits an extra leading column for the
    audio before it. Stacking that against a per-beat array one column shorter
    fails outright, and silently trimming the wrong end would misalign every
    feature against its beat. The extra column is at the front, so that is the
    end that goes.
    """
    width = matrix.shape[1]
    if width == columns:
        return matrix
    if width > columns:
        return matrix[:, width - columns:]
    # Shorter than expected: repeat the final column rather than zero-pad, so a
    # missing beat looks like "more of the same" instead of a silent gap.
    padding = np.repeat(matrix[:, -1:], columns - width, axis=1)
    return np.hstack([matrix, padding])


def _merge_short_spans(
    spans: list[tuple[float, float, float]], minimum: float
) -> list[tuple[float, float, float]]:
    """Fold sections too short to be structure into their neighbour.

    Agglomerative clustering will happily place a boundary two beats from the
    end of a track, and because labels are assigned relative to the *range* of
    section energies, a single 2ms sliver sitting at an extreme drags the
    thresholds and mislabels everything else. Energies are combined weighted by
    duration so merging cannot distort the result either.
    """
    if len(spans) <= 1:
        return spans

    merged = [list(span) for span in spans]
    index = 0
    while len(merged) > 1 and index < len(merged):
        start, end, energy = merged[index]
        if end - start >= minimum:
            index += 1
            continue

        if index > 0:
            target, absorbed_first = merged[index - 1], False
        else:
            target, absorbed_first = merged[1], True

        target_length = target[1] - target[0]
        span_length = end - start
        total = target_length + span_length
        if total > 0:
            target[2] = (target[2] * target_length + energy * span_length) / total
        if absorbed_first:
            target[0] = start
            merged.pop(0)
        else:
            target[1] = end
            merged.pop(index)

    return [tuple(span) for span in merged]


def _beat_energy(
    librosa,
    samples: np.ndarray,
    sr: int,
    hop_length: int,
    beat_frames: np.ndarray,
) -> np.ndarray:
    """Mean RMS across each beat's own span, normalised to 0..1.

    Measured over the whole interval rather than at the beat frame itself, and
    that distinction matters more than it looks. Beat tracking routinely sits a
    little off the transient — it can slip phase by half a beat where a track
    changes texture and never fully recover — and a single-frame sample then
    reports the loudest section of a song as its quietest, inverting the cut
    density so the edit cuts fast through the intro and holds through the drop.

    Averaging over the beat is immune to that, and is a better description of
    the question actually being asked: how much is going on right now.
    """
    rms = librosa.feature.rms(y=samples, hop_length=hop_length)[0]
    frames = np.clip(np.asarray(beat_frames, dtype=int), 0, len(rms) - 1)

    gaps = np.diff(frames)
    # The final beat has no successor; give it a typical beat's worth of audio
    # rather than everything to the end of the file.
    tail = int(np.median(gaps)) if gaps.size else len(rms) - int(frames[-1])
    edges = np.append(frames, min(len(rms), int(frames[-1]) + max(1, tail)))

    values = np.array([
        float(np.mean(rms[start:stop])) if stop > start else float(rms[start])
        for start, stop in zip(edges[:-1], edges[1:])
    ])

    ceiling = float(np.percentile(values, 95)) if values.size else 0.0
    if ceiling <= 0:
        return np.zeros_like(values, dtype=float)
    return np.clip(values / ceiling, 0.0, 1.0)


def _label_sections(
    spans: list[tuple[float, float, float]]
) -> tuple[Section, ...]:
    """Turn (start, end, energy) triples into labelled Sections.

    Thresholds are relative to the track: the quietest section anchors "low"
    and the loudest anchors "high". A track with no meaningful dynamic range
    is labelled medium throughout rather than having structure invented for it.
    """
    if not spans:
        return ()

    energies = np.array([span[2] for span in spans])
    spread = float(energies.max() - energies.min())

    if spread < 0.10:
        return tuple(
            Section(start, end, "medium", energy) for start, end, energy in spans
        )

    low_threshold = float(energies.min()) + spread / 3.0
    high_threshold = float(energies.min()) + 2.0 * spread / 3.0

    sections = []
    for start, end, energy in spans:
        if energy < low_threshold:
            label = "low"
        elif energy > high_threshold:
            label = "high"
        else:
            label = "medium"
        sections.append(Section(start, end, label, energy))
    return tuple(sections)
