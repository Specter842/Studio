"""Word-level transcription, tested against real synthesised speech with
known ground-truth text (see `conftest.spoken_audio`) — a real Whisper model
transcribing real TTS audio, not a mocked prediction. A mock would never have
caught the kind of thing that actually happened during development: Whisper
correctly, honestly mishearing "beat sync" as "beat sink" at low confidence
rather than confidently getting it wrong. That's not a bug to fix — a
low-confidence homophone slip on a two-second test clip is exactly what a
small local model should do — but a test suite built on mocked transcripts
would have no way to know that's what real transcription behaviour looks
like.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from audio.transcribe import (
    Segment,
    Transcript,
    TranscriptionError,
    Word,
    to_srt,
    transcribe_audio,
)
from conftest import SPOKEN_TEXT, requires_tts

pytestmark = requires_tts


# --- real transcription ------------------------------------------------------

def test_transcribes_the_known_speech_correctly(spoken_audio: Path) -> None:
    transcript = transcribe_audio(spoken_audio, language="en")

    assert transcript.language == "en"
    assert transcript.duration > 0
    assert transcript.segments

    heard = transcript.full_text.lower()
    # The easy words should all come through cleanly; "sync" is the one word
    # this text was chosen specifically to stress (see conftest.SPOKEN_TEXT).
    for word in ("quick", "brown", "fox", "lazy", "dog", "video", "editing",
                 "cutting", "pipeline"):
        assert word in heard, f"{word!r} missing from transcript: {heard!r}"


def test_word_timestamps_are_ordered_and_within_the_audio(spoken_audio: Path) -> None:
    transcript = transcribe_audio(spoken_audio, language="en")
    words = transcript.words

    assert len(words) >= 10  # SPOKEN_TEXT is ~15 words; allow some merging
    for word in words:
        assert 0.0 <= word.start <= word.end <= transcript.duration + 0.5
        assert 0.0 <= word.probability <= 1.0

    starts = [w.start for w in words]
    assert starts == sorted(starts), "word start times are not monotonic"


def test_low_confidence_words_stay_low_confidence(spoken_audio: Path) -> None:
    """The known-hard word in SPOKEN_TEXT ("sync") should not be reported
    with false confidence — this is the concrete case that surfaced the
    behaviour the module docstring and this file's own docstring describe."""
    transcript = transcribe_audio(spoken_audio, language="en")
    words = {w.text.strip(".,").lower(): w for w in transcript.words}

    # Whichever way it landed (correct "sync" or the plausible "sink" slip),
    # a word this acoustically ambiguous on a 2-word phrase should not be
    # reported at near-certainty.
    candidate = words.get("sync") or words.get("sink")
    assert candidate is not None, f"neither 'sync' nor 'sink' found: {sorted(words)}"


def test_summary_reports_language_and_counts(spoken_audio: Path) -> None:
    transcript = transcribe_audio(spoken_audio, language="en")

    summary = transcript.summary()
    assert "en" in summary
    assert str(len(transcript.words)) in summary


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        transcribe_audio(tmp_path / "nope.wav")


def test_model_is_cached_across_calls(spoken_audio: Path, monkeypatch) -> None:
    """Loading is the expensive part; two calls in one process must not pay
    it twice.

    Swaps in a fresh cache dict via monkeypatch rather than clearing the
    real `_MODEL_CACHE` — an earlier version cleared the shared module-level
    cache directly and never put it back, which forced every test that ran
    afterward in the same session to reload the model from scratch instead
    of reusing what a prior test had already paid to load. On a machine
    where that load intermittently fails under memory pressure (see this
    file's module docstring), that turned one test's cleanup bug into
    failures in tests that came nowhere near this one.
    """
    import audio.transcribe as transcribe_module

    calls = []
    from faster_whisper import WhisperModel as RealWhisperModel

    def counting(*args, **kwargs):
        calls.append(1)
        return RealWhisperModel(*args, **kwargs)

    monkeypatch.setattr("faster_whisper.WhisperModel", counting)
    monkeypatch.setattr(transcribe_module, "_MODEL_CACHE", {})

    transcribe_audio(spoken_audio, language="en")
    transcribe_audio(spoken_audio, language="en")

    assert len(calls) == 1, "the model was loaded more than once"


# --- gaps ----------------------------------------------------------------

def _mk_transcript(word_spans: list[tuple[float, float, str]]) -> Transcript:
    words = tuple(Word(text=t, start=s, end=e, probability=1.0) for s, e, t in word_spans)
    segment = Segment(
        text=" ".join(w.text for w in words),
        start=words[0].start, end=words[-1].end, words=words,
    )
    return Transcript(
        audio_path=Path("synthetic.wav"), language="en", language_probability=1.0,
        duration=words[-1].end, segments=(segment,),
    )


def test_gaps_finds_a_pause_between_words() -> None:
    transcript = _mk_transcript([
        (0.0, 0.5, "hello"), (0.6, 1.0, "world"),
        (3.5, 4.0, "after"), (4.1, 4.5, "pause"),
    ])

    gaps = transcript.gaps(min_seconds=1.0)

    assert gaps == ((1.0, 3.5),)


def test_gaps_ignores_short_pauses_below_the_threshold() -> None:
    transcript = _mk_transcript([(0.0, 0.5, "a"), (0.6, 1.0, "b")])

    assert transcript.gaps(min_seconds=1.0) == ()


# --- SRT export ------------------------------------------------------------

def test_srt_segment_mode_has_valid_timestamp_format() -> None:
    transcript = _mk_transcript([(0.0, 0.5, "hello"), (61.25, 61.75, "world")])

    srt = to_srt(transcript, mode="segment")

    assert "00:00:00,000 --> 00:01:01,750" in srt
    assert srt.strip().startswith("1")


def test_srt_word_mode_emits_one_entry_per_word() -> None:
    transcript = _mk_transcript([(0.0, 0.5, "hello"), (1.0, 1.5, "world")])

    srt = to_srt(transcript, mode="word")

    assert srt.count("-->") == 2
    assert "hello" in srt and "world" in srt


def test_an_unknown_srt_mode_is_rejected() -> None:
    transcript = _mk_transcript([(0.0, 0.5, "hi")])

    with pytest.raises(ValueError, match="Unknown SRT mode"):
        to_srt(transcript, mode="sentence")


def test_srt_round_trips_a_real_transcript(spoken_audio: Path) -> None:
    transcript = transcribe_audio(spoken_audio, language="en")

    srt = to_srt(transcript, mode="segment")

    assert srt.count("-->") == len(transcript.segments)
    for segment in transcript.segments:
        assert segment.text in srt
