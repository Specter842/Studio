"""Word-level speech transcription — free, local, CPU-only.

`faster-whisper` (CTranslate2, MIT) rather than `whisper.cpp`: pip-installable,
no separate binary/build step, fits this codebase's existing style of calling
Python libraries directly (librosa for beats, this for words) instead of
shelling out to a compiled tool the way Blender is driven. The model itself
downloads once from Hugging Face on first use (~140MB for "base") and is
cached at `~/.cache/huggingface`; every call after that is offline.

Why this exists: two independent creators' AI-editing pipelines (see
`skills/beat-sync-cutting/SKILL.md`'s Kaestral comparison, and the
"Hyperframes"/content-operating-system pattern) both build their entire
editing workflow around word-level timestamps — silence/retake removal,
caption timing, B-roll placement keyed to specific words. This pipeline had
none of that; it only ever knew about beats, never about speech.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_MODEL_SIZE = "base"

# One loaded model per (size, device, compute_type) combination, reused across
# calls in the same process. Loading is the expensive part (seconds to almost
# two minutes on first download); transcribing is fast once it's loaded, so
# paying the load cost once per process instead of once per call matters a lot
# for anything that transcribes more than one file in a session (an MCP server
# handling several tool calls, for instance).
_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}


class TranscriptionError(RuntimeError):
    """Whisper could not produce a transcript."""


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    probability: float  # 0..1, Whisper's own confidence for this word


@dataclass(frozen=True)
class Segment:
    """One contiguous stretch of speech Whisper transcribed as a unit."""

    text: str
    start: float
    end: float
    words: tuple[Word, ...]


@dataclass(frozen=True)
class Transcript:
    audio_path: Path
    language: str
    language_probability: float
    duration: float
    segments: tuple[Segment, ...]

    @property
    def words(self) -> tuple[Word, ...]:
        return tuple(word for segment in self.segments for word in segment.words)

    @property
    def full_text(self) -> str:
        return " ".join(segment.text.strip() for segment in self.segments).strip()

    def gaps(self, *, min_seconds: float = 0.5) -> tuple[tuple[float, float], ...]:
        """Silences of at least `min_seconds` between consecutive words.

        Not silences in the *audio* sense (no energy analysis) — gaps between
        recognised speech. A long pause, a retake, or a stretch of only
        filler noise Whisper didn't transcribe as words all show up here the
        same way, which is exactly what a silence/retake-removal cut wants:
        "no useful speech happened during this span," not "the waveform was
        literally at zero."
        """
        words = self.words
        found = []
        for previous, current in zip(words, words[1:]):
            gap = current.start - previous.end
            if gap >= min_seconds:
                found.append((previous.end, current.start))
        return tuple(found)

    def summary(self) -> str:
        return (
            f"{self.language} ({self.language_probability:.0%}), "
            f"{self.duration:.2f}s, {len(self.segments)} segment(s), "
            f"{len(self.words)} word(s)"
        )


def _load_model(model_size: str, device: str, compute_type: str):
    key = (model_size, device, compute_type)
    if key not in _MODEL_CACHE:
        from faster_whisper import WhisperModel

        log.info("Loading Whisper model %r (%s/%s) ...", model_size, device, compute_type)
        try:
            _MODEL_CACHE[key] = WhisperModel(
                model_size, device=device, compute_type=compute_type
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as one clear error type
            raise TranscriptionError(
                f"Could not load Whisper model {model_size!r}: {exc}"
            ) from exc
    return _MODEL_CACHE[key]


def transcribe_audio(
    audio_path: Path | str,
    *,
    model_size: str = DEFAULT_MODEL_SIZE,
    language: str | None = None,
    device: str = "cpu",
    compute_type: str = "int8",
) -> Transcript:
    """Transcribe `audio_path` with word-level timestamps.

    `language` is a prior, not a constraint — pass an ISO code (`"en"`) to
    skip language detection when it's already known; leave it `None` to let
    Whisper detect it. `model_size` trades accuracy for speed/memory: "tiny"
    and "base" run comfortably on CPU with no GPU, "small"/"medium"/"large"
    are progressively slower and better, and only worth it with a GPU
    (`device="cuda"`) unless the audio is unusually hard to transcribe.
    """
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = _load_model(model_size, device, compute_type)

    try:
        raw_segments, info = model.transcribe(
            str(audio_path), word_timestamps=True, language=language,
        )
        raw_segments = list(raw_segments)
    except Exception as exc:  # noqa: BLE001 - one error type for callers to catch
        raise TranscriptionError(f"Transcription failed for {audio_path.name}: {exc}") from exc

    segments = tuple(
        Segment(
            text=segment.text.strip(),
            start=segment.start,
            end=segment.end,
            words=tuple(
                Word(
                    text=word.word.strip(),
                    start=word.start,
                    end=word.end,
                    probability=word.probability,
                )
                for word in (segment.words or ())
            ),
        )
        for segment in raw_segments
    )

    transcript = Transcript(
        audio_path=audio_path,
        language=info.language,
        language_probability=info.language_probability,
        duration=info.duration,
        segments=segments,
    )
    log.info("Transcribed %s: %s", audio_path.name, transcript.summary())
    return transcript


def _format_timestamp(seconds: float) -> str:
    """SRT's own timestamp format: HH:MM:SS,mmm."""
    total_ms = round(seconds * 1000)
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def to_srt(transcript: Transcript, *, mode: str = "segment") -> str:
    """Render `transcript` as an SRT caption file.

    `mode="segment"` — one caption per Whisper segment (the natural
    sentence/phrase grouping, what most caption styles want).
    `mode="word"` — one caption per word (word-level "karaoke" captions, or
    input to something like the numbers-counting SRT trick — a caption file
    is just a caption file regardless of what generated the numbers in it).
    """
    if mode not in ("segment", "word"):
        raise ValueError(f"Unknown SRT mode {mode!r}; expected 'segment' or 'word'.")

    entries = transcript.words if mode == "word" else transcript.segments
    lines = []
    for index, entry in enumerate(entries, start=1):
        lines.append(str(index))
        lines.append(
            f"{_format_timestamp(entry.start)} --> {_format_timestamp(entry.end)}"
        )
        lines.append(entry.text)
        lines.append("")
    return "\n".join(lines)
