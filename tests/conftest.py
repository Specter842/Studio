"""Shared fixtures.

All test media is synthesised at session start rather than committed: a few
seconds of solid-colour video and a click track cost well under a second to
generate, keep the repo free of binaries, and make the beat positions exactly
known — which is what lets the accuracy test assert a number instead of an
impression.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config  # noqa: E402

# So FFMPEG_BIN / FFPROBE_BIN overrides in .env apply to the suite too, not
# only to runs that go through pipeline.py.
config.load_env()

from ffmpeg_tools import ffmpeg_bin, have_ffmpeg  # noqa: E402

# The click track is deliberately unambiguous: an exact tempo, a hard transient
# on every beat, and an accent on the first beat of each bar.
TEST_BPM = 120.0
TEST_BEATS = 32                      # 8 bars
TEST_SAMPLE_RATE = 22050
CLIP_SECONDS = 4.0
CLIP_WIDTH, CLIP_HEIGHT = 640, 360
CLIP_FPS = 30
CLIP_COLORS = ("red", "green", "blue", "yellow")


requires_ffmpeg = pytest.mark.skipif(
    not have_ffmpeg(),
    reason="ffmpeg/ffprobe not on PATH (or FFMPEG_BIN/FFPROBE_BIN unset in .env)",
)


def frame_metric_track(
    path: Path, *, metrics: tuple[str, ...] = ("YAVG",), crop: str | None = None
) -> dict[str, list[float]]:
    """Per-frame `signalstats` values for every frame in `path`, in order.

    One pass over the whole file, no `-ss` anywhere. Re-seeking a short,
    single-keyframe test clip to hit one exact instant turned out to be
    unreliable in practice — two independently "correct-looking" `-ss`
    invocations against the same file and timestamp read back different
    frames (both an input-seek and an accurate-seek output variant). Decoding
    the file start-to-finish and indexing by frame number sidesteps the whole
    question: there is no seek to get subtly wrong.
    """
    vf = f"{crop}," if crop else ""
    output = subprocess.run(
        [
            ffmpeg_bin(), "-hide_banner", "-i", str(path),
            "-vf", f"{vf}signalstats,metadata=print:file=-",
            "-f", "null", "-",
        ],
        capture_output=True, text=True,
    ).stdout

    tracks: dict[str, list[float]] = {metric: [] for metric in metrics}
    for line in output.splitlines():
        for metric in metrics:
            marker = f"signalstats.{metric}="
            if marker in line:
                tracks[metric].append(float(line.rsplit("=", 1)[1]))
    return tracks


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("media")


@pytest.fixture(scope="session")
def sample_clips(media_dir: Path) -> list[Path]:
    """Four short clips, each a distinct colour with a moving white box.

    Distinct colours matter: the accuracy test finds cuts with ffmpeg's scene
    detector, and it can only confirm a cut it can actually see.
    """
    if not have_ffmpeg():
        pytest.skip("ffmpeg not available")

    clips_dir = media_dir / "clips"
    clips_dir.mkdir(exist_ok=True)

    paths = []
    for index, color in enumerate(CLIP_COLORS):
        path = clips_dir / f"{index:02d}_{color}.mp4"
        if not path.exists():
            subprocess.run(
                [
                    ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
                    "-f", "lavfi",
                    "-i",
                    f"color=c={color}:s={CLIP_WIDTH}x{CLIP_HEIGHT}"
                    f":r={CLIP_FPS}:d={CLIP_SECONDS}",
                    # A moving box keeps the clip from being a single static
                    # frame, which would encode to something unrealistic.
                    "-vf",
                    f"drawbox=x='mod(t*160\\,{CLIP_WIDTH})':y=140:w=80:h=80"
                    f":color=white:t=fill",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(path),
                ],
                check=True,
                capture_output=True,
            )
        paths.append(path)
    return paths


@pytest.fixture(scope="session")
def click_track(media_dir: Path) -> Path:
    """A 120 BPM click track with a bar accent, written as a WAV."""
    import soundfile

    path = media_dir / "click_120bpm.wav"
    if path.exists():
        return path

    beat_seconds = 60.0 / TEST_BPM
    total_samples = int(TEST_SAMPLE_RATE * TEST_BEATS * beat_seconds)
    times = np.arange(total_samples) / TEST_SAMPLE_RATE

    # A quiet bass drone so RMS energy is non-zero between clicks, which is
    # what the section detector measures.
    audio = 0.05 * np.sin(2 * np.pi * 55.0 * times)

    click_samples = int(0.05 * TEST_SAMPLE_RATE)
    envelope = np.exp(-np.linspace(0.0, 12.0, click_samples))
    click_times = np.arange(click_samples) / TEST_SAMPLE_RATE

    for beat in range(TEST_BEATS):
        start = int(beat * beat_seconds * TEST_SAMPLE_RATE)
        is_downbeat = beat % 4 == 0
        frequency = 1800.0 if is_downbeat else 1200.0
        amplitude = 0.9 if is_downbeat else 0.45
        tone = amplitude * envelope * np.sin(2 * np.pi * frequency * click_times)
        audio[start:start + click_samples] += tone

    soundfile.write(path, np.clip(audio, -1.0, 1.0), TEST_SAMPLE_RATE)
    return path


@pytest.fixture(scope="session")
def structured_track(media_dir: Path) -> Path:
    """A click track with real dynamics: quiet, loud, quiet.

    The flat click track cannot tell a correct energy reading from an inverted
    one — every beat is the same. This one has a soft intro (beats 0-7), a busy
    middle with hats and snares (8-23) and a soft outro (24-31), so section
    labelling and cut density have something to actually get right.
    """
    import soundfile

    path = media_dir / "structured_120bpm.wav"
    if path.exists():
        return path

    beat_seconds = 60.0 / TEST_BPM
    total = int(TEST_SAMPLE_RATE * TEST_BEATS * beat_seconds)
    audio = 0.02 * np.sin(
        2 * np.pi * 55.0 * np.arange(total) / TEST_SAMPLE_RATE
    )

    def add(start_seconds: float, seconds: float, freq: float, amp: float, decay: float):
        length = min(int(seconds * TEST_SAMPLE_RATE), total - int(start_seconds * TEST_SAMPLE_RATE))
        if length <= 0:
            return
        offset = int(start_seconds * TEST_SAMPLE_RATE)
        envelope = np.exp(-np.linspace(0.0, decay, length))
        tone = np.sin(2 * np.pi * freq * np.arange(length) / TEST_SAMPLE_RATE)
        audio[offset:offset + length] += amp * envelope * tone

    for beat in range(TEST_BEATS):
        loud = 8 <= beat < 24
        level = 1.0 if loud else 0.25
        start = beat * beat_seconds
        add(start, 0.18, 60.0, 0.85 * level, 9.0)              # kick
        if loud:
            add(start + beat_seconds / 2, 0.05, 6000.0, 0.20, 22.0)   # hat
            if beat % 4 == 2:
                add(start, 0.12, 220.0, 0.40, 12.0)                   # snare

    soundfile.write(path, np.clip(audio, -1.0, 1.0), TEST_SAMPLE_RATE)
    return path


@pytest.fixture(scope="session")
def structured_grid(structured_track: Path):
    from audio.beat_detect import detect_beats

    return detect_beats(structured_track, start_bpm=TEST_BPM)


@pytest.fixture(scope="session")
def beat_grid(click_track: Path):
    """The click track analysed once and reused; librosa import is not cheap."""
    from audio.beat_detect import detect_beats

    return detect_beats(click_track, start_bpm=TEST_BPM)


@pytest.fixture
def settings():
    import config

    return config.load_settings(PROJECT_ROOT / "config" / "settings.yaml")


def make_settings(**overrides):
    """A Settings object from a plain nested dict, for adapter tests.

    Keeps adapter tests independent of whatever config/settings.yaml currently
    says, so tuning a default cannot silently change what a test asserts.
    """
    import config

    data = {"generator_default": "local_comfyui", "paid_adapters_enabled": False}
    data.update(overrides)
    return config.Settings(data, source=Path("<test>"))


@pytest.fixture(scope="session")
def tiny_video_bytes(media_dir: Path) -> bytes:
    """A real, small MP4 — what the mock servers hand back as a download."""
    if not have_ffmpeg():
        pytest.skip("ffmpeg not available")

    path = media_dir / "tiny.mp4"
    if not path.exists():
        subprocess.run(
            [
                ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=s=320x180:r=24:d=3",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
            ],
            check=True, capture_output=True,
        )
    return path.read_bytes()


@pytest.fixture(scope="session")
def tiny_image_bytes(media_dir: Path) -> bytes:
    """A real PNG, for the image-model path through the paid adapter."""
    if not have_ffmpeg():
        pytest.skip("ffmpeg not available")

    path = media_dir / "tiny.png"
    if not path.exists():
        subprocess.run(
            [
                ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=s=640x360",
                "-frames:v", "1", str(path),
            ],
            check=True, capture_output=True,
        )
    return path.read_bytes()
