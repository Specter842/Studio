"""Thin, shared wrapper around the ffmpeg/ffprobe binaries.

Every module that touches media goes through here so there is exactly one place
that knows how to find the binaries and how to report their failures. We shell
out to ffmpeg directly rather than using moviepy: moviepy re-encodes through
Python and is dramatically slower once a timeline has more than a handful of
segments.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

log = logging.getLogger(__name__)


class FFmpegNotFound(RuntimeError):
    """ffmpeg or ffprobe could not be located."""


class FFmpegError(RuntimeError):
    """An ffmpeg/ffprobe invocation exited non-zero."""

    def __init__(self, command: Sequence[str], returncode: int, stderr: str) -> None:
        self.command = list(command)
        self.returncode = returncode
        self.stderr = stderr
        # ffmpeg's real error is almost always in the last few lines; the rest is
        # the build banner. Surface the tail so tracebacks stay readable.
        tail = "\n".join(stderr.strip().splitlines()[-15:])
        super().__init__(
            f"{Path(self.command[0]).name} exited {returncode}\n{tail}"
        )


def _resolve(env_var: str, exe_name: str) -> str:
    """Find a binary via .env override first, then PATH."""
    override = os.environ.get(env_var, "").strip()
    if override:
        if not Path(override).is_file():
            raise FFmpegNotFound(
                f"{env_var}={override!r} but that file does not exist."
            )
        return override

    found = shutil.which(exe_name)
    if not found:
        raise FFmpegNotFound(
            f"{exe_name} was not found on PATH and {env_var} is not set in .env.\n"
            f"Install it (Windows: `winget install Gyan.FFmpeg`, "
            f"macOS: `brew install ffmpeg`, Debian/Ubuntu: `apt install ffmpeg`) "
            f"and reopen your shell."
        )
    return found


def ffmpeg_bin() -> str:
    return _resolve("FFMPEG_BIN", "ffmpeg")


def ffprobe_bin() -> str:
    return _resolve("FFPROBE_BIN", "ffprobe")


def have_ffmpeg() -> bool:
    """True when both binaries are usable. Used by tests to skip cleanly."""
    try:
        ffmpeg_bin()
        ffprobe_bin()
    except FFmpegNotFound:
        return False
    return True


def run(command: Sequence[str], *, timeout: float | None = None) -> str:
    """Run a binary, returning stdout. Raises FFmpegError on a non-zero exit."""
    log.debug("exec: %s", " ".join(str(part) for part in command))
    completed = subprocess.run(
        [str(part) for part in command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise FFmpegError(command, completed.returncode, completed.stderr or "")
    return completed.stdout


def run_with_progress(
    command: Sequence[str],
    *,
    total_seconds: float | None = None,
    timeout: float | None = None,
) -> None:
    """Run ffmpeg, logging progress as it goes.

    A long render with no output looks indistinguishable from a hang, so this
    asks ffmpeg for machine-readable progress on stdout and merges stderr into
    the same stream — one pipe, so there is no way to deadlock on a full buffer
    the way two separately-drained pipes can.
    """
    command = [str(part) for part in command]
    # Global options, so they are valid immediately after the binary name.
    command = command[:1] + ["-v", "error", "-nostats", "-progress", "pipe:1"] + command[1:]
    log.debug("exec: %s", " ".join(command))

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    # ffmpeg puts the real error in the last handful of lines; keep a window of
    # them rather than the whole log, which is mostly progress key=value pairs.
    recent: list[str] = []
    last_reported = -1.0
    started = time.monotonic()

    assert process.stdout is not None
    try:
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue

            if line.startswith("out_time_us=") and total_seconds:
                elapsed = _parse_out_time_us(line)
                # One line per 10% so a long render leaves a readable log.
                if elapsed is not None and elapsed - last_reported >= total_seconds / 10:
                    last_reported = elapsed
                    log.info(
                        "  encoding %5.1f%% (%.1fs / %.1fs)",
                        min(100.0, 100.0 * elapsed / total_seconds),
                        elapsed, total_seconds,
                    )
                continue

            if not _is_progress_line(line):
                recent.append(line)
                del recent[:-40]

            if timeout is not None and time.monotonic() - started > timeout:
                process.kill()
                raise subprocess.TimeoutExpired(command, timeout)
    finally:
        process.stdout.close()

    returncode = process.wait()
    if returncode != 0:
        raise FFmpegError(command, returncode, "\n".join(recent))


# Keys emitted by `-progress`. Anything else on the stream is diagnostic output
# worth keeping for the error message.
_PROGRESS_KEYS = frozenset({
    "frame", "fps", "stream_0_0_q", "bitrate", "total_size", "out_time_us",
    "out_time_ms", "out_time", "dup_frames", "drop_frames", "speed", "progress",
})


def _is_progress_line(line: str) -> bool:
    key, separator, _ = line.partition("=")
    return bool(separator) and key.strip() in _PROGRESS_KEYS


def _parse_out_time_us(line: str) -> float | None:
    try:
        return int(line.split("=", 1)[1]) / 1_000_000.0
    except (IndexError, ValueError):
        return None


def probe_json(path: Path) -> dict:
    """Full ffprobe dump for a media file, as parsed JSON."""
    out = run(
        [
            ffprobe_bin(),
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    return json.loads(out)


def parse_fraction(value: str | None) -> float:
    """Parse ffprobe's rational strings ('30000/1001') into a float.

    Returns 0.0 for the missing/zero-denominator cases ffprobe emits for
    streams with no meaningful frame rate.
    """
    if not value:
        return 0.0
    if "/" in value:
        numerator, _, denominator = value.partition("/")
        try:
            den = float(denominator)
            if den == 0:
                return 0.0
            return float(numerator) / den
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


@dataclass(frozen=True)
class AudioInfo:
    """What the pipeline needs to know about the music track."""

    path: Path
    duration: float
    sample_rate: int
    channels: int


def probe_audio(path: Path) -> AudioInfo:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio track not found: {path}")

    data = probe_json(path)
    streams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    if not streams:
        raise FFmpegError(
            ["ffprobe", str(path)], 1, f"{path.name} contains no audio stream."
        )
    stream = streams[0]

    duration = float(
        stream.get("duration") or data.get("format", {}).get("duration") or 0.0
    )
    return AudioInfo(
        path=path,
        duration=duration,
        sample_rate=int(stream.get("sample_rate") or 0),
        channels=int(stream.get("channels") or 0),
    )
