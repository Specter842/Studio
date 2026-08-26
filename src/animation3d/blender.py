"""Headless Blender, for 3D elements composited into the timeline.

Free, offline, no API and no per-render cost — the same rule the rest of the
default path follows.

Blender is driven as a subprocess rather than imported:

    blender --background --factory-startup --python <script> -- <args.json>

`bpy` only exists inside Blender's own bundled Python, so it cannot be a pip
dependency of this project. Keeping it behind a subprocess boundary also means a
Blender crash or a version incompatibility fails one element instead of taking
the whole render down with it.

Elements are rendered as **RGBA PNG sequences**, not video. Blender's video
output paths lose the alpha channel in most container/codec combinations, and
an image sequence is something ffmpeg composites directly and losslessly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent / "blender_scripts"

# Blender writes `<filepath><frame padded to 4>.png` by default.
FRAME_PREFIX = "frame_"
FRAME_PATTERN = "frame_%04d.png"

# Bump when text_reveal.py changes what it produces, so cached frames rendered
# by an older version are not reused.
TEXT_REVEAL_VERSION = 3

# Where Blender usually lands per platform. Checked only after BLENDER_BIN and
# PATH, and the highest version found wins.
_SEARCH_DIRS = (
    r"C:\Program Files\Blender Foundation",
    r"C:\Program Files (x86)\Blender Foundation",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Blender Foundation"),
    "/Applications",
    "/usr/share",
    "/opt",
)


class BlenderNotFound(RuntimeError):
    """Blender is not installed, or not where we looked."""


class BlenderError(RuntimeError):
    """A Blender run exited non-zero."""


@dataclass(frozen=True)
class Element:
    """A rendered 3D element: a directory of RGBA frames."""

    directory: Path
    fps: float
    frames: int
    width: int
    height: int
    pattern: str = FRAME_PATTERN

    @property
    def duration(self) -> float:
        return self.frames / self.fps if self.fps else 0.0

    @property
    def input_path(self) -> str:
        """What to hand ffmpeg as an image-sequence input."""
        return str(self.directory / self.pattern)

    def __str__(self) -> str:
        return (
            f"{self.frames} frames at {self.width}x{self.height} "
            f"({self.duration:.2f}s) in {self.directory.name}"
        )


def blender_bin() -> str:
    """Locate the Blender executable."""
    override = os.environ.get("BLENDER_BIN", "").strip()
    if override:
        if not Path(override).is_file():
            raise BlenderNotFound(f"BLENDER_BIN={override!r} is not a file.")
        return override

    found = shutil.which("blender")
    if found:
        return found

    candidates: list[tuple[tuple[int, ...], str]] = []
    for directory in _SEARCH_DIRS:
        base = Path(directory)
        if not base.is_dir():
            continue
        for entry in base.glob("[Bb]lender*"):
            for relative in ("blender.exe", "blender", "Contents/MacOS/Blender"):
                executable = entry / relative
                if executable.is_file():
                    candidates.append((_version_key(entry.name), str(executable)))
    if candidates:
        # Newest install wins, so an old 2.x left behind does not shadow a 4.x.
        return max(candidates)[1]

    raise BlenderNotFound(
        "Blender was not found. Install it from https://www.blender.org/download/ "
        "(free), then either add it to PATH or set BLENDER_BIN in .env. "
        "3D elements are optional — the rest of the pipeline runs without it."
    )


def _version_key(name: str) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", name)
    return tuple(int(number) for number in numbers) or (0,)


def have_blender() -> bool:
    try:
        blender_bin()
    except BlenderNotFound:
        return False
    return True


def blender_version() -> str:
    output = subprocess.run(
        [blender_bin(), "--version"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=120,
    ).stdout
    return output.splitlines()[0].strip() if output else "unknown"


def run_script(
    script: Path | str,
    arguments: dict[str, Any],
    *,
    timeout: float = 900.0,
) -> str:
    """Run a bpy script headlessly and return its stdout.

    Arguments go through a JSON file rather than the command line: Blender
    passes everything after `--` straight through, and quoting a JSON blob
    through a Windows shell is a reliable source of silent corruption.
    """
    script_path = Path(script)
    if not script_path.is_absolute():
        script_path = SCRIPTS_DIR / script_path
    if not script_path.is_file():
        raise BlenderError(f"Blender script not found: {script_path}")

    with tempfile.TemporaryDirectory(prefix="blender_args_") as workdir:
        args_path = Path(workdir) / "args.json"
        args_path.write_text(json.dumps(arguments), encoding="utf-8")

        command = [
            blender_bin(),
            "--background",
            # Ignore user preferences and add-ons: a headless render must not
            # depend on how somebody's interactive Blender happens to be set up.
            "--factory-startup",
            "--python", str(script_path),
            "--", str(args_path),
        ]
        log.debug("blender: %s", " ".join(command))

        completed = subprocess.run(
            command,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout,
        )

    if completed.returncode != 0:
        raise BlenderError(
            f"Blender exited {completed.returncode} running {script_path.name}\n"
            + _tail(completed.stdout, completed.stderr)
        )
    return completed.stdout


def _tail(stdout: str, stderr: str, lines: int = 25) -> str:
    """Blender puts tracebacks on stdout, so both streams matter."""
    combined = "\n".join(part for part in (stdout, stderr) if part).strip()
    return "\n".join(combined.splitlines()[-lines:])


def render_text_reveal(
    text: str,
    *,
    cache_dir: Path | str,
    width: int = 1920,
    height: int = 1080,
    fps: float = 30.0,
    seconds: float = 2.5,
    color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    style: str = "metal",
    timeout: float = 900.0,
) -> Element:
    """Render an animated 3D text element on a transparent background.

    Cached on everything that changes the pixels — a Blender render costs real
    seconds, and re-running the pipeline on the same title should not pay for
    it twice.
    """
    settings: dict[str, Any] = {
        "text": text,
        "width": int(width),
        "height": int(height),
        "fps": float(fps),
        "seconds": float(seconds),
        "color": [float(channel) for channel in color],
        "style": style,
    }

    # The script version is part of the key so that changing how an element
    # looks invalidates everything rendered by the old version, instead of
    # silently serving stale frames from the cache.
    digest = hashlib.sha256(
        json.dumps(
            {**settings, "_script": TEXT_REVEAL_VERSION}, sort_keys=True
        ).encode("utf-8")
    ).hexdigest()[:16]
    # Absolute, always. Blender resolves relative render paths against the
    # .blend file, and a factory-startup scene has no .blend file — so a
    # relative path silently writes somewhere else and the run "succeeds" with
    # zero frames.
    out_dir = (Path(cache_dir) / f"text_{digest}").resolve()

    frames = max(1, round(seconds * fps))
    if _is_complete(out_dir, frames):
        log.info("Blender cache hit: %s", out_dir.name)
        return Element(out_dir, fps, frames, int(width), int(height))

    out_dir.mkdir(parents=True, exist_ok=True)
    settings["output_prefix"] = str(out_dir / FRAME_PREFIX)

    log.info("Rendering 3D text %r with Blender (%d frames)", text, frames)
    output = run_script("text_reveal.py", settings, timeout=timeout)
    log.debug("blender stdout:\n%s", output)

    rendered = _count_frames(out_dir)
    if rendered == 0:
        raise BlenderError(
            f"Blender wrote no frames to {out_dir}. Run with -v to see its output."
        )
    if rendered != frames:
        log.warning("Expected %d frames, Blender wrote %d.", frames, rendered)

    element = Element(out_dir, fps, rendered, int(width), int(height))
    log.info("Rendered %s", element)
    return element


def _count_frames(directory: Path) -> int:
    return len(list(directory.glob(f"{FRAME_PREFIX}*.png")))


def _is_complete(directory: Path, expected: int) -> bool:
    return directory.is_dir() and _count_frames(directory) == expected
