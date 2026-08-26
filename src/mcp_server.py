"""MCP server exposing this pipeline as tools for any MCP-compatible client
(Claude Desktop, Claude Code, Cursor) to call directly — the same idea
Kaestral (github.com/prabindersinghh/Kaestral-pro) uses, deliberately
reimplemented from scratch rather than adapted from its GPL-3.0 source: the
value there is the *pattern* (expose editing primitives as MCP tools, let the
LLM be the brain), not the code, and our own engine underneath is stronger
for exactly the thing this project is for. See `skills/beat-sync-cutting/
SKILL.md` for the concrete reasoning — Kaestral's own equivalent skill has
its agent iterate cut placement one tool call at a time against a plain
tempo grid ("weaker on ambient/legato tracks", by its own admission); ours
delegates that to `plan_edit`/`render_edit`, which call the same deterministic,
render-verified assembler the CLI does. The tool's job is choosing *what*
to ask for, never re-deriving frame math an LLM is bad at doing reliably
across dozens of repeated calls.

    python src/mcp_server.py                # stdio transport

Claude Desktop / Claude Code config:
    {"mcpServers": {"video-pipeline": {"command": "python",
                     "args": ["/absolute/path/to/src/mcp_server.py"]}}}

Two tool tiers:
  * Read-only inspection (`list_clips`, `analyze_audio`, `plan_edit`,
    `list_looks`, `list_transitions`) run in-process — fast, no rendering,
    safe to call speculatively while the agent is still deciding.
  * `render_edit` shells out to `pipeline.py` through the exact same
    `orchestrator_cli.build_argv()` translation the HTTP API and the Studio
    UI already use, so a request built here, there, or by n8n all mean the
    same thing — one flag surface, three front doors.

Every tool function below is a plain, module-level, directly-callable
function — registered with the MCP server in `create_server()`, but never
*only* reachable through it. That is what makes `tests/test_mcp_server.py`
possible without standing up a full MCP session: it calls these the same
way `orchestrator_cli`'s own tests call `build_argv` directly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config  # noqa: E402
import orchestrator_cli  # noqa: E402
from audio.beat_detect import BeatDetectionError, detect_beats  # noqa: E402
from editing import assembler  # noqa: E402
from editing.assembler import PlanningError  # noqa: E402
from editing.looks import LOOKS  # noqa: E402
from editing.transitions import CUT, TRANSITIONS, VideoFormat  # noqa: E402
from ingest.local_clips import NoClipsFound, scan_folder  # noqa: E402

PROJECT_ROOT = config.PROJECT_ROOT
RENDER_TIMEOUT_SECONDS = 3600


def _settings():
    return config.load_settings(PROJECT_ROOT / "config" / "settings.yaml")


def list_clips(folder: str, recursive: bool = False) -> dict[str, Any]:
    """List usable video clips in a folder, with duration/resolution/fps.

    Call this before planning an edit — it's what plan_edit and
    render_edit's `clips` argument point at, and knowing what footage
    actually exists (durations especially) shapes what's reasonable to
    ask for: a 20s edit needs clips that add up to at least that much.
    """
    try:
        clips = scan_folder(Path(folder), recursive=recursive)
    except (NoClipsFound, NotADirectoryError, FileNotFoundError) as exc:
        return {"error": str(exc), "clips": []}

    return {
        "clips": [
            {
                "path": str(clip.path),
                "duration_seconds": round(clip.duration, 3),
                "width": clip.width,
                "height": clip.height,
                "fps": clip.fps,
                "has_audio": clip.has_audio,
            }
            for clip in clips
        ],
        "total_duration_seconds": round(sum(c.duration for c in clips), 3),
        "count": len(clips),
    }


def analyze_audio(
    audio_path: str,
    start_bpm: float | None = None,
    trim_silence: bool | None = None,
) -> dict[str, Any]:
    """Analyse a track's tempo, beat grid, downbeats, and energy sections.

    `start_bpm` is a prior, not a constraint — pass it only when the track
    keeps getting tracked at half or double tempo. Sections are labelled
    low/medium/high energy; that's what a good cut density
    (beats_per_cut in plan_edit/render_edit) should follow — hold longer
    through low-energy sections, cut faster through high ones. This runs
    librosa's beat tracker plus downbeat detection via agglomerative
    clustering, not a plain tempo grid — it distinguishes a quiet intro
    from a loud drop, which a bare BPM estimate cannot.
    """
    settings = _settings()
    try:
        grid = detect_beats(
            audio_path,
            start_bpm=start_bpm if start_bpm is not None
            else settings.get("beats.start_bpm", 120.0),
            tightness=float(settings.get("beats.tightness", 100.0)),
            beats_per_bar=int(settings.get("beats.beats_per_bar", 4)),
            trim_silence=trim_silence if trim_silence is not None
            else bool(settings.get("beats.trim_silence", True)),
        )
    except (BeatDetectionError, FileNotFoundError) as exc:
        return {"error": str(exc)}

    return {
        "bpm": round(grid.bpm, 2),
        "duration_seconds": round(grid.duration, 3),
        "beat_count": len(grid.beats),
        "beats": [round(b, 4) for b in grid.beats],
        "downbeats": [round(b, 4) for b in grid.downbeats],
        "sections": [
            {
                "start": round(s.start, 3), "end": round(s.end, 3),
                "label": s.label, "energy": round(s.energy, 3),
            }
            for s in grid.sections
        ],
        "summary": grid.summary(),
    }


def list_looks() -> list[str]:
    """Named colour-grade presets available to `--look`/render_edit."""
    return list(LOOKS)


def list_transitions() -> list[str]:
    """Every transition name available to `--transition`/render_edit —
    "cut", "crossfade", and ~58 ffmpeg xfade types (wipeleft, dissolve,
    pixelize, circleopen, zoomin, and more)."""
    return list(TRANSITIONS)


def plan_edit(
    clips: str,
    audio: str,
    duration: float | None = None,
    start: float = 0.0,
    seed: int | None = None,
    transition: str = CUT,
    width: int = 1920,
    height: int = 1080,
    fps: float = 30.0,
    beats_per_cut_low: int | None = None,
    beats_per_cut_medium: int | None = None,
    beats_per_cut_high: int | None = None,
) -> dict[str, Any]:
    """Plan the cut timeline WITHOUT rendering — see exactly which clip
    lands where before spending render time.

    Local clips only (no --brief/stock sourcing here; that needs a
    budget/cache context render_edit already carries). This calls the
    *exact same* `assembler.build_plan()` render_edit does, so what you
    see here is what you get — not a preview that might differ from the
    real render.
    """
    settings = _settings()
    try:
        clip_list = scan_folder(Path(clips))
    except (NoClipsFound, NotADirectoryError, FileNotFoundError) as exc:
        return {"error": str(exc)}

    try:
        grid = detect_beats(
            audio,
            start_bpm=settings.get("beats.start_bpm", 120.0),
            tightness=float(settings.get("beats.tightness", 100.0)),
            beats_per_bar=int(settings.get("beats.beats_per_bar", 4)),
            trim_silence=bool(settings.get("beats.trim_silence", True)),
        )
    except (BeatDetectionError, FileNotFoundError) as exc:
        return {"error": str(exc)}

    beats_per_cut = dict(
        settings.get("editing.beats_per_cut", {"low": 8, "medium": 4, "high": 2})
    )
    if beats_per_cut_low is not None:
        beats_per_cut["low"] = beats_per_cut_low
    if beats_per_cut_medium is not None:
        beats_per_cut["medium"] = beats_per_cut_medium
    if beats_per_cut_high is not None:
        beats_per_cut["high"] = beats_per_cut_high

    end = None if duration is None else start + duration
    try:
        plan = assembler.build_plan(
            clip_list, grid,
            video_format=VideoFormat(width, height, fps),
            beats_per_cut=beats_per_cut,
            min_shot_seconds=float(settings.get("editing.min_shot_seconds", 0.2)),
            max_shot_seconds=float(settings.get("editing.max_shot_seconds", 8.0)),
            snap_to_downbeats=bool(settings.get("editing.snap_to_downbeats", True)),
            transition=transition,
            crossfade_seconds=float(settings.get("editing.crossfade_seconds", 0.25)),
            clip_order=str(settings.get("editing.clip_order", "shuffle")),
            seed=seed, start=start, end=end,
        )
    except PlanningError as exc:
        return {"error": str(exc)}

    return {
        "total_duration_seconds": round(plan.total_duration, 3),
        "cut_count": len(plan.cut_times),
        "segments": [
            {
                "clip": str(segment.clip.path),
                "timeline_start": round(segment.timeline_start, 4),
                "duration": round(segment.duration, 4),
                "beat_time": round(segment.beat_time, 4),
            }
            for segment in plan.segments
        ],
        "summary": plan.summary(),
    }


def render_edit(
    audio: str,
    out: str,
    clips: str | None = None,
    brief: str = "",
    duration: float | None = None,
    start: float = 0.0,
    seed: int | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    transition: str | None = None,
    title: str = "",
    title_style: str | None = None,
    look: str | None = None,
    punch_zoom: float | None = None,
    glitch: int | None = None,
    speed_ramp: bool = False,
    ken_burns: float | None = None,
    verify: bool = False,
) -> dict[str, Any]:
    """Render the edit. Blocks until it finishes (or times out at 1h) — for
    a render expected to run long, use the HTTP orchestrator's async
    `wait:false` instead (src/orchestrator_cli.py, POST /render).

    This is the full pipeline: sourcing (local clips and/or --brief stock
    search), beat-synced cutting, and the common Phase 4 post effects (for
    full parameter coverage — LUTs, chroma key, Ken Burns pan, grain, etc.
    — drive the HTTP API directly or edit config/settings.yaml). Cost is
    always $0 unless paid_adapters_enabled is explicitly set in
    settings.yaml and a paid --generator is named.
    """
    spec = {
        "audio": audio, "out": out, "clips": clips, "brief": brief,
        "duration": duration, "start": start, "seed": seed,
        "width": width, "height": height, "fps": fps,
        "transition": transition, "title": title, "title_style": title_style,
        "look": look, "punch_zoom": punch_zoom, "glitch": glitch,
        "speed_ramp": speed_ramp, "ken_burns": ken_burns, "verify": verify,
    }
    output_root = Path(out).expanduser().resolve().parent
    try:
        argv, out_path = orchestrator_cli.build_argv(spec, output_root)
    except ValueError as exc:
        return {"status": "failed", "error": str(exc)}

    try:
        completed = subprocess.run(
            argv, cwd=str(PROJECT_ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=RENDER_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {"status": "failed", "error": f"timed out after {RENDER_TIMEOUT_SECONDS}s"}

    log = (completed.stdout or "") + (completed.stderr or "")
    return {
        "status": "succeeded" if completed.returncode == 0 else "failed",
        "out": str(out_path) if completed.returncode == 0 else None,
        "returncode": completed.returncode,
        "log_tail": "\n".join(log.splitlines()[-40:]),
    }


TOOLS = (list_clips, analyze_audio, list_looks, list_transitions, plan_edit, render_edit)


def create_server():
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        "video-pipeline",
        instructions=(
            "Local, free-first, beat-synced video editing. Typical flow: "
            "list_clips to see what footage is available, analyze_audio to "
            "get the track's tempo/beat grid/energy sections, plan_edit to "
            "see the cut timeline *before* spending render time, then "
            "render_edit with whichever look/transition/effects fit what "
            "was asked for. Read a skill under skills/ first if one matches "
            "the task — they carry judgment calls (cut density per energy "
            "level, which effects suit which genre) that aren't in any "
            "single tool's schema."
        ),
    )
    for tool in TOOLS:
        mcp.add_tool(tool)
    return mcp


if __name__ == "__main__":
    config.load_env()
    server = create_server()
    server.run()
