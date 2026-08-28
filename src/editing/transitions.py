"""ffmpeg filter-graph construction for joining segments together.

Kept separate from the assembler so that "what the timeline is" and "how ffmpeg
is told to build it" stay independent. The assembler decides durations; this
module only knows how to express them as a filter graph.

Two families of join are supported:

  cut        Hard cuts via the `concat` filter. Frame-exact, and the default:
             a beat-cut edit wants the picture to change *on* the beat, and any
             dissolve necessarily smears that moment.
  <xfade>    Every dissolve/wipe ffmpeg's own `xfade` filter knows how to do —
             all 58 of them, from a plain `fade` through wipes, slides, blurs
             and pixelizes. Picked from ffmpeg's own `-h filter=xfade` output
             rather than hand-copied, so the list can never drift from what the
             installed binary actually supports. Whichever one is named, the
             dissolve is centred on the beat (each segment is extended by half
             the transition at each join) so the beat grid holds rather than
             being progressively shortened.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from editing.speed_ramp import SpeedRamp

CUT = "cut"
CROSSFADE = "crossfade"  # kept as an alias for xfade's "fade" — the original name

# The full xfade transition library, in the order `ffmpeg -h filter=xfade`
# lists them. Hand-maintained rather than parsed from the binary at import
# time: parsing runs a subprocess on every import, and the list is part of
# ffmpeg's stable public filter API — it has not removed a transition since
# xfade was added. `available_transitions()` below is the version that trusts
# the installed binary, for anything that wants to be defensive about it.
XFADE_TRANSITIONS = (
    "fade", "wipeleft", "wiperight", "wipeup", "wipedown",
    "slideleft", "slideright", "slideup", "slidedown",
    "circlecrop", "rectcrop", "distance", "fadeblack", "fadewhite", "radial",
    "smoothleft", "smoothright", "smoothup", "smoothdown",
    "circleopen", "circleclose", "vertopen", "vertclose", "horzopen", "horzclose",
    "dissolve", "pixelize", "diagtl", "diagtr", "diagbl", "diagbr",
    "hlslice", "hrslice", "vuslice", "vdslice", "hblur", "fadegrays",
    "wipetl", "wipetr", "wipebl", "wipebr", "squeezeh", "squeezev", "zoomin",
    "fadefast", "fadeslow", "hlwind", "hrwind", "vuwind", "vdwind",
    "coverleft", "coverright", "coverup", "coverdown",
    "revealleft", "revealright", "revealup", "revealdown",
)

TRANSITIONS = (CUT, CROSSFADE) + XFADE_TRANSITIONS


@lru_cache(maxsize=1)
def available_transitions() -> tuple[str, ...]:
    """The transition names the *installed* ffmpeg actually accepts.

    `TRANSITIONS` above is what this module knows how to ask for; this is
    what will actually work on this machine. They should always match on a
    current ffmpeg — this exists so a version mismatch produces "here is what
    your ffmpeg supports" instead of a filter-graph error deep in stderr.
    Cached: it shells out once per process, not once per transition lookup.
    """
    from ffmpeg_tools import ffmpeg_bin

    try:
        result = subprocess.run(
            [ffmpeg_bin(), "-hide_banner", "-h", "filter=xfade"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return (CUT, CROSSFADE) + XFADE_TRANSITIONS

    names = tuple(
        line.split()[0]
        for line in result.stdout.splitlines()
        if line.strip().split() and line.split()[0] in XFADE_TRANSITIONS
    )
    return (CUT, CROSSFADE) + (names or XFADE_TRANSITIONS)


VIDEO_OUT = "outv"
AUDIO_OUT = "outa"

# The label the assembled-but-not-yet-graded stream is handed off under when
# post-processing (look/zoom/glitch) is attached. Shared with `effects.py` so
# a post-chain it builds always plugs into the same socket this module wires
# up — the two modules agree on this name rather than one importing the other.
PRE_POST_LABEL = "pre_out"


@dataclass(frozen=True)
class InputSpec:
    """One decoded source segment, as an entry in the ffmpeg input list.

    frames is authoritative: the assembler derives it from absolute timeline
    positions so that rounding cannot accumulate across the edit.
    """

    input_index: int
    frames: int
    # Seconds of frozen last frame to append when the source runs out early.
    pad_seconds: float = 0.0
    # See editing.speed_ramp.SpeedRamp. Disabled whenever pad_seconds > 0: the
    # source already ran short for this shot, and stacking "hold the last
    # frame" padding on top of "consume source at two different rates" is a
    # rare-enough combination that falling back to an unramped shot is worth
    # more than the complexity of supporting it.
    speed_ramp: "SpeedRamp | None" = None


@dataclass(frozen=True)
class VideoFormat:
    width: int
    height: int
    fps: float
    fit: str = "pad"            # "pad" letterboxes, "crop" fills
    pad_color: str = "black"
    pixel_format: str = "yuv420p"


def _scale_filters(fmt: VideoFormat) -> str:
    """Fit an arbitrary source into the output frame."""
    if fmt.fit == "crop":
        return (
            f"scale=w={fmt.width}:h={fmt.height}"
            f":force_original_aspect_ratio=increase:flags=bicubic,"
            f"crop={fmt.width}:{fmt.height}"
        )
    if fmt.fit != "pad":
        raise ValueError(f"Unknown video.fit {fmt.fit!r}; expected 'pad' or 'crop'.")
    return (
        f"scale=w={fmt.width}:h={fmt.height}"
        f":force_original_aspect_ratio=decrease:flags=bicubic,"
        f"pad={fmt.width}:{fmt.height}:(ow-iw)/2:(oh-ih)/2:color={fmt.pad_color}"
    )


def normalise_chain(spec: InputSpec, fmt: VideoFormat, label: str) -> str:
    """Bring one input to the output format and clamp it to an exact length.

    The trailing `trim=end_frame` is the reason cuts stay on the beat: it fixes
    the segment at an integer frame count, so a segment can never be a frame
    long or short and push every later cut off the grid. That holds whether or
    not a speed ramp ran first — the ramp changes how much *source* time the
    shot spans, never how many *output* frames it produces.
    """
    pre = [
        # Input seeking leaves a non-zero start PTS on some containers; concat
        # and xfade both assume streams begin at zero.
        "setpts=PTS-STARTPTS",
        _scale_filters(fmt),
        f"fps={format_number(fmt.fps)}",
        "setsar=1",
    ]
    tail = [f"trim=end_frame={spec.frames}", "setpts=PTS-STARTPTS", f"format={fmt.pixel_format}"]

    if spec.speed_ramp is not None and spec.pad_seconds <= 0 and spec.frames >= 2:
        from editing.speed_ramp import ramp_chain  # deferred: speed_ramp imports us

        pre_label, ramped_label = f"{label}_pre", f"{label}_ramped"
        lines = [f"[{spec.input_index}:v]" + ",".join(pre) + f"[{pre_label}]"]
        lines.extend(
            ramp_chain(
                spec.frames, fmt.fps, spec.speed_ramp,
                input_label=pre_label, output_label=ramped_label,
            )
        )
        lines.append(f"[{ramped_label}]" + ",".join(tail) + f"[{label}]")
        return ";\n".join(lines)

    filters = list(pre)
    if spec.pad_seconds > 0:
        # The source ran out before the beat did. Hold the last frame rather
        # than letting the segment come up short and drag every later cut early.
        filters.append(
            f"tpad=stop_mode=clone:stop_duration={format_number(spec.pad_seconds)}"
        )
    filters.extend(tail)
    return f"[{spec.input_index}:v]" + ",".join(filters) + f"[{label}]"


def build_video_graph(
    specs: list[InputSpec],
    fmt: VideoFormat,
    *,
    transition: str = CUT,
    crossfade_seconds: float = 0.25,
    post_chain: str | None = None,
) -> str:
    """Full video filter graph, ending in a stream labelled `outv`.

    `post_chain`, when given, is one or more already-complete filter-graph
    statements (semicolon-joined, each with its own `[in]...[out]` brackets)
    that consume `[pre_out]` and end by producing `[outv]` — this is where a
    colour-grade look, punch-in zoom, or glitch burst attaches. Handed in
    fully formed rather than built here because a punch-zoom needs its own
    `split`/`overlay` sub-graph, not just a linear chain a single label can
    carry; `effects.build_post_chain()` is what builds it. It runs once on
    the *assembled* timeline rather than once per segment, which matters for
    anything keyed to cut time: a punch-zoom window written in terms of
    absolute timeline seconds only makes sense once every segment has already
    been placed on that timeline.
    """
    if not specs:
        raise ValueError("Cannot build a video graph with no segments.")
    if transition not in TRANSITIONS:
        raise ValueError(
            f"Unknown transition {transition!r}; expected one of {TRANSITIONS}."
        )

    lines = [
        normalise_chain(spec, fmt, f"v{index}")
        for index, spec in enumerate(specs)
    ]

    pre_label = PRE_POST_LABEL if post_chain else VIDEO_OUT
    if len(specs) == 1:
        lines.append(f"[v0]null[{pre_label}]")
    elif transition == CUT:
        inputs = "".join(f"[v{index}]" for index in range(len(specs)))
        lines.append(f"{inputs}concat=n={len(specs)}:v=1:a=0[{pre_label}]")
    else:
        xfade_name = "fade" if transition == CROSSFADE else transition
        lines.extend(
            _xfade_chain(specs, fmt, crossfade_seconds, xfade_name, pre_label)
        )

    if post_chain:
        lines.append(post_chain)

    return ";\n".join(lines)


def _xfade_chain(
    specs: list[InputSpec],
    fmt: VideoFormat,
    crossfade_seconds: float,
    xfade_name: str = "fade",
    final_label: str = VIDEO_OUT,
) -> list[str]:
    """Chain xfade across every join.

    Each xfade shortens the running timeline by `crossfade_seconds`; the
    assembler compensates by extending segments, so the offsets computed here
    still land the midpoint of every dissolve on its beat. `xfade_name` picks
    which of ffmpeg's ~58 transitions plays at every join — the mechanics of
    keeping the beat grid intact are identical regardless of which one it is.
    """
    if crossfade_seconds <= 0:
        raise ValueError("crossfade_seconds must be positive for a crossfade.")

    durations = [spec.frames / fmt.fps for spec in specs]
    shortest = min(durations)
    if crossfade_seconds >= shortest:
        raise ValueError(
            f"crossfade_seconds ({crossfade_seconds:.3f}s) must be shorter than "
            f"the shortest segment ({shortest:.3f}s). Lower "
            f"editing.crossfade_seconds or raise editing.min_shot_seconds."
        )

    lines: list[str] = []
    current_label = "v0"
    running_length = durations[0]

    for index in range(1, len(specs)):
        offset = running_length - crossfade_seconds
        is_last = index == len(specs) - 1
        output_label = final_label if is_last else f"x{index}"
        lines.append(
            f"[{current_label}][v{index}]"
            f"xfade=transition={xfade_name}"
            f":duration={format_number(crossfade_seconds)}"
            f":offset={format_number(offset)}"
            f"[{output_label}]"
        )
        running_length = running_length + durations[index] - crossfade_seconds
        current_label = output_label

    return lines


def build_audio_graph(
    input_index: int,
    *,
    start: float,
    duration: float,
    fade_out_seconds: float = 0.0,
) -> str:
    """Trim the music track to the edit and optionally fade it out.

    `start` matters when the edit does not begin at the top of the track: the
    beat grid is in absolute audio time, so the audio has to be shifted by the
    same amount the video timeline was.
    """
    filters = [
        f"atrim=start={format_number(start)}:end={format_number(start + duration)}",
        "asetpts=PTS-STARTPTS",
    ]
    if fade_out_seconds > 0 and duration > fade_out_seconds:
        filters.append(
            f"afade=t=out"
            f":st={format_number(duration - fade_out_seconds)}"
            f":d={format_number(fade_out_seconds)}"
        )
    return f"[{input_index}:a]" + ",".join(filters) + f"[{AUDIO_OUT}]"


def format_number(value: float) -> str:
    """Fixed-point formatting so ffmpeg never sees scientific notation.

    ffmpeg's option parser rejects values like `1e-05`, which is exactly the
    form Python's repr picks for small crossfade or offset values.
    """
    return f"{float(value):.6f}".rstrip("0").rstrip(".") or "0"
