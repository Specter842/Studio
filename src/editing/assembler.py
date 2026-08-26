"""Turn clips + a beat grid into a rendered video.

Two distinct jobs, kept in that order:

  1. Planning  — decide which beats become cuts, then which clip and which
     part of it fills each resulting shot. Pure computation, no ffmpeg, fully
     testable and inspectable via `--dry-run`.
  2. Rendering — hand the plan to a single ffmpeg process. One process, one
     decode pass per segment, no intermediate files.

The accuracy rule that everything else follows from: cut positions are derived
from *absolute* timeline positions quantised to the frame grid, never by adding
up segment durations. Summing rounded durations lets a fraction of a frame of
error accumulate at every cut, which is how a beat-cut edit ends up visibly
behind the music by the last chorus.
"""

from __future__ import annotations

import logging
import random
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from audio.beat_detect import BeatGrid
from editing import effects, transitions
from editing.speed_ramp import SpeedRamp
from editing.speed_ramp import source_seconds_needed as ramp_source_seconds_needed
from editing.transitions import CROSSFADE, CUT, InputSpec, VideoFormat
from ffmpeg_tools import FFmpegError, ffmpeg_bin, run_with_progress
from ingest.local_clips import ClipInfo

log = logging.getLogger(__name__)


class PlanningError(RuntimeError):
    """The edit could not be planned from the given clips and audio."""


@dataclass(frozen=True)
class Segment:
    """One shot on the timeline."""

    clip: ClipInfo
    source_in: float        # seconds into the source clip
    read_seconds: float     # how much to ask ffmpeg to decode (>= duration)
    frames: int             # exact output frame count
    timeline_start: float   # seconds from the start of the output
    duration: float         # frames / fps
    pad_seconds: float      # frozen-frame padding when the source runs short
    beat_time: float        # the absolute audio time this shot's cut sits on
    # None whenever the ramp effect is off, or this particular shot needed
    # padding — a source that already ran short cannot also spend footage at
    # two different rates.
    speed_ramp: "SpeedRamp | None" = None

    @property
    def timeline_end(self) -> float:
        return self.timeline_start + self.duration


@dataclass(frozen=True)
class Effects:
    """Post-processing applied once, to the assembled timeline.

    Disabled by construction: every field defaults to "off", so a plan built
    without naming any of this renders byte-identical to how it always did.
    """

    look: str = "none"
    lut_path: str | None = None
    grade_lift: tuple[float, float, float] = (0.0, 0.0, 0.0)
    grade_gamma: tuple[float, float, float] = (0.0, 0.0, 0.0)
    grade_gain: tuple[float, float, float] = (0.0, 0.0, 0.0)
    grade_saturation: float = 1.0
    grade_contrast: float = 1.0
    grade_brightness: float = 0.0
    denoise_strength: float = 0.0
    ken_burns_amount: float = 0.0
    ken_burns_steps: int = 8
    ken_burns_direction: str = "in"
    ken_burns_alternate: bool = False
    ken_burns_pan: tuple[float, float] = (0.0, 0.0)
    light_leak_opacity: float = 0.0
    light_leak_color: tuple[int, int, int] = (255, 160, 40)
    light_leak_center: tuple[float, float] = (0.85, 0.15)
    chroma_key_color: str | None = None
    chroma_key_similarity: float = 0.18
    chroma_key_background: str = "0x000000"
    temporal_blend_frames: int = 1
    punch_zoom_amount: float = 0.0
    punch_zoom_seconds: float = 0.22
    punch_zoom_every_nth: int = 1
    glitch_shift_px: int = 0
    glitch_seconds: float = 0.08
    glitch_every_nth: int = 1
    grain_strength: int = 0
    sharpen_amount: float = 0.0
    vignette_strength: float = 0.0
    speed_ramp_enabled: bool = False
    speed_ramp_fraction: float = 0.45
    speed_ramp_slow_factor: float = 0.5
    speed_ramp_fast_factor: float = 1.3

    @property
    def is_active(self) -> bool:
        return bool(
            (self.look and self.look != "none")
            or self.lut_path
            or any(self.grade_lift) or any(self.grade_gamma) or any(self.grade_gain)
            or self.grade_saturation != 1.0 or self.grade_contrast != 1.0
            or self.grade_brightness != 0.0
            or self.denoise_strength > 0
            or self.ken_burns_amount > 0
            or self.light_leak_opacity > 0
            or self.chroma_key_color
            or self.temporal_blend_frames > 1
            or self.punch_zoom_amount > 0
            or self.glitch_shift_px > 0
            or self.grain_strength > 0
            or self.sharpen_amount != 0.0
            or self.vignette_strength > 0
            or self.speed_ramp_enabled
        )

    @property
    def speed_ramp(self) -> "SpeedRamp | None":
        if not self.speed_ramp_enabled:
            return None
        return SpeedRamp(
            fraction=self.speed_ramp_fraction,
            slow_factor=self.speed_ramp_slow_factor,
            fast_factor=self.speed_ramp_fast_factor,
        )


@dataclass(frozen=True)
class EditPlan:
    segments: tuple[Segment, ...]
    audio_path: Path
    audio_start: float
    video_format: VideoFormat
    transition: str
    crossfade_seconds: float
    audio_fade_out_seconds: float
    total_duration: float
    effects: Effects = Effects()

    @property
    def cut_times(self) -> tuple[float, ...]:
        """Timeline positions where the picture changes (excluding 0)."""
        return tuple(segment.timeline_start for segment in self.segments[1:])

    @property
    def beat_targets(self) -> tuple[float, ...]:
        """Absolute audio times of the beats those cuts were placed on."""
        return tuple(segment.beat_time for segment in self.segments[1:])

    @property
    def local_cut_times(self) -> tuple[float, ...]:
        """Cut positions relative to this plan's own first segment.

        Identical to `cut_times` for the whole edit, since the assembled
        video stream always starts its own PTS at local 0 regardless of
        `audio_start`. Not identical for a *chunk* of a longer edit passed
        through `replace()` during chunked rendering: its segments keep their
        original absolute `timeline_start`, but the pass's own rendered video
        restarts its PTS at 0 — a post-effect keyed to cut time has to be
        expressed in that pass-local time, or it looks for a cut that, from
        inside this pass, never happens.
        """
        if not self.segments:
            return ()
        origin = self.segments[0].timeline_start
        return tuple(segment.timeline_start - origin for segment in self.segments[1:])

    @property
    def local_shot_spans(self) -> tuple[tuple[float, float], ...]:
        """(start, end) for every segment, in the same pass-local time as
        `local_cut_times` — what a Ken Burns move needs, since it animates
        across a shot's *whole* span rather than reacting to the instant of
        a cut.
        """
        if not self.segments:
            return ()
        origin = self.segments[0].timeline_start
        return tuple(
            (segment.timeline_start - origin, segment.timeline_end - origin)
            for segment in self.segments
        )

    def summary(self) -> str:
        clip_names = {segment.clip.path.name for segment in self.segments}
        padded = sum(1 for segment in self.segments if segment.pad_seconds > 0)
        shortest = min(segment.duration for segment in self.segments)
        longest = max(segment.duration for segment in self.segments)
        lines = [
            f"{len(self.segments)} segments from {len(clip_names)} clip(s)",
            f"total {self.total_duration:.2f}s at "
            f"{self.video_format.width}x{self.video_format.height} "
            f"@{self.video_format.fps:g}fps",
            f"shot length {shortest:.2f}s - {longest:.2f}s",
            f"transition: {self.transition}",
        ]
        if padded:
            lines.append(f"{padded} segment(s) hold a frozen final frame")
        if self.effects.is_active:
            parts = []
            if self.effects.look != "none":
                parts.append(f"look={self.effects.look}")
            if self.effects.punch_zoom_amount > 0:
                parts.append(f"punch-zoom={self.effects.punch_zoom_amount:.0%}")
            if self.effects.glitch_shift_px > 0:
                parts.append(f"glitch={self.effects.glitch_shift_px}px")
            if self.effects.speed_ramp_enabled:
                ramped = sum(1 for s in self.segments if s.speed_ramp is not None)
                parts.append(f"speed-ramp={ramped}/{len(self.segments)} shots")
            lines.append("effects: " + ", ".join(parts))
        return "\n  ".join(lines)

    def describe_segments(self) -> str:
        rows = []
        for index, segment in enumerate(self.segments):
            rows.append(
                f"  {index:>3}  {segment.timeline_start:>7.3f}s "
                f"+{segment.duration:>5.3f}s  "
                f"beat@{segment.beat_time:>7.3f}s  "
                f"[{segment.clip.origin}] "
                f"{segment.clip.path.name} @{segment.source_in:.2f}s"
            )
        return "\n".join(rows)


# --------------------------------------------------------------------------
# 1. Which beats become cuts
# --------------------------------------------------------------------------

def choose_cut_times(
    grid: BeatGrid,
    *,
    beats_per_cut: dict[str, int],
    min_shot_seconds: float = 0.2,
    max_shot_seconds: float = 8.0,
    snap_to_downbeats: bool = True,
    start: float = 0.0,
    end: float | None = None,
) -> list[float]:
    """Pick the subset of beats the edit cuts on.

    Returns absolute audio timestamps: the first is `start`, the last is `end`,
    and every value between is a detected beat.

    Cut density follows section energy — a "high" section cuts every 2 beats
    while a "low" section holds for 8 — and candidates are nudged onto bar
    lines where one is within a beat, so shots start where the music does.
    """
    end = grid.duration if end is None else min(end, grid.duration)
    if end <= start:
        raise PlanningError(
            f"Edit window is empty: start={start:.2f}s, end={end:.2f}s."
        )

    # Only beats strictly inside the window are cut candidates; the window
    # edges themselves are always boundaries.
    candidates = [
        (index, time)
        for index, time in enumerate(grid.beats)
        if start < time < end
    ]
    if not candidates:
        log.warning(
            "No beats fall inside %.2fs-%.2fs; emitting a single shot.", start, end
        )
        return [start, end]

    beat_times = [time for _, time in candidates]
    downbeat_positions = {
        position
        for position, (index, _) in enumerate(candidates)
        if grid.is_downbeat_index(index)
    }

    cuts = [start]
    position = -1  # index into beat_times of the last cut (-1 == window start)

    while True:
        last_cut = cuts[-1]
        label = grid.label_at(last_cut)
        step = max(1, int(beats_per_cut.get(label, 4)))

        target = position + step
        if snap_to_downbeats:
            target = _snap_to_downbeat(target, downbeat_positions, len(beat_times))

        # Running past the last beat is not on its own a reason to abandon the
        # beats that remain — only to stop stepping at the section's density.
        overshoot = target >= len(beat_times)
        if overshoot:
            target = len(beat_times) - 1

        target = _respect_max_shot(
            target, position, beat_times, last_cut, max_shot_seconds
        )
        target = _respect_min_shot(target, beat_times, last_cut, min_shot_seconds)
        if target is None or target >= len(beat_times) or target <= position:
            break

        # Out of beats: take this one only if holding all the way to `end`
        # would make the closing shot too long. Otherwise let the edit finish
        # on the shot it is already in, rather than adding a stub near the end.
        if overshoot and end - last_cut <= max_shot_seconds:
            break

        cuts.append(beat_times[target])
        position = target

    # The loop stops as soon as the next planned cut would run past the last
    # beat, which can leave a tail far longer than max_shot_seconds — the final
    # shot of a 4-minute edit sitting on screen for eight seconds. Fill it with
    # whatever beats are still available, taking the latest one that fits.
    while end - cuts[-1] > max_shot_seconds:
        reachable = [
            time for time in beat_times
            if time > cuts[-1] and time - cuts[-1] <= max_shot_seconds
        ]
        if not reachable:
            break
        cuts.append(reachable[-1])

    # A sliver of a shot at the very end reads as a glitch; absorb it backwards.
    if len(cuts) > 1 and end - cuts[-1] < min_shot_seconds:
        cuts.pop()
    cuts.append(end)

    log.info(
        "Planned %d cut(s) across %.2fs (%.2fs - %.2fs).",
        len(cuts) - 2, end - start, start, end,
    )
    return cuts


def _snap_to_downbeat(
    target: int, downbeat_positions: set[int], count: int
) -> int:
    """Move `target` onto a bar line if one sits within a single beat."""
    if target in downbeat_positions:
        return target
    for offset in (-1, 1):
        neighbour = target + offset
        if 0 <= neighbour < count and neighbour in downbeat_positions:
            return neighbour
    return target


def _respect_max_shot(
    target: int,
    position: int,
    beat_times: list[float],
    last_cut: float,
    max_shot_seconds: float,
) -> int:
    """Walk the candidate earlier until the shot is short enough.

    Never walks past the next available beat: at very slow tempos a single beat
    can already exceed the maximum, and cutting off-beat to satisfy a config
    value would defeat the point of the whole pipeline.
    """
    while (
        target > position + 1
        and beat_times[target] - last_cut > max_shot_seconds
    ):
        target -= 1
    return target


def _respect_min_shot(
    target: int,
    beat_times: list[float],
    last_cut: float,
    min_shot_seconds: float,
) -> int | None:
    """Walk the candidate later until the shot is long enough to be seen."""
    while (
        target < len(beat_times)
        and beat_times[target] - last_cut < min_shot_seconds
    ):
        target += 1
    return target if target < len(beat_times) else None


# --------------------------------------------------------------------------
# 2. Which clip fills each shot
# --------------------------------------------------------------------------

class ClipPicker:
    """Hands out (clip, in-point) pairs for successive shots.

    Goals, in priority order: never repeat a clip back to back, prefer a clip
    with enough unused footage left to cover the shot, and walk forward through
    each clip so a short clip reused ten times doesn't show the same two
    seconds every time.
    """

    def __init__(
        self,
        clips: list[ClipInfo],
        *,
        order: str = "shuffle",
        seed: int | None = None,
        advance_within_clip: bool = True,
    ) -> None:
        if not clips:
            raise PlanningError("No clips available to assemble.")
        self._clips = list(clips)
        self._order = order
        self._random = random.Random(seed)
        self._advance = advance_within_clip
        self._cursor: dict[Path, float] = {clip.path: 0.0 for clip in clips}
        self._rotation: list[ClipInfo] = []
        self._position = 0
        self._last: ClipInfo | None = None
        self._reshuffle()

    def _reshuffle(self) -> None:
        rotation = list(self._clips)
        if self._order == "shuffle":
            self._random.shuffle(rotation)
            # Avoid the new pass opening with the clip the last one closed on.
            if len(rotation) > 1 and self._last is not None and rotation[0] is self._last:
                rotation[0], rotation[-1] = rotation[-1], rotation[0]
        elif self._order != "sequential":
            raise PlanningError(
                f"Unknown editing.clip_order {self._order!r}; "
                f"expected 'shuffle' or 'sequential'."
            )
        self._rotation = rotation
        self._position = 0

    def _ordered_candidates(self) -> list[tuple[int, ClipInfo]]:
        """The rotation, starting from the current position and wrapping."""
        size = len(self._rotation)
        return [
            ((self._position + step) % size, self._rotation[(self._position + step) % size])
            for step in range(size)
        ]

    def pick(self, needed: float) -> tuple[ClipInfo, float]:
        """Choose a clip and in-point able to cover `needed` seconds."""
        candidates = self._ordered_candidates()
        fresh = [
            (index, clip) for index, clip in candidates
            if clip.duration - self._cursor[clip.path] >= needed
        ]
        # With one clip there is no alternative to repeating it, so the
        # no-back-to-back rule has to be relaxed or nothing is ever pickable.
        allow_repeat = len(self._clips) == 1

        choice = (
            self._first_match(fresh, allow_repeat=allow_repeat)
            # Everything is used up: rewind clips that are long enough overall.
            # Rewinding a different clip beats repeating the previous one.
            or self._first_match(
                [(i, c) for i, c in candidates if c.duration >= needed],
                allow_repeat=allow_repeat,
                rewind=True,
            )
            # Nothing is long enough at all — take the longest and hold its
            # last frame. The timeline stays on the beat; the shot freezes.
            or self._longest(candidates)
        )

        index, clip = choice
        source_in = self._cursor[clip.path] if self._advance else 0.0
        if source_in + needed > clip.duration:
            source_in = max(0.0, min(source_in, clip.duration - needed))

        if self._advance:
            self._cursor[clip.path] = source_in + needed

        self._last = clip
        self._position = index + 1
        if self._position >= len(self._rotation):
            self._reshuffle()
        return clip, source_in

    def _first_match(
        self,
        options: list[tuple[int, ClipInfo]],
        *,
        allow_repeat: bool,
        rewind: bool = False,
    ) -> tuple[int, ClipInfo] | None:
        """First option in rotation order, skipping the previous clip.

        Returns None rather than conceding a back-to-back repeat, so the caller
        can try a later strategy that may still find a different clip.
        """
        if not allow_repeat:
            options = [
                option for option in options
                if self._last is None or option[1] is not self._last
            ]
        if not options:
            return None
        chosen = options[0]
        if rewind:
            self._cursor[chosen[1].path] = 0.0
        return chosen

    def _longest(self, candidates: list[tuple[int, ClipInfo]]) -> tuple[int, ClipInfo]:
        preferred = [
            option for option in candidates
            if self._last is None or option[1] is not self._last
        ] or candidates
        chosen = max(preferred, key=lambda option: option[1].duration)
        self._cursor[chosen[1].path] = 0.0
        return chosen


# --------------------------------------------------------------------------
# 3. Build the plan
# --------------------------------------------------------------------------

def build_plan(
    clips: list[ClipInfo],
    grid: BeatGrid,
    *,
    video_format: VideoFormat,
    beats_per_cut: dict[str, int],
    min_shot_seconds: float = 0.2,
    max_shot_seconds: float = 8.0,
    snap_to_downbeats: bool = True,
    transition: str = CUT,
    crossfade_seconds: float = 0.25,
    audio_fade_out_seconds: float = 0.0,
    clip_order: str = "shuffle",
    advance_within_clip: bool = True,
    seed: int | None = None,
    start: float = 0.0,
    end: float | None = None,
    effects: Effects | None = None,
) -> EditPlan:
    """Produce a complete, renderable timeline."""
    cuts = choose_cut_times(
        grid,
        beats_per_cut=beats_per_cut,
        min_shot_seconds=min_shot_seconds,
        max_shot_seconds=max_shot_seconds,
        snap_to_downbeats=snap_to_downbeats,
        start=start,
        end=end,
    )

    fps = video_format.fps
    # Absolute frame positions, so per-segment rounding cannot accumulate.
    frame_positions = [round((cut - start) * fps) for cut in cuts]
    picker = ClipPicker(
        clips, order=clip_order, seed=seed, advance_within_clip=advance_within_clip
    )

    # A crossfade needs extra footage on each side of the join to dissolve
    # through; half the transition at each end keeps the dissolve centred on
    # the beat so the cut still reads as landing on time.
    extra = crossfade_seconds / 2.0 if transition == CROSSFADE else 0.0
    ramp = (effects or Effects()).speed_ramp

    segments: list[Segment] = []
    for index in range(len(frame_positions) - 1):
        frames = frame_positions[index + 1] - frame_positions[index]
        if frames <= 0:
            # Two cuts landed inside one frame; the shot cannot be seen.
            continue

        head = extra if index > 0 else 0.0
        tail = extra if index < len(frame_positions) - 2 else 0.0
        extra_frames = round((head + tail) * fps)
        total_frames = frames + extra_frames

        # A ramp consumes source at a different rate than the shot's own
        # output duration — slow motion less per second, sped-up more — so it
        # needs its own, separate source-time figure from the plain one.
        # Whichever is larger is what gets asked of the clip picker, so the
        # picked clip has the best chance of covering either.
        unramped_needed = total_frames / fps
        ramped_needed = (
            ramp_source_seconds_needed(total_frames, fps, ramp)
            if ramp is not None else unramped_needed
        )
        clip, source_in = picker.pick(max(unramped_needed, ramped_needed))
        available = max(0.0, clip.duration - source_in)
        margin = 2.0 / fps  # a couple of extra frames so fps conversion can
                             # never leave a segment one frame short of trim

        def _read_and_pad(target: float) -> tuple[float, float]:
            read = min(available, target + margin)
            pad = (target - read) + margin if read < target else 0.0
            return read, pad

        # Commit to the ramp only if what was actually picked can cover it;
        # otherwise fall back to the plain, unramped requirement for this shot.
        use_ramp = ramp is not None and available + margin >= ramped_needed
        needed = ramped_needed if use_ramp else unramped_needed
        read_seconds, pad_seconds = _read_and_pad(needed)

        if use_ramp and pad_seconds > 0:
            # The ramp looked like it fit within the margin above, but the
            # actual read still came up short. Fall back to the plain
            # requirement and *recompute* read/pad against that — reusing
            # the ramped pad_seconds here was the original bug: padding sized
            # for a ~6% cheaper ramped read is not enough source for a plain
            # one, so the segment landed short of its planned frame count and
            # every cut after it drifted, compounding to seconds of error by
            # the end of the edit. Caught by --verify on a real render, not
            # by the ramp's own isolated tests, which never combined a
            # near-miss shortfall with a full multi-segment plan.
            use_ramp = False
            needed = unramped_needed
            read_seconds, pad_seconds = _read_and_pad(needed)

        segments.append(
            Segment(
                clip=clip,
                source_in=source_in,
                read_seconds=read_seconds,
                frames=total_frames,
                timeline_start=frame_positions[index] / fps,
                duration=frames / fps,
                pad_seconds=pad_seconds,
                beat_time=cuts[index],
                speed_ramp=ramp if use_ramp else None,
            )
        )

    if not segments:
        raise PlanningError(
            "The edit window produced no usable segments. "
            "Check --start/--duration and the detected beat grid."
        )

    total_duration = frame_positions[-1] / fps
    plan = EditPlan(
        segments=tuple(segments),
        audio_path=grid.audio_path,
        audio_start=start,
        video_format=video_format,
        transition=transition,
        crossfade_seconds=crossfade_seconds,
        audio_fade_out_seconds=audio_fade_out_seconds,
        total_duration=total_duration,
        effects=effects or Effects(),
    )
    log.info("Edit plan: %s", plan.summary())
    return plan


# --------------------------------------------------------------------------
# 4. Render
# --------------------------------------------------------------------------

# Every segment in a pass needs its own decoder holding a pool of reference
# frames, so peak memory grows with the number of inputs — measured at roughly
# 24MB per input on top of a ~1.1GB floor for a 1080p x264 encode. That floor
# is fixed, but the growth is not: a 200-segment edit in one pass needs several
# gigabytes and dies with an allocation failure from x264. Capping the pass and
# stitching the results losslessly makes peak memory a function of this number
# instead of the length of the edit.
#
# 24 keeps a 1080p pass around 1.6GB, which fits comfortably on an 8GB laptop.
# Lower it on a memory-constrained machine; raise it to render more crossfade
# joins in a single pass.
DEFAULT_MAX_INPUTS_PER_PASS = 24


def _encode_args(encode: dict) -> list[str]:
    return [
        "-c:v", str(encode.get("video_codec", "libx264")),
        "-crf", str(encode.get("crf", 20)),
        "-preset", str(encode.get("preset", "medium")),
        "-pix_fmt", str(encode.get("pixel_format", "yuv420p")),
    ]


def _audio_args(encode: dict) -> list[str]:
    return [
        "-c:a", str(encode.get("audio_codec", "aac")),
        "-b:a", str(encode.get("audio_bitrate", "192k")),
    ]


def build_render_command(
    plan: EditPlan,
    out_path: Path,
    graph_path: Path,
    encode: dict,
    *,
    video_only: bool = False,
) -> list[str]:
    """The exact ffmpeg argv for this plan. Split out so it can be asserted on."""
    command: list[str] = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-y"]

    for segment in plan.segments:
        # -ss before -i is an input seek: ffmpeg jumps to the nearest keyframe
        # and decodes forward, instead of decoding the whole clip and throwing
        # most of it away. With -t this decodes only the frames we use.
        command += [
            "-ss", f"{segment.source_in:.6f}",
            "-t", f"{segment.read_seconds:.6f}",
            "-i", str(segment.clip.path),
        ]
    if not video_only:
        command += ["-i", str(plan.audio_path)]

    command += [
        # The graph goes in a file: with a few hundred segments it comfortably
        # exceeds the command-line length limit, on Windows especially.
        "-filter_complex_script", str(graph_path),
        "-map", f"[{transitions.VIDEO_OUT}]",
    ]
    if not video_only:
        command += ["-map", f"[{transitions.AUDIO_OUT}]"]

    command += ["-r", f"{plan.video_format.fps:g}"]
    command += _encode_args(encode)
    if not video_only:
        command += _audio_args(encode)
        if encode.get("faststart", True):
            command += ["-movflags", "+faststart"]
    command += [str(out_path)]
    return command


def build_filter_graph(plan: EditPlan, *, video_only: bool = False) -> str:
    specs = [
        InputSpec(
            input_index=index,
            frames=segment.frames,
            pad_seconds=segment.pad_seconds,
            speed_ramp=segment.speed_ramp,
        )
        for index, segment in enumerate(plan.segments)
    ]
    post_chain = None
    if plan.effects.is_active:
        post_chain = effects.build_post_chain(
            plan.local_cut_times,
            plan.video_format,
            look=plan.effects.look,
            lut_path=plan.effects.lut_path,
            grade_lift=plan.effects.grade_lift,
            grade_gamma=plan.effects.grade_gamma,
            grade_gain=plan.effects.grade_gain,
            grade_saturation=plan.effects.grade_saturation,
            grade_contrast=plan.effects.grade_contrast,
            grade_brightness=plan.effects.grade_brightness,
            denoise_strength=plan.effects.denoise_strength,
            shot_spans=plan.local_shot_spans,
            ken_burns_amount=plan.effects.ken_burns_amount,
            ken_burns_steps=plan.effects.ken_burns_steps,
            ken_burns_direction=plan.effects.ken_burns_direction,
            ken_burns_alternate=plan.effects.ken_burns_alternate,
            ken_burns_pan=plan.effects.ken_burns_pan,
            light_leak_opacity=plan.effects.light_leak_opacity,
            light_leak_color=plan.effects.light_leak_color,
            light_leak_center=plan.effects.light_leak_center,
            chroma_key_color=plan.effects.chroma_key_color,
            chroma_key_similarity=plan.effects.chroma_key_similarity,
            chroma_key_background=plan.effects.chroma_key_background,
            temporal_blend_frames=plan.effects.temporal_blend_frames,
            punch_zoom_amount=plan.effects.punch_zoom_amount,
            punch_zoom_seconds=plan.effects.punch_zoom_seconds,
            punch_zoom_every_nth=plan.effects.punch_zoom_every_nth,
            glitch_shift_px=plan.effects.glitch_shift_px,
            glitch_seconds=plan.effects.glitch_seconds,
            glitch_every_nth=plan.effects.glitch_every_nth,
            grain_strength=plan.effects.grain_strength,
            sharpen_amount=plan.effects.sharpen_amount,
            vignette_strength=plan.effects.vignette_strength,
        )

    video = transitions.build_video_graph(
        specs,
        plan.video_format,
        transition=plan.transition,
        crossfade_seconds=plan.crossfade_seconds,
        post_chain=post_chain,
    )
    if video_only:
        return video

    audio = transitions.build_audio_graph(
        len(plan.segments),
        start=plan.audio_start,
        duration=plan.total_duration,
        fade_out_seconds=plan.audio_fade_out_seconds,
    )
    return f"{video};\n{audio}"


def render(
    plan: EditPlan,
    out_path: Path | str,
    *,
    encode: dict | None = None,
    max_inputs_per_pass: int = DEFAULT_MAX_INPUTS_PER_PASS,
    timeout: float | None = None,
) -> Path:
    """Render `plan` to `out_path`.

    Short edits go through a single ffmpeg process. Longer ones are rendered in
    capped passes and joined with the concat demuxer, so memory does not grow
    with the length of the edit.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encode = encode or {}
    limit = max(1, int(max_inputs_per_pass))

    with tempfile.TemporaryDirectory(prefix="video_pipeline_") as workdir:
        work = Path(workdir)
        try:
            if len(plan.segments) <= limit:
                log.info(
                    "Rendering %d segment(s) -> %s", len(plan.segments), out_path
                )
                _render_pass(plan, out_path, encode, work, video_only=False,
                             timeout=timeout)
            else:
                _render_chunked(plan, out_path, encode, work, limit, timeout)
        except FFmpegError as exc:
            raise _explain_if_out_of_memory(exc, limit) from exc

    log.info("Wrote %s", out_path)
    return out_path


class RenderOutOfMemory(RuntimeError):
    """ffmpeg could not allocate what the pass needed."""


# What ffmpeg, x264 and the allocator say when they run out of room.
_OUT_OF_MEMORY_MARKERS = (
    "malloc of size",
    "cannot allocate memory",
    "out of memory",
    "unable to allocate",
    "av_buffer_alloc",
)


def _explain_if_out_of_memory(exc: FFmpegError, limit: int) -> Exception:
    """Turn an allocation failure into an error that says what to change."""
    text = (exc.stderr or "").lower()
    if not any(marker in text for marker in _OUT_OF_MEMORY_MARKERS):
        return exc
    return RenderOutOfMemory(
        f"ffmpeg ran out of memory rendering {limit} segment(s) per pass. "
        f"Lower editing/render `max_inputs_per_pass` in config/settings.yaml, "
        f"or pass --max-inputs-per-pass {max(1, limit // 2)}. "
        f"Reducing the output resolution also helps, since each pass holds one "
        f"decoder per segment at source resolution.\n\n{exc}"
    )


def _render_pass(
    plan: EditPlan,
    out_path: Path,
    encode: dict,
    work: Path,
    *,
    video_only: bool,
    timeout: float | None,
) -> None:
    graph = build_filter_graph(plan, video_only=video_only)
    log.debug("filter graph:\n%s", graph)

    graph_path = work / f"graph_{out_path.stem}.txt"
    graph_path.write_text(graph, encoding="utf-8")

    command = build_render_command(
        plan, out_path, graph_path, encode, video_only=video_only
    )
    run_with_progress(command, total_seconds=plan.total_duration, timeout=timeout)


def _render_chunked(
    plan: EditPlan,
    out_path: Path,
    encode: dict,
    work: Path,
    limit: int,
    timeout: float | None,
) -> None:
    chunks = [
        plan.segments[start:start + limit]
        for start in range(0, len(plan.segments), limit)
    ]
    log.info(
        "Rendering %d segment(s) in %d pass(es) of up to %d -> %s",
        len(plan.segments), len(chunks), limit, out_path,
    )
    if plan.transition == CROSSFADE:
        # Said out loud rather than quietly degraded: a dissolve cannot span two
        # separate encodes, so the joins between passes come out as hard cuts.
        log.warning(
            "%d join(s) between passes will be hard cuts, not crossfades. "
            "Raise editing.max_inputs_per_pass to render more of the edit at "
            "once if the machine has the memory for it.",
            len(chunks) - 1,
        )
    if plan.effects.is_active and len(chunks) > 1:
        # The final join is a stream copy specifically to avoid a second
        # encode; a filter can't run there, so a punch-zoom or glitch that
        # would have landed exactly on a pass boundary is skipped for that
        # one cut instead of applied. The colour grade is unaffected — it
        # reapplies per pass, not per cut.
        log.warning(
            "Punch-zoom/glitch at the %d join(s) between passes will be "
            "skipped for that cut (a hard join, same as the crossfade "
            "limitation above). The colour grade is unaffected.",
            len(chunks) - 1,
        )

    parts: list[Path] = []
    for index, chunk in enumerate(chunks):
        part = work / f"part_{index:04d}.mp4"
        chunk_plan = replace(
            plan,
            segments=chunk,
            total_duration=sum(segment.duration for segment in chunk),
        )
        log.info("  pass %d/%d (%d segments)", index + 1, len(chunks), len(chunk))
        _render_pass(chunk_plan, part, encode, work, video_only=True,
                     timeout=timeout)
        parts.append(part)

    _join_parts(parts, plan, out_path, encode, work, timeout)


def _join_parts(
    parts: list[Path],
    plan: EditPlan,
    out_path: Path,
    encode: dict,
    work: Path,
    timeout: float | None,
) -> None:
    """Stitch the rendered passes together and add the music.

    Video is stream-copied: every pass used identical encoder settings, so the
    parts share a bitstream configuration and can be concatenated without
    re-encoding. Nothing is decoded twice and nothing loses quality.
    """
    list_path = work / "parts.txt"
    list_path.write_text(
        "\n".join(f"file '{_escape_concat_path(part)}'" for part in parts) + "\n",
        encoding="utf-8",
    )

    audio_graph = transitions.build_audio_graph(
        1,
        start=plan.audio_start,
        duration=plan.total_duration,
        fade_out_seconds=plan.audio_fade_out_seconds,
    )

    command = [
        ffmpeg_bin(), "-hide_banner", "-nostdin", "-y",
        # -safe 0 because the list holds absolute paths.
        "-f", "concat", "-safe", "0", "-i", str(list_path),
        "-i", str(plan.audio_path),
        "-filter_complex", audio_graph,
        "-map", "0:v", "-map", f"[{transitions.AUDIO_OUT}]",
        "-c:v", "copy",
        *_audio_args(encode),
    ]
    if encode.get("faststart", True):
        command += ["-movflags", "+faststart"]
    command += [str(out_path)]

    log.info("  joining %d pass(es) and muxing audio", len(parts))
    run_with_progress(command, total_seconds=plan.total_duration, timeout=timeout)


def _escape_concat_path(path: Path) -> str:
    """Format a path for the concat demuxer's single-quoted `file` directive."""
    # Forward slashes work on Windows and avoid backslash escaping entirely;
    # a literal quote has to be closed, escaped and reopened.
    return path.resolve().as_posix().replace("'", "'\\''")
