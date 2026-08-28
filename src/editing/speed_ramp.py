"""Speed ramp: a shot that opens in slow motion and whips up to speed right
into the cut. Probably the single biggest driver of the "edit" look this
whole effects pass exists for — a beat-synced cut is already correct timing,
but a ramp is what makes the *motion* read as intentional rather than just
footage that happened to be trimmed to length.

Built as two constant-speed stages, not one continuously-eased curve.
`setpts`'s expression language has `PTS`, `N`, `TB` — no `t` in seconds, the
same limitation already found (and worked around a different way) while
building the punch-zoom effect — so a smooth acceleration curve isn't
practical to write as a single expression here either. Two fixed-speed
pieces, slow then fast, reads as an intentional "whip" rather than
approximating a smooth ramp badly, which happens to match the genre's own
visual signature: real edits whip-pan, they don't ease.

This is a *per-segment* concern, unlike the look/zoom/glitch effects in
`effects.py`, which run once on the assembled timeline. A speed ramp changes
how much source footage a shot consumes before it is ever placed on that
timeline, so it has to be decided while the plan is still being built, not
after.
"""

from __future__ import annotations

from dataclasses import dataclass

from editing.transitions import format_number


@dataclass(frozen=True)
class SpeedRamp:
    """fraction is the portion of the shot's OUTPUT duration spent slow."""

    fraction: float = 0.45
    slow_factor: float = 0.5     # < 1 = slow motion
    fast_factor: float = 1.3     # > 1 = sped up

    def __post_init__(self) -> None:
        if not 0.0 < self.fraction < 1.0:
            raise ValueError(f"fraction must be between 0 and 1, got {self.fraction}")
        if self.slow_factor <= 0 or self.fast_factor <= 0:
            raise ValueError("slow_factor and fast_factor must be positive.")


def stage_frames(frames: int, ramp: SpeedRamp) -> tuple[int, int]:
    """Output frame counts for the slow and fast stages, always >= 1 each."""
    n1 = min(frames - 1, max(1, round(frames * ramp.fraction)))
    return n1, frames - n1


def source_seconds_needed(frames: int, fps: float, ramp: SpeedRamp) -> float:
    """Source footage a ramped segment consumes.

    Not the same as its output duration: slow motion shows less source per
    second of output, sped-up shows more, and the two only cancel out by
    coincidence. `assembler.build_plan` uses this instead of `frames / fps`
    when asking the clip picker for a shot, so a ramped segment never comes
    up short of the footage the ramp itself is about to spend.
    """
    n1, n2 = stage_frames(frames, ramp)
    return (n1 / fps) * ramp.slow_factor + (n2 / fps) * ramp.fast_factor


def ramp_chain(
    frames: int,
    fps: float,
    ramp: SpeedRamp,
    *,
    input_label: str,
    output_label: str,
) -> list[str]:
    """The two-stage speed chain.

    Consumes `input_label` — expected to already be scaled/padded/setsar'd —
    and produces `output_label` at exactly `fps`, spanning exactly the source
    time this stage's `source_seconds_needed` promised. The caller still
    applies `trim=end_frame=`/`format=` afterward exactly as it does for an
    unramped segment; this only handles the speed change.

    The trailing `fps=` here is not optional. `setpts=PTS/factor` divides
    every timestamp by a non-round number, so the result does not land back
    on exact `1/fps` boundaries — confirmed by rendering: without this,
    `trim=end_frame=N` still (correctly) passes exactly N *frames* of stream,
    but those N frames carry slightly irregular timestamps, and the final
    render's own `-r fps` then pads extra frames to reconcile them against a
    clean timeline, silently landing more frames on screen than the plan
    accounted for. Re-snapping to the frame grid immediately, the same way
    every unramped segment already does, is what makes `trim=end_frame`
    downstream of this actually exact rather than exact-looking.
    """
    n1, n2 = stage_frames(frames, ramp)
    source1 = (n1 / fps) * ramp.slow_factor

    slow_src, fast_src = f"{output_label}_slowsrc", f"{output_label}_fastsrc"
    slow, fast = f"{output_label}_slow", f"{output_label}_fast"
    joined = f"{output_label}_joined"

    return [
        f"[{input_label}]split=2[{slow_src}][{fast_src}]",
        f"[{slow_src}]trim=end={format_number(source1)},"
        f"setpts=(PTS-STARTPTS)/{format_number(ramp.slow_factor)}[{slow}]",
        f"[{fast_src}]trim=start={format_number(source1)},"
        f"setpts=(PTS-STARTPTS)/{format_number(ramp.fast_factor)}[{fast}]",
        f"[{slow}][{fast}]concat=n=2:v=1:a=0[{joined}]",
        f"[{joined}]fps={format_number(fps)}[{output_label}]",
    ]
