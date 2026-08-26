"""Ken Burns: a slow zoom (and optional pan) across a shot's own duration —
AE's keyframed camera moves, applied to still-feeling footage.

`zoompan`'s documented accumulator recipe (`z='min(zoom+0.002,1.5)'`) and a
plain `scale`/`crop` driven by `t` with `eval=frame` were both tried first and
both produced zero visible motion on this ffmpeg build — the same class of
failure already found building punch-zoom (see `effects.py`) and speed ramp.
So this is built the same proven way punch-zoom is: not a continuously
animated expression, but discrete fixed-parameter crop/scale states switched
with `overlay`'s `enable=`, confirmed to actually respond per frame.

The difference from punch-zoom is the switching pattern. Punch-zoom is one
brief window, on then off, around a single instant. A Ken Burns move spans an
entire shot and only ever goes one direction, so every step across the whole
timeline — not just within one shot — is flattened into a single ordered
list and layered with `enable='gte(t, step_start)'`: "on from here onward."
Step N's overlay simply paints over step N-1's once its own moment arrives
and stays on top until step N+1 does the same, including across a shot
boundary — a staircase standing in for a ramp, the same trick film only
needs ~24 discrete frames a second to read as continuous motion.

An earlier version of this got that boundary wrong: it treated each shot's
first step as a fresh, *unconditional* crop rather than one more gated
overlay, which would have silently replaced the entire visible frame from
t=0 onward the moment a second shot's stages were appended — caught before
it was ever rendered, by re-reading what the loop actually built rather than
trusting that "reuses the proven overlay pattern" was enough on its own.
"""

from __future__ import annotations

from editing.transitions import VideoFormat, format_number

DIRECTIONS = ("in", "out")


def _zoom_at(progress: float, amount: float, direction: str) -> float:
    if direction == "in":
        return 1.0 + amount * progress
    return 1.0 + amount * (1.0 - progress)


def _step_stage(
    fmt: VideoFormat,
    *,
    progress: float,
    amount: float,
    direction: str,
    pan: tuple[float, float],
    source_label: str,
    out_label: str,
) -> str:
    zoom = _zoom_at(progress, amount, direction)
    crop_w = max(2, round(fmt.width / zoom / 2) * 2)
    crop_h = max(2, round(fmt.height / zoom / 2) * 2)

    center_x = 0.5 + pan[0] * progress
    center_y = 0.5 + pan[1] * progress
    x = round(center_x * fmt.width - crop_w / 2)
    y = round(center_y * fmt.height - crop_h / 2)
    x = max(0, min(fmt.width - crop_w, x))
    y = max(0, min(fmt.height - crop_h, y))

    return (
        f"[{source_label}]crop=w={crop_w}:h={crop_h}:x={x}:y={y},"
        f"scale=w={fmt.width}:h={fmt.height}[{out_label}]"
    )


def ken_burns_stages(
    shot_spans: tuple[tuple[float, float], ...],
    fmt: VideoFormat,
    *,
    zoom_amount: float = 0.12,
    steps: int = 8,
    direction: str = "in",
    alternate: bool = False,
    pan: tuple[float, float] = (0.0, 0.0),
    input_label: str,
    output_label: str,
    tag: str,
) -> list[str] | None:
    """One staircased zoom(+pan) move per shot in `shot_spans`.

    `direction` is "in" or "out"; with `alternate=True` it flips for every
    successive shot (in, out, in, out, ...) rather than repeating the same
    direction on every cut, which reads as more intentional across a whole
    edit. `pan` is (dx, dy) as a fraction of frame size — total drift of the
    crop centre by the end of each shot's move; (0, 0) is a pure zoom.
    `shot_spans` must be in the same local (pass-relative) time this whole
    package's post-chain always uses — the first span is expected to start
    at (or very near) 0.
    """
    if not shot_spans or zoom_amount <= 0 or steps < 2:
        return None

    # Flatten every shot's steps into one ordered (activation_time, progress,
    # direction) list. This is the fix for the boundary bug described above:
    # a shot's own first step is just one more entry here, gated at that
    # shot's start time exactly like every other step, never a special case.
    keyframes: list[tuple[float, float, str]] = []
    for shot_index, (start, end) in enumerate(shot_spans):
        duration = end - start
        if duration <= 0:
            continue
        shot_direction = direction
        if alternate and shot_index % 2 == 1:
            shot_direction = "out" if direction == "in" else "in"
        for step in range(steps):
            progress = step / (steps - 1)
            step_start = start + duration * (step / steps)
            keyframes.append((step_start, progress, shot_direction))

    if not keyframes:
        return None

    lines: list[str] = []

    # The very first keyframe overall — and only this one — has nothing
    # before it, so it is the unconditional base layer rather than a gated
    # overlay.
    t0, progress0, direction0 = keyframes[0]
    base_label = f"{tag}_0"
    lines.append(_step_stage(
        fmt, progress=progress0, amount=zoom_amount, direction=direction0,
        pan=pan, source_label=input_label, out_label=base_label,
    ))

    layer = base_label
    last_index = len(keyframes) - 1
    for index, (t, progress, shot_direction) in enumerate(keyframes[1:], start=1):
        src_label = f"{tag}_{index}_src"
        lines.append(_step_stage(
            fmt, progress=progress, amount=zoom_amount, direction=shot_direction,
            pan=pan, source_label=input_label, out_label=src_label,
        ))
        nxt = output_label if index == last_index else f"{tag}_{index}"
        lines.append(
            f"[{layer}][{src_label}]overlay=x=0:y=0"
            f":enable='gte(t,{format_number(t)})'[{nxt}]"
        )
        layer = nxt

    if layer != output_label:
        # Only one keyframe existed in total (a single shot with the loop
        # above never running) — route the base layer to the output label.
        lines.append(f"[{layer}]null[{output_label}]")

    return lines
