"""Post-processing on the assembled timeline: colour grade, punch-zoom, glitch.

All three run once, after every segment has already been placed — not per
source clip — because punch-zoom and glitch are keyed to absolute cut time,
which only exists once the timeline is assembled.

Every technique here was verified against the installed ffmpeg by rendering a
known test pattern and measuring actual pixel values, not assumed from
filter docs — two assumptions that looked reasonable turned out to be wrong:

  * `scale`/`crop` do not re-evaluate `t` (seconds) per frame on this build,
    despite `-h filter=scale` listing `w`/`h` as expression-typed. A
    dynamically-sized crop window driven by `t` silently never changes.
  * `crop` does not support the `enable` timeline option at all — ffmpeg
    refuses to start with "Timeline ('enable' option) not supported with
    filter 'crop'", not a silent no-op.

What *is* confirmed working: `rgbashift` honours `enable=<expr>` with fixed
shift amounts (used for glitch), and `overlay` honours `enable=<expr>` to
switch a whole second input in and out (used for punch-zoom: a permanently
zoomed-in copy of the frame, laid over the normal one only in short windows
around each cut).
"""

from __future__ import annotations

from editing import camera, compositing, finishing
from editing.looks import custom_grade, look_filters, lut_filter
from editing.transitions import PRE_POST_LABEL, VIDEO_OUT, VideoFormat, format_number


def _pulse_windows(cut_times: tuple[float, ...], window: float) -> str:
    """`between()` terms, one per cut, summed so ffmpeg treats it as OR."""
    return "+".join(
        f"between(t,{format_number(cut - window / 2)},{format_number(cut + window / 2)})"
        for cut in cut_times
    )


def glitch_filters(
    cut_times: tuple[float, ...],
    *,
    seconds: float = 0.08,
    shift_px: int = 6,
) -> str | None:
    """A brief RGB channel split at each cut — red one way, blue the other.

    A fixed offset rather than an animated one: `rgbashift`'s shift amounts
    are plain integers, not per-frame expressions, so the "glitch" comes from
    snapping the effect on and off (via `enable`) rather than easing it —
    which reads correctly for this effect anyway, since real glitches don't
    ease in.
    """
    if not cut_times or shift_px <= 0:
        return None
    shift = int(shift_px)
    windows = _pulse_windows(cut_times, seconds)
    return f"rgbashift=rh={shift}:bh=-{shift}:edge=smear:enable='{windows}'"


def _punch_zoom_stage(
    cut_times: tuple[float, ...],
    fmt: VideoFormat,
    *,
    amount: float,
    seconds: float,
    input_label: str,
    output_label: str,
    tag: str,
) -> list[str] | None:
    """A permanently-zoomed-in copy of the frame, shown only around each cut.

    `scale`/`crop` cannot animate a zoom smoothly on this ffmpeg build (see
    module docstring), so this does not ease in — it snaps to the zoomed
    frame for the window and snaps back. Verified by rendering: a bordered
    test frame drops from a distinct outer-edge value to the fully-zoomed
    interior value only inside the requested window, and returns immediately
    after.
    """
    if not cut_times or amount <= 0:
        return None

    crop_w = max(2, round(fmt.width / (1 + amount) / 2) * 2)
    crop_h = max(2, round(fmt.height / (1 + amount) / 2) * 2)
    base = f"{tag}_base"
    src = f"{tag}_src"
    zoomed = f"{tag}_zoomed"
    windows = _pulse_windows(cut_times, seconds)

    return [
        f"[{input_label}]split=2[{base}][{src}]",
        f"[{src}]crop=w={crop_w}:h={crop_h},scale=w={fmt.width}:h={fmt.height}[{zoomed}]",
        f"[{base}][{zoomed}]overlay=x=0:y=0:enable='{windows}'[{output_label}]",
    ]


def build_post_chain(
    cut_times: tuple[float, ...],
    fmt: VideoFormat,
    *,
    shot_spans: tuple[tuple[float, float], ...] = (),
    ken_burns_amount: float = 0.0,
    ken_burns_steps: int = 8,
    ken_burns_direction: str = "in",
    ken_burns_alternate: bool = False,
    ken_burns_pan: tuple[float, float] = (0.0, 0.0),
    look: str = "none",
    lut_path: str | None = None,
    grade_lift: tuple[float, float, float] = (0.0, 0.0, 0.0),
    grade_gamma: tuple[float, float, float] = (0.0, 0.0, 0.0),
    grade_gain: tuple[float, float, float] = (0.0, 0.0, 0.0),
    grade_saturation: float = 1.0,
    grade_contrast: float = 1.0,
    grade_brightness: float = 0.0,
    denoise_strength: float = 0.0,
    light_leak_opacity: float = 0.0,
    light_leak_color: tuple[int, int, int] = (255, 160, 40),
    light_leak_center: tuple[float, float] = (0.85, 0.15),
    chroma_key_color: str | None = None,
    chroma_key_similarity: float = 0.18,
    chroma_key_background: str = "0x000000",
    temporal_blend_frames: int = 1,
    punch_zoom_amount: float = 0.0,
    punch_zoom_seconds: float = 0.22,
    punch_zoom_every_nth: int = 1,
    glitch_shift_px: int = 0,
    glitch_seconds: float = 0.08,
    glitch_every_nth: int = 1,
    grain_strength: int = 0,
    sharpen_amount: float = 0.0,
    vignette_strength: float = 0.0,
    input_label: str = PRE_POST_LABEL,
    output_label: str = VIDEO_OUT,
) -> str | None:
    """The full post-processing sub-graph, every effect this package offers
    strung into one ordered pipeline.

    Order, and why: LUT and the numeric 3-way grade and the named look preset
    all stack (a project can use any combination — the LUT setting the base
    look, the wheels nudging it, all still free); denoise before anything
    that would amplify noise; Ken Burns next, since it is a camera move on
    the settled base image, and everything after it (light leak, chroma key,
    punch-zoom, glitch) should ride along with wherever it has panned/zoomed
    to rather than the other way around; light leak and chroma key are
    compositing swaps; temporal blend and punch-zoom are motion/spatial;
    glitch is the final distortion layer; grain sits on top of everything,
    the way real film grain physically would; sharpen and vignette are the
    last-pass finishing touches. Returns None when every effect is disabled,
    so the caller can skip attaching a post-chain entirely rather than pay
    for a stage that does nothing.
    """
    stages: list[tuple[str, object]] = []

    if lut_path:
        stages.append(("linear", lut_filter(lut_path)))

    grade_chain = custom_grade(
        lift=grade_lift, gamma=grade_gamma, gain=grade_gain,
        saturation=grade_saturation, contrast=grade_contrast,
        brightness=grade_brightness,
    )
    if grade_chain:
        stages.append(("linear", grade_chain))

    look_chain = look_filters(look) if look and look != "none" else ""
    if look_chain:
        stages.append(("linear", look_chain))

    denoise_chain = finishing.denoise_filter(denoise_strength)
    if denoise_chain:
        stages.append(("linear", denoise_chain))

    if ken_burns_amount > 0 and shot_spans:
        stages.append(("ken_burns", None))

    if light_leak_opacity > 0:
        stages.append(("light_leak", None))

    if chroma_key_color:
        stages.append(("chroma_key", None))

    blend_chain = compositing.temporal_blend_filter(temporal_blend_frames)
    if blend_chain:
        stages.append(("linear", blend_chain))

    if punch_zoom_amount > 0:
        selected = cut_times[::max(1, punch_zoom_every_nth)]
        if selected:
            stages.append(("zoom", selected))

    if glitch_shift_px > 0:
        selected = cut_times[::max(1, glitch_every_nth)]
        if selected:
            stages.append(("glitch", selected))

    grain_chain = compositing.grain_filter(grain_strength)
    if grain_chain:
        stages.append(("linear", grain_chain))

    sharpen_chain = finishing.sharpen_filter(sharpen_amount)
    if sharpen_chain:
        stages.append(("linear", sharpen_chain))

    vignette_chain = finishing.vignette_filter(vignette_strength)
    if vignette_chain:
        stages.append(("linear", vignette_chain))

    if not stages:
        return None

    lines: list[str] = []
    current = input_label
    for index, (kind, payload) in enumerate(stages):
        is_last = index == len(stages) - 1
        nxt = output_label if is_last else f"post{index}"
        tag = f"post{index}"

        if kind == "linear":
            lines.append(f"[{current}]{payload}[{nxt}]")
        elif kind == "zoom":
            lines.extend(_punch_zoom_stage(
                payload, fmt,
                amount=punch_zoom_amount, seconds=punch_zoom_seconds,
                input_label=current, output_label=nxt, tag=tag,
            ))
        elif kind == "glitch":
            glitch_chain = glitch_filters(
                payload, seconds=glitch_seconds, shift_px=glitch_shift_px
            )
            lines.append(f"[{current}]{glitch_chain}[{nxt}]")
        elif kind == "ken_burns":
            lines.extend(camera.ken_burns_stages(
                shot_spans, fmt,
                zoom_amount=ken_burns_amount, steps=ken_burns_steps,
                direction=ken_burns_direction, alternate=ken_burns_alternate,
                pan=ken_burns_pan, input_label=current, output_label=nxt, tag=tag,
            ))
        elif kind == "light_leak":
            lines.extend(compositing.light_leak_stages(
                fmt, input_label=current, output_label=nxt, tag=tag,
                color=light_leak_color, center=light_leak_center,
                opacity=light_leak_opacity,
            ))
        else:  # chroma_key
            lines.extend(compositing.chroma_key_stages(
                fmt, input_label=current, output_label=nxt, tag=tag,
                key_color=chroma_key_color, similarity=chroma_key_similarity,
                background=chroma_key_background,
            ))

        current = nxt

    return ";\n".join(lines)
