"""Layered/blended effects: film grain, light leaks, motion blur / ghosting,
chroma key. The AE "blending modes + overlay stock" and Fusion "green screen"
categories, built from generated sources rather than licensed footage — a
light leak here is a `geq` gradient, not a purchased overlay clip, which is
what keeps it free rather than merely cheap.

Two things confirmed by rendering before anything here was written, matching
the pattern already found while building punch-zoom and speed ramp:

  * `geq`'s per-pixel expressions (X, Y, N — pixel coordinates and frame
    number) evaluate correctly per frame on this ffmpeg build. This is NOT
    the same mechanism as a filter's own named parameters (`scale`'s w/h,
    `drawbox`'s x/y) evaluated against `t`, which do not.
  * `tmix` genuinely averages a sliding window of N *real* consecutive
    frames — confirmed against a source whose luma encodes its own frame
    index, output frame i exactly equals mean(raw[i-N+1 : i+1]). An earlier
    attempt to prove this against a `drawbox` position animated with `t`
    showed no motion at all, which would have been read as "tmix doesn't
    blur anything" — it was `drawbox`'s `t` expression not moving, the same
    class of failure as `scale`/`crop`/`zoompan`/`drawtext`, not tmix.
"""

from __future__ import annotations


# --------------------------------------------------------------------------
# Grain
# --------------------------------------------------------------------------

def grain_filter(strength: int = 8) -> str:
    """Temporal random noise. `strength` is ffmpeg's own 0-100 `noise` scale."""
    strength = max(0, min(100, int(strength)))
    if strength == 0:
        return ""
    return f"noise=alls={strength}:allf=t+u"


# --------------------------------------------------------------------------
# Light leak: a warm radial glow generated in-graph (no stock footage, no
# network fetch) and blended in with `screen`, which only ever brightens —
# the correct blend mode for a leak, since a leak is light being added, not
# colour being mixed.
# --------------------------------------------------------------------------

def light_leak_stages(
    fmt,
    *,
    input_label: str,
    output_label: str,
    tag: str,
    color: tuple[int, int, int] = (255, 160, 40),
    center: tuple[float, float] = (0.85, 0.15),
    radius_frac: float = 0.5,
    opacity: float = 0.6,
) -> list[str]:
    """`color` is 0-255 RGB. `center` is (x, y) as a fraction of the frame,
    so the same call works at any resolution. `radius_frac` is the glow's
    falloff radius as a fraction of frame width.
    """
    r, g, b = color
    # geq's per-pixel expressions use uppercase W/H/X/Y/N for frame width,
    # height, pixel position, and frame number — confirmed by hitting the
    # lowercase-w version's parse error directly: geq does not fall back to
    # scale/crop-style lowercase `w`/`h`.
    cx, cy = f"(W*{center[0]:g})", f"(H*{center[1]:g})"
    sigma_sq = f"(2*(W*{radius_frac:g})*(W*{radius_frac:g}))"
    dist_sq = f"((X-{cx})*(X-{cx})+(Y-{cy})*(Y-{cy}))"
    leak = f"{tag}_leak"
    return [
        f"color=c=black:s={fmt.width}x{fmt.height}:r={fmt.fps:g},format=rgb24,"
        f"geq=r='{r}*exp(-{dist_sq}/{sigma_sq})'"
        f":g='{g}*exp(-{dist_sq}/{sigma_sq})'"
        f":b='{b}*exp(-{dist_sq}/{sigma_sq})'[{leak}]",
        # shortest=1: the leak source has no `d=`, so it generates frames
        # forever by design (it only needs to outlast whatever it's blended
        # with). `blend` defaults `shortest` to false regardless of which
        # input is finite — confirmed by hitting the hang this fixes.
        f"[{input_label}][{leak}]blend=all_mode=screen:all_opacity={opacity:g}:shortest=1[{output_label}]",
    ]


# --------------------------------------------------------------------------
# Temporal blend: motion blur and ghosting/double-exposure are the same
# mechanism (average a sliding window of recent frames) at different window
# sizes — a small window reads as motion blur, a large one as a trailing
# ghost/echo.
# --------------------------------------------------------------------------

def temporal_blend_filter(frames: int = 3) -> str:
    """`frames` is the sliding-window size. 1 is a no-op (returns "")."""
    frames = max(1, int(frames))
    if frames == 1:
        return ""
    weights = " ".join("1" for _ in range(frames))
    return f"tmix=frames={frames}:weights='{weights}'"


# --------------------------------------------------------------------------
# Chroma key: key one colour to transparent, composite over a replacement.
# --------------------------------------------------------------------------

def chroma_key_stages(
    fmt,
    *,
    input_label: str,
    output_label: str,
    tag: str,
    key_color: str = "0x00FF00",
    similarity: float = 0.18,
    blend: float = 0.05,
    background: str | None = "0x000000",
) -> list[str]:
    """Key `key_color` out of `input_label`.

    `key_color`/`background` are anything ffmpeg's colour parser accepts
    ("green", "#00FF00", "0x00FF00"). With `background` given (the default),
    the keyed result is composited over a flat colour of that size, producing
    an opaque `output_label` — usable directly as a plain post-chain stage.
    With `background=None`, `output_label` keeps its alpha channel instead,
    for a caller compositing over something else (another video branch, a
    generated texture) with its own `overlay`.
    """
    keyed = f"{tag}_keyed"
    stage1 = (
        f"[{input_label}]chromakey=color={key_color}:similarity={similarity:g}"
        f":blend={blend:g}[{keyed if background is not None else output_label}]"
    )
    if background is None:
        return [stage1]

    bg = f"{tag}_bg"
    return [
        stage1,
        f"color=c={background}:s={fmt.width}x{fmt.height}:r={fmt.fps:g}[{bg}]",
        # `color` with no `d=` generates frames forever; `shortest=1` is what
        # stops the render at the real (finite) keyed stream instead of
        # running until killed. Confirmed the hang this fixes by hitting it:
        # this graph, minus `shortest=1`, made ffmpeg simply never finish.
        f"[{bg}][{keyed}]overlay=x=0:y=0:shortest=1[{output_label}]",
    ]
