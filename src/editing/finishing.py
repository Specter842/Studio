"""Finishing touches: vignette, sharpen/soften, denoise. The last-pass polish
tools in Resolve's colour page. All constant-parameter filters — none of them
touch the per-frame-expression territory that turned out to be unreliable on
this ffmpeg build (see `speed_ramp.py`/`effects.py` for that story), so these
are lower-risk and verified more lightly: one rendered check each is enough
to confirm the parameter actually moves the pixels in the right direction.
"""

from __future__ import annotations


def vignette_filter(strength: float = 0.5) -> str:
    """0 disables it. `strength` maps onto vignette's lens `angle` in
    radians — a *larger* angle darkens the corners more. (An initial version
    of this had that backwards, computing a smaller angle for higher
    strength; caught by a test comparing two renders' actual corner
    brightness, not by reasoning about the formula.)"""
    strength = max(0.0, min(1.0, strength))
    if strength == 0:
        return ""
    angle = 0.35 + strength * (1.3 - 0.35)  # subtle .. heavy, in radians
    return f"vignette=angle={angle:.4f}"


def sharpen_filter(amount: float = 1.0) -> str:
    """0 disables it. Positive sharpens, negative softens — `unsharp`'s own
    luma_amount range, passed straight through."""
    amount = max(-2.0, min(5.0, amount))
    if amount == 0:
        return ""
    return f"unsharp=luma_amount={amount:g}"


def denoise_filter(strength: float = 4.0) -> str:
    """0 disables it. Spatial+temporal luma/chroma denoise via `hqdn3d`,
    scaled off one `strength` knob rather than exposing all four of
    hqdn3d's own parameters — chroma denoised at half the luma strength,
    since over-denoising colour first shows up as smeared edges."""
    strength = max(0.0, strength)
    if strength == 0:
        return ""
    return f"hqdn3d={strength:g}:{strength / 2:g}:{strength:g}:{strength / 2:g}"
