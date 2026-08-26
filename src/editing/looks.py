"""Color grade presets, applied once to the assembled edit.

Each look is a plain ffmpeg filter-chain fragment — no brackets, no labels —
meant to be spliced into the graph after the transition/concat stage and
before the final `format=`. Built entirely from filters already shipped in
ffmpeg (`eq`, `curves`, `colorbalance`, `vignette`); nothing here needs a LUT
file, a network fetch, or anything paid.

Parameter names and ranges are taken from the installed ffmpeg's own
`-h filter=eq` / `-h filter=curves` / `-h filter=colorbalance` output, not
guessed, so a preset here is a filter graph that is known to parse.
"""

from __future__ import annotations

NONE = "none"

# name -> ordered list of filter invocations (each a complete "name=args" or
# bare "name" token; joined with commas by look_filters()).
_LOOKS: dict[str, list[str]] = {
    NONE: [],
    # The reference: crushed shadows, desaturated, high contrast, a cool
    # shift toward blue in the highlights, edges darkened.
    "blackout": [
        "curves=preset=increase_contrast",
        "eq=contrast=1.18:brightness=-0.055:saturation=0.72:gamma=0.92",
        "colorbalance=bs=0.10:bm=0.04:rs=-0.06",
        "vignette=angle=PI/4.3",
    ],
    # Classic blockbuster grade: shadows pushed cyan/teal, highlights pushed
    # warm/orange, so skin tones and fire/headlights read as the "pop".
    "teal_orange": [
        "colorbalance=rs=-0.12:gs=0.04:bs=0.16:rh=0.14:gh=0.02:bh=-0.12",
        "eq=contrast=1.08:saturation=1.15",
    ],
    # Saturated and contrasty — the "hype edit" default look.
    "punchy": [
        "eq=contrast=1.22:saturation=1.35:gamma=1.04",
        "curves=preset=medium_contrast",
    ],
    # Bleach-bypass: desaturated, lifted, high contrast, slightly cool.
    "bleach": [
        "eq=saturation=0.35:contrast=1.28:brightness=0.03",
        "curves=preset=strong_contrast",
        "colorbalance=bs=0.05:bh=0.05",
    ],
    # Faded, warm, slightly lifted blacks — old-footage feel.
    "vintage": [
        "curves=preset=vintage",
        "eq=saturation=0.85:contrast=0.96",
    ],
}

LOOKS = tuple(_LOOKS)


def look_filters(name: str) -> str:
    """The filter-chain fragment for `name`, or "" for `"none"`.

    Returns a comma-joined fragment with no leading/trailing comma, ready to
    be appended to an existing filter chain — callers add the comma.
    """
    if name not in _LOOKS:
        raise ValueError(f"Unknown look {name!r}; expected one of {LOOKS}.")
    return ",".join(_LOOKS[name])


# --------------------------------------------------------------------------
# 3D LUTs: any .cube from any free source (Resolve ships hundreds; sites like
# RocketStock and IWLTBAP give away more), not just the five presets above.
# --------------------------------------------------------------------------

def _escape_lut_path(path: str) -> str:
    """Make a filesystem path safe inside an ffmpeg filter option value.

    ffmpeg's filter syntax uses `:` to separate a filter's own options — a
    Windows path's drive-letter colon (`C:\\...`) collides with that
    unless escaped. Backslashes need the same treatment. Verified against a
    real drive-letter path rather than assumed: an unescaped `C:\\` here
    parses as filter option `C` with no value and fails, not silently.
    """
    return path.replace("\\", "\\\\").replace(":", "\\:")


def lut_filter(path: str, *, interp: str = "tetrahedral") -> str:
    """A `.cube` (or `.3dl`/`.dat`/`.m3d`) LUT applied via `lut3d`.

    `interp` is passed straight to ffmpeg's own `lut3d` interpolation modes
    (tetrahedral, trilinear, nearest, pyramid, prism) — tetrahedral is its own
    default and the one most LUT authors tune against.
    """
    return f"lut3d=file='{_escape_lut_path(path)}':interp={interp}"


# --------------------------------------------------------------------------
# General 3-way colour correction: lift (shadows) / gamma (midtones) /
# gain (highlights) per RGB channel, the same shape as a Resolve primary
# wheel. The five presets above are convenient names for points in this same
# space; this is the space itself, for anything that needs a grade built from
# numbers rather than picked from a list — e.g. a grade synthesized to match
# a reference video's own measured colour statistics.
# --------------------------------------------------------------------------

def custom_grade(
    *,
    lift: tuple[float, float, float] = (0.0, 0.0, 0.0),
    gamma: tuple[float, float, float] = (0.0, 0.0, 0.0),
    gain: tuple[float, float, float] = (0.0, 0.0, 0.0),
    saturation: float = 1.0,
    contrast: float = 1.0,
    brightness: float = 0.0,
) -> str:
    """Build a grade from numbers instead of a preset name.

    `lift`/`gamma`/`gain` are (r, g, b) in [-1, 1], matching `colorbalance`'s
    own shadow/midtone/highlight ranges exactly — passed straight through, not
    rescaled, so a caller reasoning in "Resolve wheel" terms gets what it asked
    for. `saturation`/`contrast`/`brightness` map onto `eq` the same way the
    fixed presets already do.
    """
    parts = []
    rs, gs, bs = lift
    rm, gm, bm = gamma
    rh, gh, bh = gain
    if any((rs, gs, bs, rm, gm, bm, rh, gh, bh)):
        parts.append(
            f"colorbalance=rs={rs:g}:gs={gs:g}:bs={bs:g}"
            f":rm={rm:g}:gm={gm:g}:bm={bm:g}"
            f":rh={rh:g}:gh={gh:g}:bh={bh:g}"
        )
    if saturation != 1.0 or contrast != 1.0 or brightness != 0.0:
        parts.append(f"eq=saturation={saturation:g}:contrast={contrast:g}:brightness={brightness:g}")
    return ",".join(parts)
