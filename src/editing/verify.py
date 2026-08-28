"""Measure how close the rendered cuts actually landed to the beats.

Phase 1's definition of done is a number, not an impression: cuts within
~100ms of a detected beat. This re-reads the finished MP4 with ffmpeg's scene
detector and compares where the picture actually changes against where the
plan said it would, so the claim can be checked rather than eyeballed.

Scene detection is a heuristic and it under-reports by design. ffmpeg scores a
frame as `min(mean absolute difference from the previous frame, change in that
difference)` — so in a run of cuts of similar visual magnitude, each one makes
the next look unremarkable and the score collapses. A steadily-paced montage
can easily have a quarter of its cuts go unscored.

So a missed cut here means "could not confirm", never "landed wrong". That is
why `max_error` is reported over the cuts that *were* detected and misses are
listed separately, and why `within_tolerance` ignores misses entirely.

Post effects (punch-zoom, glitch, a speed ramp's own whip-fast stage) are
visually dramatic *on purpose*, and the scene detector cannot tell "this is a
big jump because it's a cut" from "this is a big jump because the shot just
whipped up to 1.3x speed" — both trip the same threshold. Matching used to go
detection-first: for every detected moment, find the nearest expected cut and
call the distance its error. That let one spurious mid-shot detection from an
effect claim the "error" against whichever real cut happened to be nearest —
sometimes a whole shot away — inflating `max_error` into the seconds even
though every actual cut had landed exactly on time (confirmed independently:
byte-exact total frame count, and every segment's rendered frame count
matching its plan exactly in isolation). Matching now goes expected-first:
each *real* cut looks for the nearest detection within a bounded search
window, and detections that match no expected cut at all are reported
separately as `extra` — informational, likely effect-induced — rather than
folded into the accuracy numbers they would otherwise have corrupted.

Ken Burns specifically defeats this detector *completely* (0/N confirmed),
confirmed down to the smallest setting tried (2 steps, 8% zoom) — not
"sometimes interferes" like punch-zoom/glitch/speed-ramp, which still leave
most cuts confirmable. Every crop+scale step resamples the entire frame, a
jump about as large as an actual cut, and the scene score is explicitly
relative (`min(diff, change in diff)` — see `detect_scene_cuts`'s use of it
below), so two comparably-sized jumps close in time suppress each other
rather than one winning. This is a property of the detector, not the
render: total frame count and duration stay byte-exact with Ken Burns
active, verified independently every time it was tested. There is no
larger `search_window` that fixes this — the detections are not slightly
mis-timed, they mostly never fire at all.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from ffmpeg_tools import ffmpeg_bin, run

log = logging.getLogger(__name__)

_PTS_TIME = re.compile(r"pts_time:([0-9]+\.?[0-9]*)")

# Distinct shots score well above this; it is low enough to catch cuts between
# similar-looking footage without firing on fast camera motion within a shot.
DEFAULT_SCENE_THRESHOLD = 0.3


@dataclass(frozen=True)
class CutAccuracyReport:
    expected: tuple[float, ...]
    detected: tuple[float, ...]
    errors: tuple[float, ...]      # |matched detection - expected|, per confirmed cut
    missed: tuple[float, ...]      # expected cuts with no detection nearby
    # Detections that matched no expected cut within the search window — most
    # often a punch-zoom/glitch burst or a speed ramp's fast stage, not a
    # timing problem. Reported for visibility, never folded into the errors
    # above: an effect being visually dramatic is the point, not a defect.
    extra: tuple[float, ...]
    tolerance: float

    @property
    def max_error(self) -> float:
        return max(self.errors) if self.errors else 0.0

    @property
    def mean_error(self) -> float:
        return sum(self.errors) / len(self.errors) if self.errors else 0.0

    @property
    def within_tolerance(self) -> bool:
        # `all([])` is vacuously True — without the explicit `errors` check,
        # a run where the detector confirmed *nothing* (every cut missed)
        # reports PASS, which is backwards: zero evidence is not the same
        # claim as "every cut landed on time," and reporting it that way is
        # worse than useless — it hides exactly the situation (heavy
        # per-shot effects like Ken Burns overwhelming the scene detector's
        # own frame-difference heuristic) where a human most needs to know
        # this check could not actually verify anything.
        return bool(self.errors) and all(error <= self.tolerance for error in self.errors)

    def summary(self) -> str:
        confirmed = len(self.errors)
        if confirmed == 0:
            # Not "0/8, max error 0ms" — that reads like a clean pass at a
            # glance. Say plainly that nothing was confirmed.
            line = (
                f"0/{len(self.expected)} cuts confirmed by scene detection — "
                f"nothing to verify against -> FAIL"
            )
        else:
            line = (
                f"{confirmed}/{len(self.expected)} cuts confirmed by scene detection; "
                f"max error {self.max_error * 1000:.0f}ms, "
                f"mean {self.mean_error * 1000:.0f}ms "
                f"(tolerance {self.tolerance * 1000:.0f}ms) -> "
                f"{'PASS' if self.within_tolerance else 'FAIL'}"
            )
        if self.extra:
            line += (
                f"  [{len(self.extra)} extra detection(s) near no expected cut — "
                f"likely a punch-zoom/glitch/speed-ramp/Ken-Burns moment, not a timing miss]"
            )
        if confirmed < len(self.expected) / 2 and confirmed > 0:
            line += (
                f"  [confirmation rate is low ({confirmed}/{len(self.expected)}) — "
                f"heavy per-shot effects (Ken Burns especially) can overwhelm the "
                f"scene detector even when the render itself is exactly on time; "
                f"check total duration/frame count independently before assuming "
                f"a real problem]"
            )
        return line


def detect_scene_cuts(
    video_path: Path | str, *, threshold: float = DEFAULT_SCENE_THRESHOLD
) -> list[float]:
    """Timestamps where the picture visibly changes, in seconds."""
    output = run([
        ffmpeg_bin(),
        "-hide_banner", "-nostdin", "-v", "error",
        "-i", str(video_path),
        "-an",
        "-vf", f"select='gt(scene,{threshold})',metadata=print:file=-",
        "-f", "null", "-",
    ])
    return [float(match) for match in _PTS_TIME.findall(output)]


def verify_cut_accuracy(
    video_path: Path | str,
    expected_cut_times: list[float],
    *,
    tolerance: float = 0.1,
    threshold: float = DEFAULT_SCENE_THRESHOLD,
    search_window: float | None = None,
) -> CutAccuracyReport:
    """Compare rendered cut positions against the planned ones.

    `search_window` bounds how far a detection may sit from an expected cut
    and still count as *that cut's* evidence, rather than an unrelated
    detection whose distance would otherwise corrupt the accuracy numbers.
    Defaults to `max(5 * tolerance, 0.5s)` — wide enough that a genuinely
    late/early cut is still caught and reported as a real error, narrow
    enough that a punch-zoom burst three shots away cannot masquerade as
    evidence for a cut it has nothing to do with.
    """
    window = search_window if search_window is not None else max(5 * tolerance, 0.5)
    detected = detect_scene_cuts(video_path, threshold=threshold)
    expected = sorted(expected_cut_times)

    remaining = list(detected)
    errors: list[float] = []
    missed: list[float] = []
    for cut in expected:
        if remaining:
            nearest = min(remaining, key=lambda moment: abs(moment - cut))
            distance = abs(nearest - cut)
        else:
            nearest, distance = None, None

        if nearest is not None and distance <= window:
            errors.append(distance)
            remaining.remove(nearest)
        else:
            missed.append(cut)

    report = CutAccuracyReport(
        expected=tuple(expected),
        detected=tuple(detected),
        errors=tuple(errors),
        missed=tuple(missed),
        extra=tuple(remaining),
        tolerance=tolerance,
    )
    log.info("Cut accuracy: %s", report.summary())
    return report
