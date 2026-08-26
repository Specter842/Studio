"""Matching detected scene-changes against planned cuts.

Regression coverage for a real bug: the original matcher went
detection-first (for every detection, find the nearest expected cut and call
the distance its error), which let one spurious detection from an effect —
punch-zoom, glitch, a speed ramp's fast stage — claim "error" against
whichever real cut happened to be nearest, however far away. A real render
with all three effects active showed this directly: byte-exact total frame
count and every segment's rendered length matching its plan exactly in
isolation, yet the old matcher reported a multi-second "error" because a
mid-shot speed-ramp whip got matched to a cut a full shot away.

`detect_scene_cuts` is monkeypatched to a canned list for all but one test —
what's under test here is the matching arithmetic, not ffmpeg's scene
detector, and a canned list makes the edge cases exact instead of "whatever
this test pattern happens to trigger."
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import requires_ffmpeg
from editing import verify


def check(monkeypatch, detected: list[float], expected: list[float], **kwargs):
    monkeypatch.setattr(verify, "detect_scene_cuts", lambda *a, **k: detected)
    return verify.verify_cut_accuracy(Path("unused.mp4"), expected, **kwargs)


def test_a_perfect_match_has_zero_error_and_nothing_missed_or_extra(monkeypatch):
    report = check(monkeypatch, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

    assert report.max_error == 0.0
    assert report.missed == ()
    assert report.extra == ()
    assert report.within_tolerance


def test_a_cut_with_no_nearby_detection_is_missed(monkeypatch):
    report = check(monkeypatch, [1.0, 3.0], [1.0, 2.0, 3.0], tolerance=0.1)

    assert report.missed == (2.0,)
    assert len(report.errors) == 2


def test_a_detection_far_from_any_cut_is_extra_not_an_error(monkeypatch):
    """The bug this module exists to prevent: a detection nowhere near any
    real cut must not corrupt max_error just because it has *a* nearest
    neighbour — every detection has one, however far away."""
    report = check(
        monkeypatch,
        detected=[1.0, 5.0],       # 5.0 is nowhere near the one expected cut
        expected=[1.0],
        search_window=0.5,
    )

    assert report.errors == (0.0,)
    assert report.extra == (5.0,)
    assert report.max_error == 0.0


def test_the_real_world_case_many_spurious_detections_near_few_real_cuts(monkeypatch):
    """Mirrors the actual failure: effects fire spurious detections all over
    a 16s edit, but the handful of real cuts still have a close match."""
    expected = [3.333, 6.933, 10.567, 11.533, 12.433, 13.4, 14.333, 15.267]
    detected = expected + [0.92, 1.12, 2.0, 4.5, 7.8, 9.1]  # effect noise

    report = check(monkeypatch, detected, expected, tolerance=0.1, search_window=0.5)

    assert report.missed == ()
    assert report.max_error == pytest.approx(0.0, abs=1e-9)
    assert set(report.extra) == {0.92, 1.12, 2.0, 4.5, 7.8, 9.1}
    assert report.within_tolerance


def test_within_tolerance_ignores_missed_and_extra(monkeypatch):
    report = check(
        monkeypatch,
        detected=[1.02, 99.0],  # 99.0 is extra; 1.02 is within tolerance of 1.0
        expected=[1.0, 2.0],    # 2.0 goes missed
        tolerance=0.05,
    )

    assert report.missed == (2.0,)
    assert 99.0 in report.extra
    assert report.within_tolerance  # the one matched error (0.02s) is inside 0.05s


def test_within_tolerance_is_false_when_a_matched_cut_is_late(monkeypatch):
    report = check(monkeypatch, [1.5], [1.0], tolerance=0.1, search_window=1.0)

    assert report.errors == (0.5,)
    assert not report.within_tolerance


def test_zero_confirmed_is_fail_not_a_vacuous_pass(monkeypatch):
    """Regression: `all(x <= tol for x in [])` is vacuously True, so without
    an explicit check, a run where the detector confirmed nothing at all
    (surfaced by a real Ken Burns render where heavy per-shot staircasing
    overwhelmed the scene detector) reported PASS with 'max error 0ms' —
    reading like a clean result when zero cuts were actually verified."""
    report = check(monkeypatch, detected=[], expected=[1.0, 2.0, 3.0])

    assert report.missed == (1.0, 2.0, 3.0)
    assert report.errors == ()
    assert not report.within_tolerance
    assert "FAIL" in report.summary()
    assert "PASS" not in report.summary()


def test_summary_mentions_extras_only_when_present(monkeypatch):
    clean = check(monkeypatch, [1.0], [1.0])
    noisy = check(monkeypatch, [1.0, 50.0], [1.0], search_window=0.5)

    assert "extra" not in clean.summary()
    assert "extra" in noisy.summary()


def test_two_close_expected_cuts_each_claim_a_different_detection(monkeypatch):
    """Each detection can satisfy at most one expected cut — otherwise two
    cuts 50ms apart could both report a perfect match off a single detection
    that only really confirms one of them."""
    report = check(
        monkeypatch,
        detected=[1.00, 1.05],
        expected=[1.00, 1.05],
        tolerance=0.02,
    )

    assert report.missed == ()
    assert len(report.errors) == 2
    assert report.extra == ()


@requires_ffmpeg
def test_a_real_render_with_a_genuine_hard_cut_is_detected(tmp_path: Path) -> None:
    """Light integration check: an actual black-to-white hard cut is
    findable by the real scene detector, not just the synthetic matcher
    above."""
    import subprocess

    from ffmpeg_tools import ffmpeg_bin

    out = tmp_path / "cut.mp4"
    subprocess.run([
        ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=64x64:r=10:d=1",
        "-f", "lavfi", "-i", "color=c=white:s=64x64:r=10:d=1",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[outv]",
        "-map", "[outv]", "-pix_fmt", "yuv420p", str(out),
    ], check=True, capture_output=True)

    report = verify.verify_cut_accuracy(out, [1.0], tolerance=0.15)

    assert report.missed == ()
    assert report.within_tolerance
