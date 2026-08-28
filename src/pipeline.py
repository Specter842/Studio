"""Phase 1 entry point: local clips + one audio track -> one beat-cut MP4.

    python src/pipeline.py --clips ./inputs/local_clips \
                           --audio ./inputs/track.mp3 \
                           --out ./output/final.mp4

This run makes zero network calls. Nothing here imports the generators package;
stock footage and AI generation arrive in Phase 2 and feed the same assembler.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make `src/` importable whether this file is run as a script, imported by the
# tests, or invoked through the root-level pipeline.py shim.
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config  # noqa: E402
import sourcing  # noqa: E402
from audio.beat_detect import detect_beats  # noqa: E402
from budget import Budget  # noqa: E402
from editing import assembler, compositor  # noqa: E402
from editing.assembler import Effects  # noqa: E402
from editing.looks import LOOKS  # noqa: E402
from editing.transitions import TRANSITIONS, VideoFormat  # noqa: E402
from editing.verify import verify_cut_accuracy  # noqa: E402
from ffmpeg_tools import FFmpegNotFound, ffmpeg_bin, ffprobe_bin  # noqa: E402

log = logging.getLogger("pipeline")

# Phase 1's accuracy target, also the default for --verify.
CUT_TOLERANCE_SECONDS = 0.100


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Cut a folder of clips to the beat of an audio track.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--clips", type=Path, default=Path("./inputs/local_clips"),
        help="Folder of source video files.",
    )
    parser.add_argument(
        "--audio", type=Path, default=Path("./inputs/track.mp3"),
        help="Music track driving the edit.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("./output/final.mp4"),
        help="Output MP4 path.",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="settings.yaml to use (defaults to config/settings.yaml).",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Search the clips folder recursively.",
    )

    sources = parser.add_argument_group("clip sources (Phase 2)")
    sources.add_argument(
        "--brief", default="",
        help="What the video is about. Drives stock search and AI generation. "
             "Separate multiple searches with ';'.",
    )
    sources.add_argument(
        "--stock-per-query", type=int, default=None,
        help="Licensed stock clips to fetch per query (Pexels/Pixabay). Free.",
    )
    sources.add_argument(
        "--generate", type=int, default=None,
        help="AI clips to generate. 0 by default: generation needs a GPU and "
             "takes minutes per clip.",
    )
    sources.add_argument(
        "--generator", default=None,
        help="Adapter to generate with. Defaults to generator_default in "
             "settings.yaml (local_comfyui). Paid adapters need "
             "paid_adapters_enabled: true.",
    )
    sources.add_argument(
        "--no-local", action="store_true",
        help="Ignore the local clips folder and build only from sourced clips.",
    )
    sources.add_argument(
        "--cache-dir", type=Path, default=Path("./inputs/cache"),
        help="Where downloaded and generated assets are kept and reused.",
    )

    output = parser.add_argument_group("output format")
    output.add_argument("--width", type=int, default=None)
    output.add_argument("--height", type=int, default=None)
    output.add_argument("--fps", type=float, default=None)
    output.add_argument(
        "--fit", choices=("pad", "crop"), default=None,
        help="Letterbox the source or fill the frame.",
    )

    editing = parser.add_argument_group("editing")
    editing.add_argument(
        "--transition", choices=TRANSITIONS, default=None,
        help="Hard cuts (frame-exact) or crossfades centred on the beat.",
    )
    editing.add_argument(
        "--start", type=float, default=0.0,
        help="Skip this many seconds of the audio before starting the edit.",
    )
    editing.add_argument(
        "--duration", type=float, default=None,
        help="Cap the output length in seconds (default: the whole track).",
    )
    editing.add_argument(
        "--seed", type=int, default=None,
        help="Seed for clip shuffling, for reproducible edits.",
    )
    editing.add_argument(
        "--start-bpm", type=float, default=None,
        help="Tempo prior for beat tracking; use if the tempo is halved/doubled.",
    )

    fx = parser.add_argument_group("post effects (Phase 4)")
    fx.add_argument(
        "--look", choices=LOOKS, default=None,
        help="Colour grade applied to the whole edit. Free, pure ffmpeg.",
    )
    fx.add_argument(
        "--lut", type=Path, default=None, metavar="FILE",
        help="A .cube (or .3dl/.dat/.m3d) LUT file to apply. Stacks with "
             "--look rather than replacing it.",
    )
    fx.add_argument(
        "--denoise", type=float, default=None, metavar="STRENGTH",
        help="Spatial+temporal denoise (ffmpeg hqdn3d). 0 disables it.",
    )
    fx.add_argument(
        "--sharpen", type=float, default=None, metavar="AMOUNT",
        help="Sharpen (positive) or soften (negative), -2..5. 0 disables it.",
    )
    fx.add_argument(
        "--vignette", type=float, default=None, metavar="STRENGTH",
        help="Darken the corners, 0..1. 0 disables it.",
    )
    fx.add_argument(
        "--grain", type=int, default=None, metavar="STRENGTH",
        help="Film grain, 0..100. 0 disables it.",
    )
    fx.add_argument(
        "--motion-blur", type=int, default=None, metavar="FRAMES",
        help="Blend N consecutive frames — motion blur at a small N, a "
             "trailing ghost/double-exposure look at a larger one. 1 disables it.",
    )
    fx.add_argument(
        "--light-leak", type=float, default=None, metavar="OPACITY",
        help="A warm glow blended into one corner, 0..1 opacity. 0 disables it.",
    )
    fx.add_argument(
        "--chroma-key", default=None, metavar="COLOR",
        help="Key this colour to transparent and composite over "
             "--chroma-key-background (e.g. 'green', '0x00FF00').",
    )
    fx.add_argument(
        "--chroma-key-background", default=None, metavar="COLOR",
        help="Background colour behind a chroma key. Default black.",
    )
    fx.add_argument(
        "--ken-burns", type=float, default=None, metavar="AMOUNT",
        help="A slow zoom across each shot's own duration (0..~0.4 is "
             "reasonable). 0 disables it. Note: --verify's scene detector "
             "cannot see through this effect (confirmed down to 0/N cuts "
             "even at the smallest setting) — every crop+scale step touches "
             "every pixel, about as large a jump as a real cut, so the two "
             "mask each other. The render itself is unaffected; trust output "
             "duration/frame count over --verify when both are in use.",
    )
    fx.add_argument(
        "--ken-burns-direction", choices=("in", "out"), default=None,
        help="Zoom in or out across each shot.",
    )
    fx.add_argument(
        "--ken-burns-alternate", action="store_true",
        help="Flip direction every other shot instead of repeating it.",
    )
    fx.add_argument(
        "--ken-burns-steps", type=int, default=None,
        help="Discrete zoom levels per shot — more is smoother and costs "
             "more filter-graph complexity.",
    )
    fx.add_argument(
        "--punch-zoom", type=float, default=None, metavar="AMOUNT",
        help="Brief zoom-in on every cut, as a fraction (0.15 = 15%%). "
             "0 or omitted disables it.",
    )
    fx.add_argument(
        "--punch-zoom-seconds", type=float, default=None,
        help="How long the punch-zoom holds before snapping back.",
    )
    fx.add_argument(
        "--punch-zoom-every-nth", type=int, default=None,
        help="Only punch-zoom every Nth cut, for edits cut too fast to zoom on all of them.",
    )
    fx.add_argument(
        "--glitch", type=int, default=None, metavar="PIXELS",
        help="RGB channel-split burst on every cut, in pixels of shift. "
             "0 or omitted disables it.",
    )
    fx.add_argument(
        "--glitch-seconds", type=float, default=None,
        help="How long the glitch burst lasts.",
    )
    fx.add_argument(
        "--glitch-every-nth", type=int, default=None,
        help="Only glitch every Nth cut.",
    )
    fx.add_argument(
        "--speed-ramp", action="store_true",
        help="Every shot opens in slow motion and whips up to speed into the "
             "cut. Free, pure ffmpeg (setpts/concat).",
    )
    fx.add_argument(
        "--speed-ramp-fraction", type=float, default=None,
        help="Portion of each shot's own duration spent in the slow stage.",
    )
    fx.add_argument(
        "--speed-ramp-slow", type=float, default=None, metavar="FACTOR",
        help="Playback speed during the slow stage (< 1 = slow motion).",
    )
    fx.add_argument(
        "--speed-ramp-fast", type=float, default=None, metavar="FACTOR",
        help="Playback speed during the fast stage (> 1 = sped up).",
    )

    elements = parser.add_argument_group("3D elements (Phase 3)")
    elements.add_argument(
        "--title", default="",
        help="Render this text as an animated 3D title with Blender and "
             "composite it over the edit. Free, offline, no GPU required.",
    )
    elements.add_argument(
        "--title-at", type=float, default=0.0,
        help="When the title appears, in seconds.",
    )
    elements.add_argument(
        "--title-seconds", type=float, default=2.5,
        help="How long the title stays on screen.",
    )
    elements.add_argument(
        "--title-style", choices=("metal", "matte", "neon"), default=None,
        help="Look of the 3D title.",
    )
    elements.add_argument(
        "--title-position", choices=tuple(compositor.POSITIONS), default=None,
        help="Where the title sits in frame.",
    )

    guards = parser.add_argument_group("guards and diagnostics")
    guards.add_argument(
        "--max-spend-usd", type=float, default=None,
        help="Hard ceiling on estimated paid-API spend for this run.",
    )
    guards.add_argument(
        "--max-inputs-per-pass", type=int, default=None,
        help="Segments rendered per ffmpeg pass; lower this if a render runs "
             "out of memory.",
    )
    guards.add_argument(
        "--dry-run", action="store_true",
        help="Plan the edit and print it, but do not render.",
    )
    guards.add_argument(
        "--verify", action="store_true",
        help="After rendering, measure where cuts actually landed.",
    )
    guards.add_argument(
        "--tolerance", type=float, default=CUT_TOLERANCE_SECONDS,
        help="Cut accuracy tolerance in seconds, used by --verify.",
    )
    guards.add_argument("-v", "--verbose", action="store_true")
    return parser


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
        stream=sys.stdout,
        # Rebind to the current sys.stdout on every call. Without this the
        # handler keeps a reference to whatever stdout was at first use, which
        # breaks any caller that swaps it out (pytest's capsys, notably).
        force=True,
    )
    # numba (via librosa) is extremely chatty at DEBUG.
    for noisy in ("numba", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _resolve_video_format(args: argparse.Namespace, settings) -> VideoFormat:
    return VideoFormat(
        width=int(args.width or settings.get("video.width", 1920)),
        height=int(args.height or settings.get("video.height", 1080)),
        fps=float(args.fps or settings.get("video.fps", 30)),
        fit=str(args.fit or settings.get("video.fit", "pad")),
        pad_color=str(settings.get("video.pad_color", "black")),
        pixel_format=str(settings.get("encode.pixel_format", "yuv420p")),
    )


def run_pipeline(args: argparse.Namespace) -> int:
    settings = config.load(args.config)
    log.info("Settings: %s", settings.source)

    # Free-first is enforced here, not just documented: Phase 1 has no
    # generation step at all, and the budget makes that visible in the log.
    budget = Budget(
        max_spend_usd=(
            args.max_spend_usd
            if args.max_spend_usd is not None
            else float(settings.get("budget.max_spend_usd", 0.0))
        ),
        paid_adapters_enabled=settings.paid_adapters_enabled,
    )
    if settings.paid_adapters_enabled:
        log.warning(
            "paid_adapters_enabled is TRUE. Spend ceiling for this run: $%.2f",
            budget.max_spend_usd,
        )

    try:
        log.debug("ffmpeg:  %s", ffmpeg_bin())
        log.debug("ffprobe: %s", ffprobe_bin())
    except FFmpegNotFound as exc:
        log.error("%s", exc)
        return 2

    video_format = _resolve_video_format(args, settings)
    sourced = sourcing.gather_clips(
        local_dir=None if args.no_local else args.clips,
        brief=args.brief,
        settings=settings,
        budget=budget,
        cache_dir=args.cache_dir,
        video_format=video_format,
        stock_per_query=args.stock_per_query,
        generate_count=args.generate,
        generator_name=args.generator,
        recursive=args.recursive,
        seed=args.seed,
    )

    grid = detect_beats(
        args.audio,
        start_bpm=(
            args.start_bpm
            if args.start_bpm is not None
            else settings.get("beats.start_bpm", 120.0)
        ),
        tightness=float(settings.get("beats.tightness", 100.0)),
        beats_per_bar=int(settings.get("beats.beats_per_bar", 4)),
        trim_silence=bool(settings.get("beats.trim_silence", True)),
    )

    end = None if args.duration is None else args.start + args.duration
    plan = assembler.build_plan(
        sourced.clips,
        grid,
        video_format=video_format,
        beats_per_cut=settings.get(
            "editing.beats_per_cut", {"low": 8, "medium": 4, "high": 2}
        ),
        min_shot_seconds=float(settings.get("editing.min_shot_seconds", 0.2)),
        max_shot_seconds=float(settings.get("editing.max_shot_seconds", 8.0)),
        snap_to_downbeats=bool(settings.get("editing.snap_to_downbeats", True)),
        transition=str(args.transition or settings.get("editing.transition", "cut")),
        crossfade_seconds=float(settings.get("editing.crossfade_seconds", 0.25)),
        audio_fade_out_seconds=float(
            settings.get("editing.audio_fade_out_seconds", 0.5)
        ),
        clip_order=str(settings.get("editing.clip_order", "shuffle")),
        advance_within_clip=bool(settings.get("editing.advance_within_clip", True)),
        seed=args.seed,
        start=args.start,
        end=end,
        effects=Effects(
            look=str(args.look or settings.get("effects.look", "none")),
            lut_path=(
                str(args.lut) if args.lut is not None
                else (settings.get("effects.lut_path") or None)
            ),
            grade_lift=tuple(settings.get("effects.grade_lift", [0.0, 0.0, 0.0])),
            grade_gamma=tuple(settings.get("effects.grade_gamma", [0.0, 0.0, 0.0])),
            grade_gain=tuple(settings.get("effects.grade_gain", [0.0, 0.0, 0.0])),
            grade_saturation=float(settings.get("effects.grade_saturation", 1.0)),
            grade_contrast=float(settings.get("effects.grade_contrast", 1.0)),
            grade_brightness=float(settings.get("effects.grade_brightness", 0.0)),
            denoise_strength=float(
                args.denoise
                if args.denoise is not None
                else settings.get("effects.denoise_strength", 0.0)
            ),
            ken_burns_amount=float(
                args.ken_burns
                if args.ken_burns is not None
                else settings.get("effects.ken_burns_amount", 0.0)
            ),
            ken_burns_steps=int(
                args.ken_burns_steps
                if args.ken_burns_steps is not None
                else settings.get("effects.ken_burns_steps", 8)
            ),
            ken_burns_direction=str(
                args.ken_burns_direction
                if args.ken_burns_direction is not None
                else settings.get("effects.ken_burns_direction", "in")
            ),
            ken_burns_alternate=bool(
                args.ken_burns_alternate
                or settings.get("effects.ken_burns_alternate", False)
            ),
            ken_burns_pan=tuple(settings.get("effects.ken_burns_pan", [0.0, 0.0])),
            light_leak_opacity=float(
                args.light_leak
                if args.light_leak is not None
                else settings.get("effects.light_leak_opacity", 0.0)
            ),
            light_leak_color=tuple(
                settings.get("effects.light_leak_color", [255, 160, 40])
            ),
            light_leak_center=tuple(
                settings.get("effects.light_leak_center", [0.85, 0.15])
            ),
            chroma_key_color=(
                args.chroma_key if args.chroma_key is not None
                else (settings.get("effects.chroma_key_color") or None)
            ),
            chroma_key_similarity=float(
                settings.get("effects.chroma_key_similarity", 0.18)
            ),
            chroma_key_background=str(
                args.chroma_key_background
                if args.chroma_key_background is not None
                else settings.get("effects.chroma_key_background", "0x000000")
            ),
            temporal_blend_frames=int(
                args.motion_blur
                if args.motion_blur is not None
                else settings.get("effects.temporal_blend_frames", 1)
            ),
            grain_strength=int(
                args.grain
                if args.grain is not None
                else settings.get("effects.grain_strength", 0)
            ),
            sharpen_amount=float(
                args.sharpen
                if args.sharpen is not None
                else settings.get("effects.sharpen_amount", 0.0)
            ),
            vignette_strength=float(
                args.vignette
                if args.vignette is not None
                else settings.get("effects.vignette_strength", 0.0)
            ),
            punch_zoom_amount=float(
                args.punch_zoom
                if args.punch_zoom is not None
                else settings.get("effects.punch_zoom_amount", 0.0)
            ),
            punch_zoom_seconds=float(
                args.punch_zoom_seconds
                if args.punch_zoom_seconds is not None
                else settings.get("effects.punch_zoom_seconds", 0.22)
            ),
            punch_zoom_every_nth=int(
                args.punch_zoom_every_nth
                if args.punch_zoom_every_nth is not None
                else settings.get("effects.punch_zoom_every_nth", 1)
            ),
            glitch_shift_px=int(
                args.glitch
                if args.glitch is not None
                else settings.get("effects.glitch_shift_px", 0)
            ),
            glitch_seconds=float(
                args.glitch_seconds
                if args.glitch_seconds is not None
                else settings.get("effects.glitch_seconds", 0.08)
            ),
            glitch_every_nth=int(
                args.glitch_every_nth
                if args.glitch_every_nth is not None
                else settings.get("effects.glitch_every_nth", 1)
            ),
            speed_ramp_enabled=bool(
                args.speed_ramp or settings.get("effects.speed_ramp_enabled", False)
            ),
            speed_ramp_fraction=float(
                args.speed_ramp_fraction
                if args.speed_ramp_fraction is not None
                else settings.get("effects.speed_ramp_fraction", 0.45)
            ),
            speed_ramp_slow_factor=float(
                args.speed_ramp_slow
                if args.speed_ramp_slow is not None
                else settings.get("effects.speed_ramp_slow_factor", 0.5)
            ),
            speed_ramp_fast_factor=float(
                args.speed_ramp_fast
                if args.speed_ramp_fast is not None
                else settings.get("effects.speed_ramp_fast_factor", 1.3)
            ),
        ),
    )

    if args.dry_run:
        print("\nPlanned edit (nothing rendered):")
        print(plan.describe_segments())
        _report(sourced, budget)
        return 0

    encode = settings.get("encode", {}) or {}
    # With a title the edit is rendered to a scratch file first and the element
    # burned on in a second pass; without one it goes straight to --out.
    render_target = (
        args.out.with_name(f"{args.out.stem}_base{args.out.suffix}")
        if args.title else args.out
    )

    assembler.render(
        plan,
        render_target,
        encode=encode,
        max_inputs_per_pass=int(
            args.max_inputs_per_pass
            if args.max_inputs_per_pass is not None
            else settings.get(
                "render.max_inputs_per_pass",
                assembler.DEFAULT_MAX_INPUTS_PER_PASS,
            )
        ),
    )

    if args.title:
        _add_title(args, settings, video_format, render_target, encode)

    exit_code = 0
    if args.verify:
        report = verify_cut_accuracy(
            args.out, list(plan.cut_times), tolerance=args.tolerance
        )
        print(f"\n{report.summary()}")
        if not report.within_tolerance:
            log.error("Cuts fell outside the %.0fms tolerance.", args.tolerance * 1000)
            exit_code = 1

    print(f"\nWrote {args.out}")
    _report(sourced, budget)
    return exit_code


def _add_title(
    args: argparse.Namespace,
    settings,
    video_format: VideoFormat,
    base_path: Path,
    encode: dict,
) -> None:
    """Render the 3D title and composite it over the finished edit.

    Imported lazily so a run without --title never touches Blender, and failing
    softly: an unavailable Blender should cost you the title, not the video you
    already spent minutes rendering. The base render is promoted to --out in
    that case so the run still produces the file it promised.
    """
    from animation3d.blender import BlenderError, BlenderNotFound, render_text_reveal

    try:
        element = render_text_reveal(
            args.title,
            cache_dir=Path(args.cache_dir) / "elements",
            width=video_format.width,
            height=video_format.height,
            fps=video_format.fps,
            seconds=args.title_seconds,
            color=_parse_colour(settings.get("elements.title_color", "1,1,1")),
            style=str(
                args.title_style or settings.get("elements.title_style", "metal")
            ),
        )
        compositor.composite(
            base_path,
            [compositor.Overlay(
                input_path=element.input_path,
                fps=element.fps,
                start=args.title_at,
                duration=element.duration,
                fade=float(settings.get("elements.title_fade_seconds", 0.3)),
                position=str(
                    args.title_position
                    or settings.get("elements.title_position", "center")
                ),
            )],
            args.out,
            encode=encode,
        )
        base_path.unlink(missing_ok=True)
    except (BlenderNotFound, BlenderError) as exc:
        log.error("Title skipped: %s", exc)
        base_path.replace(args.out)


def _parse_colour(value) -> tuple[float, float, float]:
    """Accept "1,0.8,0.2" or a YAML list. Falls back to white."""
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        parts = list(value)[:3]
    else:
        parts = str(value).split(",")
    try:
        channels = [max(0.0, min(1.0, float(part))) for part in parts]
    except (TypeError, ValueError):
        return (1.0, 1.0, 1.0)
    return tuple(channels[:3]) if len(channels) >= 3 else (1.0, 1.0, 1.0)


def _report(sourced: sourcing.SourcingResult, budget: Budget) -> None:
    """Close every run with where the clips came from and what it cost."""
    print(f"Sources: {sourced.summary()}")
    if sourced.warnings:
        # Printed, not just logged: a run that silently fell back to local-only
        # because a key was missing looks identical to one that worked.
        print("Warnings:")
        for warning in sourced.warnings:
            print(f"  - {warning}")
    print(budget.report())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    try:
        return run_pipeline(args)
    except KeyboardInterrupt:
        log.error("Interrupted.")
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report, don't traceback
        log.error("%s: %s", type(exc).__name__, exc)
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
