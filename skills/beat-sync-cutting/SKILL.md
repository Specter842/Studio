---
name: Beat-synced cutting
description: Cut footage to a track's rhythm — analyze the beat grid with analyze_audio, choose cut density from its energy sections, preview with plan_edit, then render. Use when the edit should feel locked to music.
---

# Beat-synced cutting

**Never place cuts by computing beat times yourself and calling a low-level
edit tool per cut.** That is the approach a comparable tool (Kaestral) takes,
and its own skill doc admits the weakness this one exists to avoid: a plain
tempo grid is "excellent for percussive music, weaker on ambient/legato
tracks," with no fallback offered. Frame-by-frame arithmetic repeated by hand
across dozens of cuts is exactly the kind of task an LLM does unreliably —
one dropped frame of rounding at cut 3 compounds by cut 30. `render_edit`
already delegates this to a deterministic assembler that computes every cut
from absolute frame positions (never accumulated durations) and has been
render-verified to land cuts at 0ms error against the beat grid. **Your job
in this skill is choosing the right *parameters* for `render_edit` — cut
density, transition, effects — never re-deriving the cut times yourselves.**

## 1. See what's available

- `list_clips(folder)` — durations, resolution, fps of everything on hand.
  A 30s edit needs clips that add up to at least that much; check before
  promising a duration the footage can't cover.
- `analyze_audio(audio_path)` — returns `bpm`, `beats`, `downbeats`, and
  `sections`, each labelled `low`/`medium`/`high` energy with its own score.
  This is a real librosa beat tracker plus downbeat detection via
  agglomerative clustering — not a bare tempo estimate — so it correctly
  tells a quiet intro from a loud drop.

## 2. Choose cut density from the sections, not a fixed rate

`render_edit` (and the `plan_edit` preview) take `beats_per_cut` per energy
label — how many beats a shot holds before the next cut, chosen once per
section rather than uniformly across the whole track:

| Energy | Typical beats_per_cut | Feel |
|---|---|---|
| low | 8 | held shots, breathing room — an intro, a verse |
| medium | 4 | steady montage pace |
| high | 2 | fast cutting — a chorus, a drop |

Read the `sections` list from `analyze_audio` and pick densities that follow
it, rather than asking for one flat rate across a track that visibly has an
intro/build/drop shape. A track with `sections: low/high/low` should cut
slow, then fast, then slow again — that shape is exactly what the section
labels are for.

## 3. Preview before rendering

Call `plan_edit` with the same `clips`/`audio`/`duration`/`beats_per_cut_*`
you're about to render with. It returns the actual segment list — which clip,
what timeline position, what duration — computed by the *same* assembler
`render_edit` uses, not an approximation. Check the shot lengths look
reasonable (`min_shot_seconds`/`max_shot_seconds` already guard against
40ms flicker cuts and multi-second holds, but a plan with only 2 segments
for a 30s edit usually means the beats_per_cut chosen was too sparse for
the footage on hand) before spending render time.

## 4. Render, then choose effects to match the track's energy

Once the cut plan looks right, call `render_edit` with the same parameters
plus whichever look/transition/effects suit what was actually asked for:

- **Hype / high-energy edit**: `look="punchy"` or `"teal_orange"`,
  `punch_zoom` on the cuts, `speed_ramp=True` for the whip-into-cut feel,
  `glitch` sparingly on the hardest hits only (every_nth, not every cut).
- **Cinematic / calm**: `look="teal_orange"` or leave it unset, `transition`
  a soft crossfade or a slow xfade type (`list_transitions()` for the full
  ~58-name list) instead of hard cuts, no punch-zoom/glitch.
- **Retro / stylized**: `look="vintage"` or `"bleach"`.

Don't reach for every effect at once by default — a montage that's just
correctly cut on the beat, with no grade or punch, is already the hard part
solved. Add effects because the brief called for a specific feel, not
because they exist.

## Notes / limits

- `plan_edit` and `render_edit`'s cut planning is local-clips-only; a
  `--brief` for stock/generated footage goes through `render_edit` directly
  (it runs the full sourcing pipeline pipeline.py does), not through
  `plan_edit`'s preview.
- `analyze_audio` is still an estimator, not ground truth — sanity-check a
  `bpm` that looks implausible (half or double what the track sounds like)
  against `--start-bpm` as a prior rather than trusting it blindly.
- Cost is always $0 on this path. Nothing here calls a paid service.
