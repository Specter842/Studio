# video_pipeline

A phased video generation and editing pipeline. The default path costs nothing
per run: local footage, local analysis, local rendering. Paid APIs are optional
adapters behind a common interface, disabled in config, and never invoked
unless somebody turns them on deliberately.

**Status: all three phases complete.**

```bash
# Phase 1: cut a folder of clips to a track
python pipeline.py --clips ./inputs/local_clips --audio ./inputs/track.mp3 --out ./output/final.mp4

# Phase 2: pull in licensed stock footage too
python pipeline.py --audio ./inputs/track.mp3 --brief "rainy city at night; neon signs" --out ./output/final.mp4

# Phase 3: add a 3D title, and/or drive it all over HTTP from n8n
python pipeline.py --audio ./inputs/track.mp3 --title "BEAT CUT" --out ./output/final.mp4
python src/orchestrator_cli.py          # POST /render on 127.0.0.1:8712 — also serves the studio UI at /
```

Everything above runs at **$0.00**. The only paid pieces are two opt-in
adapters that cannot even be constructed unless you deliberately enable them.

---

## Setup

ffmpeg and ffprobe must be on `PATH`:

```bash
winget install Gyan.FFmpeg
```

(macOS: `brew install ffmpeg`; Debian/Ubuntu: `apt install ffmpeg`.) If you
would rather not touch `PATH`, set `FFMPEG_BIN` and `FFPROBE_BIN` in `.env`.

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env      # then fill in whichever keys you want
```

Phase 1 needs no keys at all. Phase 2's stock sourcing needs two free ones.

---

## Where clips come from

Three sources, all producing the same `ClipInfo` objects. `assembler.py` cannot
tell them apart, which is the seam the whole design hangs on.

| Source | Cost | Needs | Flag |
| --- | --- | --- | --- |
| Local folder | free | nothing | `--clips ./inputs/local_clips` |
| Pexels + Pixabay | free | a free API key each | `--brief "..."` |
| ComfyUI (local AI) | free | a discrete GPU, 8–16 GB VRAM | `--generate N` |
| fal.ai gateway | **paid** | opt-in + a key | `--generator fal_gateway` |

`--brief` drives both search and generation. Separate multiple searches with
`;`. Generation is **0 by default** — not a free-first compromise, but because
it costs minutes per clip and needs a GPU, so it should be asked for rather
than assumed. When you do ask, it runs through `generator_default`, which is
the local, free adapter.

### Stock footage (the practical free path)

Get the keys — both are free, no card, about two minutes:

- Pexels: https://www.pexels.com/api/
- Pixabay: https://pixabay.com/api/docs/

Put them in `.env`, then:

```bash
python pipeline.py --audio ./inputs/track.mp3 --brief "aerial coastline; storm clouds" --stock-per-query 3 --out ./output/final.mp4
```

Downloads land in `inputs/cache/stock/` and are reused across runs. Neither
service requires attribution, but `CREDITS.json` and `CREDITS.md` are written
next to the cache anyway — it costs nothing and it is the difference between
being able to credit contributors and not.

**Not supported, deliberately:** pulling video from YouTube, TikTok, Instagram
or arbitrary sites. That is a copyright and terms-of-service problem, not a
technical one, and no amount of care in the code fixes it. Likewise no
unofficial Midjourney wrappers — Midjourney has no public API, and the wrappers
that exist automate Discord in violation of its terms and get accounts banned.
FLUX through ComfyUI or Nano Banana through fal.ai are the real-API routes to a
similar look.

### Local AI generation (ComfyUI)

1. Install ComfyUI and put a video checkpoint in `models/`. **Wan 2.2** first —
   best VRAM-to-quality ratio on consumer cards, runs on 8–16 GB with GGUF
   quantization. HunyuanVideo 1.5 and LTX-2.3 are drop-in alternates.
2. ComfyUI **Settings → enable Dev Mode**.
3. Build the graph in the visual editor until one clip comes out right, then
   **Save (API Format)** into `config/comfyui_workflows/`.
4. Point `comfyui.workflow` in `settings.yaml` at it.

You do **not** have to transcribe node IDs. The adapter reads the exported
graph and follows its links to find the positive prompt, the negative prompt,
the geometry, the frame count and the seed — following links is the only way to
tell the positive text encoder from the negative one, since they are the same
node class. Check what it found:

```bash
python src/generators/local_comfyui.py --inspect config/comfyui_workflows/wan22_t2v_api.json
```

If a guess is wrong, paste the printed block into `comfyui.inputs` in
`settings.yaml` and correct it. Explicit config always wins.

```bash
python pipeline.py --audio ./inputs/track.mp3 --brief "slow drone shot over a frozen lake" --generate 3 --out ./output/final.mp4
```

### Paid generation (optional, off)

`fal_gateway` fronts Seedance, Kling, Veo3, FLUX and Nano Banana behind one
auth pattern. It is for the specific shot where local generation genuinely is
not good enough, and somebody has decided to pay. To enable it you must, on
purpose: set `paid_adapters_enabled: true`, name a model in `fal.model`, put
`FAL_KEY` in `.env`, and pass `--generator fal_gateway`. Missing any one of
those is a clear error, never a silent charge.

Fill in `fal.costs` from fal.ai's current pricing before relying on the budget
guard being accurate. No prices are hardcoded here — they change, and a stale
number in a repo is worse than no number.

---

## Useful flags

| Flag | Effect |
| --- | --- |
| `--dry-run` | Print the planned timeline (with each shot's source), render nothing. |
| `--verify` | After rendering, measure how far each cut landed from its beat. |
| `--brief "a; b"` | What the video is about. Drives stock search and generation. |
| `--stock-per-query N` | Stock clips per search query. |
| `--generate N` | AI clips to generate this run. |
| `--generator NAME` | Which adapter generates. Defaults to `local_comfyui`. |
| `--no-local` | Build only from sourced clips, ignoring the clips folder. |
| `--max-spend-usd N` | Hard ceiling on estimated paid spend for this run. |
| `--seed 42` | Reproducible clip shuffling and generation seeds. |
| `--duration 30` / `--start 12` | Cap the length / skip into the track. |
| `--transition crossfade` | Dissolve instead of cut, centred on the beat. |
| `--start-bpm 140` | Tempo prior, for tracks tracked at half or double time. |
| `--width/--height/--fps/--fit` | Override the output format. |
| `--max-inputs-per-pass 8` | Fewer segments per ffmpeg pass, if a render runs out of memory. |
| `-v` | Debug logging, including the full ffmpeg filter graph. |

Everything not passed on the command line comes from `config/settings.yaml`.

---

## Post effects (Phase 4)

The After Effects/DaVinci Resolve/Fusion feature set, built as free ffmpeg
filters rather than by scripting the actual paid apps. Every knob defaults to
off — a run that never mentions this section renders exactly as it always
did — and every effect here was verified by rendering a real test pattern and
measuring the actual pixels, not by trusting a filter's documentation. That
discipline caught three real bugs during development: `geq` silently
rejecting lowercase `w`/`h` (it wants `W`/`H`), `overlay`/`blend` defaulting
`shortest=false` regardless of which input is actually finite (a generated
light-leak/chroma-key background with no fixed duration hung the render
indefinitely until `shortest=1` was added explicitly), and — the one that
mattered most — a scene-detection verifier that matched detections to their
*nearest* expected cut rather than the other way around, so one spurious
detection from an effect could report a multi-second "error" against a real
cut that had actually landed exactly on time.

| Flag | Effect |
| --- | --- |
| `--look NAME` | A named colour-grade preset: `blackout`, `teal_orange`, `punchy`, `bleach`, `vintage`. |
| `--lut FILE` | Any `.cube`/`.3dl`/`.dat`/`.m3d` LUT. Stacks with `--look`. |
| `--denoise N` | Spatial+temporal denoise. |
| `--sharpen N` | Sharpen (positive) or soften (negative). |
| `--vignette N` | Darken the corners, 0–1. |
| `--grain N` | Film grain, 0–100. |
| `--motion-blur N` | Blend N consecutive frames — motion blur at a small N, a trailing ghost/double-exposure look at a larger one. |
| `--light-leak N` | A warm glow blended into one corner, generated in-graph — no stock footage needed. |
| `--chroma-key COLOR` / `--chroma-key-background COLOR` | Key a colour to transparent and composite over a flat background. |
| `--punch-zoom N` / `--glitch N` | A brief zoom or RGB-split burst on every cut. |
| `--speed-ramp` | Every shot opens in slow motion and whips up to speed into the cut. |
| `--ken-burns N` | A slow zoom across each shot's own duration. `--ken-burns-direction`, `--ken-burns-alternate`, `--ken-burns-steps`. |

**Continuous per-frame animation does not work on this ffmpeg build.**
`scale`/`crop` driven by `t`, `zoompan`'s documented accumulator recipe, and
`drawbox`/`drawtext` position expressions were all tried and all produced
zero visible motion, confirmed by rendering and measuring, not assumed.
Everything that looks animated here — punch-zoom, Ken Burns, the speed
ramp's two-stage split — is actually a small number of fixed, pre-computed
states switched with `overlay`'s `enable=`, which does reliably respond
per-frame. Ken Burns staircases through this in discrete steps rather than
easing smoothly; more `--ken-burns-steps` looks smoother at the cost of
filter-graph size.

**`--verify` cannot see through `--ken-burns`.** Confirmed down to the
smallest setting tried (2 steps, 8% zoom): 0 of N cuts confirmed, every
time. Every Ken Burns step resamples the whole frame — a jump about as
large as a real cut — and the scene detector's score is explicitly
*relative*, so two comparably-sized jumps close together suppress each
other rather than one winning. This is a property of the detector, not the
render: total output duration and frame count stay byte-exact with Ken
Burns active, independently confirmed every time. Punch-zoom/glitch/speed-
ramp interfere less severely (most cuts stay confirmable; a handful of
effect-induced detections are reported separately as "extra," not folded
into the accuracy numbers) — Ken Burns is the one that defeats it outright.

---

## How it fits together

```
inputs/local_clips/*.mp4 ──▶ ingest/local_clips.py  ─┐
                                                     │
--brief ──▶ ingest/stock_fetch.py  (Pexels/Pixabay) ─┤
        └─▶ generators/…                            ─┤   all become ClipInfo
              base_adapter.py   (one interface)      │
              local_comfyui.py  (default, free)      │
              fal_gateway.py    (optional, paid)     │
                                                     ▼
track.mp3 ──▶ audio/beat_detect.py ──────▶ editing/assembler.py
              (beats, downbeats,           (which beat, which clip)
               sections)                            │
                                          editing/transitions.py
                                                    ▼
                                            output/final.mp4
```

`sourcing.py` is the only place that knows there is more than one source, and
it is deliberately forgiving: a missing API key or an unreachable ComfyUI costs
you those clips and nothing else. Every run ends with a `Sources:` line and any
warnings printed, not just logged — a run that quietly fell back to local-only
because a key was missing otherwise looks identical to one that worked.

---

## Four design decisions worth knowing about

**Cut positions are quantised against absolute time, never accumulated.** Each
cut's frame index is `round(beat_time * fps)`, and segment lengths are the
differences between those. Summing rounded per-segment durations instead lets a
fraction of a frame of error build up at every cut, which is how an edit ends up
visibly behind the music by the last chorus. Segments are then clamped with
`trim=end_frame=N`, so a segment cannot be a frame long or short.

**Cut density follows the music, not a fixed interval.** `beat_detect.py`
splits the track into sections by timbre and harmony, labels each low/medium/
high by energy, and `assembler.py` cuts every 8/4/2 beats accordingly — holding
through an intro and cutting fast through a drop. Candidates are also nudged
onto bar lines where one is within a beat. This follows the approach in
[BeatSync Engine](https://github.com/Merserk/BeatSync-Engine).

**Rendering is capped at 24 segments per ffmpeg pass.** Each segment in a pass
needs its own decoder holding reference frames — measured at ~24 MB per input on
top of a ~1.1 GB floor for a 1080p x264 encode. That floor is fixed, the growth
is not, so a 200-segment edit in one pass wants several gigabytes and dies with
`x264 [error]: malloc ... failed`. Longer edits render in capped passes joined
with the concat demuxer at `-c:v copy` — nothing decoded twice, no quality lost.
The one visible consequence: a crossfade cannot span a pass boundary, so those
joins become hard cuts and the run says so.

**Budget is reserved before the call, not after.** A ledger updated only on
success cannot stop a loop that keeps failing and retrying, and a retried
request is a second billable call. Adapters charge once per *attempt*. Only
server errors are retried — a 400 will be rejected identically next time, so
retrying it just buys the same answer twice.

---

## Cost guard

`src/budget.py` tracks estimated spend and halts when it exceeds
`--max-spend-usd` (default: `budget.max_spend_usd`, `0.0`). Three things have
to fail before a surprise bill is possible:

1. A paid adapter cannot even be **constructed** while `paid_adapters_enabled`
   is false — the guard is at the factory, so no code path reaches a paid API
   by accident.
2. Every attempt is charged before it is made.
3. Zero-cost adapters record nothing, so the ledger is a precise record of money
   at risk rather than a call log.

Every run prints its total. On the default path that is always
`Estimated run cost: $0.00 (no paid calls made)`.

---

## Tests

```bash
python -m pytest
```

154 tests, a few seconds. No committed binaries and no network: test media is
synthesised at session start, and the adapters run against **mock HTTP servers
speaking the real Pexels, Pixabay, ComfyUI and fal.ai protocols** — real
sockets, real query strings, real headers, real streaming downloads. Mocking
httpx instead would replace exactly the parts of an API client where the bugs
live.

`tests/test_pipeline_e2e.py` renders with outbound sockets blocked, then
measures the result frame by frame: every planned cut must appear at exactly
its frame and nowhere else.

Blender tests skip cleanly when Blender is not installed, the orchestrator
tests skip without FastAPI, and `tests/test_transcribe.py` (plus the two
`transcribe`-related cases in `test_mcp_server.py`) skip without Windows
SAPI available to generate known-ground-truth test speech — real TTS audio
through the real Whisper model, not a mocked transcript, the same principle
as the click track. The first run downloads the Whisper model (~140MB) from
Hugging Face; every run after that is offline, cached at
`~/.cache/huggingface`.

**On a memory-constrained machine**, tests can fail in a full run while passing
individually — Blender needs roughly a gigabyte to start, faster-whisper's
model load is the single biggest individual allocation in the whole suite
(a caught allocation failure under load, and once an outright process crash,
both observed while this was being built — never a bug in the code itself,
confirmed by the same test passing cleanly once memory was available), and
the effects suite (`test_looks`, `test_compositing`, `test_finishing`,
`test_speed_ramp`, `test_verify`, plus every rendered assertion in
`test_effects`) launches a real ffmpeg subprocess per assertion, on top of
pytest, librosa and ffmpeg all already running. The tell is
`[WinError 1455] The paging file is too small`, an `mkl_malloc`/`MemoryError`
from faster-whisper, tests erroring (not failing) only in a full run, or
different tests failing on each run. Enabling a page file fixes it; so does
splitting the run:

```bash
python -m pytest tests/test_beat_detect.py tests/test_assembler.py tests/test_local_clips.py tests/test_budget.py tests/test_transitions.py tests/test_generators.py tests/test_local_comfyui.py tests/test_fal_gateway.py tests/test_heygen.py
python -m pytest tests/test_stock_fetch.py tests/test_sourcing.py tests/test_pipeline_e2e.py
python -m pytest tests/test_looks.py tests/test_effects.py tests/test_compositing.py tests/test_finishing.py tests/test_speed_ramp.py tests/test_verify.py
python -m pytest tests/test_animation3d.py tests/test_orchestrator.py
python -m pytest tests/test_transcribe.py tests/test_mcp_server.py
```

---

## 3D elements (Blender, free)

Install Blender from [blender.org](https://www.blender.org/download/) — that is
the whole setup. It is found automatically on PATH or in the usual install
locations; `BLENDER_BIN` in `.env` overrides that. It is **optional**: without
it you lose 3D titles and nothing else.

```bash
python pipeline.py --audio track.mp3 --title "BEAT CUT" --title-style neon --title-at 1.5
```

Blender is driven as a subprocess (`--background --factory-startup --python`),
never imported: `bpy` only exists inside Blender's own Python, so it cannot be
a pip dependency, and the subprocess boundary means a Blender crash costs you
one element instead of the whole render.

Elements are rendered as **RGBA PNG sequences**, not video — Blender's video
writers drop alpha in most codec combinations, and ffmpeg composites an image
sequence directly and losslessly. Compositing is a separate ffmpeg pass, so it
cannot disturb the beat alignment; the audio is stream-copied through
untouched.

Rendering is CPU-bound without a GPU (roughly a second a frame at 640×360 on
integrated graphics), so elements are cached on their content **and** on the
script version — changing how a title looks invalidates the old frames instead
of silently serving them.

## Orchestration (n8n, free, self-hosted)

```bash
python src/orchestrator_cli.py            # http://127.0.0.1:8712
```

`POST /render` takes a job document and runs the pipeline; `GET /jobs/{id}` and
`GET /health` support it. Runs execute as subprocesses, so the server stays
answerable during a four-minute render and a crash in ffmpeg takes down one job
rather than the orchestrator. n8n never touches media — it calls this and
reacts to the result.

Import `orchestration/n8n_workflow.json`, set your sheet ID, and it polls a
`Jobs` sheet for rows with an empty `status`, triggers a render, and writes back
`done` or `failed` with the output path. A disabled folder-watch trigger is
included as an alternative to Google Sheets.

Two safety defaults, because this endpoint accepts filesystem paths and runs an
encoder: it binds to loopback only, and it **refuses to start on a non-local
host** unless `ORCHESTRATOR_TOKEN` is set in `.env`. Output paths are confined
to the output root — a job supplies a filename, never a path, so
`../../../etc/x.mp4` becomes `output/x.mp4`.

## Studio (browser UI)

```bash
python src/orchestrator_cli.py
```

Open `http://127.0.0.1:8712/` — the orchestrator serves the UI itself, so
there is nothing separate to build or start. It's three stages you scroll
through — **Source** (clips, audio, brief), **Shape** (duration, format,
transition, 3D title), **Render** (spend cap, generator choice, the button
itself) — that stack on top of each other as you scroll, each posting to the
same `/render` and `/jobs/{id}` routes n8n uses. No separate account, no
tracking, nothing sent anywhere but your own machine.

`web/index.html` and `web/assets/` are static files with no build step and no
framework — edit and reload. Rendered output is served back from
`/media/<filename>`, confined to the output root the same way `/render`'s
`out` field is.

**Known gap:** the Studio UI's form predates the Phase 4 effects toolkit —
`/render` and the MCP server both accept `look`/`punch_zoom`/`glitch`/
`speed_ramp`/`ken_burns` today, the HTML form doesn't have fields for them
yet. Reachable now via curl, the MCP tools, or n8n; not yet from the page
itself.

## MCP server (Claude Desktop/Code, Cursor)

```bash
python src/mcp_server.py            # stdio transport
```

```json
{"mcpServers": {"video-pipeline": {"command": "python",
                 "args": ["/absolute/path/to/src/mcp_server.py"]}}}
```

Seven tools, all free, all local: `list_clips`, `analyze_audio`, `transcribe`
(word-level speech timestamps — captions via `write_srt_to`, silence/retake
candidates via the returned `gaps`), `list_looks`, `list_transitions`,
`plan_edit` (the cut timeline, computed but not rendered — see exactly what
will happen before spending render time), `render_edit` (the real thing,
blocking, same `orchestrator_cli.build_argv` translation the HTTP API and
Studio use — one flag surface, three front doors). `skills/beat-sync-cutting/
SKILL.md` is the first playbook; more belong in `skills/` the same way.

`transcribe` uses `faster-whisper` (CTranslate2, MIT) — pip-installable, no
separate binary the way Blender needs one. The model (~140MB for `"base"`)
downloads once from Hugging Face and is cached at `~/.cache/huggingface`;
every call after that is offline. Loading it is the one place in this repo
that reliably surfaces this machine's memory ceiling (no page file, see
Tests below) — it held up fine in isolation and in most combined runs, but
hit both a caught allocation failure and one outright process crash under
heavy concurrent load while this was being built. Real transcription,
correct word-level timestamps, and appropriately low confidence on a
genuinely ambiguous word were all confirmed working when memory allowed;
nothing pointed to a bug in the code itself.

The pattern — expose editing primitives as MCP tools, keep domain judgment
in read-on-demand skill docs instead of a hardcoded rule engine — is
borrowed from [Kaestral](https://github.com/prabindersinghh/Kaestral-pro),
reimplemented from scratch rather than adapted from its GPL-3.0 source. Not
just a licensing call: Kaestral's own `beat-sync-cutting` skill has its
agent compute cut placement by hand, tool call by tool call, against a
plain tempo grid — "excellent for percussive music, weaker on ambient/
legato tracks," by its own admission, with no fallback offered. Every skill
here instead delegates cut timing to `plan_edit`/`render_edit`, which call
the same deterministic, render-verified `assembler.build_plan()` the CLI
does. An LLM choosing *parameters* — cut density per energy section,
transition, which effects fit the brief — plays to what it's actually good
at; an LLM re-deriving frame arithmetic across dozens of repeated tool
calls is exactly the kind of task that drifts.

## Avatars (HeyGen, paid, off)

`generators/heygen.py` exists for the case where a project genuinely needs a
talking head. There is no free tier — HeyGen bills per minute — so it is gated
exactly like `fal_gateway` and is never part of a default run. The shipped
config deliberately leaves `avatar_id` and `voice_id` empty, so an accidental
`--generator heygen` fails loudly instead of billing.
