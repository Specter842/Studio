# ComfyUI workflows

Put your exported workflow JSON here and point `comfyui.workflow` in
`config/settings.yaml` at it.

**No workflow is shipped in this folder on purpose.** A ComfyUI graph is tied
to the exact node packs, model filenames and checkpoint versions installed on
one machine — a JSON copied from somewhere else fails to load, and it fails in
a way that looks like a bug in this pipeline rather than a missing model. The
thirty seconds it takes to export your own is better spent than the hour spent
debugging someone else's.

## Exporting one

1. Install ComfyUI and put a video checkpoint in its `models/` folder. Start
   with **Wan 2.2** — best VRAM-to-quality ratio on consumer cards, runs on
   8–16 GB with GGUF quantization. HunyuanVideo 1.5 and LTX-2.3 are drop-in
   alternates once the loop works.
2. **Settings → enable Dev Mode.** Without it there is no API-format export.
3. Build the graph in the visual editor and run it until one clip comes out the
   way you want. Getting this right in the UI first is much faster than
   debugging it through HTTP.
4. **Save (API Format)** → save it here, e.g. `wan22_t2v_api.json`.

The plain **Save** button produces a different schema (`{"nodes": [...],
"links": [...]}`) that the API will not accept. The adapter detects that case
and says so, but it is easy to hit by accident.

## Checking it

```bash
python src/generators/local_comfyui.py --inspect config/comfyui_workflows/wan22_t2v_api.json
```

This lists every node with its settable inputs, then prints the parameter map
the adapter worked out by following the graph's links — which text encoder is
the positive prompt, which is the negative, where width/height/frame count and
seed live.

If anything is wrong or missing, copy the printed block into `comfyui.inputs`
in `settings.yaml` and correct it. Explicit config always wins over detection.

## Notes

- Frame count: the adapter writes `seconds x fps` into whichever input it
  identified as the length. Many video models only accept specific values
  (Wan 2.2 wants `4n+1`), so pick `sourcing.generated_seconds` to land on one.
- Resolution is written from the pipeline's output format. Generating at the
  full output size wastes time — the assembler scales anyway, so a smaller
  generation resolution is usually the better trade.
