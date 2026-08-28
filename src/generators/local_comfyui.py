"""The default generator: a locally running ComfyUI instance. Cost: $0.

Setup, once:

  1. Install ComfyUI per its own instructions and download a video checkpoint
     into `models/`. Wan 2.2 first — it has the best VRAM-to-quality ratio for
     consumer cards and runs on 8-16GB with GGUF quantisation. HunyuanVideo 1.5
     and LTX-2.3 are drop-in alternates once the loop works.
  2. In ComfyUI: Settings -> enable **Dev Mode**.
  3. Build the workflow in the visual editor until one clip comes out the way
     you want, then **Save (API Format)**. That JSON is what this adapter POSTs.
  4. Point `comfyui.workflow` in settings.yaml at the saved file.

Nothing else needs configuring: the adapter reads the graph and works out where
the prompt, dimensions, frame count and seed live by following the links, so
node IDs (which differ between workflows) do not have to be transcribed by
hand. `comfyui.inputs` in settings.yaml overrides any of that when the guess is
wrong.

To see what it found:

    python src/generators/local_comfyui.py --inspect path/to/workflow_api.json
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import config
from config import PROJECT_ROOT
from ffmpeg_tools import ffmpeg_bin, probe_json, run
from generators.base_adapter import (
    GenerationError,
    GenerationRequest,
    GeneratorAdapter,
    register,
)

log = logging.getLogger(__name__)

DEFAULT_HOST = "http://127.0.0.1:8188"

# ComfyUI puts saved media under a different key depending on which save node
# the workflow ends with, and new ones keep appearing. Rather than enumerate
# them, any list of objects carrying a "filename" is treated as output.
_FILENAME_KEY = "filename"

# Containers the assembler can use directly; anything else is normalised.
_DIRECTLY_USABLE = {".mp4", ".mov", ".mkv", ".webm"}


class ComfyUINotRunning(GenerationError):
    """Nothing is answering at the configured ComfyUI host."""


class WorkflowError(GenerationError):
    """The workflow file is missing, malformed, or rejected by ComfyUI."""


@register
class LocalComfyUIAdapter(GeneratorAdapter):
    """Generates clips through ComfyUI's dev-mode HTTP API."""

    name = "local_comfyui"
    is_paid = False
    estimated_cost_per_call_usd = 0.0  # local inference; never touches the ledger

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        settings = self.settings
        # .env wins, so a non-default port does not need a settings.yaml edit.
        self.host = str(
            config.secret("COMFYUI_HOST") or settings.get("comfyui.host", DEFAULT_HOST)
        ).rstrip("/")
        self.timeout_seconds = float(settings.get("comfyui.timeout_seconds", 900))
        self.poll_seconds = float(settings.get("comfyui.poll_seconds", 2.0))
        self.workflow_path = self._resolve_workflow(
            settings.get("comfyui.workflow", "")
        )
        self.input_overrides = settings.get("comfyui.inputs", {}) or {}
        self._client_id = str(uuid.uuid4())

    # -- public API -------------------------------------------------------

    def generate(self, prompt: str, **kwargs) -> Path:
        request = self.build_request(prompt, **kwargs)
        destination = self.cached_path(request)
        if destination.exists():
            log.info("ComfyUI cache hit: %s", destination.name)
            return destination

        # Free, so this records nothing — called anyway so that every adapter
        # goes through the same path and the accounting stays uniform.
        self.reserve_budget(note=request.prompt[:60])

        workflow = self._prepare_workflow(request)
        prompt_id = self._submit(workflow)
        outputs = self._await_outputs(prompt_id)
        downloaded = self._download_first(outputs, request)

        return self._normalise(downloaded, destination)

    def health(self) -> dict[str, Any]:
        """Ask ComfyUI what it is. Raises if it is not there."""
        client = self._http()
        try:
            response = client.get(f"{self.host}/system_stats", timeout=10.0)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001 - all failures mean the same thing
            raise ComfyUINotRunning(
                f"No ComfyUI at {self.host} ({type(exc).__name__}: {exc}).\n"
                f"Start it (`python main.py` in the ComfyUI folder) or set "
                f"COMFYUI_HOST in .env if it listens elsewhere."
            ) from exc

    # -- workflow handling ------------------------------------------------

    @staticmethod
    def _resolve_workflow(configured: str) -> Path:
        if not configured:
            raise WorkflowError(
                "comfyui.workflow is not set in config/settings.yaml. Export a "
                "workflow from ComfyUI with Save (API Format) and point that "
                "setting at the file."
            )
        path = Path(configured)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    def load_workflow(self) -> dict[str, Any]:
        if not self.workflow_path.is_file():
            raise WorkflowError(
                f"ComfyUI workflow not found: {self.workflow_path}\n"
                f"In ComfyUI: enable Dev Mode in Settings, build the graph, "
                f"then Save (API Format) and save it there. The plain 'Save' "
                f"format is a different schema and will not work."
            )
        try:
            workflow = json.loads(self.workflow_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise WorkflowError(f"{self.workflow_path} is not valid JSON: {exc}") from exc

        if not isinstance(workflow, dict) or not workflow:
            raise WorkflowError(f"{self.workflow_path} is empty or not a JSON object.")
        if "nodes" in workflow and "links" in workflow:
            raise WorkflowError(
                f"{self.workflow_path} looks like a UI-format workflow, not an "
                f"API-format one. Re-export it with Save (API Format)."
            )
        return workflow

    def _prepare_workflow(self, request: GenerationRequest) -> dict[str, Any]:
        workflow = self.load_workflow()
        mapping = detect_input_map(workflow)
        mapping.update(self.input_overrides)

        values: dict[str, Any] = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "width": request.width,
            "height": request.height,
            "frames": request.frames,
            "fps": request.fps,
        }
        if request.seed is not None:
            values["seed"] = request.seed

        for key, dotted_path in mapping.items():
            if key not in values or values[key] in (None, ""):
                continue
            try:
                _set_by_path(workflow, dotted_path, values[key])
            except (KeyError, TypeError) as exc:
                raise WorkflowError(
                    f"comfyui.inputs maps {key!r} to {dotted_path!r}, which does "
                    f"not exist in {self.workflow_path.name}. Run "
                    f"`python src/generators/local_comfyui.py --inspect "
                    f"{self.workflow_path}` to see the real node IDs."
                ) from exc

        log.debug("ComfyUI parameter map: %s", mapping)
        return workflow

    # -- HTTP -------------------------------------------------------------

    def _http(self):
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise GenerationError(
                "httpx is required for the ComfyUI adapter: pip install httpx"
            ) from exc
        return httpx.Client(follow_redirects=True)

    def _submit(self, workflow: dict[str, Any]) -> str:
        self.health()
        client = self._http()
        response = client.post(
            f"{self.host}/prompt",
            json={"prompt": workflow, "client_id": self._client_id},
            timeout=60.0,
        )
        if response.status_code >= 400:
            # ComfyUI returns per-node validation detail here; it is the single
            # most useful thing to show when a workflow will not run.
            raise WorkflowError(
                f"ComfyUI rejected the workflow (HTTP {response.status_code}):\n"
                f"{_pretty(response.text)}"
            )

        payload = response.json()
        prompt_id = payload.get("prompt_id")
        if not prompt_id:
            raise WorkflowError(f"ComfyUI returned no prompt_id: {payload}")
        if payload.get("node_errors"):
            raise WorkflowError(
                f"ComfyUI reported node errors:\n{_pretty(payload['node_errors'])}"
            )
        log.info("ComfyUI queued prompt %s", prompt_id)
        return str(prompt_id)

    def _await_outputs(self, prompt_id: str) -> list[dict[str, Any]]:
        """Poll /history until the job finishes, fails, or the timeout expires."""
        client = self._http()
        deadline = time.monotonic() + self.timeout_seconds
        started = time.monotonic()
        announced = 0.0

        while time.monotonic() < deadline:
            response = client.get(f"{self.host}/history/{prompt_id}", timeout=30.0)
            response.raise_for_status()
            history = response.json() or {}
            entry = history.get(prompt_id)

            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise GenerationError(
                        f"ComfyUI failed to run the workflow:\n"
                        f"{_pretty(status.get('messages', status))}"
                    )
                files = _collect_output_files(entry.get("outputs", {}))
                if files:
                    log.info(
                        "ComfyUI finished prompt %s in %.1fs",
                        prompt_id, time.monotonic() - started,
                    )
                    return files
                if status.get("completed"):
                    raise GenerationError(
                        f"ComfyUI completed prompt {prompt_id} but saved no "
                        f"files. The workflow needs a save node (SaveWEBM, "
                        f"SaveAnimatedWEBP or VHS_VideoCombine) on its output."
                    )

            elapsed = time.monotonic() - started
            if elapsed - announced >= 30.0:
                announced = elapsed
                log.info("  still generating (%.0fs elapsed)", elapsed)
            time.sleep(self.poll_seconds)

        raise GenerationError(
            f"ComfyUI did not finish prompt {prompt_id} within "
            f"{self.timeout_seconds:.0f}s. Raise comfyui.timeout_seconds, or "
            f"lower the resolution or frame count."
        )

    def _download_first(
        self, outputs: list[dict[str, Any]], request: GenerationRequest
    ) -> Path:
        entry = outputs[0]
        if len(outputs) > 1:
            log.debug("ComfyUI produced %d files; using the first.", len(outputs))

        params = {
            "filename": entry.get(_FILENAME_KEY, ""),
            "subfolder": entry.get("subfolder", ""),
            "type": entry.get("type", "output"),
        }
        client = self._http()
        response = client.get(f"{self.host}/view", params=params, timeout=300.0)
        response.raise_for_status()

        suffix = Path(params["filename"]).suffix or ".mp4"
        raw_path = self.cache_dir / f"{self.name}_raw_{request.cache_key(self.name)}{suffix}"
        raw_path.write_bytes(response.content)
        log.info("Downloaded %s (%.1f MB)", raw_path.name, len(response.content) / 1e6)
        return raw_path

    def _normalise(self, source: Path, destination: Path) -> Path:
        """Make sure whatever ComfyUI saved is something the assembler can cut.

        Animated WEBP and GIF are common ComfyUI outputs and neither carries the
        timing metadata the timeline needs, so anything that is not already a
        normal video container is transcoded once, here.
        """
        if source.suffix.lower() in _DIRECTLY_USABLE and _has_video_stream(source):
            if source != destination:
                source.replace(destination)
            return destination

        log.info("Transcoding %s to mp4 for the timeline", source.name)
        run([
            ffmpeg_bin(), "-hide_banner", "-nostdin", "-y", "-v", "error",
            "-i", str(source),
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(destination),
        ])
        source.unlink(missing_ok=True)
        return destination


# --------------------------------------------------------------------------
# Reading an exported workflow
# --------------------------------------------------------------------------

def detect_input_map(workflow: dict[str, Any]) -> dict[str, str]:
    """Work out where each parameter lives in an API-format workflow.

    Uses the graph rather than guesswork. In API format a sampler node names its
    conditioning inputs explicitly — `"positive": ["6", 0]` — so following those
    links identifies which text encoder is the prompt and which is the negative,
    which is the one thing that cannot be guessed from class names alone.

    Returns dotted paths like `{"prompt": "6.inputs.text"}`. Anything it cannot
    find is simply absent, and `comfyui.inputs` in settings.yaml overrides it.
    """
    mapping: dict[str, str] = {}

    sampler_id, sampler = _find_node(
        workflow, lambda node: {"positive", "negative"} <= set(node.get("inputs", {}))
    )
    if sampler:
        for key, field in (("prompt", "positive"), ("negative_prompt", "negative")):
            linked = sampler.get("inputs", {}).get(field)
            text_field = _text_field_of(workflow, linked)
            if text_field:
                mapping[key] = text_field

    # Latent/video nodes carry the output geometry and the frame count.
    for field, keys in (
        ("width", ("width",)),
        ("height", ("height",)),
        ("frames", ("length", "num_frames", "video_frames", "batch_size")),
    ):
        node_id, node = _find_node(
            workflow,
            lambda node, keys=keys: any(key in node.get("inputs", {}) for key in keys)
            and "width" in node.get("inputs", {}),
        )
        if not node:
            continue
        for key in keys:
            if key in node.get("inputs", {}):
                mapping[field] = f"{node_id}.inputs.{key}"
                break

    for field, keys in (("seed", ("seed", "noise_seed")), ("fps", ("fps", "frame_rate"))):
        node_id, node = _find_node(
            workflow,
            lambda node, keys=keys: any(key in node.get("inputs", {}) for key in keys),
        )
        if not node:
            continue
        for key in keys:
            if key in node.get("inputs", {}):
                mapping[field] = f"{node_id}.inputs.{key}"
                break

    return mapping


def _text_field_of(workflow: dict[str, Any], link: Any) -> str | None:
    """Resolve a `["node_id", slot]` link to that node's literal text input."""
    if not isinstance(link, list) or not link:
        return None
    node = workflow.get(str(link[0]))
    if not isinstance(node, dict):
        return None
    for field in ("text", "prompt", "string"):
        if isinstance(node.get("inputs", {}).get(field), str):
            return f"{link[0]}.inputs.{field}"
    return None


def _find_node(workflow: dict[str, Any], predicate) -> tuple[str | None, dict | None]:
    """First node satisfying `predicate`, in node-ID order for determinism."""
    for node_id in sorted(workflow, key=_sort_key):
        node = workflow[node_id]
        if isinstance(node, dict) and predicate(node):
            return node_id, node
    return None, None


def _sort_key(node_id: str):
    return (0, int(node_id)) if node_id.isdigit() else (1, node_id)


def _set_by_path(workflow: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    node: Any = workflow
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def _collect_output_files(outputs: dict[str, Any]) -> list[dict[str, Any]]:
    """Every saved file in a history entry, whatever key the save node used."""
    files: list[dict[str, Any]] = []
    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        for value in node_output.values():
            if not isinstance(value, list):
                continue
            files.extend(
                item for item in value
                if isinstance(item, dict) and item.get(_FILENAME_KEY)
            )
    return files


def _has_video_stream(path: Path) -> bool:
    try:
        streams = probe_json(path).get("streams", [])
    except Exception:  # noqa: BLE001 - unreadable means "needs transcoding"
        return False
    return any(stream.get("codec_type") == "video" for stream in streams)


def _pretty(value: Any) -> str:
    if isinstance(value, str):
        return value[:2000]
    try:
        return json.dumps(value, indent=2)[:2000]
    except (TypeError, ValueError):
        return str(value)[:2000]


# --------------------------------------------------------------------------
# `--inspect`: show what the detector found in a workflow
# --------------------------------------------------------------------------

def _inspect(path: Path) -> int:
    workflow = json.loads(path.read_text(encoding="utf-8"))
    if "nodes" in workflow and "links" in workflow:
        print(f"{path} is UI format. Re-export with Save (API Format).")
        return 1

    print(f"{path}\n{len(workflow)} nodes\n")
    for node_id in sorted(workflow, key=_sort_key):
        node = workflow[node_id]
        title = node.get("_meta", {}).get("title", "")
        literals = [
            key for key, value in node.get("inputs", {}).items()
            if not isinstance(value, list)
        ]
        print(f"  {node_id:>4}  {node.get('class_type', '?'):<28} {title}")
        if literals:
            print(f"        settable: {', '.join(literals)}")

    print("\nDetected parameter map (paste into comfyui.inputs to override):")
    detected = detect_input_map(workflow)
    if not detected:
        print("  nothing detected — set comfyui.inputs by hand")
    for key, dotted in sorted(detected.items()):
        print(f"  {key + ':':<18} \"{dotted}\"")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a ComfyUI API workflow.")
    parser.add_argument("--inspect", type=Path, required=True)
    raise SystemExit(_inspect(parser.parse_args().inspect))
