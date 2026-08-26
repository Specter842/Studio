"""The default (free) generator, driven against a mock ComfyUI.

The mock speaks ComfyUI's real dev-mode API — POST /prompt, poll
/history/{id}, download from /view — so everything except the model inference
itself is exercised: workflow parsing, parameter injection, queueing, polling,
error surfacing, download and normalisation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from budget import Budget
from conftest import make_settings, requires_ffmpeg
from generators.local_comfyui import (
    ComfyUINotRunning,
    LocalComfyUIAdapter,
    WorkflowError,
    detect_input_map,
)
from mock_http import MockServer, Request, Response, routed

# A synthetic API-format workflow. Not a real Wan 2.2 graph — it exists to
# exercise the detector, and mirrors the shape ComfyUI exports: a sampler that
# names its positive and negative conditioning by node reference.
WORKFLOW = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 0, "steps": 20, "cfg": 6.0,
            "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["40", 0],
        },
    },
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "wan22.safetensors"}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["4", 1]}},
    "40": {
        "class_type": "EmptyLatentVideo",
        "inputs": {"width": 512, "height": 512, "length": 33, "batch_size": 1},
    },
    "50": {"class_type": "SaveWEBM", "inputs": {"fps": 24, "images": ["3", 0]}},
}


@pytest.fixture(autouse=True)
def _isolate_comfyui_host(monkeypatch):
    """COMFYUI_HOST in .env deliberately outranks settings.yaml.

    These tests point the adapter at a mock server through settings, so a real
    .env on the machine running them must not reach in and redirect it.
    """
    monkeypatch.delenv("COMFYUI_HOST", raising=False)


@pytest.fixture
def workflow_file(tmp_path: Path) -> Path:
    path = tmp_path / "workflow_api.json"
    path.write_text(json.dumps(WORKFLOW), encoding="utf-8")
    return path


def make_adapter(host: str, workflow: Path, tmp_path: Path, **overrides):
    settings = make_settings(
        comfyui={
            "host": host,
            "workflow": str(workflow),
            "poll_seconds": 0.01,
            "timeout_seconds": 10,
            **overrides,
        }
    )
    return LocalComfyUIAdapter(
        settings=settings,
        budget=Budget(),
        cache_dir=tmp_path / "cache",
    )


def comfy_handler(video: bytes, *, polls_before_ready: int = 1):
    """A mock ComfyUI that finishes after `polls_before_ready` history calls."""
    state = {"polls": 0}

    def history(request: Request) -> Response:
        state["polls"] += 1
        if state["polls"] < polls_before_ready:
            return Response.json({})  # queued, not in history yet
        return Response.json({
            "abc123": {
                "status": {"status_str": "success", "completed": True},
                "outputs": {
                    "50": {"gifs": [
                        {"filename": "out.mp4", "subfolder": "", "type": "output"}
                    ]}
                },
            }
        })

    return routed({
        "/system_stats": lambda r: Response.json({"system": {"comfyui_version": "test"}}),
        "/prompt": lambda r: Response.json({"prompt_id": "abc123", "node_errors": {}}),
        "/history/abc123": history,
        "/view": lambda r: Response.binary(video),
    })


# --- workflow reading -----------------------------------------------------

def test_detector_follows_the_graph_to_find_positive_and_negative_prompts() -> None:
    """The one thing class names alone cannot tell you."""
    mapping = detect_input_map(WORKFLOW)

    assert mapping["prompt"] == "6.inputs.text"
    assert mapping["negative_prompt"] == "7.inputs.text"


def test_detector_finds_geometry_frames_and_seed() -> None:
    mapping = detect_input_map(WORKFLOW)

    assert mapping["width"] == "40.inputs.width"
    assert mapping["height"] == "40.inputs.height"
    assert mapping["frames"] == "40.inputs.length"
    assert mapping["seed"] == "3.inputs.seed"
    assert mapping["fps"] == "50.inputs.fps"


def test_ui_format_workflows_are_rejected_with_the_fix(tmp_path: Path) -> None:
    path = tmp_path / "ui.json"
    path.write_text(json.dumps({"nodes": [], "links": []}), encoding="utf-8")
    adapter = make_adapter("http://127.0.0.1:1", path, tmp_path)

    with pytest.raises(WorkflowError, match="Save \\(API Format\\)"):
        adapter.load_workflow()


def test_a_missing_workflow_says_how_to_export_one(tmp_path: Path) -> None:
    adapter = make_adapter("http://127.0.0.1:1", tmp_path / "absent.json", tmp_path)

    with pytest.raises(WorkflowError, match="Dev Mode"):
        adapter.load_workflow()


def test_an_unset_workflow_setting_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(WorkflowError, match="comfyui.workflow is not set"):
        LocalComfyUIAdapter(
            settings=make_settings(comfyui={"host": "http://x"}),
            budget=Budget(),
            cache_dir=tmp_path,
        )


def test_comfyui_host_in_env_overrides_settings(
    workflow_file: Path, tmp_path: Path, monkeypatch
) -> None:
    """So a non-default port needs a .env line, not a settings.yaml edit."""
    monkeypatch.setenv("COMFYUI_HOST", "http://10.0.0.5:9000")

    adapter = make_adapter("http://127.0.0.1:8188", workflow_file, tmp_path)

    assert adapter.host == "http://10.0.0.5:9000"


def test_a_blank_env_host_leaves_settings_in_charge(
    workflow_file: Path, tmp_path: Path, monkeypatch
) -> None:
    """`.env.example` ships COMFYUI_HOST blank; that must mean 'unset'."""
    monkeypatch.setenv("COMFYUI_HOST", "")

    adapter = make_adapter("http://127.0.0.1:7777", workflow_file, tmp_path)

    assert adapter.host == "http://127.0.0.1:7777"


def test_parameters_are_written_into_the_workflow(
    workflow_file: Path, tmp_path: Path
) -> None:
    adapter = make_adapter("http://127.0.0.1:1", workflow_file, tmp_path)
    request = adapter.build_request(
        "a neon street", negative_prompt="blurry",
        width=768, height=432, fps=24.0, seconds=2.0, seed=99,
    )

    prepared = adapter._prepare_workflow(request)

    assert prepared["6"]["inputs"]["text"] == "a neon street"
    assert prepared["7"]["inputs"]["text"] == "blurry"
    assert prepared["40"]["inputs"]["width"] == 768
    assert prepared["40"]["inputs"]["height"] == 432
    assert prepared["40"]["inputs"]["length"] == 48  # 2.0s x 24fps
    assert prepared["3"]["inputs"]["seed"] == 99


def test_config_overrides_beat_detection(workflow_file: Path, tmp_path: Path) -> None:
    adapter = make_adapter(
        "http://127.0.0.1:1", workflow_file, tmp_path,
        inputs={"prompt": "7.inputs.text"},
    )
    request = adapter.build_request("override me")

    prepared = adapter._prepare_workflow(request)

    assert prepared["7"]["inputs"]["text"] == "override me"


def test_a_bad_override_names_the_inspect_command(
    workflow_file: Path, tmp_path: Path
) -> None:
    adapter = make_adapter(
        "http://127.0.0.1:1", workflow_file, tmp_path,
        inputs={"prompt": "999.inputs.text"},
    )

    with pytest.raises(WorkflowError, match="--inspect"):
        adapter._prepare_workflow(adapter.build_request("x"))


# --- talking to ComfyUI ---------------------------------------------------

@requires_ffmpeg
def test_generate_returns_a_usable_clip(
    workflow_file: Path, tmp_path: Path, tiny_video_bytes: bytes
) -> None:
    with MockServer(comfy_handler(tiny_video_bytes)) as server:
        adapter = make_adapter(server.url, workflow_file, tmp_path)
        path = adapter.generate("a neon street", seconds=2.0, width=320, height=180)

    assert path.is_file()
    assert path.stat().st_size > 0

    from ingest.local_clips import probe_clip
    assert probe_clip(path).is_usable


@requires_ffmpeg
def test_the_submitted_workflow_carries_the_prompt(
    workflow_file: Path, tmp_path: Path, tiny_video_bytes: bytes
) -> None:
    with MockServer(comfy_handler(tiny_video_bytes)) as server:
        adapter = make_adapter(server.url, workflow_file, tmp_path)
        adapter.generate("a neon street at night")

        submitted = next(r for r in server.requests if r.path == "/prompt").json()

    assert submitted["prompt"]["6"]["inputs"]["text"] == "a neon street at night"
    assert submitted["client_id"]


@requires_ffmpeg
def test_it_polls_until_the_job_appears(
    workflow_file: Path, tmp_path: Path, tiny_video_bytes: bytes
) -> None:
    with MockServer(comfy_handler(tiny_video_bytes, polls_before_ready=3)) as server:
        adapter = make_adapter(server.url, workflow_file, tmp_path)
        adapter.generate("slow one")

    assert server.count("/history/abc123") == 3


@requires_ffmpeg
def test_a_second_identical_request_is_served_from_cache(
    workflow_file: Path, tmp_path: Path, tiny_video_bytes: bytes
) -> None:
    """Regenerating costs minutes; identical requests must not pay it twice."""
    with MockServer(comfy_handler(tiny_video_bytes)) as server:
        adapter = make_adapter(server.url, workflow_file, tmp_path)
        first = adapter.generate("same prompt", seed=1)
        second = adapter.generate("same prompt", seed=1)

        assert server.count("/prompt") == 1

    assert first == second


@requires_ffmpeg
def test_generation_never_charges_the_budget(
    workflow_file: Path, tmp_path: Path, tiny_video_bytes: bytes
) -> None:
    """Local inference is free, and the ledger has to keep saying so."""
    with MockServer(comfy_handler(tiny_video_bytes)) as server:
        adapter = make_adapter(server.url, workflow_file, tmp_path)
        adapter.generate("free as in beer")

    assert adapter.budget.spent_usd == 0.0
    assert adapter.budget.report().startswith("Estimated run cost: $0.00")


def test_a_stopped_comfyui_says_how_to_start_it(
    workflow_file: Path, tmp_path: Path
) -> None:
    # Port 1 is never listening.
    adapter = make_adapter("http://127.0.0.1:1", workflow_file, tmp_path)

    with pytest.raises(ComfyUINotRunning, match="Start it"):
        adapter.generate("anything")


def test_validation_errors_from_comfyui_are_surfaced(
    workflow_file: Path, tmp_path: Path
) -> None:
    handler = routed({
        "/system_stats": lambda r: Response.json({}),
        "/prompt": lambda r: Response.json(
            {"error": {"message": "missing checkpoint"},
             "node_errors": {"4": "ckpt_name not found"}},
            status=400,
        ),
    })

    with MockServer(handler) as server:
        adapter = make_adapter(server.url, workflow_file, tmp_path)
        with pytest.raises(WorkflowError, match="ckpt_name not found"):
            adapter.generate("anything")


def test_a_failed_run_is_reported_not_hung(workflow_file: Path, tmp_path: Path) -> None:
    handler = routed({
        "/system_stats": lambda r: Response.json({}),
        "/prompt": lambda r: Response.json({"prompt_id": "abc123"}),
        "/history/abc123": lambda r: Response.json({
            "abc123": {
                "status": {"status_str": "error",
                           "messages": [["execution_error", {"exception_message": "OOM"}]]},
                "outputs": {},
            }
        }),
    })

    with MockServer(handler) as server:
        adapter = make_adapter(server.url, workflow_file, tmp_path)
        with pytest.raises(Exception, match="OOM"):
            adapter.generate("too big")


def test_a_workflow_with_no_save_node_says_so(
    workflow_file: Path, tmp_path: Path
) -> None:
    handler = routed({
        "/system_stats": lambda r: Response.json({}),
        "/prompt": lambda r: Response.json({"prompt_id": "abc123"}),
        "/history/abc123": lambda r: Response.json({
            "abc123": {"status": {"completed": True}, "outputs": {}}
        }),
    })

    with MockServer(handler) as server:
        adapter = make_adapter(server.url, workflow_file, tmp_path)
        with pytest.raises(Exception, match="save node"):
            adapter.generate("anything")


def test_a_job_that_never_finishes_times_out(workflow_file: Path, tmp_path: Path) -> None:
    handler = routed({
        "/system_stats": lambda r: Response.json({}),
        "/prompt": lambda r: Response.json({"prompt_id": "abc123"}),
        "/history/abc123": lambda r: Response.json({}),
    })

    with MockServer(handler) as server:
        adapter = make_adapter(
            server.url, workflow_file, tmp_path,
            timeout_seconds=0.3, poll_seconds=0.05,
        )
        with pytest.raises(Exception, match="did not finish"):
            adapter.generate("forever")
