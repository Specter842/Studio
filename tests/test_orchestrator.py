"""The HTTP trigger n8n calls.

Driven through FastAPI's TestClient, which exercises the real routing,
validation and dependency wiring rather than calling the handlers directly —
the 422-instead-of-a-body class of bug only shows up through the stack.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import orchestrator_cli
from conftest import PROJECT_ROOT, requires_ffmpeg

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def output_root(tmp_path: Path) -> Path:
    root = tmp_path / "out"
    root.mkdir()
    return root


@pytest.fixture
def client(output_root: Path, monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_TOKEN", raising=False)
    return TestClient(orchestrator_cli.create_app(output_root))


# --- output confinement ---------------------------------------------------

@pytest.mark.parametrize("requested", [
    "../../../../etc/passwd.mp4",
    "/absolute/elsewhere.mp4",
    r"..\..\windows\system32\evil.mp4",
    "sub/dir/nested.mp4",
])
def test_outputs_cannot_escape_the_output_root(requested, output_root: Path) -> None:
    """A job supplies a filename, never a path."""
    resolved = orchestrator_cli.resolve_output(requested, output_root)

    assert resolved.parent == output_root.resolve()


def test_an_extension_is_added_when_missing(output_root: Path) -> None:
    assert orchestrator_cli.resolve_output("clip", output_root).suffix == ".mp4"


def test_an_unnamed_output_gets_a_unique_name(output_root: Path) -> None:
    first = orchestrator_cli.resolve_output(None, output_root)
    second = orchestrator_cli.resolve_output(None, output_root)

    assert first != second
    assert first.suffix == ".mp4"


# --- request translation --------------------------------------------------

def test_a_job_becomes_pipeline_arguments(output_root: Path) -> None:
    argv, out_path = orchestrator_cli.build_argv(
        {
            "audio": "track.mp3", "clips": "./clips", "brief": "a rainy city",
            "duration": 30, "seed": 7, "width": 1280, "generate": 2,
            "title": "HELLO", "no_local": True, "verify": True,
        },
        output_root,
    )

    assert argv[1].endswith("pipeline.py")
    for flag, value in (
        ("--audio", "track.mp3"), ("--clips", "./clips"),
        ("--brief", "a rainy city"), ("--duration", "30"),
        ("--seed", "7"), ("--width", "1280"), ("--generate", "2"),
        ("--title", "HELLO"),
    ):
        assert flag in argv and argv[argv.index(flag) + 1] == value
    assert "--no-local" in argv and "--verify" in argv
    assert out_path.parent == output_root.resolve()


def test_phase4_effects_flow_through_to_pipeline_arguments(output_root: Path) -> None:
    """Regression: build_argv predates the Phase 4 effects toolkit and knew
    nothing about it — every one of these was silently dropped until this
    was fixed, meaning neither the Studio UI nor n8n could reach look,
    punch-zoom, glitch, speed-ramp, or Ken Burns at all."""
    argv, _ = orchestrator_cli.build_argv(
        {
            "audio": "t.mp3", "look": "teal_orange", "lut": "mine.cube",
            "denoise": 0.4, "sharpen": 0.2, "vignette": 0.3, "grain": 12,
            "motion_blur": 2, "light_leak": 0.25,
            "chroma_key": "green", "chroma_key_background": "blue",
            "ken_burns": 0.15, "ken_burns_direction": "out",
            "ken_burns_alternate": True, "ken_burns_steps": 6,
            "punch_zoom": 0.2, "punch_zoom_seconds": 0.3, "punch_zoom_every_nth": 2,
            "glitch": 8, "glitch_seconds": 0.1, "glitch_every_nth": 3,
            "speed_ramp": True, "speed_ramp_fraction": 0.4,
            "speed_ramp_slow": 0.6, "speed_ramp_fast": 1.4,
            "fit": "crop", "start_bpm": 140.0, "tolerance": 0.05,
        },
        output_root,
    )

    for flag, value in (
        ("--look", "teal_orange"), ("--lut", "mine.cube"),
        ("--denoise", "0.4"), ("--sharpen", "0.2"), ("--vignette", "0.3"),
        ("--grain", "12"), ("--motion-blur", "2"), ("--light-leak", "0.25"),
        ("--chroma-key", "green"), ("--chroma-key-background", "blue"),
        ("--ken-burns", "0.15"), ("--ken-burns-direction", "out"),
        ("--ken-burns-steps", "6"),
        ("--punch-zoom", "0.2"), ("--punch-zoom-seconds", "0.3"),
        ("--punch-zoom-every-nth", "2"),
        ("--glitch", "8"), ("--glitch-seconds", "0.1"), ("--glitch-every-nth", "3"),
        ("--speed-ramp-fraction", "0.4"), ("--speed-ramp-slow", "0.6"),
        ("--speed-ramp-fast", "1.4"),
        ("--fit", "crop"), ("--start-bpm", "140.0"), ("--tolerance", "0.05"),
    ):
        assert flag in argv and argv[argv.index(flag) + 1] == value, flag

    assert "--ken-burns-alternate" in argv
    assert "--speed-ramp" in argv


def test_absent_fields_produce_no_flags(output_root: Path) -> None:
    """So the pipeline's own defaults stay in charge of anything unspecified."""
    argv, _ = orchestrator_cli.build_argv({"audio": "t.mp3"}, output_root)

    for flag in ("--brief", "--seed", "--duration", "--title", "--no-local"):
        assert flag not in argv


def test_zero_is_passed_through_not_treated_as_absent(output_root: Path) -> None:
    argv, _ = orchestrator_cli.build_argv(
        {"audio": "t.mp3", "generate": 0, "start": 0.0}, output_root
    )

    assert argv[argv.index("--generate") + 1] == "0"
    assert argv[argv.index("--start") + 1] == "0.0"


def test_audio_is_required(output_root: Path) -> None:
    with pytest.raises(ValueError, match="audio"):
        orchestrator_cli.build_argv({"clips": "./x"}, output_root)


# --- routes ---------------------------------------------------------------

def test_health_reports_the_output_root(client, output_root: Path) -> None:
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert Path(body["output_root"]) == output_root.resolve()
    assert body["auth_required"] is False


def test_the_body_is_parsed_as_json_not_query_parameters(client) -> None:
    """Regression: a locally-defined request model made FastAPI treat the
    whole body as a missing query parameter and reject valid jobs with 422."""
    response = client.post("/render", json={"audio": "does-not-exist.mp3"})

    assert response.status_code == 200
    assert "job_id" in response.json()


def test_a_job_without_audio_is_rejected(client) -> None:
    assert client.post("/render", json={"brief": "no audio"}).status_code == 422


def test_a_failing_render_reports_failed_rather_than_raising(client) -> None:
    body = client.post("/render", json={"audio": "no-such-file.mp3"}).json()

    assert body["status"] == "failed"
    assert body["returncode"] not in (0, None)
    assert body["log_tail"]


def test_an_unknown_job_is_a_404(client) -> None:
    assert client.get("/jobs/nope").status_code == 404


def test_a_token_is_enforced_when_configured(output_root: Path, monkeypatch) -> None:
    monkeypatch.setenv("ORCHESTRATOR_TOKEN", "s3cret")
    guarded = TestClient(orchestrator_cli.create_app(output_root))

    assert guarded.post("/render", json={"audio": "x.mp3"}).status_code == 401
    assert guarded.post(
        "/render", json={"audio": "x.mp3"},
        headers={"Authorization": "Bearer wrong"},
    ).status_code == 401
    assert guarded.post(
        "/render", json={"audio": "x.mp3"},
        headers={"Authorization": "Bearer s3cret"},
    ).status_code == 200


def test_binding_a_public_host_without_a_token_is_refused(monkeypatch) -> None:
    """This endpoint takes filesystem paths and runs an encoder."""
    monkeypatch.delenv("ORCHESTRATOR_TOKEN", raising=False)
    monkeypatch.setattr(orchestrator_cli.config, "load_env", lambda *a, **k: None)

    assert orchestrator_cli.main(["--host", "0.0.0.0"]) == 2


@requires_ffmpeg
def test_a_real_render_can_be_triggered_over_http(
    client, sample_clips: list[Path], click_track: Path, output_root: Path
) -> None:
    """Phase 3's definition of done: a run started without touching the CLI."""
    response = client.post("/render", json={
        "audio": str(click_track),
        "clips": str(sample_clips[0].parent),
        "out": "from_http.mp4",
        "duration": 4, "width": 320, "height": 180, "fps": 30, "seed": 2,
    })
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "succeeded", body["log_tail"]
    assert Path(body["out"]).is_file()
    assert "Estimated run cost: $0.00" in body["log_tail"]

    followed = client.get(f"/jobs/{body['job_id']}").json()
    assert followed["status"] == "succeeded"


# --- the shipped workflow -------------------------------------------------

def test_the_n8n_workflow_is_valid_importable_json() -> None:
    path = PROJECT_ROOT / "orchestration" / "n8n_workflow.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))

    assert workflow["name"]
    assert workflow["nodes"] and workflow["connections"]

    names = {node["name"] for node in workflow["nodes"]}
    for node in workflow["nodes"]:
        assert node["type"].startswith("n8n-nodes-base.")
        assert node["id"] and node["position"]
    # Every connection must point at a node that exists, or the import breaks.
    for source, outputs in workflow["connections"].items():
        assert source in names, f"connection from unknown node {source}"
        for branch in outputs["main"]:
            for link in branch:
                assert link["node"] in names, f"connection to unknown node {link}"


def test_the_workflow_calls_the_orchestrator_route() -> None:
    path = PROJECT_ROOT / "orchestration" / "n8n_workflow.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))

    http_nodes = [
        node for node in workflow["nodes"]
        if node["type"] == "n8n-nodes-base.httpRequest"
    ]

    assert http_nodes, "the workflow never calls the pipeline"
    node = http_nodes[0]
    assert node["parameters"]["method"] == "POST"
    assert node["parameters"]["url"].endswith("/render")
    assert str(orchestrator_cli.DEFAULT_PORT) in node["parameters"]["url"]


def test_the_workflow_never_touches_media_itself() -> None:
    """n8n calls the pipeline and reacts; it must not process video."""
    path = PROJECT_ROOT / "orchestration" / "n8n_workflow.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))

    types = {node["type"] for node in workflow["nodes"]}
    forbidden = {
        "n8n-nodes-base.ffmpeg",
        "n8n-nodes-base.editImage",
        "n8n-nodes-base.executeCommand",
    }

    assert not (types & forbidden), f"workflow does media work itself: {types & forbidden}"
