"""The optional paid adapter.

Most of these tests are about money, not video. The adapter itself is
straightforward; what has to be right is that it cannot run when it should not,
that every attempt is counted before it happens, and that a retry loop hits the
ceiling rather than the card.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from budget import Budget, BudgetExceeded, PaidAdapterDisabled
from conftest import make_settings, requires_ffmpeg
from generators import fal_gateway, get_adapter
from generators.fal_gateway import FalError, FalGatewayAdapter, _first_asset_url
from mock_http import MockServer, Request, Response, routed

MODEL = "fal-ai/test/text-to-video"


def make_adapter(
    tmp_path: Path,
    *,
    ceiling: float = 10.0,
    enabled: bool = True,
    model: str = MODEL,
    costs: dict | None = None,
    **fal_overrides,
):
    settings = make_settings(
        paid_adapters_enabled=enabled,
        fal={
            "model": model,
            "poll_seconds": 0.01,
            "timeout_seconds": 10,
            "backoff_seconds": 0.01,
            "costs": costs if costs is not None else {MODEL: 0.20},
            **fal_overrides,
        },
    )
    budget = Budget(max_spend_usd=ceiling, paid_adapters_enabled=enabled)
    return FalGatewayAdapter(
        settings=settings, budget=budget, cache_dir=tmp_path / "cache"
    ), budget


def fal_handler(asset_bytes: bytes, *, suffix: str = ".mp4", failures: int = 0):
    """A mock fal queue: submit, poll to COMPLETED, fetch result, download."""
    state = {"submits": 0}

    def submit(request: Request) -> Response:
        state["submits"] += 1
        if state["submits"] <= failures:
            return Response.json({"detail": "server exploded"}, status=500)
        return Response.json({
            "request_id": "req-1",
            "status_url": f"{state['url']}/status",
            "response_url": f"{state['url']}/result",
        })

    def result(request: Request) -> Response:
        return Response.json({"video": {"url": f"{state['url']}/asset{suffix}"}})

    handler = routed({
        f"/{MODEL}": submit,
        "/status": lambda r: Response.json({"status": "COMPLETED"}),
        "/result": result,
        f"/asset{suffix}": lambda r: Response.binary(asset_bytes),
    })
    return handler, state


@pytest.fixture
def fal_env(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "fal-test-key")
    return monkeypatch


# --- the guard ------------------------------------------------------------

def test_it_cannot_be_built_while_paid_adapters_are_disabled(
    tmp_path: Path, fal_env
) -> None:
    with pytest.raises(PaidAdapterDisabled, match="paid_adapters_enabled"):
        get_adapter(
            "fal_gateway",
            settings=make_settings(paid_adapters_enabled=False, fal={"model": MODEL}),
            budget=Budget(max_spend_usd=100.0),
            cache_dir=tmp_path,
        )


def test_no_model_is_chosen_for_you(tmp_path: Path, fal_env) -> None:
    """Nothing that costs money per call gets a convenient default."""
    with pytest.raises(FalError, match="fal.model is not set"):
        make_adapter(tmp_path, model="")


def test_a_missing_key_points_at_dotenv(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("FAL_KEY", raising=False)

    with pytest.raises(FalError, match="FAL_KEY is not set in .env"):
        make_adapter(tmp_path)


def test_an_undeclared_model_cost_falls_back_pessimistically(
    tmp_path: Path, fal_env, caplog
) -> None:
    adapter, _ = make_adapter(tmp_path, costs={}, default_cost_usd=0.75)

    assert adapter.estimated_cost_per_call_usd == 0.75
    assert "No declared cost" in caplog.text


def test_a_declared_model_cost_is_used(tmp_path: Path, fal_env) -> None:
    adapter, _ = make_adapter(tmp_path, costs={MODEL: 0.33})

    assert adapter.estimated_cost_per_call_usd == 0.33


# --- money ----------------------------------------------------------------

@requires_ffmpeg
def test_a_successful_call_charges_once(
    tmp_path: Path, fal_env, tiny_video_bytes: bytes
) -> None:
    handler, state = fal_handler(tiny_video_bytes)
    with MockServer(handler) as server:
        state["url"] = server.url
        fal_env.setattr(fal_gateway, "QUEUE_URL", server.url)
        adapter, budget = make_adapter(tmp_path)

        path = adapter.generate("a neon street", seconds=1.0, width=320, height=180)

    assert path.is_file()
    assert budget.spent_usd == pytest.approx(0.20)


def test_every_retry_is_charged_before_it_happens(
    tmp_path: Path, fal_env, tiny_video_bytes: bytes
) -> None:
    """A budget updated only on success cannot stop a loop that keeps failing."""
    handler, state = fal_handler(tiny_video_bytes, failures=2)
    with MockServer(handler) as server:
        state["url"] = server.url
        fal_env.setattr(fal_gateway, "QUEUE_URL", server.url)
        adapter, budget = make_adapter(tmp_path, max_retries=3)

        adapter.generate("eventually works", seconds=1.0, width=320, height=180)

    # Two failed attempts plus the successful one.
    assert budget.spent_usd == pytest.approx(0.60)
    assert len(budget.entries) == 3


def test_a_retry_loop_hits_the_ceiling_instead_of_the_card(
    tmp_path: Path, fal_env
) -> None:
    handler, state = fal_handler(b"", failures=99)
    with MockServer(handler) as server:
        state["url"] = server.url
        fal_env.setattr(fal_gateway, "QUEUE_URL", server.url)
        adapter, budget = make_adapter(tmp_path, ceiling=0.50, max_retries=100)

        with pytest.raises(BudgetExceeded):
            adapter.generate("never works")

    assert budget.spent_usd == pytest.approx(0.40)  # 2 x 0.20, third refused


def test_a_rejected_request_is_not_retried(tmp_path: Path, fal_env) -> None:
    """400 means the request is wrong; asking again buys the same answer."""
    state: dict = {}

    def submit(request: Request) -> Response:
        return Response.json({"detail": "invalid prompt"}, status=400)

    with MockServer(routed({f"/{MODEL}": submit})) as server:
        state["url"] = server.url
        fal_env.setattr(fal_gateway, "QUEUE_URL", server.url)
        adapter, budget = make_adapter(tmp_path, max_retries=5)

        with pytest.raises(FalError, match="400"):
            adapter.generate("bad")

    assert server.count(f"/{MODEL}") == 1
    assert budget.spent_usd == pytest.approx(0.20)


@requires_ffmpeg
def test_a_cache_hit_costs_nothing(
    tmp_path: Path, fal_env, tiny_video_bytes: bytes
) -> None:
    handler, state = fal_handler(tiny_video_bytes)
    with MockServer(handler) as server:
        state["url"] = server.url
        fal_env.setattr(fal_gateway, "QUEUE_URL", server.url)
        adapter, budget = make_adapter(tmp_path)

        adapter.generate("same", seed=1, seconds=1.0, width=320, height=180)
        adapter.generate("same", seed=1, seconds=1.0, width=320, height=180)

        assert server.count(f"/{MODEL}") == 1

    assert budget.spent_usd == pytest.approx(0.20)


def test_a_failed_job_is_reported(tmp_path: Path, fal_env) -> None:
    state: dict = {}

    with MockServer(routed({
        f"/{MODEL}": lambda r: Response.json({
            "status_url": f"{state['url']}/status",
            "response_url": f"{state['url']}/result",
        }),
        "/status": lambda r: Response.json({"status": "FAILED", "error": "nsfw"}),
    })) as server:
        state["url"] = server.url
        fal_env.setattr(fal_gateway, "QUEUE_URL", server.url)
        adapter, _ = make_adapter(tmp_path, max_retries=1)

        with pytest.raises(FalError, match="FAILED"):
            adapter.generate("rejected")


def test_a_stuck_job_times_out_and_warns_it_may_still_bill(
    tmp_path: Path, fal_env
) -> None:
    state: dict = {}

    with MockServer(routed({
        f"/{MODEL}": lambda r: Response.json({
            "status_url": f"{state['url']}/status",
            "response_url": f"{state['url']}/result",
        }),
        "/status": lambda r: Response.json({"status": "IN_PROGRESS"}),
    })) as server:
        state["url"] = server.url
        fal_env.setattr(fal_gateway, "QUEUE_URL", server.url)
        adapter, _ = make_adapter(
            tmp_path, max_retries=1, timeout_seconds=0.3, poll_seconds=0.05
        )

        with pytest.raises(FalError, match="still be running and billable"):
            adapter.generate("slow")


# --- assets ---------------------------------------------------------------

@requires_ffmpeg
def test_an_image_model_result_becomes_a_moving_clip(
    tmp_path: Path, fal_env, tiny_image_bytes: bytes
) -> None:
    """FLUX and Nano Banana come through the same gateway but return stills.

    A locked-off frozen frame in the middle of a beat-cut edit reads as a
    rendering fault, so a still is turned into a slow pan of the right length.
    """
    handler, state = fal_handler(tiny_image_bytes, suffix=".png")
    with MockServer(handler) as server:
        state["url"] = server.url
        fal_env.setattr(fal_gateway, "QUEUE_URL", server.url)
        adapter, _ = make_adapter(tmp_path)

        path = adapter.generate(
            "a still image", seconds=2.0, width=320, height=180, fps=24.0
        )

    from ingest.local_clips import probe_clip
    clip = probe_clip(path)

    assert clip.is_usable
    assert clip.duration == pytest.approx(2.0, abs=0.15)
    assert (clip.width, clip.height) == (320, 180)


@pytest.mark.parametrize("payload,expected", [
    ({"video": {"url": "http://a/v.mp4"}}, "http://a/v.mp4"),
    ({"images": [{"url": "http://a/i.png"}]}, "http://a/i.png"),
    ({"output": {"url": "http://a/o.webm"}}, "http://a/o.webm"),
    ({"url": "http://a/bare.mp4"}, "http://a/bare.mp4"),
    ({"videos": ["http://a/list.mp4"]}, "http://a/list.mp4"),
    ({"nothing": 1}, None),
])
def test_asset_urls_are_found_across_model_response_shapes(payload, expected) -> None:
    """Response schemas differ per model; one gateway should not mean one shape."""
    assert _first_asset_url(payload) == expected


def test_no_asset_in_the_response_is_an_error(tmp_path: Path, fal_env) -> None:
    state: dict = {}

    with MockServer(routed({
        f"/{MODEL}": lambda r: Response.json({
            "status_url": f"{state['url']}/status",
            "response_url": f"{state['url']}/result",
        }),
        "/status": lambda r: Response.json({"status": "COMPLETED"}),
        "/result": lambda r: Response.json({"logs": ["done"]}),
    })) as server:
        state["url"] = server.url
        fal_env.setattr(fal_gateway, "QUEUE_URL", server.url)
        adapter, _ = make_adapter(tmp_path, max_retries=1)

        with pytest.raises(FalError, match="no downloadable asset"):
            adapter.generate("empty")


def test_the_api_key_is_sent_as_a_key_header(
    tmp_path: Path, fal_env, tiny_video_bytes: bytes
) -> None:
    handler, state = fal_handler(tiny_video_bytes)
    with MockServer(handler) as server:
        state["url"] = server.url
        fal_env.setattr(fal_gateway, "QUEUE_URL", server.url)
        adapter, _ = make_adapter(tmp_path)
        try:
            adapter.generate("x", seconds=1.0, width=320, height=180)
        except Exception:
            pass

        submit = next(r for r in server.requests if r.path == f"/{MODEL}")

    assert submit.headers["authorization"] == "Key fal-test-key"
