"""The HeyGen avatar adapter — paid, no free tier, off by default.

Same discipline as the fal gateway, and tested the same way: what matters is
that it cannot run unless somebody deliberately turned it on, and that every
attempt is counted before it is made.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from budget import Budget, BudgetExceeded, PaidAdapterDisabled
from conftest import make_settings
from generators import get_adapter, heygen as heygen_module
from generators.heygen import HeyGenAdapter, HeyGenError
from mock_http import MockServer, Request, Response, routed

AVATAR = {"avatar_id": "avatar_123", "voice_id": "voice_456"}


def make_adapter(tmp_path: Path, *, ceiling=10.0, enabled=True, **overrides):
    settings = make_settings(
        paid_adapters_enabled=enabled,
        heygen={
            **AVATAR,
            "poll_seconds": 0.01,
            "timeout_seconds": 10,
            "backoff_seconds": 0.01,
            "cost_per_call_usd": 0.80,
            **overrides,
        },
    )
    budget = Budget(max_spend_usd=ceiling, paid_adapters_enabled=enabled)
    return HeyGenAdapter(
        settings=settings, budget=budget, cache_dir=tmp_path / "cache"
    ), budget


def heygen_handler(video_bytes: bytes, *, failures: int = 0):
    state = {"submits": 0}

    def generate(request: Request) -> Response:
        state["submits"] += 1
        if state["submits"] <= failures:
            return Response.json({"detail": "upstream"}, status=502)
        return Response.json({"data": {"video_id": "vid-1"}})

    def status(request: Request) -> Response:
        return Response.json({"data": {
            "status": "completed",
            "video_url": f"{state['url']}/asset.mp4",
        }})

    handler = routed({
        "/v2/video/generate": generate,
        "/v1/video_status.get": status,
        "/asset.mp4": lambda r: Response.binary(video_bytes),
    })
    return handler, state


@pytest.fixture
def heygen_env(monkeypatch):
    monkeypatch.setenv("HEYGEN_API_KEY", "hg-test-key")
    return monkeypatch


def point_at(monkeypatch, server_url: str) -> None:
    monkeypatch.setattr(
        heygen_module, "GENERATE_URL", f"{server_url}/v2/video/generate"
    )
    monkeypatch.setattr(
        heygen_module, "STATUS_URL", f"{server_url}/v1/video_status.get"
    )


# --- the guard ------------------------------------------------------------

def test_it_cannot_be_built_while_paid_adapters_are_disabled(
    tmp_path: Path, heygen_env
) -> None:
    with pytest.raises(PaidAdapterDisabled, match="paid_adapters_enabled"):
        get_adapter(
            "heygen",
            settings=make_settings(paid_adapters_enabled=False, heygen=AVATAR),
            budget=Budget(max_spend_usd=100.0),
            cache_dir=tmp_path,
        )


def test_it_is_never_the_default_generator() -> None:
    import config
    from config import PROJECT_ROOT

    settings = config.load_settings(PROJECT_ROOT / "config" / "settings.yaml")

    assert settings.generator_default != "heygen"
    assert settings.paid_adapters_enabled is False


def test_the_shipped_config_leaves_the_avatar_unset() -> None:
    """So an accidental --generator heygen fails loudly instead of billing."""
    import config
    from config import PROJECT_ROOT

    settings = config.load_settings(PROJECT_ROOT / "config" / "settings.yaml")

    assert not settings.get("heygen.avatar_id")
    assert not settings.get("heygen.voice_id")


def test_a_missing_avatar_says_how_to_list_them(tmp_path: Path, heygen_env) -> None:
    with pytest.raises(HeyGenError, match="v2/avatars"):
        make_adapter(tmp_path, avatar_id="", voice_id="")


def test_a_missing_key_points_at_dotenv(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("HEYGEN_API_KEY", raising=False)

    with pytest.raises(HeyGenError, match="HEYGEN_API_KEY is not set in .env"):
        make_adapter(tmp_path)


# --- money ----------------------------------------------------------------

def test_a_successful_call_charges_once(
    tmp_path: Path, heygen_env, tiny_video_bytes: bytes
) -> None:
    handler, state = heygen_handler(tiny_video_bytes)
    with MockServer(handler) as server:
        state["url"] = server.url
        point_at(heygen_env, server.url)
        adapter, budget = make_adapter(tmp_path)

        path = adapter.generate("Hello and welcome.", width=1280, height=720)

    assert path.is_file()
    assert budget.spent_usd == pytest.approx(0.80)


def test_every_retry_is_charged_before_it_happens(
    tmp_path: Path, heygen_env, tiny_video_bytes: bytes
) -> None:
    handler, state = heygen_handler(tiny_video_bytes, failures=2)
    with MockServer(handler) as server:
        state["url"] = server.url
        point_at(heygen_env, server.url)
        adapter, budget = make_adapter(tmp_path, max_retries=3)

        adapter.generate("Eventually works.")

    assert budget.spent_usd == pytest.approx(2.40)
    assert len(budget.entries) == 3


def test_a_retry_loop_hits_the_ceiling(tmp_path: Path, heygen_env) -> None:
    handler, state = heygen_handler(b"", failures=99)
    with MockServer(handler) as server:
        state["url"] = server.url
        point_at(heygen_env, server.url)
        adapter, budget = make_adapter(tmp_path, ceiling=2.0, max_retries=100)

        with pytest.raises(BudgetExceeded):
            adapter.generate("Never works.")

    assert budget.spent_usd == pytest.approx(1.60)


def test_a_rejected_request_is_not_retried(tmp_path: Path, heygen_env) -> None:
    state: dict = {}
    with MockServer(routed({
        "/v2/video/generate": lambda r: Response.json(
            {"detail": "script too long"}, status=400
        ),
    })) as server:
        state["url"] = server.url
        point_at(heygen_env, server.url)
        adapter, budget = make_adapter(tmp_path, max_retries=5)

        with pytest.raises(HeyGenError, match="400"):
            adapter.generate("Bad script.")

    assert server.count("/v2/video/generate") == 1
    assert budget.spent_usd == pytest.approx(0.80)


def test_a_cache_hit_costs_nothing(
    tmp_path: Path, heygen_env, tiny_video_bytes: bytes
) -> None:
    handler, state = heygen_handler(tiny_video_bytes)
    with MockServer(handler) as server:
        state["url"] = server.url
        point_at(heygen_env, server.url)
        adapter, budget = make_adapter(tmp_path)

        adapter.generate("Same line.", seed=1)
        adapter.generate("Same line.", seed=1)

        assert server.count("/v2/video/generate") == 1

    assert budget.spent_usd == pytest.approx(0.80)


# --- protocol details -----------------------------------------------------

def test_an_error_returned_with_http_200_is_still_an_error(
    tmp_path: Path, heygen_env
) -> None:
    """HeyGen reports some rejections with a 200 and an `error` field."""
    state: dict = {}
    with MockServer(routed({
        "/v2/video/generate": lambda r: Response.json(
            {"error": {"message": "quota exceeded"}}
        ),
    })) as server:
        state["url"] = server.url
        point_at(heygen_env, server.url)
        adapter, _ = make_adapter(tmp_path, max_retries=1)

        with pytest.raises(HeyGenError, match="quota exceeded"):
            adapter.generate("Anything.")


def test_a_failed_render_is_reported(tmp_path: Path, heygen_env) -> None:
    state: dict = {}
    with MockServer(routed({
        "/v2/video/generate": lambda r: Response.json({"data": {"video_id": "v"}}),
        "/v1/video_status.get": lambda r: Response.json(
            {"data": {"status": "failed", "error": "avatar unavailable"}}
        ),
    })) as server:
        state["url"] = server.url
        point_at(heygen_env, server.url)
        adapter, _ = make_adapter(tmp_path, max_retries=1)

        with pytest.raises(HeyGenError, match="avatar unavailable"):
            adapter.generate("Anything.")


def test_a_stuck_job_warns_that_it_may_still_bill(tmp_path: Path, heygen_env) -> None:
    state: dict = {}
    with MockServer(routed({
        "/v2/video/generate": lambda r: Response.json({"data": {"video_id": "v"}}),
        "/v1/video_status.get": lambda r: Response.json(
            {"data": {"status": "processing"}}
        ),
    })) as server:
        state["url"] = server.url
        point_at(heygen_env, server.url)
        adapter, _ = make_adapter(
            tmp_path, max_retries=1, timeout_seconds=0.3, poll_seconds=0.05
        )

        with pytest.raises(HeyGenError, match="still be rendering and billable"):
            adapter.generate("Slow one.")


def test_the_key_is_sent_as_an_x_api_key_header(
    tmp_path: Path, heygen_env, tiny_video_bytes: bytes
) -> None:
    handler, state = heygen_handler(tiny_video_bytes)
    with MockServer(handler) as server:
        state["url"] = server.url
        point_at(heygen_env, server.url)
        adapter, _ = make_adapter(tmp_path)
        adapter.generate("Hello.")

        submit = next(
            r for r in server.requests if r.path == "/v2/video/generate"
        )

    assert submit.headers["x-api-key"] == "hg-test-key"
    body = submit.json()
    assert body["video_inputs"][0]["voice"]["input_text"] == "Hello."
    assert body["video_inputs"][0]["character"]["avatar_id"] == "avatar_123"
