"""The adapter interface, the registry, and the free-first guard.

The guard is the important part: these tests assert that there is no way to
reach a paid adapter without deliberately enabling one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from budget import Budget, BudgetExceeded, PaidAdapterDisabled
from conftest import make_settings
from generators import available, get_adapter
from generators.base_adapter import (
    GenerationRequest,
    GeneratorAdapter,
    UnknownAdapter,
    register,
)


class _FakeFree(GeneratorAdapter):
    name = "_test_free"
    is_paid = False
    estimated_cost_per_call_usd = 0.0

    def generate(self, prompt: str, **kwargs) -> Path:
        self.reserve_budget(note=prompt)
        return Path("free.mp4")


class _FakePaid(GeneratorAdapter):
    name = "_test_paid"
    is_paid = True
    estimated_cost_per_call_usd = 0.25

    def generate(self, prompt: str, **kwargs) -> Path:
        self.reserve_budget(note=prompt)
        return Path("paid.mp4")


@pytest.fixture(scope="module", autouse=True)
def _register_fakes():
    register(_FakeFree)
    register(_FakePaid)


def build(adapter_class, tmp_path, *, enabled=False, ceiling=0.0):
    budget = Budget(max_spend_usd=ceiling, paid_adapters_enabled=enabled)
    settings = make_settings(paid_adapters_enabled=enabled)
    return adapter_class(settings=settings, budget=budget, cache_dir=tmp_path), budget


# --- registry -------------------------------------------------------------

def test_the_real_adapters_are_registered() -> None:
    assert "local_comfyui" in available()
    assert "fal_gateway" in available()


def test_the_default_generator_is_the_free_local_one() -> None:
    """The free-first rule, as it appears in the shipped config."""
    import config
    from config import PROJECT_ROOT

    settings = config.load_settings(PROJECT_ROOT / "config" / "settings.yaml")

    assert settings.generator_default == "local_comfyui"
    assert settings.paid_adapters_enabled is False


def test_unknown_adapter_names_list_what_is_available(tmp_path: Path) -> None:
    with pytest.raises(UnknownAdapter, match="local_comfyui"):
        get_adapter(
            "nope",
            settings=make_settings(),
            budget=Budget(),
            cache_dir=tmp_path,
        )


def test_a_paid_adapter_must_declare_a_price() -> None:
    """Without a number the budget guard has nothing to count."""
    class _Priceless(GeneratorAdapter):
        name = "_test_priceless"
        is_paid = True
        estimated_cost_per_call_usd = 0.0

        def generate(self, prompt: str, **kwargs) -> Path:
            return Path("x")

    with pytest.raises(ValueError, match="no per-call cost"):
        register(_Priceless)


# --- the free-first guard -------------------------------------------------

def test_paid_adapters_cannot_be_constructed_while_disabled(tmp_path: Path) -> None:
    with pytest.raises(PaidAdapterDisabled, match="paid_adapters_enabled"):
        get_adapter(
            "_test_paid",
            settings=make_settings(paid_adapters_enabled=False),
            budget=Budget(max_spend_usd=100.0),
            cache_dir=tmp_path,
        )


def test_free_adapters_are_constructed_while_paid_are_disabled(tmp_path: Path) -> None:
    adapter = get_adapter(
        "_test_free",
        settings=make_settings(paid_adapters_enabled=False),
        budget=Budget(),
        cache_dir=tmp_path,
    )

    assert adapter.name == "_test_free"
    assert adapter.is_paid is False


def test_paid_adapters_are_constructed_once_enabled(tmp_path: Path) -> None:
    adapter = get_adapter(
        "_test_paid",
        settings=make_settings(paid_adapters_enabled=True),
        budget=Budget(max_spend_usd=1.0, paid_adapters_enabled=True),
        cache_dir=tmp_path,
    )

    assert adapter.is_paid is True


# --- budget integration ---------------------------------------------------

def test_a_free_adapter_leaves_the_ledger_empty(tmp_path: Path) -> None:
    adapter, budget = build(_FakeFree, tmp_path)

    for _ in range(50):
        adapter.generate("anything")

    assert budget.spent_usd == 0.0
    assert budget.entries == []


def test_a_paid_adapter_charges_per_call(tmp_path: Path) -> None:
    adapter, budget = build(_FakePaid, tmp_path, enabled=True, ceiling=1.0)

    adapter.generate("one")
    adapter.generate("two")

    assert budget.spent_usd == pytest.approx(0.50)


def test_a_paid_adapter_stops_at_the_ceiling(tmp_path: Path) -> None:
    adapter, budget = build(_FakePaid, tmp_path, enabled=True, ceiling=0.60)

    adapter.generate("one")
    adapter.generate("two")
    with pytest.raises(BudgetExceeded):
        adapter.generate("three")

    assert budget.spent_usd == pytest.approx(0.50)


# --- requests and caching -------------------------------------------------

def test_frames_are_derived_from_seconds_and_fps() -> None:
    assert GenerationRequest("x", seconds=4.0, fps=24.0).frames == 96
    assert GenerationRequest("x", seconds=0.0, fps=24.0).frames == 1


def test_identical_requests_share_a_cache_key() -> None:
    first = GenerationRequest("a neon street", seconds=4.0, seed=7)
    second = GenerationRequest("a neon street", seconds=4.0, seed=7)

    assert first.cache_key("x") == second.cache_key("x")


@pytest.mark.parametrize(
    "changed",
    [
        {"prompt": "a different street"},
        {"seed": 8},
        {"seconds": 5.0},
        {"width": 1920},
        {"negative_prompt": "blurry"},
        {"extra": {"guidance": 7}},
    ],
)
def test_any_meaningful_change_changes_the_cache_key(changed: dict) -> None:
    base = dict(prompt="a neon street", seconds=4.0, seed=7)
    assert (
        GenerationRequest(**base).cache_key("x")
        != GenerationRequest(**{**base, **changed}).cache_key("x")
    )


def test_the_adapter_name_is_part_of_the_cache_key() -> None:
    """Two adapters given the same prompt must not collide in the cache."""
    request = GenerationRequest("a neon street")

    assert request.cache_key("local_comfyui") != request.cache_key("fal_gateway")


def test_loose_kwargs_split_into_known_fields_and_extra(tmp_path: Path) -> None:
    adapter, _ = build(_FakeFree, tmp_path)

    request = adapter.build_request(
        "a prompt", seconds=6.0, width=1280, guidance_scale=7.5
    )

    assert request.seconds == 6.0
    assert request.width == 1280
    assert request.extra == {"guidance_scale": 7.5}
