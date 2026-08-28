"""The cost guard.

Phase 1 never charges anything, but this is the module that has to be correct
*before* anyone enables a paid adapter, not after the first surprise invoice.
"""

from __future__ import annotations

import pytest

from budget import Budget, BudgetExceeded, PaidAdapterDisabled


def test_free_adapters_never_touch_the_ledger() -> None:
    budget = Budget(max_spend_usd=0.0, paid_adapters_enabled=False)

    charged = budget.charge("local_comfyui", unit_cost_usd=0.0, units=25)

    assert charged == 0.0
    assert budget.spent_usd == 0.0
    assert budget.entries == []
    assert budget.report() == "Estimated run cost: $0.00 (no paid calls made)"


def test_a_paid_call_is_refused_while_paid_adapters_are_disabled() -> None:
    budget = Budget(max_spend_usd=100.0, paid_adapters_enabled=False)

    with pytest.raises(PaidAdapterDisabled, match="paid_adapters_enabled"):
        budget.charge("fal_gateway", unit_cost_usd=0.25)

    assert budget.spent_usd == 0.0


def test_charges_accumulate_while_under_the_ceiling() -> None:
    budget = Budget(max_spend_usd=1.0, paid_adapters_enabled=True)

    budget.charge("fal_gateway", unit_cost_usd=0.25, units=2)
    budget.charge("heygen", unit_cost_usd=0.10)

    assert budget.spent_usd == pytest.approx(0.60)
    assert budget.remaining_usd == pytest.approx(0.40)
    assert budget.by_adapter() == {
        "fal_gateway": pytest.approx(0.50),
        "heygen": pytest.approx(0.10),
    }


def test_a_charge_over_the_ceiling_halts_and_is_not_recorded() -> None:
    budget = Budget(max_spend_usd=1.0, paid_adapters_enabled=True)
    budget.charge("fal_gateway", unit_cost_usd=0.90)

    with pytest.raises(BudgetExceeded, match="ceiling"):
        budget.charge("fal_gateway", unit_cost_usd=0.20)

    # The refused call must leave no trace, or a caught-and-retried failure
    # would inflate the ledger without any money having moved.
    assert budget.spent_usd == pytest.approx(0.90)
    assert len(budget.entries) == 1


def test_a_runaway_retry_loop_is_stopped_by_the_ceiling() -> None:
    """The scenario the module exists for: a bug that keeps calling."""
    budget = Budget(max_spend_usd=0.50, paid_adapters_enabled=True)

    calls = 0
    with pytest.raises(BudgetExceeded):
        for _ in range(10_000):
            budget.charge("fal_gateway", unit_cost_usd=0.05)
            calls += 1

    assert calls == 10
    assert budget.spent_usd == pytest.approx(0.50)


def test_the_ceiling_is_inclusive() -> None:
    budget = Budget(max_spend_usd=0.30, paid_adapters_enabled=True)

    budget.charge("fal_gateway", unit_cost_usd=0.30)

    assert budget.spent_usd == pytest.approx(0.30)


def test_negative_costs_are_rejected() -> None:
    budget = Budget(max_spend_usd=1.0, paid_adapters_enabled=True)

    with pytest.raises(ValueError):
        budget.charge("fal_gateway", unit_cost_usd=-1.0)
    with pytest.raises(ValueError):
        budget.charge("fal_gateway", unit_cost_usd=1.0, units=-1)


def test_report_breaks_spend_down_by_adapter() -> None:
    budget = Budget(max_spend_usd=5.0, paid_adapters_enabled=True)
    budget.charge("fal_gateway", unit_cost_usd=0.25, units=2, note="seedance")
    budget.charge("heygen", unit_cost_usd=1.00, note="avatar")

    report = budget.report()

    assert "$1.5000" in report
    assert "fal_gateway: $0.5000 across 2 call(s)" in report
    assert "heygen: $1.0000 across 1 call(s)" in report
