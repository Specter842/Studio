"""Cost guard.

The default pipeline path is free, so in a normal run this module records
nothing and reports $0.00. It exists for the moment somebody flips
`paid_adapters_enabled: true`: at that point a retry loop or an off-by-one in a
generation step is capable of producing a real invoice, and the only reliable
defence is a counter that hard-stops the run.

Two rules make that work:

  1. Adapters charge *before* they make the call, not after. A budget that is
     only updated on success cannot stop a loop that keeps failing and retrying.
  2. A zero-cost adapter (anything local) never records an entry, so the ledger
     stays a precise record of money actually at risk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator

log = logging.getLogger(__name__)


class BudgetExceeded(RuntimeError):
    """A charge would push the run past its spend ceiling. The run must stop."""


class PaidAdapterDisabled(RuntimeError):
    """A paid adapter was invoked while paid_adapters_enabled is false."""


@dataclass(frozen=True)
class LedgerEntry:
    adapter: str
    unit_cost_usd: float
    units: int
    note: str = ""

    @property
    def total_usd(self) -> float:
        return self.unit_cost_usd * self.units


@dataclass
class Budget:
    """Tracks estimated spend for a single pipeline run."""

    max_spend_usd: float = 0.0
    paid_adapters_enabled: bool = False
    entries: list[LedgerEntry] = field(default_factory=list)

    @property
    def spent_usd(self) -> float:
        return sum(entry.total_usd for entry in self.entries)

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.max_spend_usd - self.spent_usd)

    def charge(
        self,
        adapter: str,
        unit_cost_usd: float,
        units: int = 1,
        note: str = "",
    ) -> float:
        """Reserve spend for a call that is about to be made.

        Call this immediately *before* the paid request, once per attempt —
        including retries, since a retried request is a second billable call.

        Returns the newly reserved amount. Raises BudgetExceeded or
        PaidAdapterDisabled instead of reserving, in which case nothing is
        recorded and the caller must not make the request.
        """
        if unit_cost_usd < 0:
            raise ValueError(f"{adapter}: unit_cost_usd may not be negative.")
        if units < 0:
            raise ValueError(f"{adapter}: units may not be negative.")

        cost = unit_cost_usd * units

        # Free adapters are invisible to the ledger, per the free-first design.
        if cost == 0:
            return 0.0

        if not self.paid_adapters_enabled:
            raise PaidAdapterDisabled(
                f"{adapter} declares a cost of ${cost:.4f} but "
                f"paid_adapters_enabled is false. Enable it in "
                f"config/settings.yaml, deliberately, to allow paid calls."
            )

        projected = self.spent_usd + cost
        if projected > self.max_spend_usd:
            raise BudgetExceeded(
                f"{adapter} would take this run to ${projected:.4f}, over the "
                f"${self.max_spend_usd:.4f} ceiling "
                f"(already spent ${self.spent_usd:.4f}). "
                f"Raise --max-spend-usd if this is expected."
            )

        entry = LedgerEntry(adapter, unit_cost_usd, units, note)
        self.entries.append(entry)
        log.info(
            "budget: %s +$%.4f (run total $%.4f of $%.4f)",
            adapter, cost, self.spent_usd, self.max_spend_usd,
        )
        return cost

    def by_adapter(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for entry in self.entries:
            totals[entry.adapter] = totals.get(entry.adapter, 0.0) + entry.total_usd
        return totals

    def report(self) -> str:
        """One-line-per-adapter summary, printed at the end of every run."""
        if not self.entries:
            return "Estimated run cost: $0.00 (no paid calls made)"
        lines = [f"Estimated run cost: ${self.spent_usd:.4f}"]
        for adapter, total in sorted(self.by_adapter().items()):
            calls = sum(e.units for e in self.entries if e.adapter == adapter)
            lines.append(f"  {adapter}: ${total:.4f} across {calls} call(s)")
        return "\n".join(lines)

    def __iter__(self) -> Iterator[LedgerEntry]:
        return iter(self.entries)
