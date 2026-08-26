"""The one interface every clip generator implements.

The point of this file is that `assembler.py` never learns which adapter made a
clip. Swapping local Wan 2.2 for a paid Seedance call is a line in
settings.yaml, not a rewrite — every adapter takes a prompt and returns a path
to a finished file on disk.

Two invariants are enforced here rather than left to each adapter to remember:

  * A paid adapter cannot be constructed at all while `paid_adapters_enabled`
    is false. The guard is at the factory, so there is no code path that
    reaches a paid API by accident.
  * Cost is reserved through `reserve_budget()` *before* the request goes out.
    Adapters declare a per-call price as a class attribute; local adapters
    declare 0.0 and never touch the ledger.
"""

from __future__ import annotations

import abc
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from budget import Budget, PaidAdapterDisabled

log = logging.getLogger(__name__)


class GenerationError(RuntimeError):
    """An adapter could not produce an asset."""


class UnknownAdapter(KeyError):
    """No adapter is registered under that name."""


@dataclass(frozen=True)
class GenerationRequest:
    """The parameters every adapter understands.

    Adapters are free to accept more via **kwargs, but must handle these — the
    pipeline fills them from the output format so a generated clip does not
    need rescaling before it hits the timeline.
    """

    prompt: str
    seconds: float = 5.0
    width: int = 1280
    height: int = 720
    fps: float = 24.0
    seed: int | None = None
    negative_prompt: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def frames(self) -> int:
        return max(1, round(self.seconds * self.fps))

    def cache_key(self, adapter_name: str) -> str:
        """Stable digest of everything that changes the output.

        Regenerating a clip costs minutes locally and real money on a paid
        adapter, so identical requests must resolve to the same file.
        """
        payload = json.dumps(
            {
                "adapter": adapter_name,
                "prompt": self.prompt,
                "negative_prompt": self.negative_prompt,
                "seconds": round(self.seconds, 3),
                "width": self.width,
                "height": self.height,
                "fps": round(self.fps, 3),
                "seed": self.seed,
                "extra": self.extra,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class GeneratorAdapter(abc.ABC):
    """Base class for every clip generator."""

    #: Name used in settings.yaml and on the CLI.
    name: ClassVar[str] = ""
    #: True if calling this adapter costs money.
    is_paid: ClassVar[bool] = False
    #: Estimated USD per call, used by the budget guard. Must be > 0 if is_paid.
    estimated_cost_per_call_usd: ClassVar[float] = 0.0

    def __init__(
        self,
        *,
        settings,
        budget: Budget,
        cache_dir: Path | str,
    ) -> None:
        self.settings = settings
        self.budget = budget
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def generate(self, prompt: str, **kwargs) -> Path:
        """Produce one clip and return the local path to it.

        Blocks until the asset exists on disk. Accepts the fields of
        `GenerationRequest` as keyword arguments.
        """

    # -- helpers shared by every adapter ----------------------------------

    def build_request(self, prompt: str, **kwargs) -> GenerationRequest:
        """Normalise loose kwargs into a GenerationRequest."""
        known = {
            field_name
            for field_name in GenerationRequest.__dataclass_fields__
            if field_name != "extra"
        }
        core = {key: value for key, value in kwargs.items() if key in known}
        extra = {key: value for key, value in kwargs.items() if key not in known}
        return GenerationRequest(prompt=prompt, extra=extra, **core)

    def cached_path(self, request: GenerationRequest, suffix: str = ".mp4") -> Path:
        return self.cache_dir / f"{self.name}_{request.cache_key(self.name)}{suffix}"

    def reserve_budget(self, units: int = 1, note: str = "") -> float:
        """Reserve spend before making a request. Raises to halt the run.

        Called once per *attempt*, retries included — a retried request is a
        second billable call, and a budget that only counts successes cannot
        stop a loop that keeps failing.
        """
        return self.budget.charge(
            self.name, self.estimated_cost_per_call_usd, units=units, note=note
        )

    def __repr__(self) -> str:
        price = (
            f"${self.estimated_cost_per_call_usd:.4f}/call"
            if self.is_paid
            else "free"
        )
        return f"<{type(self).__name__} {self.name} ({price})>"


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

_REGISTRY: dict[str, type[GeneratorAdapter]] = {}


def register(adapter_class: type[GeneratorAdapter]) -> type[GeneratorAdapter]:
    """Class decorator that makes an adapter selectable by name."""
    if not adapter_class.name:
        raise ValueError(f"{adapter_class.__name__} must set a `name`.")
    if adapter_class.is_paid and adapter_class.estimated_cost_per_call_usd <= 0:
        # Without a declared price the budget guard has nothing to count, and a
        # runaway loop would be invisible to it.
        raise ValueError(
            f"{adapter_class.__name__} is paid but declares no per-call cost; "
            f"the budget guard cannot protect a price it does not know."
        )
    _REGISTRY[adapter_class.name] = adapter_class
    return adapter_class


def available() -> list[str]:
    return sorted(_REGISTRY)


def adapter_class(name: str) -> type[GeneratorAdapter]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise UnknownAdapter(
            f"No generator named {name!r}. Available: {', '.join(available())}"
        ) from None


def get_adapter(
    name: str,
    *,
    settings,
    budget: Budget,
    cache_dir: Path | str,
) -> GeneratorAdapter:
    """Build the named adapter, refusing paid ones unless they are enabled.

    This is the single chokepoint for the free-first rule. Nothing else in the
    codebase constructs an adapter directly.
    """
    cls = adapter_class(name)
    if cls.is_paid and not settings.paid_adapters_enabled:
        raise PaidAdapterDisabled(
            f"{name} is a paid adapter (about "
            f"${cls.estimated_cost_per_call_usd:.4f} per call) and "
            f"paid_adapters_enabled is false in config/settings.yaml. "
            f"Set it to true, deliberately, to allow paid calls."
        )
    if cls.is_paid:
        log.warning(
            "Using PAID adapter %s at ~$%.4f per call. Run ceiling: $%.2f",
            name, cls.estimated_cost_per_call_usd, budget.max_spend_usd,
        )
    return cls(settings=settings, budget=budget, cache_dir=cache_dir)
