"""Loads config/settings.yaml and .env.

Two separate concerns, deliberately kept apart:
  * settings.yaml holds behaviour — committed, reviewable, safe to share.
  * .env holds secrets — gitignored, never read by anything but the adapters
    that need a specific key.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

# video_pipeline/ — the project root, two levels up from src/config.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


class Settings:
    """Dotted-path read access over the parsed settings.yaml."""

    def __init__(self, data: dict[str, Any], source: Path | None = None) -> None:
        self._data = data
        self.source = source

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """settings.get("editing.beats_per_cut.high") -> 2"""
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return copy.deepcopy(node) if isinstance(node, (dict, list)) else node

    def require(self, dotted_key: str) -> Any:
        sentinel = object()
        value = self.get(dotted_key, sentinel)
        if value is sentinel:
            raise KeyError(
                f"Missing required setting {dotted_key!r} in {self.source}"
            )
        return value

    @property
    def paid_adapters_enabled(self) -> bool:
        return bool(self.get("paid_adapters_enabled", False))

    @property
    def generator_default(self) -> str:
        return str(self.get("generator_default", "local_comfyui"))

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)


def load_env(env_path: Path | None = None) -> None:
    """Load .env into os.environ. Missing file is fine — Phase 1 needs no keys.

    Existing environment variables win, so CI and shell exports are not
    clobbered by a stale local .env.
    """
    path = Path(env_path) if env_path else DEFAULT_ENV_PATH
    if not path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is a declared dependency
        return
    load_dotenv(path, override=False)


def load_settings(settings_path: Path | None = None) -> Settings:
    path = Path(settings_path) if settings_path else DEFAULT_SETTINGS_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Settings file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level.")
    return Settings(data, source=path)


def load(settings_path: Path | None = None, env_path: Path | None = None) -> Settings:
    """Normal entry point: .env first, then settings."""
    load_env(env_path)
    return load_settings(settings_path)


def secret(name: str) -> str | None:
    """Read a secret from the environment.

    Adapters call this instead of touching os.environ directly, so there is one
    place to audit for key access. Returns None for unset or placeholder values
    left over from .env.example.
    """
    value = os.environ.get(name, "").strip()
    if not value or value.startswith("your_"):
        return None
    return value
