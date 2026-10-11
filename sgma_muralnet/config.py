"""Configuration loading: a YAML dict with attribute access, merged onto defaults."""
from __future__ import annotations

import os
from argparse import Namespace
from typing import Any, Dict, Optional

import yaml

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    "config", "default.yaml")


class Config:
    """Attribute-style access to configuration values loaded from YAML."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self._dict: Dict[str, Any] = {}
        if cfg is not None:
            self._dict.update(cfg)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        # Merge the file onto the built-in defaults so partial configs work.
        defaults: Dict[str, Any] = {}
        if os.path.exists(_DEFAULT_CONFIG_PATH):
            with open(_DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
                defaults = yaml.safe_load(f) or {}
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls({**defaults, **data})

    @classmethod
    def from_defaults(cls) -> "Config":
        return cls.from_yaml(_DEFAULT_CONFIG_PATH)

    def update(self, other: Dict[str, Any]) -> None:
        self._dict.update(other)

    def apply_args(self, args: Namespace) -> None:
        """Apply non-None argparse fields as overrides."""
        for key, value in vars(args).items():
            if value is not None and key in self._dict:
                self._dict[key] = value

    def __getattr__(self, name: str) -> Any:
        # Only called when normal attribute lookup fails.
        try:
            return self._dict[name]
        except KeyError:
            raise AttributeError(f"config has no attribute {name!r}") from None

    def get(self, name: str, default: Any = None) -> Any:
        return self._dict.get(name, default)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._dict)

    def __repr__(self) -> str:
        return f"Config({self._dict!r})"
