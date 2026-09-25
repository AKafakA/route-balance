"""Strict schemas for the files the benchmark and the analysis read and write.

Every key is required, unknown keys are rejected, nothing is coerced (no "1" for 1, no true for 1), and
non-finite numbers are refused. JSON text goes through routebalance.config.loads_strict, which also
rejects duplicate keys.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from routebalance.config import ConfigError, loads_strict

M = TypeVar("M", bound=BaseModel)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


def read_json(path: Path) -> Any:
    try:
        text = Path(path).read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise ConfigError(f"cannot read {path}: {e}") from e
    try:
        return loads_strict(text)
    except ConfigError as e:
        raise ConfigError(f"{path}: {e}") from e


def parse(model: type[M], data: Any, what: str) -> M:
    try:
        return model.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"invalid {what}: {e}") from e
