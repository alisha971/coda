"""The frozen neural training configuration.

Searched **once** on the real condition's validation set, then frozen, hashed,
and reused unchanged for every donor condition and every ablation. The hash is
asserted equal across runs (donor-condition-tuning row). Only the
partner mode and the donor assignment may differ between conditions, and
neither is part of this configuration.

The values come from :class:`src.settings.Settings` (``neural_training``): the
specification leaves them to the implementer and requires only that they
be identical across conditions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    optimiser: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    dropout: float = 0.1
    grad_clip_norm: float = 1.0
    batch_size: int = 64
    max_epochs: int = 40
    patience: int = 5
    speaker_dropout: float = 0.2
    heteroscedastic_after_warm_start: bool = False
    warm_start_epochs: int = 5
    amp: bool = False                 # only if it measurably helps

    @classmethod
    def from_settings(cls, settings=None) -> TrainConfig:
        from src.settings import Settings

        v = (settings or Settings()).neural_training
        return cls(optimiser=v["optimiser"], learning_rate=float(v["learning_rate"]),
                   weight_decay=float(v["weight_decay"]), dropout=float(v["dropout"]),
                   grad_clip_norm=float(v["grad_clip_norm"]), batch_size=int(v["batch_size"]),
                   max_epochs=int(v["max_epochs"]), patience=int(v["patience"]),
                   speaker_dropout=float(v["speaker_dropout"]))

    def hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def search_grid(base: TrainConfig, grid: dict[str, list]) -> list[TrainConfig]:
    """The small, fixed search, run once on the real condition's validation set."""
    keys = sorted(grid)
    return [replace(base, **dict(zip(keys, vals, strict=True)))
            for vals in product(*(grid[k] for k in keys))]


def freeze(cfg: TrainConfig, path: Path, search_record: dict) -> str:
    """Write the chosen configuration; refuses to replace an existing frozen file."""
    if path.exists():
        raise FileExistsError(f"{path} is frozen; a new search would be a new study decision")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": asdict(cfg), "hash": cfg.hash(),
                                "search": search_record}, indent=2), encoding="utf-8")
    return cfg.hash()


def load_frozen(path: Path) -> TrainConfig:
    d = json.loads(path.read_text(encoding="utf-8"))
    cfg = TrainConfig(**d["config"])
    if cfg.hash() != d["hash"]:
        raise AssertionError("frozen configuration does not match its recorded hash")
    return cfg


def assert_same_config(hashes: dict[str, str]) -> None:
    if len(set(hashes.values())) != 1:
        raise AssertionError(f"training configuration differs across conditions: {hashes}")


__all__ = ["TrainConfig", "search_grid", "freeze", "load_frozen", "assert_same_config"]
