"""Normalisation statistics fitted on the training split only.

Every statistic stored here records the rows it was fitted on (a hash of the
fit index) and the split it came from, so that a later step can assert, at load
time, that nothing outside training contributed to it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class ProvenanceError(AssertionError):
    """A statistic was fitted on rows outside the training split."""


def index_hash(ids) -> str:
    return hashlib.sha256(json.dumps(sorted(map(str, ids))).encode()).hexdigest()[:12]


@dataclass
class FittedStats:
    """A fitted transform's parameters plus where they came from."""

    name: str
    params: dict
    fit_ids: list[str]
    split_hash: str
    source: str = "train"
    extra: dict = field(default_factory=dict)

    @property
    def fit_hash(self) -> str:
        return index_hash(self.fit_ids)

    def assert_training_only(self, train_ids) -> None:
        if self.source != "train":
            raise ProvenanceError(f"{self.name} was fitted on '{self.source}', not train")
        outside = set(map(str, self.fit_ids)) - set(map(str, train_ids))
        if outside:
            raise ProvenanceError(f"{self.name} was fitted on {len(outside)} non-training rows")

    def record(self) -> dict:
        return {"name": self.name, "source": self.source, "n_fit": len(self.fit_ids),
                "fit_hash": self.fit_hash, "split_hash": self.split_hash, **self.extra}


def speaker_stats(prosody: pd.DataFrame, dims: list[str], train_turn_ids,
                  split_hash: str) -> FittedStats:
    """Per-speaker mean and SD (ddof = 1) over usable training turns."""
    train_turn_ids = set(train_turn_ids)
    p = prosody[prosody["turn_id"].isin(train_turn_ids) & prosody["usable"].astype(bool)]
    g = p.groupby("speaker")[dims]
    params = {"mean": g.mean(), "sd": g.std(ddof=1), "n": g.size()}
    return FittedStats("speaker_stats", params, p["turn_id"].astype(str).tolist(), split_hash)


def column_scaler(X: np.ndarray, fit_rows: np.ndarray, ids, split_hash: str,
                  name: str) -> FittedStats:
    """Column mean and SD on the fit rows; zero-variance columns get SD 1."""
    Xf = X[fit_rows]
    mu = Xf.mean(axis=0)
    sd = Xf.std(axis=0, ddof=0)
    sd = np.where(sd > 0, sd, 1.0)
    return FittedStats(name, {"mean": mu, "sd": sd}, [str(i) for i in np.asarray(ids)[fit_rows]],
                       split_hash)


__all__ = ["ProvenanceError", "FittedStats", "index_hash", "speaker_stats", "column_scaler"]
