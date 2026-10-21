"""Run directories: identified, never overwritten, labelled by what data they hold.

Every analysis run writes into one directory

    results/<experiment>/<data_kind>__<stage>__m<measurement>__s<split>__<run_key>/

- ``data_kind`` is ``synthetic`` (a fixture or dry run), ``pilot`` (the 40-session
  development subset; never confirmatory, Prereg §5) or ``sample`` (the
  registered 250-session sample). Every output file inherits the label through
  its directory and through ``run_record.json``.
- ``m<measurement>`` is the signature of the measurement settings in force
  (``src.settings.MEASUREMENT_FIELDS``), so results computed under different
  definitions cannot overwrite each other or be mixed by accident.
- ``s<split>`` is the split-manifest hash.
- ``run_key`` hashes the configuration, so identical configurations resolve to
  the same directory and different ones never collide.

A directory whose status is ``completed`` is never written again. A
``running`` or ``failed`` directory can be resumed only when the caller asks for
it explicitly, and the attempt is logged. Individual files are never
overwritten. Exceptions are captured into ``status.json`` with the traceback
before being re-raised.
"""

from __future__ import annotations

import hashlib
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATA_KINDS = ("synthetic", "pilot", "sample")


class RunExists(RuntimeError):
    """A completed run already occupies this directory."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonable(obj: Any):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def run_key(config: dict) -> str:
    blob = json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via a temporary file and rename, so an interrupted write (a Colab runtime
    dying while Drive is syncing) never leaves a truncated checkpoint behind."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_checkpoint(path: Path) -> dict | None:
    """Content of a checkpoint file, or None if it is missing or unreadable."""
    try:
        return read_json(path)
    except (FileNotFoundError, json.JSONDecodeError, KeyError, UnicodeDecodeError):
        return None


class RunWriter:
    """Context manager owning one run directory."""

    def __init__(self, results_root: Path, experiment: str, data_kind: str, stage: str,
                 measurement_signature: str, split_hash: str, config: dict,
                 provenance: dict | None = None, resume: bool = False):
        if data_kind not in DATA_KINDS:
            raise ValueError(f"data_kind must be one of {DATA_KINDS}")
        self.config = {"experiment": experiment, "data_kind": data_kind, "stage": stage,
                       "measurement_signature": measurement_signature,
                       "split_hash": split_hash, **config}
        self.key = run_key(self.config)
        name = f"{data_kind}__{stage}__m{measurement_signature}__s{split_hash}__{self.key}"
        self.dir = Path(results_root) / experiment / name
        status_path = self.dir / "status.json"
        if status_path.exists():
            st = json.loads(status_path.read_text(encoding="utf-8"))
            if st.get("status") == "completed":
                raise RunExists(f"{self.dir} is a completed run and is never overwritten")
            if not resume:
                raise RunExists(f"{self.dir} exists with status {st.get('status')!r}; "
                                "pass resume=True to continue it")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.provenance = provenance or {}
        self._attempt = self._next_attempt()

    def _next_attempt(self) -> int:
        log = self.dir / "attempts.jsonl"
        n = sum(1 for _ in log.open(encoding="utf-8")) if log.exists() else 0
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"attempt": n + 1, "started_utc": _now()}) + "\n")
        return n + 1

    # -- context ---------------------------------------------------------

    def __enter__(self) -> RunWriter:
        from src.config import environment_record

        record = {"run_key": self.key, "directory": self.dir.name, "config": self.config,
                  "environment": environment_record(),
                  "provenance": self.provenance, "created_utc": _now(),
                  "attempt": self._attempt}
        rec_path = self.dir / f"run_record_attempt{self._attempt}.json"
        rec_path.write_text(json.dumps(_jsonable(record), indent=2, default=str),
                            encoding="utf-8")
        self._status("running")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self._status("completed")
        else:
            self._status("failed", error=repr(exc),
                         traceback="".join(traceback.format_exception(exc_type, exc, tb)))
        return False

    def _status(self, status: str, **extra) -> None:
        (self.dir / "status.json").write_text(
            json.dumps({"status": status, "updated_utc": _now(), "attempt": self._attempt,
                        "data_kind": self.config["data_kind"], **extra}, indent=2),
            encoding="utf-8")

    # -- outputs ---------------------------------------------------------

    def _path(self, name: str) -> Path:
        p = self.dir / name
        if p.exists():
            raise RunExists(f"{p} already exists; outputs are never overwritten")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def set_aside(self, name: str) -> Path | None:
        """Rename an unreadable checkpoint so it can be recomputed; it is kept, never
        deleted, as evidence of the interruption (``<name>.unreadable.<attempt>``)."""
        p = self.dir / name
        if not p.exists():
            return None
        target = p.with_name(f"{p.name}.unreadable.{self._attempt}")
        os.replace(p, target)
        return target

    def json(self, name: str, obj) -> Path:
        p = self._path(name if name.endswith(".json") else f"{name}.json")
        payload = {"data_kind": self.config["data_kind"], "run_key": self.key,
                   "content": _jsonable(obj)}
        _atomic_write_text(p, json.dumps(payload, indent=2, default=str))
        return p

    def table(self, name: str, df: pd.DataFrame) -> Path:
        p = self._path(name if name.endswith(".csv") else f"{name}.csv")
        df.assign(data_kind=self.config["data_kind"]).to_csv(p, index=False)
        return p

    def array(self, name: str, **arrays) -> Path:
        """Numeric arrays (predictions, null draws). Binary: kept out of git by .gitignore."""
        p = self._path(name if name.endswith(".npz") else f"{name}.npz")
        np.savez_compressed(p, data_kind=self.config["data_kind"], **arrays)
        return p


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))["content"]


def find_runs(results_root: Path, experiment: str, data_kind: str | None = None,
              completed_only: bool = True) -> list[Path]:
    """Run directories of an experiment, optionally filtered by data kind."""
    base = Path(results_root) / experiment
    if not base.exists():
        return []
    out = []
    for d in sorted(base.iterdir()):
        st = d / "status.json"
        if not st.exists():
            continue
        s = json.loads(st.read_text(encoding="utf-8"))
        if completed_only and s.get("status") != "completed":
            continue
        if data_kind and not d.name.startswith(f"{data_kind}__"):
            continue
        out.append(d)
    return out


__all__ = ["DATA_KINDS", "RunExists", "RunWriter", "run_key", "read_json", "find_runs"]
