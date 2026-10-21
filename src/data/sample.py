"""From the 40-session pilot to the registered 250-session sample.

The selection rule is fixed by the preregistration: sessions are taken in the
seeded order of ``manifests/coda_download_manifest.csv``; a session that fails
channel QC is replaced by the next session in that order; processing stops when
250 sessions have passed ("processing continues down the same
order until 250 sessions pass channel QC"). Nothing here looks at any delivery
measure, so the sample cannot be chosen on the outcome.

The pilot is the first 40 accepted sessions of the same order. It stays a
separate, traceable stage: every output carries ``stage = pilot`` or
``stage = sample``, and :func:`sample_manifest` records for each session whether
it belongs to the pilot, the planned sample, or the reserve that replaced a
failure.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from src.preprocessing.channel_qc import usable_sessions


def sample_manifest(manifest: pd.DataFrame, channel_qc: pd.DataFrame, target: int,
                    n_pilot: int) -> pd.DataFrame:
    """One row per session walked in rank order, with its fate and reason."""
    ranked = (manifest.drop_duplicates("session_key")
              .assign(selection_rank=lambda d: d["selection_rank"].astype(int))
              .sort_values("selection_rank")[["session_key", "selection_rank", "role"]])
    passed = usable_sessions(channel_qc) if len(channel_qc) else pd.Series(dtype=bool)
    rows, accepted = [], 0
    for r in ranked.itertuples(index=False):
        if accepted >= target:
            break
        if r.session_key not in passed.index:
            status, reason = "not_measured", "audio pass has not processed this session"
        elif not bool(passed[r.session_key]):
            status, reason = "excluded", "no interaction with both channels passing QC"
        else:
            status, reason = "accepted", None
            accepted += 1
        rows.append({"session_key": r.session_key, "selection_rank": r.selection_rank,
                     "manifest_role": r.role, "status": status, "reason": reason,
                     "accepted_index": accepted - 1 if status == "accepted" else None})
    out = pd.DataFrame(rows)
    out["stage"] = [("pilot" if (s == "accepted" and i is not None and i < n_pilot) else
                     "sample" if s == "accepted" else None)
                    for s, i in zip(out["status"], out["accepted_index"], strict=True)]
    out.attrs["complete"] = accepted >= target
    return out


def sample_audit(sm: pd.DataFrame, manifest: pd.DataFrame, target: int, n_pilot: int) -> dict:
    """Checks on the selected sample; raises on any violation of the registered rule."""
    acc = sm[sm["status"] == "accepted"]
    if not acc["selection_rank"].is_monotonic_increasing:
        raise AssertionError("accepted sessions are not in selection order")
    if (sm["status"] == "not_measured").any() and len(acc) < target:
        status = "incomplete: sessions remain to be measured"
    elif len(acc) < target:
        status = "incomplete: manifest exhausted before the target"
    else:
        status = "complete"
    parts = manifest[manifest["session_key"].isin(acc["session_key"])]
    per_speaker = parts.groupby("participant_key")["session_key"].nunique()
    if (per_speaker > 1).any():
        raise AssertionError("a speaker appears in more than one accepted session")
    pilot = acc[acc["stage"] == "pilot"]
    pilot_manifest = manifest.loc[manifest["role"] == "pilot", "session_key"].unique()
    return {"status": status, "target": target, "accepted": int(len(acc)),
            "pilot_accepted": int(len(pilot)),
            "pilot_is_manifest_pilot": bool(set(pilot["session_key"]) <= set(pilot_manifest)),
            "excluded_for_qc": int((sm["status"] == "excluded").sum()),
            "reserve_used": int((acc["manifest_role"] == "reserve").sum()),
            "speakers": int(per_speaker.size),
            "hash": hashlib.sha256(json.dumps(acc["session_key"].tolist()).encode())
            .hexdigest()[:12]}


__all__ = ["sample_manifest", "sample_audit"]
