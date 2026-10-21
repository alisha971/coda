"""J1 — the condition sweep, the H2 coupling map, and the mixed-effects re-check.

Four conditions on the speaker-disjoint split (primary). The dyad-disjoint
contrast is run by calling the same driver with the contrast split's
``AnalysisData`` (stage name ``..._dyad_disjoint``); that contrast is degenerate on this sample
(one session per speaker).

Outputs (one run directory, never overwritten):
``j1_summary.json`` (conditions, the three increments with both units and
session-bootstrap intervals, coverage, effective df, λ diagnostics),
``j1_errors.npz`` (per-pair squared errors, so intervals can be recomputed),
``coupling_map.json`` (H2 requirements 1-4; requirement 5 is added by J2),
``mixed_effects.json`` (the re-check, if run), ``droppage_by_condition.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.coupling_map import h2_battery
from src.config import RidgeConfig
from src.experiments.gates import require
from src.experiments.layer1 import fit_j1
from src.experiments.pipeline import DIMS, AnalysisData
from src.experiments.preflight import require_preflight
from src.experiments.runio import RunWriter
from src.models.rrr import fit_rrr
from src.settings import Settings, as_record, measurement_signature


def mixed_effects_frame(data: AnalysisData, rows: np.ndarray | None = None) -> pd.DataFrame:
    """Columns for the re-check: targets, partner dims, Z (text PCs and metadata)."""
    rows = np.ones(len(data.pairs), bool) if rows is None else rows
    df = pd.DataFrame(data.Y[rows], columns=[f"y_{d}" for d in DIMS])
    for j, d in enumerate(DIMS):
        df[f"a_{d}"] = data.A[rows, j]
    z = np.hstack([data.Z.reply_text, data.Z.partner_text, data.Z.metadata])[rows]
    zn = ([f"rpc{j}" for j in range(data.Z.reply_text.shape[1])]
          + [f"ppc{j}" for j in range(data.Z.partner_text.shape[1])] + data.Z.metadata_names)
    for j, n in enumerate(zn):
        df[n] = (z[:, j] - z[:, j].mean()) / (z[:, j].std() or 1.0)
    df["session_key"] = data.groups[rows]
    df["replier"] = data.pairs.loc[rows, "replier"].to_numpy()
    df.attrs["z_cols"] = zn
    return df


def run_j1(data: AnalysisData, *, results_root: Path, data_kind: str, stage: str,
           settings: Settings | None = None, ridge: RidgeConfig | None = None,
           preflight_report=None, gates: dict | None = None, n_boot: int | None = None,
           h2_boot: int = 200, mixed_effects: bool = True, seed: int = 20260912) -> Path:
    require_preflight(preflight_report, data_kind)
    settings = settings or Settings()
    ridge = ridge or RidgeConfig()
    label = require(gates or {}, "j1")
    cfg = {"ridge": ridge.__dict__, "n_boot": n_boot, "h2_boot": h2_boot, "seed": seed,
           "settings": as_record(settings), "output_label": label}
    with RunWriter(results_root, "j1", data_kind, stage, measurement_signature(settings),
                   data.split_hash, cfg, provenance=data.provenance()) as run:
        res = fit_j1(data, ridge, settings, n_boot=n_boot, seed=seed)
        eng = res.pop("_engine_full")
        res.pop("_reduced_pred")
        errors = res.pop("errors")
        preds = res.pop("predictions")
        res["output_label"] = label
        res["confirmatory"] = label == "confirmatory"
        run.json("j1_summary", res)
        run.array("j1_errors", **{k: v.astype(np.float32) for k, v in errors.items()},
                  **{f"pred_{k}": v.astype(np.float32) for k, v in preds.items()},
                  eval_groups=data.groups[eng.zc.eval_idx].astype(str))
        n = len(data.pairs)
        run.table("droppage_by_condition", pd.DataFrame(
            [{"condition": c, "pairs_used": n, "note": "one common analysis set"}
             for c in res["conditions"]]))

        ref = fit_rrr(eng, data.A, data.blocks, ranks=ridge.rank_grid)
        fold = np.full(n, -1)
        fold[eng.zc.fit_idx] = eng.zc.fold
        battery = h2_battery(data.Z.full(), data.A, data.Y, data.groups, data.train,
                             data.blocks, ref, eng, None, h2_boot, fold, seed)
        battery["lambda_at_grid_edge"] = ref.at_grid_edge
        run.json("coupling_map", battery)

        if mixed_effects:
            from src.statistics.mixed_effects import partner_model

            df = mixed_effects_frame(data)
            out = [partner_model(df, f"y_{d}", [f"a_{x}" for x in DIMS], df.attrs["z_cols"])
                   for d in DIMS]
            run.json("mixed_effects", {"label": "confirmatory re-check (drop-first list)",
                                       "models": out})
    return run.dir


__all__ = ["run_j1", "mixed_effects_frame"]
