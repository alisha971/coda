"""J3 — calibration: the positive control, the detectability curve, the MDE, G1, G2.

Run and reported **before any real result**. For each
coupling form (identity, cross-dimension) and each target partial R² in
``configs/j3_injection.yaml``, R replicates of Y* are tested against the N1 and
N2 donor nulls. β = 0 gives the null calibration (G1); the power curve gives
the MDE per block and in aggregate (G2).

Outputs: ``j3_levels.json`` (every level, replicate p-values, rejection rates
with binomial intervals), ``detectability.json`` (power per level, MDE),
``gate_G1.json``, ``gate_G2.json``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import RidgeConfig
from src.controls.injection import mde_from_power
from src.experiments.gates import g1, g2
from src.experiments.layer1 import (
    content_fit_for_injection,
    j3_level,
    merge_level_chunks,
)
from src.experiments.pipeline import AnalysisData
from src.experiments.preflight import require_preflight
from src.experiments.run_j2 import draw_donors
from src.experiments.runio import RunWriter, read_checkpoint
from src.settings import Settings, as_record, measurement_signature


def run_j3(data: AnalysisData, turns, prosody, *, results_root: Path, data_kind: str,
           stage: str, n_draws: int, targets: list[float], settings: Settings | None = None,
           ridge: RidgeConfig | None = None, preflight_report=None,
           couplings=("identity_matched_dimensions", "cross_dimension_variant"),
           replicates: int | None = None, replicates_beta0: int | None = None,
           seed: int = 20260912, resume: bool = False,
           chunk_replicates: int | None = None) -> tuple[Path, dict]:
    """J3 calibration, chunked and resumable.

    Each β level runs in chunks of at most ``chunk_replicates`` replicates
    (``Settings.injection``), which bounds memory at large R. Every chunk is written
    as ``j3_level_<k>_c<j>.json`` when it completes; with ``resume=True`` an
    interrupted run continues in the same run directory and skips finished
    chunks. Donors are seeded, and each replicate is seeded by (level seed,
    replicate index), so neither the chunk size nor an interruption changes any
    number.
    """
    require_preflight(preflight_report, data_kind)
    settings = settings or Settings()
    ridge = ridge or RidgeConfig()
    inj, g1_rule = settings.injection, settings.g1_rule
    alpha = float(g1_rule["alpha"])
    R = replicates or int(inj["replicates_per_beta"])
    R0 = replicates_beta0 or int(g1_rule["replicates_beta0"])
    chunk = int(chunk_replicates or inj.get("chunk_replicates", 250))
    cfg = {"ridge": ridge.__dict__, "n_draws": n_draws, "targets": targets, "seed": seed,
           "replicates": R, "replicates_beta0": R0, "couplings": list(couplings),
           "chunk_replicates": chunk, "residual_resampling": inj["residual_resampling"],
           "settings": as_record(settings)}
    with RunWriter(results_root, "j3", data_kind, stage, measurement_signature(settings),
                   data.split_hash, cfg, provenance=data.provenance(), resume=resume) as run:
        donors = draw_donors(data, turns, prosody, n_draws, seed + 100)
        donors = {"N1": donors["N1"], "N2": donors["N2"]}
        content = content_fit_for_injection(data, ridge)
        plan = [(0.0, "identity_matched_dimensions", R0, seed)]
        plan += [(t, c, R, seed + 1 + k) for c in couplings for k, t in enumerate(targets)]
        levels = []
        for i, (t, c, reps, sd) in enumerate(plan):
            parts = []
            for j, start in enumerate(range(0, reps, chunk)):
                done = run.dir / f"j3_level_{i:02d}_c{j:03d}.json"
                cached = read_checkpoint(done) if resume else None
                if cached is not None:
                    parts.append(cached)
                    continue
                if resume and done.exists():        # unreadable: kept aside, recomputed
                    run.set_aside(done.name)
                part = j3_level(data, ridge, content, t, c, inj["cross_variant_permutation"],
                                reps, donors, alpha, sd,
                                resampling=inj["residual_resampling"],
                                replicate_ids=range(start, min(start + chunk, reps)))
                run.json(f"j3_level_{i:02d}_c{j:03d}", part)
                parts.append(part)
            levels.append(merge_level_chunks(parts, alpha))
        run.json("j3_levels", levels)
        g1_res = g1({s: levels[0][s]["rejection_aggregate"] for s in ("N1", "N2")}, settings,
                    n_draws=n_draws)
        run.json("gate_G1", g1_res.as_dict())
        power_level = float(settings.mde_power)
        curve = {}
        for c in couplings:
            lv = [x for x in levels[1:] if x["coupling"] == c]
            t = [x["target_partial_r2"] for x in lv]
            p_agg = [x["N1"]["rejection_aggregate"]["rate"] for x in lv]
            n_blk = len(lv[0]["N1"]["rejection_block"])
            p_blk = [[x["N1"]["rejection_block"][b]["rate"] for x in lv] for b in range(n_blk)]
            realised = [x["realised_aggregate_mean"] for x in lv]
            curve[c] = {"targets": t, "power_aggregate": p_agg, "power_block": p_blk,
                        "mde_aggregate": mde_from_power(t, p_agg, power_level),
                        "mde_block": [mde_from_power(t, pb, power_level) for pb in p_blk],
                        "realised_partial_r2": realised,
                        # The same curve against the partial R² actually realised on the
                        # held-out pairs. β is set from training variances, so the realised
                        # effect can fall short of the target; the larger of the two MDEs
                        # is the conservative floor.
                        "mde_aggregate_realised": mde_from_power(realised, p_agg, power_level)}
        primary = curve.get("identity_matched_dimensions", next(iter(curve.values())))
        both = [v for v in (primary["mde_aggregate"], primary["mde_aggregate_realised"])
                if v is not None]
        mde = {"aggregate": max(both) if both else None,
               "aggregate_target_scale": primary["mde_aggregate"],
               "aggregate_realised_scale": primary["mde_aggregate_realised"],
               "block": primary["mde_block"], "power_level": power_level,
               "coupling": "identity_matched_dimensions",
               "rule": "the larger of the target-scale and realised-scale MDE"}
        run.json("detectability", {"curves": curve, "mde": mde, "alpha": alpha})
        g2_res = g2(mde, settings)
        run.json("gate_G2", g2_res.as_dict())
    return run.dir, {"G1": g1_res, "G2": g2_res, "mde": mde,
                     "beta0": {s: levels[0][s]["rejection_aggregate"] for s in ("N1", "N2")},
                     "realised_mean": float(np.mean([x["realised_aggregate_mean"]
                                                      for x in levels[1:]]))}


__all__ = ["run_j3"]
