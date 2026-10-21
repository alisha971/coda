"""J4 — function dependence (Level E): G4, the interaction test, per-stratum Δ, G5.

Steps:

1. **Labels** from text only (``src.analysis.situation``). The interrogative
   rule is implemented; the remaining categories need a text-only classifier,
   which is not set. Without it, J4 stops after the rule-based labels with
   status BLOCKED.
2. **G4** from hand labels (300 seeded pairs) against the automatic ones.
3. **Interaction**: per target dimension, the function × partner-delivery term
   in the mixed model; one H3 p-value by ``Settings.h3_aggregation``.
4. **Per-stratum Δ** (descriptive companion): J1's increment refitted within
   each confirmatory stratum, with session-bootstrap intervals.
5. **G5** from the Holm-corrected H3 p-value, with per-stratum detectability.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.situation import (
    confirmatory_strata,
    g4_verdict,
    kappa_per_category,
)
from src.config import RidgeConfig
from src.experiments.gates import GateResult, g5, require
from src.experiments.layer1 import fit_j1
from src.experiments.pipeline import DIMS, AnalysisData
from src.experiments.preflight import require_preflight
from src.experiments.run_j1 import mixed_effects_frame
from src.experiments.runio import RunWriter
from src.settings import SettingNotSet, Settings, as_record, measurement_signature
from src.statistics.mixed_effects import aggregate_h3, interaction_model


def run_j4(data: AnalysisData, labels: pd.DataFrame, hand: pd.DataFrame | None, *,
           results_root: Path, data_kind: str, stage: str, h2_h4_p: dict,
           settings: Settings | None = None, ridge: RidgeConfig | None = None,
           preflight_report=None, gates: dict | None = None, n_boot: int = 1000,
           seed: int = 20260912, per_stratum_mde: dict | None = None) -> tuple[Path, dict]:
    """``labels``: pair_id, function. ``hand``: pair_id, hand_label (300 rows).

    ``per_stratum_mde`` maps each confirmatory stratum to its calibrated MDE
    (from a J3 run restricted to that stratum); without it a null H3 is not
    credible under G5.
    """
    require_preflight(preflight_report, data_kind)
    settings = settings or Settings()
    ridge = ridge or RidgeConfig()
    cfg = {"n_boot": n_boot, "seed": seed, "settings": as_record(settings)}
    with RunWriter(results_root, "j4", data_kind, stage, measurement_signature(settings),
                   data.split_hash, cfg, provenance=data.provenance()) as run:
        lab = data.pairs[["pair_id"]].merge(labels, on="pair_id", how="left")["function"]
        if (lab == "unlabelled_classifier_blocked").any() or lab.isna().any():
            out = {"status": "BLOCKED", "reason": "no text-only classifier is set; "
                   "only interrogative-rule labels exist",
                   "label_counts": lab.value_counts(dropna=False).to_dict()}
            run.json("j4_status", out)
            return run.dir, out
        if hand is None:
            raise SettingNotSet("G4 needs the 300 hand labels")
        merged = hand.merge(labels, on="pair_id", how="inner")
        kappas = kappa_per_category(merged["hand_label"], merged["function"])
        g4 = g4_verdict(kappas)
        run.json("gate_G4", g4)
        gates = {**(gates or {}),
                 "G4": GateResult("G4", g4["status"], g4["criterion"], "Prereg §10",
                                  {"kappas": kappas})}
        label = require(gates, "j4")
        strata = confirmatory_strata(lab, kappas)
        run.table("strata", strata)
        conf = strata.loc[strata["role"] == "confirmatory", "category"].tolist()

        df = mixed_effects_frame(data)
        df["function"] = lab.to_numpy()
        keep = df["function"].isin(conf).to_numpy()
        inter = {}
        if len(conf) >= 2:
            for d in DIMS:
                inter[d] = interaction_model(df[keep], f"y_{d}", [f"a_{x}" for x in DIMS],
                                             df.attrs["z_cols"])
            h3_p = aggregate_h3({d: (None if v["fit_failed"] else v["wald_interaction"]["p"])
                                 for d, v in inter.items()},
                                settings.h3_aggregation)
        else:
            h3_p = None
        run.json("interaction", {"output_label": label, "per_dimension": inter, "h3_p": h3_p})

        per_stratum = {}
        for c in conf:
            rows = (lab == c).to_numpy()
            r = fit_j1(data, ridge, settings, rows=rows, n_boot=n_boot, seed=seed)
            per_stratum[c] = {"pairs": int(rows.sum()),
                              **r["comparisons"]["partner_delivery_over_partner_text"]}
        run.json("per_stratum_delta", {"label": "descriptive companion",
                                       "strata": per_stratum})
        from src.statistics.permutation import holm

        family = {k: v for k, v in {**h2_h4_p, "H3": h3_p}.items() if v is not None}
        corrected = holm(family, alpha=float(settings.g3_rule["alpha"])) if family else {}
        run.json("holm_family", corrected)
        # Per-stratum detectability needs a J3 calibration restricted to each
        # stratum, which is not part of this run: G5 can then pass only
        # on a significant interaction, never on a null.
        power = {"per_stratum_mde": per_stratum_mde or {},
                 "pairs": {c: v["pairs"] for c, v in per_stratum.items()}}
        g5_res = g5(corrected.get("H3", {}).get("p_holm"),
                    float(settings.g3_rule["alpha"]), power, conf)
        run.json("gate_G5", g5_res.as_dict())
    return run.dir, {"G4": g4, "G5": g5_res, "holm": corrected,
                     "n_confirmatory_strata": len(conf),
                     "mean_stratum_delta": float(np.mean([v["aggregate_partial_r2"]
                                                          for v in per_stratum.values()]))
                     if per_stratum else None}


__all__ = ["run_j4"]
