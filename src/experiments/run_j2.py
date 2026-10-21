"""J2 — the null family: N1 and N2 donor refits, the primary test, H4, G3.

Draws donors (seeded, saved, validated), refits ``plus_partner_delivery`` with
each draw in place of A, and compares the observed increment with both null
distributions. Also computes H2 requirement 5 (leading coupling strength
against the N1 refits) and the Holm family.

Outputs: ``donors_N1.npz``, ``donors_N2.npz`` (exact reproducibility),
``null_draws.npz``, ``j2_tests.json`` (H1 vs N1, H4 vs N2, per-block p-values,
H2 null comparison), ``gate_G3.json``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analysis.coupling_map import h2_verdict, stability_null
from src.config import DataConfig, RidgeConfig
from src.controls.donors import assert_donor_validity, draw_n1, draw_n2, save_donors
from src.experiments.gates import g3, require
from src.experiments.layer1 import caches, donor_null, h1_h4_tests
from src.experiments.pipeline import AnalysisData
from src.experiments.preflight import require_preflight
from src.experiments.runio import RunWriter, read_json
from src.models.ridge import RidgeEngine
from src.models.rrr import fit_rrr, leading_strength
from src.settings import Settings, as_record, measurement_signature
from src.statistics.permutation import donor_p_value
from src.targets.normalise import standardise_donors


def draw_donors(data: AnalysisData, turns, prosody, n_draws: int, seed: int) -> dict:
    usable = set(prosody.loc[prosody["usable"].astype(bool), "turn_id"])
    n1, n1_report = draw_n1(data.pairs, turns, n_draws, seed)
    n2 = draw_n2(data.pairs, turns, usable, n_draws, seed + 1,
                 DataConfig().n2_min_turn_distance)
    assert_donor_validity(data.pairs, n1, "N1_cross_dyad", turns)
    assert_donor_validity(data.pairs, n2, "N2_within_dyad_distant", turns,
                          DataConfig().n2_min_turn_distance)
    return {"N1": n1, "N2": n2, "n1_report": n1_report}


def run_j2(data: AnalysisData, turns, prosody, *, results_root: Path, data_kind: str,
           stage: str, n_draws: int, j1_dir: Path, settings: Settings | None = None,
           ridge: RidgeConfig | None = None, preflight_report=None, gates: dict | None = None,
           seed: int = 20260912) -> tuple[Path, dict]:
    """``j1_dir`` is the completed J1 run whose observed increment is tested.

    The observed statistic and the reduced model's test predictions are read
    from J1's stored outputs, so the test split is not evaluated a second time
    for the J1 conditions (the test split is touched once per condition).
    """
    require_preflight(preflight_report, data_kind)
    settings = settings or Settings()
    ridge = ridge or RidgeConfig()
    label = require(gates or {}, "j2")
    cfg = {"ridge": ridge.__dict__, "n_draws": n_draws, "seed": seed,
           "settings": as_record(settings), "output_label": label}
    with RunWriter(results_root, "j2", data_kind, stage, measurement_signature(settings),
                   data.split_hash, cfg, provenance=data.provenance()) as run:
        d = draw_donors(data, turns, prosody, n_draws, seed)
        for scheme in ("N1", "N2"):
            h = save_donors(run.dir / f"donors_{scheme}.npz", d[scheme], data.pairs["pair_id"],
                            scheme, seed)
            run.json(f"donors_{scheme}_record", {"content_hash": h, "draws": n_draws})
        run.table("n1_matching_report", d["n1_report"])
        j1_summary = read_json(Path(j1_dir) / "j1_summary.json")
        observed = j1_summary["comparisons"]["partner_delivery_over_partner_text"]
        red = np.load(Path(j1_dir) / "j1_errors.npz")["pred_plus_partner_text"].astype(np.float64)
        eng = RidgeEngine(caches(data, ridge)["full"], data.Y, data.blocks, ridge.lambda_grid)
        if red.shape != (len(eng.zc.eval_idx), data.Y.shape[1]):
            raise ValueError("the J1 run does not match this analysis set")
        run.json("j1_source", {"j1_run": Path(j1_dir).name})
        nulls = {s: donor_null(data, eng, red, d[s]) for s in ("N1", "N2")}
        run.array("null_draws", **{f"{s}_{k}": v for s, nl in nulls.items()
                                   for k, v in nl.items()})
        tests = h1_h4_tests(observed, nulls["N1"], nulls["N2"])
        ref = fit_rrr(eng, data.A, data.blocks, ranks=ridge.rank_grid)
        h2_null = [leading_strength(eng, standardise_donors(data.pairs, d["N1"][:, [k]],
                                                            data.standardiser)[:, 0, :],
                                    ref.lam_rel) for k in range(n_draws)]
        tests["H2_leading_strength_vs_N1"] = donor_p_value(
            leading_strength(eng, data.A, ref.lam_rel), np.asarray(h2_null))
        # H2 verdict: stability statistics against the same statistics refitted
        # with N1 donor partner matrices (a seeded subset).
        battery = read_json(Path(j1_dir) / "coupling_map.json")
        battery["null"] = tests["H2_leading_strength_vs_N1"]
        fold = np.full(len(data.pairs), -1)
        fold[eng.zc.fit_idx] = eng.zc.fold
        h2 = settings.h2_stability
        k_use, h2_boot = min(int(h2["null_draws"]), n_draws), int(h2["n_boot"])
        donor_A = [standardise_donors(data.pairs, d["N1"][:, [k]], data.standardiser)[:, 0, :]
                   for k in range(k_use)]
        h2_stats = stability_null(data.Z.full(), donor_A, data.Y, data.groups, data.train,
                                  data.blocks, battery["lambda_rel"], battery["rank"], fold,
                                  h2_boot, seed + 7)
        run.json("h2_stability_null", {"draws": k_use, "n_boot": h2_boot, "stats": h2_stats})
        tests["H2_verdict"] = h2_verdict(battery, h2_stats, 0.05)
        tests["observed"] = observed
        tests["output_label"] = label
        run.json("j2_tests", tests)
        gate = g3(tests["H1_vs_N1"]["p"], tests["H4_vs_N2"]["p"], nulls["N1"]["aggregate"],
                  nulls["N2"]["aggregate"], settings)
        run.json("gate_G3", gate.as_dict())
    return run.dir, {"G3": gate, "tests": tests}


__all__ = ["run_j2", "draw_donors"]
