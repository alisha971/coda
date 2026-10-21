"""H4 identification (shared drift versus local coupling) and G2 sensitivity reporting.

Synthetic only. The drift test is a regression test for the identification
failure: a corpus with shared drift and no local coupling must not be read as
evidence of a local mechanism.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.experiments.gates import g2_sensitivity, g3
from src.settings import Settings


@pytest.fixture(scope="module")
def drift_corpus():
    from src.experiments import dry_run as dr

    # The design of the identification study (40 sessions x 80 turns); smaller
    # corpora leave too few test sessions to show the pattern.
    c = dr.make_corpus(40, 80, coupling=0.0, seed=100, drift_sd=1.0, drift_rho=0.95)
    data, prosody, _ = dr.prepare(c, Path(tempfile.mkdtemp()))
    return c, data, prosody


def test_exact_distance_donors_are_the_partner_at_that_distance(drift_corpus):
    from src.experiments.h4_identification import exact_distance_donors

    c, data, prosody = drift_corpus
    usable = set(prosody.loc[prosody["usable"].astype(bool), "turn_id"])
    donors, ok = exact_distance_donors(data, c.turns, usable, 4, 5, seed=1)
    t = c.turns.set_index("turn_id")
    for i in np.flatnonzero(ok)[:200]:
        row = data.pairs.iloc[i]
        k = t.at[row["partner_turn_id"], "turn_index"]
        for d in donors[i]:
            assert t.at[d, "speaker"] == row["partner_speaker"]
            assert t.at[d, "interaction_key"] == row["interaction_key"]
            assert abs(int(t.at[d, "turn_index"]) - int(k)) == 4
    with pytest.raises(ValueError):
        exact_distance_donors(data, c.turns, usable, 3, 5, seed=1)


@pytest.mark.integration
def test_shared_drift_without_local_coupling_is_not_read_as_a_local_mechanism(drift_corpus):
    """Regression: the registered N1/N2 ordering is produced by drift alone."""
    from src.config import RidgeConfig
    from src.controls.donors import draw_n1, draw_n2
    from src.experiments.h4_identification import increment_and_nulls

    c, data, prosody = drift_corpus
    usable = set(prosody.loc[prosody["usable"].astype(bool), "turn_id"])
    n1, _ = draw_n1(data.pairs, c.turns, 19, 1)
    n2 = draw_n2(data.pairs, c.turns, usable, 19, 2)
    rows = np.ones(len(data.pairs), bool)
    res = increment_and_nulls(data, data.Z.full(), rows, {"N1": n1, "N2": n2},
                              RidgeConfig(nested_cv_folds=3))
    assert res["N2"]["p"] <= 0.05 and res["N1"]["p"] <= 0.05   # the misleading ordering
    gate = g3(res["N1"]["p"], res["N2"]["p"], np.array([res["N1"]["null_median"]] * 19),
              np.array([res["N2"]["null_median"]] * 19), Settings())
    assert gate.evidence["pattern"] == "adjacent"
    assert gate.evidence["mechanism"] == "not identified"


def test_g2_sensitivity_reports_power_and_never_passes():
    levels = [{"target_partial_r2": t, "rejection": {"rate": r}}
              for t, r in ((0.0, 0.05), (0.0025, 0.3), (0.005, 0.7), (0.01, 0.95))]
    rep = g2_sensitivity(levels)
    assert rep["status"].startswith("SENSITIVITY ONLY")
    by = {r["candidate_partial_r2"]: r for r in rep["candidates"]}
    assert by[0.0025]["power"] == pytest.approx(0.3) and by[0.0025]["measured_on_grid"]
    assert not by[0.005]["detectable_at_power_level"] and by[0.01]["detectable_at_power_level"]
    assert "passed" not in rep


def test_power_projection_runs_and_detects_a_large_effect():
    from src.experiments.power_projection import project

    r = project(n_sessions=30, pairs_per_session=60, z_dense=20, replicates=4, draws=9,
                targets=(0.0, 0.05), folds=3)
    big = [x for x in r["levels"] if x["target_partial_r2"] == 0.05][0]
    assert big["rejection"]["rate"] >= 0.75
    assert "assumptions" in r


def test_calibration_grid_measures_the_g2_sensitivity_values():
    import yaml

    cfg = Path(__file__).resolve().parents[1] / "configs" / "j3_injection.yaml"
    grid = yaml.safe_load(cfg.read_text(encoding="utf-8"))["injection"]["beta_targets_partial_r2"]
    assert {0.0025, 0.005, 0.01} <= set(grid)
    assert min(grid) >= 0.002 and max(grid) <= 0.05        # inside the registered span


def test_injection_content_fit_can_include_partner_text(drift_corpus):
    """Sensitivity option: the full-Z content fit differs from Z_content and
    the registered default is unchanged."""
    from src.config import RidgeConfig
    from src.experiments.layer1 import content_fit_for_injection

    _, data, _ = drift_corpus
    cfg = RidgeConfig(nested_cv_folds=3)
    a = content_fit_for_injection(data, cfg)
    b = content_fit_for_injection(data, cfg, source="full")
    rows = a["rows"] & b["rows"]
    assert rows.any() and not np.allclose(a["fhat"][rows], b["fhat"][rows])
    with pytest.raises(ValueError):
        content_fit_for_injection(data, cfg, source="partner")


@pytest.mark.integration
def test_per_stratum_calibration_returns_an_mde_per_stratum(drift_corpus):
    """Each stratum gets its own power curve and MDE (or None if never reached)."""
    from src.config import RidgeConfig
    from src.controls.donors import draw_n1, draw_n2
    from src.experiments.gates import g5
    from src.experiments.layer1 import content_fit_for_injection, stratum_mde

    c, data, prosody = drift_corpus
    usable = set(prosody.loc[prosody["usable"].astype(bool), "turn_id"])
    donors = {"N1": draw_n1(data.pairs, c.turns, 9, 1)[0],
              "N2": draw_n2(data.pairs, c.turns, usable, 9, 2)}
    cfg = RidgeConfig(nested_cv_folds=3)
    inj = content_fit_for_injection(data, cfg)
    half = np.arange(len(data.pairs)) % 2 == 0           # two strata spanning every session
    strata = {"a": half, "b": ~half}
    out = stratum_mde(data, cfg, inj, strata, [0.02, 0.2], 4, donors, 0.05, 0.8, seed=3)
    assert set(out) == {"a", "b"}
    for v in out.values():
        assert len(v["power"]) == 2 and v["pairs"] > 0
        assert v["mde"] is None or 0 < v["mde"] <= 0.2
    gate = g5(0.5, 0.05, {"per_stratum_mde": {k: v["mde"] for k, v in out.items()}}, ["a", "b"])
    expected = "PASS" if all(v["mde"] is not None for v in out.values()) else "FAIL"
    assert gate.evidence["provisional_outcome"] == expected


def test_within_speaker_lag1_detects_self_persistence_and_not_its_absence():
    """D-AC: the diagnostic reads serial dependence of a speaker's own delivery."""
    from src.analysis.serial_dependence import within_speaker_lag1
    from src.experiments import dry_run as dr

    res = {}
    for ar in (0.0, 0.6):
        c = dr.make_corpus(30, 80, coupling=0.0, seed=7, self_ar=ar)
        data, _, _ = dr.prepare(c, Path(tempfile.mkdtemp()))
        res[ar] = within_speaker_lag1(data)
    assert len(res[0.0]) == 8 and (res[0.0]["n_pairs"] > 0).all()
    r = {k: v["r_residual_on_Z_content"].mean() for k, v in res.items()}
    assert r[0.6] > r[0.0] + 0.1
