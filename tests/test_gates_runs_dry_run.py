"""Gates, the stop rule, run directories, preflight, and the end-to-end dry run (Gate G).

The dry run is a software check on synthetic data. Passing it says the pipeline
is connected and recovers a planted effect; it says nothing about conversation.
"""

import json
from dataclasses import replace

import numpy as np
import pytest

from src.experiments.gates import (
    GateBlocked,
    GateResult,
    g1,
    g2,
    g3,
    require,
    stop_rule,
)
from src.experiments.preflight import PreflightFailed, preflight, require_preflight
from src.experiments.runio import RunExists, RunWriter, find_runs, read_json
from src.settings import Settings


def settings(**over):
    """Settings with the original binomial G1 rule and no G2 effect range."""
    base = dict(g1_rule={"alpha": 0.05, "replicates_beta0": 200,
                         "acceptance": "rejection_rate_within_binomial_95ci_of_alpha"},
                g2_plausible_range=None)
    return replace(Settings(), **{**base, **over})


def rate(r, lo, hi):
    return {"rate": r, "ci95": [lo, hi]}


# -- gates ------------------------------------------------------------------


def test_g1_is_provisional_under_development_settings():
    rates = {"N1": rate(0.05, 0.02, 0.09), "N2": rate(0.06, 0.03, 0.1)}
    res = g1(rates, settings().development())
    assert res.status == "PROVISIONAL" and res.evidence["provisional_outcome"] == "PASS"
    d = settings()
    assert g1(rates, d).status == "PASS"
    assert g1({"N1": rate(0.2, 0.15, 0.25), "N2": rate(0.06, 0.03, 0.1)}, d).status == "FAIL"


def test_g2_is_blocked_without_a_sourced_effect_range():
    res = g2({"aggregate": 0.01}, settings())
    assert res.status == "BLOCKED" and "g2_plausible_range" in res.settings


def test_g2_uses_the_effect_range_bound_once_supplied():
    d = settings(g2_plausible_range=0.02)
    assert g2({"aggregate": 0.01}, d).status == "PASS"
    assert g2({"aggregate": 0.03}, d).status == "FAIL"


@pytest.mark.parametrize("p1,p2,shift,pattern", [
    (0.001, 0.001, 1.0, "adjacent"),
    (0.001, 0.4, 1.0, "distant"),
    (0.5, 0.6, 1.0, "none"),
    (0.001, 0.001, 0.0, "adjacent"),      # the pattern shared drift produces
    (0.5, 0.01, 0.0, "inconsistent"),
])
def test_g3_reports_patterns_and_never_a_mechanism(p1, p2, shift, pattern):
    """H4 identification: G3 never says "local adaptation"."""
    rng = np.random.default_rng(0)
    n1 = rng.normal(0, 0.01, 500)
    n2 = rng.normal(shift * 0.05, 0.01, 500)
    ev = g3(p1, p2, n1, n2, settings()).evidence
    assert ev["pattern"] == pattern
    assert ev["mechanism"] == "not identified"
    assert "local adaptation" not in ev["reading"].replace(
        "Local adaptation and shared dependence", "")


def test_g1_failure_stops_everything_downstream():
    gates = {"G1": GateResult("G1", "FAIL", "", "")}
    for step in ("j1", "j2", "h2", "j4", "neural"):
        with pytest.raises(GateBlocked):
            require(gates, step)


def test_j1_is_confirmatory_only_when_g1_and_g2_pass():
    ok = {"G1": GateResult("G1", "PASS", "", ""), "G2": GateResult("G2", "PASS", "", "")}
    assert require(ok, "j1") == "confirmatory"
    assert require({**ok, "G2": GateResult("G2", "BLOCKED", "", "")}, "j1").startswith(
        "not_confirmatory")
    assert require({}, "j1").startswith("not_confirmatory")


def test_neural_gates_never_block_layer_one():
    gates = {"G1": GateResult("G1", "PASS", "", ""), "G2": GateResult("G2", "PASS", "", ""),
             "T1": GateResult("T1", "FAIL", "", ""), "T2": GateResult("T2", "FAIL", "", "")}
    assert require(gates, "j1") == "confirmatory"
    assert require(gates, "neural").startswith("corroborating")


def test_stop_rule():
    s = stop_rule({"delta_n1": True, "delta_n2": True, "mde_stated": True, "g4": "PASS",
                   "g5": "PASS", "g1": "PASS", "g2": "PASS"})
    assert s["complete"] and s["claims_licensed"]
    s = stop_rule({"delta_n1": True, "delta_n2": True, "mde_stated": False, "g5": "PASS"})
    assert not s["complete"]
    s = stop_rule({"delta_n1": True, "delta_n2": True, "mde_stated": True, "g4": "FAIL",
                   "g4_no_category": True, "g1": "PASS", "g2": "BLOCKED"})
    assert s["complete"] and "Level D" in s["part_ii_how"] and not s["claims_licensed"]
    assert not stop_rule({"t1": "PASS", "t2": "PASS"})["neural_arm_finished"]


# -- run directories --------------------------------------------------------


def test_run_directories_are_labelled_and_never_overwritten(tmp_path):
    kw = dict(results_root=tmp_path, experiment="j1", data_kind="synthetic", stage="t",
              measurement_signature="abc", split_hash="def", config={"x": 1})
    with RunWriter(**kw) as run:
        run.json("summary", {"v": 1.5})
        with pytest.raises(RunExists):
            run.json("summary", {"v": 2})
    assert run.dir.name.startswith("synthetic__t__mabc__sdef__")
    assert read_json(run.dir / "summary.json") == {"v": 1.5}
    assert json.loads((run.dir / "summary.json").read_text())["data_kind"] == "synthetic"
    with pytest.raises(RunExists, match="never overwritten"):
        RunWriter(**kw)
    other = RunWriter(**{**kw, "measurement_signature": "zzz"})   # another variant: new dir
    assert other.dir != run.dir
    assert len(find_runs(tmp_path, "j1", "synthetic")) == 1


def test_a_failed_run_records_its_exception_and_resumes_only_on_request(tmp_path):
    kw = dict(results_root=tmp_path, experiment="j2", data_kind="pilot", stage="t",
              measurement_signature="a", split_hash="b", config={})
    with pytest.raises(ZeroDivisionError):
        with RunWriter(**kw) as run:
            raise ZeroDivisionError("simulated failure inside a run")
    st = json.loads((run.dir / "status.json").read_text())
    assert st["status"] == "failed" and "ZeroDivisionError" in st["traceback"]
    with pytest.raises(RunExists):
        RunWriter(**kw)
    with RunWriter(**kw, resume=True):
        pass
    assert json.loads((run.dir / "status.json").read_text())["attempt"] == 2


def test_unknown_data_kind_is_refused(tmp_path):
    with pytest.raises(ValueError):
        RunWriter(tmp_path, "j1", "real", "t", "a", "b", {})


# -- preflight --------------------------------------------------------------


def test_real_data_runs_need_a_passing_preflight(tmp_path):
    with pytest.raises(PreflightFailed):
        require_preflight(None, "pilot")
    require_preflight(None, "synthetic")
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    # without a split manifest the preflight refuses
    rep = preflight("j1_j2", "sample", root)
    failed = {c["check"] for c in rep.checks if c["critical"] and not c["ok"]}
    assert not rep.ok and "split manifest valid" in failed
    with pytest.raises(PreflightFailed):
        require_preflight(rep, "sample")


# -- the end-to-end dry run -------------------------------------------------


@pytest.mark.integration
def test_dry_run_recovers_a_planted_coupling_and_finds_none_without_it(tmp_path):
    from src.experiments.dry_run import run

    planted = run(tmp_path / "on", n_sessions=24, coupling=0.6, n_draws=9)
    assert planted["observed_aggregate_partial_r2"] > 0.03
    assert planted["p_vs_N1"] <= 1 / 9
    assert planted["G2"] == "NOT_EVALUATED"          # reporting only; never PASS
    assert not planted["stop_rule"]["claims_licensed"]   # development settings: gates provisional
    null = run(tmp_path / "off", n_sessions=24, coupling=0.0, n_draws=9)
    # Without coupling the increment is near zero. Its p-value is not asserted:
    # with 9 draws a valid null gives p = 0 one time in ten. Calibration of the
    # p-value is checked across replicates by the β = 0 machinery (G1), not here.
    assert abs(null["observed_aggregate_partial_r2"]) < 0.02


def test_j2_reuses_j1_outputs_instead_of_reading_the_test_split_again():
    """The test set is touched once per condition."""
    import inspect

    from src.experiments import run_j2

    src = inspect.getsource(run_j2.run_j2)
    assert "fit_j1(" not in src
    assert "j1_dir" in inspect.signature(run_j2.run_j2).parameters


def test_g5_null_is_credible_only_with_a_per_stratum_mde_for_every_stratum():
    """Whole-sample power or a partial per-stratum set does not make a null credible."""
    from src.experiments.gates import g5

    strata = ["question", "backchannel"]
    assert g5(0.40, 0.05, None, strata).evidence["provisional_outcome"] == "FAIL"
    part = {"per_stratum_mde": {"question": 0.004}}
    assert g5(0.40, 0.05, part, strata).evidence["provisional_outcome"] == "FAIL"
    full = {"per_stratum_mde": {"question": 0.004, "backchannel": 0.006}}
    assert g5(0.40, 0.05, full, strata).evidence["provisional_outcome"] == "PASS"
    assert g5(0.01, 0.05, None, strata).evidence["provisional_outcome"] == "PASS"
    assert g5(None, 0.05, None, strata).status == "NOT_RUN"


@pytest.mark.integration
def test_j3_resumes_after_an_interruption_with_identical_levels(tmp_path, monkeypatch):
    """A J3 run interrupted mid-way continues in the same directory, skips finished chunks,
    and reproduces an uninterrupted, unchunked run exactly (chunk size changes nothing)."""
    from src.config import RidgeConfig
    from src.experiments import dry_run as dr
    from src.experiments import run_j3 as rj

    c = dr.make_corpus(30, coupling=0.3, seed=4)
    data, prosody, _ = dr.prepare(c, tmp_path / "cache")
    kw = dict(data_kind="synthetic", stage="resume", n_draws=9, targets=[0.02, 0.05],
              ridge=RidgeConfig(nested_cv_folds=3), couplings=("identity_matched_dimensions",),
              replicates=3, replicates_beta0=4, seed=1)
    ref_dir, _ = rj.run_j3(data, c.turns, prosody, results_root=tmp_path / "a",
                           chunk_replicates=2, **kw)
    ref = json.loads((ref_dir / "j3_levels.json").read_text())["content"]
    one_dir, _ = rj.run_j3(data, c.turns, prosody, results_root=tmp_path / "u",
                           chunk_replicates=100, **kw)
    one = json.loads((one_dir / "j3_levels.json").read_text())["content"]
    # Chunk size changes no p-value or decision; continuous statistics agree to rounding
    # (BLAS summation order depends on matrix width).
    for x, y in zip(ref, one, strict=True):
        for s_ in ("N1", "N2"):
            assert x[s_]["p_aggregate"] == y[s_]["p_aggregate"]
            assert x[s_]["p_block"] == y[s_]["p_block"]
            assert x[s_]["rejection_aggregate"] == y[s_]["rejection_aggregate"]
        np.testing.assert_allclose(x["observed_block"], y["observed_block"], rtol=1e-10)

    real, calls = rj.j3_level, []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt("simulated disconnect")
        return real(*a, **k)

    monkeypatch.setattr(rj, "j3_level", flaky)
    with pytest.raises(KeyboardInterrupt):
        rj.run_j3(data, c.turns, prosody, results_root=tmp_path / "b", chunk_replicates=2,
                  **kw)
    monkeypatch.setattr(rj, "j3_level", real)
    done = sorted(p.name for p in (tmp_path / "b").rglob("j3_level_*.json"))
    assert done == ["j3_level_00_c000.json", "j3_level_00_c001.json"]
    # a checkpoint truncated by a dying runtime is recomputed, not trusted
    bad = next((tmp_path / "b").rglob("j3_level_00_c001.json"))
    bad.write_text(bad.read_text()[:40], encoding="utf-8")
    out_dir, _ = rj.run_j3(data, c.turns, prosody, results_root=tmp_path / "b", resume=True,
                           chunk_replicates=2, **kw)
    got = json.loads((out_dir / "j3_levels.json").read_text())["content"]
    assert list(out_dir.glob("j3_level_00_c001.json.unreadable.*"))   # kept as evidence
    for lv in got + ref:
        lv.pop("chunks")
    assert got == ref


def test_mc_test_size_is_exact_and_bradley_g1_cutoffs_match_the_validation():
    """G1 validation: size (floor(aB)+1)/(B+1); with the
    Bradley rule a scheme passes at <= 61/1000 and <= 130/2000 rejections."""
    from src.experiments.gates import g1, mc_test_size

    assert mc_test_size(0.05, 500) == pytest.approx(26 / 501)
    assert mc_test_size(0.05, 499) == pytest.approx(0.05)
    assert mc_test_size(0.05, 39) == pytest.approx(0.05)
    d = replace(Settings(), confirmatory=False)       # the Bradley rule is the default

    def rate(k, n):
        return {"rejections": k, "replicates": n, "rate": k / n, "ci95": [0, 1]}

    for n, kmax in ((1000, 61), (2000, 130)):
        ok = g1({"N1": rate(kmax, n), "N2": rate(10, n)}, d, n_draws=500)
        bad = g1({"N1": rate(kmax + 1, n), "N2": rate(10, n)}, d, n_draws=500)
        assert ok.evidence["provisional_outcome"] == "PASS"
        assert bad.evidence["provisional_outcome"] == "FAIL"
    assert ok.evidence["mc_test_exact_size"] == pytest.approx(26 / 501)
    # a conservative rate is not a failure
    low = g1({"N1": rate(5, 2000), "N2": rate(3, 2000)}, d)
    assert low.evidence["provisional_outcome"] == "PASS"


def test_g2_reporting_only_never_passes_and_licenses_only_with_g1_and_a_stated_mde():
    """The registered G2 rule: G2 is NOT_EVALUATED, never PASS; J1 is confirmatory only with G1 PASS
    and the registered reporting-only rule with the MDE stated."""
    from src.experiments.gates import GateResult, g2, require, stop_rule

    gate = g2({"aggregate": 0.0016}, Settings())
    assert gate.status == "NOT_EVALUATED"
    g1_pass = GateResult("G1", "PASS", "", "", {})
    assert require({"G1": g1_pass, "G2": gate}, "j1") == "confirmatory"
    assert require({"G1": GateResult("G1", "PROVISIONAL", "", "", {}), "G2": gate},
                   "j1").startswith("not_confirmatory")
    dev = Settings().development()
    assert require({"G1": g1_pass, "G2": g2({"aggregate": 0.0016}, dev)},
                   "j1").startswith("not_confirmatory")
    s = stop_rule({"g1": "PASS", "g2": "NOT_EVALUATED", "mde_stated": True})
    assert s["claims_licensed"] and "size matters" in s["claims_note"]
    assert not stop_rule({"g1": "PASS", "g2": "NOT_EVALUATED", "mde_stated": False})[
        "claims_licensed"]
