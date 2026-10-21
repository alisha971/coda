"""Sample scaling (pilot → 250) and a smoke run of the neural driver on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from src.data.sample import sample_audit, sample_manifest


def manifest(n_sessions=12, n_pilot=3, n_sample=6):
    rows = []
    for r in range(n_sessions):
        role = "pilot" if r < n_pilot else "sample" if r < n_sample else "reserve"
        for p in ("a", "b"):
            rows.append({"session_key": f"S{r:02d}", "selection_rank": r, "role": role,
                         "participant_key": f"P{r:02d}{p}"})
    return pd.DataFrame(rows)


def qc(sessions_failing=(), measured=None):
    rows = []
    for r in range(measured or 12):
        s = f"S{r:02d}"
        fail = "missing" if s in sessions_failing else None
        rows += [{"interaction_key": f"{s}_I1", "failure": fail},
                 {"interaction_key": f"{s}_I1", "failure": None}]
    return pd.DataFrame(rows)


def test_failures_are_replaced_in_rank_order_and_the_pilot_stays_first():
    m = manifest()
    sm = sample_manifest(m, qc(sessions_failing={"S01", "S04"}), target=6, n_pilot=3)
    acc = sm[sm["status"] == "accepted"]
    assert acc["session_key"].tolist() == ["S00", "S02", "S03", "S05", "S06", "S07"]
    assert acc.loc[acc["stage"] == "pilot", "session_key"].tolist() == ["S00", "S02", "S03"]
    a = sample_audit(sm, m, 6, 3)
    assert a["status"] == "complete" and a["reserve_used"] == 2 and a["excluded_for_qc"] == 2


def test_unmeasured_sessions_leave_the_sample_incomplete():
    m = manifest()
    sm = sample_manifest(m, qc(measured=4), target=6, n_pilot=3)
    assert sample_audit(sm, m, 6, 3)["status"].startswith("incomplete")


def test_a_reused_speaker_is_caught():
    m = manifest()
    m.loc[m["session_key"] == "S02", "participant_key"] = ["P00a", "P02b"]
    sm = sample_manifest(m, qc(), target=6, n_pilot=3)
    with pytest.raises(AssertionError, match="speaker"):
        sample_audit(sm, m, 6, 3)


@pytest.mark.integration
def test_neural_driver_runs_every_job_kind_with_identical_capacity(tmp_path):
    pytest.importorskip("torch")
    from src.experiments import dry_run as dr
    from src.experiments.run_neural import Job, run_neural
    from src.training.config import TrainConfig

    corpus = dr.make_corpus(14, coupling=0.5, seed=1)
    data, prosody, _ = dr.prepare(corpus, tmp_path / "cache")
    rng = np.random.default_rng(0)
    n = len(data.pairs)
    frames = rng.normal(size=(n, 12, 5, 128)).astype(np.float16)
    lengths = rng.integers(1, 13, n)

    def donor(draw):
        r = np.random.default_rng(100 + draw)
        return frames[r.permutation(n)], lengths[r.permutation(n)]

    jobs = [Job("no_partner_acoustic", "null_token", 0), Job("ssl_sequence", "real", 0),
            Job("interpretable_vector", "delivery_vector_interpretable", 0),
            Job("frozen_single_source", "real", 0, freeze_source="wavlm_l6"),
            Job("donor_N1", "donor", 0, "N1", 0)]
    res = run_neural(data, frames, lengths, {"N1": donor}, results_root=tmp_path / "results",
                     data_kind="synthetic", stage="smoke",
                     cfg=TrainConfig(max_epochs=1, batch_size=128), jobs=jobs)
    assert set(res["jobs"]) == {"ssl_sequence|seed0|draw-1", "interpretable_vector|seed0|draw-1",
                                "frozen_single_source|seed0|draw-1", "donor_N1|seed0|draw0"}
    assert res["trainable_parameters_total"] > 1_000_000


def test_deferred_neural_arm_refuses_real_data_and_allows_synthetic():
    """Neural arm deferred: the code stays testable on synthetic data only."""
    from src.experiments.run_neural import NeuralArmDeferred, require_neural_not_deferred
    from src.settings import Settings

    d = Settings()
    assert d.neural_arm == "deferred"
    require_neural_not_deferred(d, "synthetic")
    for kind in ("sample", "pilot"):
        with pytest.raises(NeuralArmDeferred):
            require_neural_not_deferred(d, kind)
    with pytest.raises(RuntimeError, match="deferred"):
        from src.representations.ssl_extract import load_extractor

        load_extractor({"revisions": {"wavlm": "x"}}, device="cpu")
