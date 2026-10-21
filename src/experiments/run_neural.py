"""The neural arm driver. Corroborating evidence only.

Every job trains the same architecture with the same frozen configuration on
the same speaker-disjoint split; only the information reaching the partner
pathway differs. Jobs:

| Job | Partner pathway | Seeds |
|---|---|---|
| ablation 1 — ``no_partner_acoustic`` | learned null-token sequence | 3 |
| ablation 2 — ``ssl_sequence`` (the full pathway) | partner SSL frames | 3 |
| ablation 3 — ``interpretable_vector`` | 8 partner dimensions, length-1 sequence | 3 |
| ablation 4 — ``frozen_single_source`` | SSL frames, mixer frozen to the chosen source | 3 |
| donors N1, N2 | a donor turn's SSL frames | 30 draws × 1 seed |

The neural Δ is the block partial R² of a job over ablation 1 (the neural
equivalent of Z + partner text), on the test split, read once per job. Seed
spread is reported separately from donor-null spread. Neural p-values
never enter the confirmatory family and carry no confirmatory α.
Once T1 and T2 pass and the four ablations are run, the arm is finished.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.evaluation.metrics import block_partial_r2
from src.experiments.pipeline import AnalysisData
from src.experiments.preflight import require_preflight
from src.experiments.runio import RunExists, RunWriter
from src.models.neural_context_model import (
    Architecture,
    NeuralContextModel,
    assert_identical_capacity,
)
from src.models.ssl_layer_mixer import SOURCES
from src.settings import Settings, as_record, measurement_signature
from src.training.config import TrainConfig, assert_same_config
from src.training.trainer import ArrayData, predict, seed_everything, train


@dataclass(frozen=True)
class Job:
    name: str
    mode: str
    seed: int
    donor_scheme: str = "none"
    draw: int = -1
    freeze_source: str | None = None


class NeuralArmDeferred(RuntimeError):
    """Raised when the neural arm is deferred and real data is passed in."""


def require_neural_not_deferred(settings: Settings, data_kind: str) -> None:
    """While the neural arm is deferred, its code runs on synthetic data only.

    The code and its tests are kept so that a later, separately registered study
    can run them; deferral prevents any real-data neural result from being
    produced under this preregistration.
    """
    if settings.neural_arm == "deferred" and data_kind != "synthetic":
        raise NeuralArmDeferred("the neural arm is deferred; "
                                f"refusing to run on {data_kind!r} data")


def job_list(seeds=(0, 1, 2), n_draws: int = 30, ablation4_source: str = "wavlm_l6") -> list[Job]:
    jobs = []
    for s in seeds:
        jobs += [Job("no_partner_acoustic", "null_token", s),
                 Job("ssl_sequence", "real", s),
                 Job("interpretable_vector", "delivery_vector_interpretable", s),
                 Job("frozen_single_source", "real", s, freeze_source=ablation4_source)]
    for scheme in ("N1", "N2"):
        jobs += [Job(f"donor_{scheme}", "donor", seeds[0], scheme, k) for k in range(n_draws)]
    return jobs


def build_arrays(data: AnalysisData, frames: np.ndarray, lengths: np.ndarray) -> dict:
    """Model inputs, row-aligned with ``data.pairs``. Unseen speakers get index 0."""
    reps = data.pairs["replier"].to_numpy()
    train_spk = sorted(set(reps[data.train]))
    idx = {s: i + 1 for i, s in enumerate(train_spk)}
    meta = data.Z.metadata
    mu = meta[data.train].mean(0)
    sd = np.where(meta[data.train].std(0) > 0, meta[data.train].std(0), 1)
    return {"frames": frames, "lengths": np.maximum(lengths, 1).astype(np.int64),
            "partner_vector": data.A.astype(np.float32),
            "partner_text": data.Z.partner_text.astype(np.float32),
            "reply_text": data.Z.reply_text.astype(np.float32),
            "metadata": ((meta - mu) / sd).astype(np.float32),
            "speaker": np.array([idx.get(s, 0) for s in reps], dtype=np.int64),
            "y": data.Y.astype(np.float32)}, len(train_spk) + 1


def make_model(arch: Architecture, job: Job) -> NeuralContextModel:
    """A fresh model whose initial weights are fixed by the job's seed.

    Seeding happens here, before construction: seeding only inside the training
    loop would leave weight initialisation uncontrolled.
    """
    seed_everything(job.seed)
    m = NeuralContextModel(arch)
    if job.freeze_source is not None:
        m.mixer.freeze_to(SOURCES.index(job.freeze_source))
    return m


def run_neural(data: AnalysisData, frames: np.ndarray, lengths: np.ndarray,
               donor_frames: dict[str, callable] | None, *, results_root: Path,
               data_kind: str, stage: str, cfg: TrainConfig, jobs: list[Job],
               settings: Settings | None = None, preflight_report=None,
               arch_overrides: dict | None = None) -> dict:
    """Run (or resume) every job; returns block partial R² per job against ablation 1.

    ``donor_frames[scheme](draw)`` returns (frames, lengths) of that draw's donor
    turns, row-aligned with ``data.pairs``.
    """
    require_preflight(preflight_report, data_kind)
    settings = settings or Settings()
    require_neural_not_deferred(settings, data_kind)
    arrays, n_spk = build_arrays(data, frames, lengths)
    # Input widths come from the data (64 text components and 128 SSL components on
    # the real data); every registered size of the architecture stays fixed.
    arch = Architecture(n_speakers=n_spk, max_frames=frames.shape[1], ssl_dim=frames.shape[3],
                        text_dim=data.Z.reply_text.shape[1], n_metadata=data.Z.metadata.shape[1],
                        dropout=cfg.dropout, **(arch_overrides or {}))
    models = {j.name + str(j.seed): make_model(arch, j) for j in jobs[:4]}
    n_params = assert_identical_capacity(models)
    assert_same_config({j.name: cfg.hash() for j in jobs})
    tr, va, te = (np.flatnonzero(m) for m in (data.train, data.val, data.test))
    blocks = [b.tolist() for b in data.blocks]
    results = {"trainable_parameters_total": n_params, "config_hash": cfg.hash(), "jobs": {}}
    preds = {}
    for job in jobs:
        a = dict(arrays)
        if job.mode == "donor":
            if donor_frames is None:
                raise ValueError("donor jobs need donor frames")
            a["frames"], dl = donor_frames[job.donor_scheme](job.draw)
            a["lengths"] = np.maximum(dl, 1).astype(np.int64)
        conf = {"job": asdict(job), "train_config": asdict(cfg), "config_hash": cfg.hash(),
                "architecture": asdict(arch), "settings": as_record(settings)}
        try:
            with RunWriter(results_root, "neural", data_kind, stage,
                           measurement_signature(settings), data.split_hash, conf,
                           provenance={"label": "corroborating; no confirmatory α"},
                           resume=True) as run:
                model = make_model(arch, job)
                hist = train(model, ArrayData(a), tr, va, job.mode, cfg, job.seed, blocks,
                             checkpoint_dir=Path(results_root).parent / "checkpoints" /
                             run.dir.name)
                out = predict(model, ArrayData(a), te, job.mode,
                              ledger=run.dir / "test_reads.jsonl")
                run.json("history", hist)
                run.array("test_predictions", mean=out["mean"], log_var=out["log_var"],
                          pooled=out["pooled"])
                preds[job] = out["mean"]
        except RunExists:
            continue
    ref = {j.seed: p for j, p in preds.items() if j.name == "no_partner_acoustic"}
    y = data.Y[te]
    for job, p in preds.items():
        if job.name == "no_partner_acoustic" or job.seed not in ref:
            continue
        pr2 = block_partial_r2(y, p, ref[job.seed], data.blocks)
        results["jobs"][f"{job.name}|seed{job.seed}|draw{job.draw}"] = {
            "block_partial_r2": pr2.tolist(), "aggregate": float(pr2.mean())}
    return results


__all__ = ["Job", "job_list", "build_arrays", "make_model", "run_neural",
           "NeuralArmDeferred", "require_neural_not_deferred"]
