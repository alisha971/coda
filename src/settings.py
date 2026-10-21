"""Method settings of the study, in one place.

Every implementation choice that changes a result and is not fixed by a module's
own signature is a field of :class:`Settings`. The defaults are the values used
for all reported results; the preregistration lists them. Modules take a
``settings`` argument (default ``Settings()``) and read these fields only.

``confirmatory=False`` (see :meth:`Settings.development`) makes the gates report
``PROVISIONAL``. Synthetic dry runs use it: they exercise the mechanics with tiny
replicate counts, so their gates must not issue verdicts.

:meth:`Settings.signature` hashes the values and is written into every run
record; :func:`measurement_signature` hashes only the fields that change a
measured quantity and is part of every output name, so results computed under
different definitions can never overwrite each other or be mixed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields, replace
from typing import Any


class SettingNotSet(RuntimeError):
    """A step needs a setting that has no value."""


def _d(**kw):
    return field(default_factory=lambda: dict(kw))


@dataclass(frozen=True)
class Settings:
    # -- measurement of the eight delivery dimensions ----------------------
    #: all | exclude_overlap | exclude_overlap_energy
    overlap_handling: str = "exclude_overlap_energy"
    #: fixed (75-500 Hz, as stored) | speaker_adaptive
    pitch_range: str = "fixed"
    #: rule_based | dictionary
    syllable_method: str = "dictionary"
    turn_measurement: dict = _d(intensity_averaging="energy", intensity_range="p95_p5",
                                boundary_min_voiced_frames=10)
    min_turns_for_speaker_stats: int = 10

    # -- estimation and inference ------------------------------------------
    variance_model: dict = _d(target="squared_out_of_fold_residuals",
                              floor_fraction_of_mean_squared_residual=0.01)
    bootstrap: dict = _d(resamples=2000, unit="session", interval="bootstrap_t",
                         simulated_worst_case_coverage=0.934, levels=[0.68, 0.95])
    h2_stability: dict = _d(rule="donor_calibrated_subspace_stability", null_draws=19,
                            n_boot=50)
    near_duplicate_rule: dict = _d(shingle_words=5, jaccard_threshold=0.5)
    #: drop | keep_as_second_random_partition
    dyad_disjoint_contrast: str = "drop"
    #: holm_min_over_dimensions
    h3_aggregation: str = "holm_min_over_dimensions"

    # -- calibration and gates ---------------------------------------------
    injection: dict = _d(
        replicates_per_beta=50, residual_resampling="session_block_ordered",
        chunk_replicates=250,
        cross_variant_permutation={
            "mean_log_f0": "articulation_rate",
            "log_f0_range_p95_p5": "intensity_range",
            "log_f0_slope_final_300ms": "pause_ratio",
            "mean_intensity": "mean_log_f0",
            "intensity_range": "longest_internal_pause",
            "articulation_rate": "log_f0_range_p95_p5",
            "pause_ratio": "log_f0_slope_final_300ms",
            "longest_internal_pause": "mean_intensity"})
    g1_rule: dict = _d(alpha=0.05, replicates_beta0=2000,
                       acceptance="bradley_liberal_upper_bound")
    mde_power: float = 0.80
    #: "reporting_only" (MDE and power stated, never a pass), a number (an MDE bound), or None
    g2_plausible_range: Any = "reporting_only"
    g3_rule: dict = _d(alpha=0.05, n2_vs_n1="n2_null_median_above_n1_null_q95")

    # -- neural arm ---------------------------------------------------------
    #: deferred | registered. While deferred the arm runs on synthetic data only.
    neural_arm: str = "deferred"
    #: revisions must be pinned (set to {"wavlm": ..., "hubert": ...}) before extraction
    ssl_stack: dict = _d(wavlm="microsoft/wavlm-base-plus", hubert="facebook/hubert-base-ls960",
                         revisions=None, sample_rate_hz=16000, pca_fit_max_turns=20000,
                         shard_turns=500)
    neural_training: dict = _d(
        optimiser="adamw", learning_rate=3.0e-4, weight_decay=0.01, dropout=0.1,
        grad_clip_norm=1.0, batch_size=64, max_epochs=40, patience=5,
        speaker_embedding_dim=16, speaker_dropout=0.2,
        search_grid={"learning_rate": [1.0e-4, 3.0e-4, 1.0e-3], "dropout": [0.1, 0.2]})
    neural_gate_rules: dict = _d(t1_recovery="p_vs_30_draw_n1_at_most_alpha", alpha=0.05,
                                 t2_replicates_beta0=3)
    ablation4_source: str = "wavlm_l6"

    # -- situation analysis -------------------------------------------------
    hand_label_protocol: dict = _d(n_pairs=300, seed=20260912, labeller="study_author",
                                   blind_to_classifier=True)

    # -- reporting mode ----------------------------------------------------
    confirmatory: bool = True

    def development(self) -> Settings:
        """The same values with ``confirmatory=False`` (gates report PROVISIONAL)."""
        return replace(self, confirmatory=False)

    def signature(self) -> str:
        """A short hash of every setting value (not of ``confirmatory``)."""
        return _digest({f.name: getattr(self, f.name) for f in fields(self)
                        if f.name != "confirmatory"})


#: Fields whose value changes a measured quantity.
MEASUREMENT_FIELDS = ("overlap_handling", "pitch_range", "syllable_method",
                      "turn_measurement", "min_turns_for_speaker_stats")


def _digest(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:10]


def measurement_signature(settings: Settings | None = None) -> str:
    s = settings or Settings()
    return _digest({k: getattr(s, k) for k in MEASUREMENT_FIELDS})


def as_record(settings: Settings | None = None) -> dict:
    """The values in force, for a run record."""
    s = settings or Settings()
    return {f.name: getattr(s, f.name) for f in fields(s)}


__all__ = ["Settings", "SettingNotSet", "MEASUREMENT_FIELDS", "measurement_signature",
           "as_record"]
