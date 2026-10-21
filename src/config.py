"""Configuration objects and run-identifier hashing.

Every fit in CODA is keyed by a hash of its configuration. Results are written
under that identifier and never overwritten, which is what lets the leakage
register assert that the configuration hash is identical across conditions
wherever the capacity-control protocol requires it, and what makes every
reported number traceable to the run that produced it.

Two properties matter and are tested:

1. Identical configurations hash identically, regardless of key order or of
   which runtime produced them. Otherwise a Colab run and a local run of the
   same fit would look like different runs.
2. Configurations differing in any field hash differently. Otherwise two
   conditions could silently collide and overwrite each other's results.

Fields that describe *where* a run happened -- runtime, host, device -- are
recorded in the run record but excluded from the hash, so that the same fit
reproduced on another machine resolves to the same identifier.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

# --------------------------------------------------------------------------
# Vocabulary of the study
# --------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Target vector size: EIGHT dimensions in four blocks (2 + 1 + 2 + 3), as in
# PREREGISTRATION.md section 4.
#
# Confirmatory inference is at block level -- four tests, equal weight per
# block -- so the count affects the delivery head width, the probe width and
# within-block orthogonalisation, and nothing else.
# ---------------------------------------------------------------------------

#: The target delivery dimensions, grouped into the four blocks that carry
#: confirmatory inference. Four tests, not one per dimension.
TARGET_BLOCKS: dict[str, tuple[str, ...]] = {
    "pitch_level_span": ("mean_log_f0", "log_f0_range_p95_p5"),
    "pitch_boundary": ("log_f0_slope_final_300ms",),
    "energy": ("mean_intensity", "intensity_range"),
    "timing": ("articulation_rate", "pause_ratio", "longest_internal_pause"),
}

TARGET_DIMENSIONS: tuple[str, ...] = tuple(
    dim for dims in TARGET_BLOCKS.values() for dim in dims
)

#: Eight. Asserted rather than assumed, so a later edit to TARGET_BLOCKS that
#: changed the target vector could not pass unnoticed.
N_TARGET_DIMENSIONS = len(TARGET_DIMENSIONS)

#: The four information conditions of the J1 condition sweep.
Condition = Literal[
    "Z_content",
    "plus_partner_text",
    "plus_partner_delivery",
    "plus_self_previous",
]

#: Donor schemes. N1 destroys the pairing; N2 destroys only local adjacency.
DonorScheme = Literal["none", "N1_cross_dyad", "N2_within_dyad_distant"]

#: Evaluation splits. Speaker-disjoint is primary; dyad-disjoint is a contrast.
SplitScheme = Literal["speaker_disjoint", "dyad_disjoint"]

#: How the partner-acoustic pathway is fed. Parameter shapes never change
#: between these; only the information available does.
PartnerMode = Literal["real", "donor", "null_token", "delivery_vector_interpretable"]


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DataConfig:
    """Corpus, sampling and split settings."""

    corpus: str = "seamless_naturalistic"
    n_conversations: int = 250
    pilot_n_conversations: int = 40
    selection_seed: int = 20260912
    split_scheme: SplitScheme = "speaker_disjoint"
    train_frac: float = 0.70
    val_frac: float = 0.10
    test_frac: float = 0.20
    split_seed: int = 20260912
    #: Minimum turn distance for an N2 donor (PREREGISTRATION.md section 6).
    n2_min_turn_distance: int = 10


@dataclass(frozen=True)
class FeatureConfig:
    """Target extraction and conditioning-set settings."""

    #: Praat via parselmouth, matching the feature set's source work.
    f0_backend: str = "parselmouth"
    f0_floor_hz: float = 75.0
    f0_ceiling_hz: float = 500.0
    frame_step_s: float = 0.01
    boundary_window_s: float = 0.300
    min_voiced_frames: int = 10
    #: Praat autocorrelation pitch settings, pinned explicitly.
    max_candidates: int = 15
    very_accurate: bool = False
    silence_threshold: float = 0.03
    voicing_threshold: float = 0.45
    octave_cost: float = 0.01
    octave_jump_cost: float = 0.35
    voiced_unvoiced_cost: float = 0.14
    intensity_subtract_mean: bool = True
    #: Where pitch is analysed. "vad" analyses the
    #: released voice-activity segments only; "vad_plus_transcript" adds the
    #: regions the transcript places speech in, which the detector can miss.
    pitch_segment_source: str = "vad_plus_transcript"
    pitch_segment_merge_gap_s: float = 0.050
    text_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_pca_dim: int = 64
    ssl_pca_dim: int = 128
    ssl_max_frames: int = 400
    ssl_tail_seconds: float = 8.0


@dataclass(frozen=True)
class RidgeConfig:
    """Layer 1, the statistical instrument. Deterministic: no seeds."""

    #: λ as a multiple of the number of fitting rows. Predictor columns
    #: are standardised, so the eigenvalues of XᵀX scale with n; a relative grid
    #: means the same shrinkage on the pilot and on the full sample. A selected
    #: λ at either end of the grid is flagged in every result.
    lambda_grid: tuple[float, ...] = (
        1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0,
    )
    #: Reduced-rank ridge ranks. The partner→reply coefficient block is 8 × 8,
    #: so its rank is at most 8 (an earlier grid ran to 9, a remnant of the
    #: nine-dimension count corrected in PREREGISTRATION §4).
    rank_grid: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
    nested_cv_folds: int = 5
    n_donor_draws: int = 500
    n_bootstrap: int = 2000


@dataclass(frozen=True)
class NeuralConfig:
    """Layer 2. Frozen before J1 and reused unchanged in every condition.

    The capacity-control protocol requires these to be identical across the
    real condition, every donor condition and every ablation. Only
    ``partner_mode`` and the donor assignment may differ, and neither changes
    a parameter shape.
    """

    d_model: int = 192
    n_blocks: int = 3
    n_heads: int = 4
    ffn_dim: int = 384
    ssl_sources: tuple[str, ...] = (
        "wavlm_l3", "wavlm_l6", "wavlm_l9", "wavlm_l12", "hubert_final",
    )
    n_donor_draws: int = 30
    seeds: tuple[int, ...] = (0, 1, 2)


@dataclass(frozen=True)
class RunConfig:
    """A complete, hashable description of one fit."""

    experiment: str
    condition: Condition
    donor_scheme: DonorScheme = "none"
    partner_mode: PartnerMode = "real"
    seed: int | None = None
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    ridge: RidgeConfig = field(default_factory=RidgeConfig)
    neural: NeuralConfig | None = None
    notes: str = ""

    # -- hashing ---------------------------------------------------------

    #: Excluded from the hash: free-text, and anything describing *where* a
    #: run happened. A Colab run and a local run of the same fit must resolve
    #: to the same identifier.
    HASH_EXCLUDED: tuple[str, ...] = ("notes",)

    def hashable_dict(self) -> dict[str, Any]:
        """The configuration as a canonical dict, excluding non-semantic fields."""
        raw = asdict(self)
        for key in self.HASH_EXCLUDED:
            raw.pop(key, None)
        return _canonical(raw)

    def run_id(self) -> str:
        """A stable 16-character identifier for this configuration."""
        payload = json.dumps(
            self.hashable_dict(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    # -- provenance ------------------------------------------------------

    def run_record(self) -> dict[str, Any]:
        """The full record written beside every result.

        Carries the environment the run happened in, which the hash excludes.
        The plan requires the runtime to be named because the work is split
        between a local machine and Colab.
        """
        return {
            "run_id": self.run_id(),
            "config": self.hashable_dict(),
            "notes": self.notes,
            "environment": environment_record(),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def result_dir(self, root: Path | str = "results") -> Path:
        """Where this run's artifacts live. Never overwritten across configs."""
        return Path(root) / f"{self.experiment}__{self.run_id()}"


def _canonical(obj: Any) -> Any:
    """Normalise for hashing: tuples become lists, dict keys are sorted."""
    if isinstance(obj, dict):
        return {k: _canonical(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list | tuple):
        return [_canonical(v) for v in obj]
    if is_dataclass(obj) and not isinstance(obj, type):
        return _canonical(asdict(obj))
    return obj


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


def detect_runtime() -> str:
    """Return "colab" or "local".

    The plan splits work between a local machine and Colab, and every run
    record names which produced it.
    """
    try:
        import google.colab  # type: ignore  # noqa: F401
    except ImportError:
        return "local"
    return "colab"


def git_commit() -> str | None:
    """The commit this code was run from, or None if unavailable.

    Colab sessions clone the repository, so this links every cloud
    artifact back to a specific commit.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def environment_record() -> dict[str, Any]:
    """Environment provenance. Recorded, never hashed.

    The Colab copy of the repository has no git history, so the code
    fingerprint is what identifies the code there.
    """
    try:
        from src.runtime import code_fingerprint

        fingerprint = code_fingerprint()
    except Exception:  # outside a project tree
        fingerprint = None
    return {
        "runtime": detect_runtime(),
        "git_commit": git_commit(),
        "code_fingerprint": fingerprint,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def assert_same_capacity(configs: list[RunConfig]) -> None:
    """Assert every config shares one neural configuration.

    The capacity-control protocol requires identical architecture, optimiser
    and budget across the real condition, every donor condition and every
    ablation, so that the comparison is the same model with different
    information rather than different models. Raises if that is violated.
    """
    neural = [c.neural for c in configs]
    if any(n is None for n in neural):
        raise ValueError("assert_same_capacity requires neural configs on all runs")
    first, *rest = neural
    for other, cfg in zip(rest, configs[1:], strict=False):
        if other != first:
            raise AssertionError(
                "Neural configuration differs across conditions, which breaks "
                f"capacity control: {cfg.experiment}/{cfg.condition}"
            )


__all__ = [
    "TARGET_BLOCKS", "TARGET_DIMENSIONS",
    "N_TARGET_DIMENSIONS",
    "Condition", "DonorScheme", "SplitScheme", "PartnerMode",
    "DataConfig", "FeatureConfig", "RidgeConfig", "NeuralConfig", "RunConfig",
    "detect_runtime", "git_commit", "environment_record", "assert_same_capacity",
]
