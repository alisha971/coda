"""The tracked YAML configuration and the code must not drift apart.

`configs/` is what a reader inspects to see how the study was run; `src/config.py`
is what actually runs. If the two disagree, the repository documents one study
and executes another. These tests make that impossible to do quietly.
"""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from src.config import (  # noqa: E402
    N_TARGET_DIMENSIONS,
    TARGET_BLOCKS,
    DataConfig,
    FeatureConfig,
    NeuralConfig,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def load(name: str) -> dict:
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def features() -> dict:
    return load("features.yaml")


# -- targets ---------------------------------------------------------------


def test_target_blocks_match_the_code(features):
    """The eight dimensions and their four blocks are defined once, not twice."""
    from_yaml = {k: tuple(v) for k, v in features["target_blocks"].items()}
    assert from_yaml == TARGET_BLOCKS


def test_yaml_target_count_is_eight(features):
    n = sum(len(v) for v in features["target_blocks"].values())
    assert n == N_TARGET_DIMENSIONS == 8


def test_four_blocks_carry_confirmatory_inference(features):
    assert len(features["target_blocks"]) == 4


# -- prosody instrument ----------------------------------------------------


def test_prosody_parameters_match_the_code(features):
    """Pinned in both places; an F0 default that drifted would move every target."""
    p = features["prosody"]
    cfg = FeatureConfig()
    assert p["backend"] == cfg.f0_backend == "parselmouth"
    assert p["f0_floor_hz"] == cfg.f0_floor_hz
    assert p["f0_ceiling_hz"] == cfg.f0_ceiling_hz
    assert p["boundary_window_s"] == cfg.boundary_window_s == 0.300


def test_frozen_encoders_are_declared_frozen(features):
    assert features["text"]["frozen"] is True
    assert features["ssl"]["frozen"] is True


def test_ssl_cache_budget_matches_the_code(features):
    s = features["ssl"]
    cfg = FeatureConfig()
    assert s["max_frames"] == cfg.ssl_max_frames == 400
    assert s["tail_seconds"] == cfg.ssl_tail_seconds == 8.0
    assert s["pca_dim"] == cfg.ssl_pca_dim == 128
    assert len(s["sources"]) == 5


def test_ssl_sources_match_the_neural_config(features):
    assert tuple(features["ssl"]["sources"]) == NeuralConfig().ssl_sources


# -- corpus and splits -----------------------------------------------------


def test_sample_size_matches_the_code():
    corpus = load("corpus.yaml")
    cfg = DataConfig()
    assert corpus["n_conversations"] == cfg.n_conversations == 250
    assert corpus["pilot_n_conversations"] == cfg.pilot_n_conversations == 40


def test_sample_is_inside_the_specified_range():
    n = load("corpus.yaml")["n_conversations"]
    assert 250 <= n <= 350, "sample size must stay inside the specified range"


def test_audio_is_process_and_discard():
    assert load("corpus.yaml")["process_and_discard"] is True


def test_split_fractions_match_the_code_and_sum_to_one():
    primary = load("splits.yaml")["primary"]
    cfg = DataConfig()
    assert (primary["train"], primary["val"], primary["test"]) == (
        cfg.train_frac, cfg.val_frac, cfg.test_frac
    )
    assert primary["train"] + primary["val"] + primary["test"] == pytest.approx(1.0)


def test_primary_split_is_speaker_disjoint():
    """The claim is about unseen speakers, so the primary split must test that."""
    assert load("splits.yaml")["primary"]["scheme"] == "speaker_disjoint"


def test_near_duplicates_are_clustered_before_splitting():
    assert load("splits.yaml")["primary"]["cluster_near_duplicates"] is True


# -- controls --------------------------------------------------------------


def test_donor_draw_counts_match_the_code():
    donors = load("j2_donors.yaml")
    assert donors["n_draws_layer1"] == 500
    assert donors["n_draws_layer2"] == NeuralConfig().n_donor_draws == 30


def test_n2_distance_matches_the_code():
    donors = load("j2_donors.yaml")
    assert donors["N2_within_dyad_distant"]["min_turn_distance"] == \
        DataConfig().n2_min_turn_distance == 10


def test_both_negative_controls_are_present():
    donors = load("j2_donors.yaml")
    assert "N1_cross_dyad" in donors and "N2_within_dyad_distant" in donors


def test_injection_spans_the_specified_partial_r2_range():
    betas = load("j3_injection.yaml")["injection"]["beta_targets_partial_r2"]
    assert min(betas) <= 0.002 and max(betas) >= 0.05
    assert load("j3_injection.yaml")["injection"]["beta_zero_calibration"] is True


# -- J1 conditions ---------------------------------------------------------


def test_j1_has_exactly_four_conditions_on_two_splits():
    j1 = load("j1_conditions.yaml")
    assert len(j1["conditions"]) == 4
    assert set(j1["splits"]) == {"speaker_disjoint", "dyad_disjoint"}


def test_content_only_condition_excludes_partner_text():
    """The baseline every delta is measured against must not see partner text."""
    conds = {c["name"]: c for c in load("j1_conditions.yaml")["conditions"]}
    assert "partner_text" not in conds["Z_content"]["inputs"]


# -- situation analysis ----------------------------------------------------


def test_situation_gates_match_the_specification():
    j4 = load("j4_situation.yaml")
    assert j4["reliability"]["hand_labelled_pairs"] == 300
    assert j4["reliability"]["min_kappa_for_confirmatory"] == 0.6
    assert j4["reliability"]["min_pairs_per_stratum"] == 200


def test_other_category_is_excluded_not_modelled():
    assert "other" in load("j4_situation.yaml")["categories"]


# -- neural capacity control ----------------------------------------------


def test_frozen_neural_config_matches_the_code():
    arch = load("neural_frozen.yaml")["architecture"]
    cfg = NeuralConfig()
    assert arch["d_model"] == cfg.d_model == 192
    assert arch["n_blocks"] == cfg.n_blocks == 3
    assert arch["n_heads"] == cfg.n_heads == 4
    assert arch["ffn_dim"] == cfg.ffn_dim == 384


def test_head_width_matches_the_target_vector():
    assert load("neural_frozen.yaml")["architecture"]["output_dim"] == N_TARGET_DIMENSIONS


def test_exactly_one_cross_attention_block():
    """One block, and it generates nothing. Not a decoder."""
    assert load("neural_frozen.yaml")["architecture"]["fusion_blocks"] == 1


def test_three_seeds_per_condition():
    assert load("neural_frozen.yaml")["training"]["seeds"] == list(NeuralConfig().seeds)


@pytest.mark.parametrize(
    "flag",
    [
        "identical_across_conditions",
        "partner_pathway_never_removed",
        "assert_parameter_count_identical",
        "assert_config_hash_identical",
        "no_test_set_model_selection",
    ],
)
def test_capacity_control_flags_are_all_on(flag):
    """Capacity control is the thing that makes the neural comparison mean anything."""
    assert load("neural_frozen.yaml")["capacity_control"][flag] is True
