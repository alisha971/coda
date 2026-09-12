"""Run-identifier hashing: the first slice of the integrity suite.

Provenance rests on two properties. Identical configurations must hash
identically, or the same fit reproduced elsewhere would look like a different
run. Configurations differing in any field must hash differently, or two
conditions could collide and overwrite each other's results.
"""

from dataclasses import replace

import pytest

from src.config import (
    N_TARGET_DIMENSIONS,
    TARGET_BLOCKS,
    TARGET_DIMENSIONS,
    DataConfig,
    NeuralConfig,
    RidgeConfig,
    RunConfig,
    assert_same_capacity,
)


@pytest.fixture
def base() -> RunConfig:
    return RunConfig(experiment="j1", condition="plus_partner_delivery")


# -- stability -------------------------------------------------------------


def test_identical_configs_hash_identically(base):
    assert base.run_id() == RunConfig(
        experiment="j1", condition="plus_partner_delivery"
    ).run_id()


def test_hash_is_stable_across_calls(base):
    assert len({base.run_id() for _ in range(5)}) == 1


def test_notes_do_not_affect_the_hash(base):
    """Free text is provenance, not identity."""
    assert replace(base, notes="rerun after a Colab preemption").run_id() == base.run_id()


def test_run_id_shape(base):
    rid = base.run_id()
    assert len(rid) == 16 and all(c in "0123456789abcdef" for c in rid)


# -- sensitivity -----------------------------------------------------------


@pytest.mark.parametrize(
    "change",
    [
        {"condition": "Z_content"},
        {"donor_scheme": "N1_cross_dyad"},
        {"donor_scheme": "N2_within_dyad_distant"},
        {"partner_mode": "null_token"},
        {"partner_mode": "delivery_vector_interpretable"},
        {"seed": 1},
        {"experiment": "j2"},
    ],
)
def test_any_semantic_change_changes_the_hash(base, change):
    assert replace(base, **change).run_id() != base.run_id()


def test_nested_config_changes_change_the_hash(base):
    """A change buried in a sub-config must still surface in the identifier."""
    assert replace(base, data=DataConfig(n_conversations=300)).run_id() != base.run_id()
    assert replace(base, data=DataConfig(split_seed=7)).run_id() != base.run_id()
    assert replace(base, ridge=RidgeConfig(nested_cv_folds=10)).run_id() != base.run_id()


def test_donor_schemes_do_not_collide(base):
    ids = {
        replace(base, donor_scheme=s).run_id()
        for s in ("none", "N1_cross_dyad", "N2_within_dyad_distant")
    }
    assert len(ids) == 3


def test_result_dirs_are_distinct_across_conditions(base):
    dirs = {
        replace(base, condition=c).result_dir()
        for c in ("Z_content", "plus_partner_text",
                  "plus_partner_delivery", "plus_self_previous")
    }
    assert len(dirs) == 4, "conditions must not share a results directory"


# -- targets ---------------------------------------------------------------


def test_four_blocks_with_unique_dimension_names():
    assert len(TARGET_BLOCKS) == 4
    assert len(set(TARGET_DIMENSIONS)) == len(TARGET_DIMENSIONS)


def test_block_sizes_match_the_specification_table():
    """Transcribed from the section 8.1 table, which is where they are named."""
    assert [len(v) for v in TARGET_BLOCKS.values()] == [2, 1, 2, 3]


def test_target_vector_is_eight_dimensions():
    """Eight, not the nine the specification's prose claims. See PREREGISTRATION.md.

    Guards against a later edit silently changing the size of the target.
    """
    assert N_TARGET_DIMENSIONS == 8
    assert sum(len(v) for v in TARGET_BLOCKS.values()) == N_TARGET_DIMENSIONS


# -- capacity control ------------------------------------------------------


def test_same_capacity_passes_when_only_information_differs():
    """The scientific comparison: same model, same capacity, different input."""
    neural = NeuralConfig()
    runs = [
        RunConfig(experiment="neural", condition="plus_partner_delivery",
                  partner_mode=m, neural=neural)
        for m in ("real", "donor", "null_token", "delivery_vector_interpretable")
    ]
    assert_same_capacity(runs)


def test_same_capacity_fails_when_the_architecture_differs():
    runs = [
        RunConfig(experiment="neural", condition="plus_partner_delivery",
                  neural=NeuralConfig()),
        RunConfig(experiment="neural", condition="plus_partner_delivery",
                  partner_mode="null_token", neural=NeuralConfig(d_model=256)),
    ]
    with pytest.raises(AssertionError, match="capacity control"):
        assert_same_capacity(runs)


# -- provenance ------------------------------------------------------------


def test_run_record_names_the_runtime(base):
    record = base.run_record()
    assert record["environment"]["runtime"] in {"local", "colab"}
    assert record["run_id"] == base.run_id()
    assert "notes" not in record["config"], "notes must stay out of the hashed config"
