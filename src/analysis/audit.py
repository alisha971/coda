"""Tables T1-T3 and the leakage register's evidence.

**T3, the leakage register.** Each route of the leakage register is mapped to the evidence
that covers it: named tests in the integrity suite, an executable assertion, a
reported check, or the preregistration's timestamp. :func:`verify_register`
confirms every named test exists, so the register cannot cite a test that was
renamed or deleted.

**T1, sample description** (with trainable parameter count and the dyad-level
effective sample size) and **T2, the experiment matrix with outcomes** are
assembled from stored run outputs.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

LEAKAGE_REGISTER = [
    ("reply audio / duration into inputs", "per-pair firewall (src/data/dataset.py)",
     ["tests/test_splits_firewall.py::test_firewall_refuses_the_pairs_own_reply_and_serves_it_as_context",
      "tests/test_splits_firewall.py::test_vectorised_gather_applies_the_same_rule",
      "tests/test_splits_firewall.py::test_temporal_order_violation_is_detected"]),
    ("test-speaker standardisation", "held-out mode excludes target and context turns",
     ["tests/test_standardisation.py::test_held_out_statistics_exclude_the_target_and_its_context_turns",
      "tests/test_standardisation.py::test_held_out_value_matches_an_independent_computation"]),
    ("train/test normalisation", "train-only statistics; held-out targets near, not exactly, 0/1",
     ["tests/test_standardisation.py::test_held_out_targets_come_out_near_but_not_exactly_zero_one",
      "tests/test_standardisation.py::test_training_mode_uses_training_statistics_only"]),
    ("speaker contamination", "speaker-disjoint split asserted on participant keys",
     ["tests/test_splits_firewall.py::test_sessions_sharing_a_speaker_are_kept_together",
      "tests/test_splits_firewall.py::test_a_naive_per_session_split_would_leak_and_is_caught"]),
    ("donor validity", "donor ≠ partner, same split, N2 shares session/speaker, ≥ 10 turns",
     ["tests/test_controls_inference.py::test_n1_donors_satisfy_every_registered_property",
      "tests/test_controls_inference.py::test_n2_donors_share_session_interaction_speaker_and_are_10_turns_away",
      "tests/test_controls_inference.py::test_invalid_n1_assignments_are_rejected",
      "tests/test_controls_inference.py::test_a_too_close_n2_donor_is_rejected"]),
    ("hyperparameter leakage", "nested CV inside training; test rows never read in selection",
     ["tests/test_ridge_rrr.py::test_lambda_selection_never_reads_evaluation_rows",
      "tests/test_ridge_rrr.py::test_inner_folds_keep_sessions_whole",
      "tests/test_gates_runs_dry_run.py::test_run_directories_are_labelled_and_never_overwritten"]),
    ("function-label leakage", "classifier input schema is text only",
     ["tests/test_injection_coupling_situation.py::test_classifier_sees_text_only"]),
    ("multiplicity", "one primary test; PREREGISTRATION.md committed before J1",
     ["git timestamp of PREREGISTRATION.md (commit e6c6504)"]),
    ("SSL extraction", "pure function of audio; no label input",
     ["tests/test_neural.py::test_extraction_takes_no_label_input"]),
    ("SSL fine-tuning", "encoders frozen; assert_frozen on load",
     ["src/representations/ssl_extract.py::assert_frozen (executed at load)"]),
    ("SSL feature PCA", "fitted on training partner turns only",
     ["tests/test_neural.py::test_ssl_pca_refuses_non_training_turns"]),
    ("layer-mixture selection", "α is a trained parameter",
     ["tests/test_neural.py::test_alpha_is_a_trained_parameter_not_a_selection"]),
    ("neural model selection", "early stopping on validation only; test read once and logged",
     ["tests/test_neural.py::test_training_reduces_validation_loss_and_never_reads_test_rows"]),
    ("architecture selection", "architecture frozen in PREREGISTRATION.md §7",
     ["git timestamp of PREREGISTRATION.md (commit e6c6504)"]),
    ("donor-condition tuning", "frozen configuration hash identical across conditions",
     ["tests/test_neural.py::test_frozen_training_config_is_hashed_and_cannot_be_replaced",
      "tests/test_neural.py::test_parameter_count_is_identical_in_every_partner_mode_and_ablation"]),
    ("representation normalisation", "text PCA and orthogonalisation fitted on train only",
     ["tests/test_text_and_z.py::test_pca_is_fitted_on_training_rows_only",
      "tests/test_standardisation.py::test_transform_fitted_on_train_is_reused_unchanged_on_held_out_rows"]),
    ("speaker shortcut through SSL", "speaker-identity probe reported beside every neural Δ",
     ["tests/test_neural.py::test_probes_recover_planted_structure"]),
]


def verify_register(root: Path) -> pd.DataFrame:
    """Every cited test exists as a function in its file."""
    rows = []
    for route, mechanism, evidence in LEAKAGE_REGISTER:
        for ev in evidence:
            if "::" in ev and ev.startswith("tests/"):
                path, name = ev.split("::")
                tree = ast.parse((root / path).read_text(encoding="utf-8"))
                found = any(isinstance(n, ast.FunctionDef) and n.name == name
                            for n in ast.walk(tree))
            else:
                found = None
            rows.append({"route": route, "mechanism": mechanism, "evidence": ev,
                         "exists": found})
    df = pd.DataFrame(rows)
    missing = df[df["exists"] == False]  # noqa: E712
    if len(missing):
        cited = missing["evidence"].tolist()
        raise AssertionError(f"leakage register cites missing tests: {cited}")
    return df


def table_t1(provenance: dict, trainable_parameters: int | None) -> pd.DataFrame:
    rows = [("sessions (dyads): effective sample size", provenance["n_sessions"]),
            ("pairs in the analysis set", provenance["n_pairs"])]
    rows += [(f"pairs, {k}", v) for k, v in provenance["pairs_by_split"].items()]
    rows.append(("trainable parameters (Layer 2)", trainable_parameters))
    return pd.DataFrame(rows, columns=["quantity", "value"])


def table_t2(j1: dict, j2: dict, gates: dict) -> pd.DataFrame:
    rows = []
    for comp, v in j1["comparisons"].items():
        rows.append({"experiment": "J1", "comparison": comp,
                     "aggregate_partial_r2": v["aggregate_partial_r2"],
                     "loglik_increment": v["aggregate_loglik_increment"]})
    rows.append({"experiment": "J2", "comparison": "H1 vs N1", "p": j2["H1_vs_N1"]["reported"]})
    rows.append({"experiment": "J2", "comparison": "H4 vs N2", "p": j2["H4_vs_N2"]["reported"]})
    for g, r in gates.items():
        rows.append({"experiment": "gate", "comparison": g, "status": r["status"]})
    return pd.DataFrame(rows)


__all__ = ["LEAKAGE_REGISTER", "verify_register", "table_t1", "table_t2"]
