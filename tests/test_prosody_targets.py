"""The eight target dimensions, checked against values computed by hand.

Synthetic frame tracks with known F0 and intensity stand in for audio, so each
expected value can be derived on paper. The cases cover the registered
definitions, the missingness rules, and the measurement options (overlap handling,
pitch range, syllable method, intensity).
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.features.prosody import (
    FAIL_BOUNDARY_TOO_FEW_VOICED,
    FAIL_NO_WORDS,
    FAIL_TOO_FEW_VOICED,
    MeasurementNotAvailable,
    ProsodySettings,
    compute_turn_prosody,
    droppage_by_dimension,
    file_loudness_offset,
    measure_turn,
)
from src.features.syllables import count_syllables, rule_based_syllables
from src.features.tracks import Tracks
from src.settings import Settings

STEP = 0.01
T0 = STEP / 2   # frame k is centred at (k + 0.5) * step, as the audio pass stores it


def tracks(seconds: float, f0_fn, db_fn):
    n = int(seconds / STEP)
    t = T0 + np.arange(n) * STEP
    return t, np.asarray(f0_fn(t), dtype=np.float32), np.asarray(db_fn(t), dtype=np.float32)


S = ProsodySettings()


def run(words, text, f0, db, start, end, settings=S, partner=None):
    return measure_turn(start, end, np.asarray(words, float), text, T0, f0, T0, db, STEP,
                        settings, partner)


# -- pitch ------------------------------------------------------------------


def test_constant_pitch_has_known_mean_zero_range_zero_slope():
    _, f0, db = tracks(2.0, lambda t: np.full_like(t, 200.0), lambda t: np.full_like(t, 60.0))
    m = run([[0.0, 1.0]], "hello there", f0, db, 0.0, 1.0)
    assert m["mean_log_f0"] == pytest.approx(np.log(200.0), abs=1e-5)
    assert m["log_f0_range_p95_p5"] == pytest.approx(0.0, abs=1e-6)
    assert m["log_f0_slope_final_300ms"] == pytest.approx(0.0, abs=1e-4)
    assert m["usable"]


def test_boundary_slope_recovers_a_known_log_f0_ramp():
    """ln F0 rising at 0.5 per second: the slope over the last 300 ms is 0.5."""
    _, f0, db = tracks(2.0, lambda t: 150.0 * np.exp(0.5 * t), lambda t: np.full_like(t, 60.0))
    m = run([[0.0, 1.5]], "a rising turn", f0, db, 0.0, 1.5)
    assert m["log_f0_slope_final_300ms"] == pytest.approx(0.5, rel=1e-3)
    assert m["n_boundary_voiced"] == 30


def test_pitch_dimensions_are_missing_below_the_voiced_minimum():
    _, f0, db = tracks(1.0, lambda t: np.where(t < 0.05, 180.0, 0.0), lambda t: t * 0 + 50)
    m = run([[0.0, 0.8]], "mm", f0, db, 0.0, 0.8)
    assert np.isnan(m["mean_log_f0"]) and m["fail_mean_log_f0"] == FAIL_TOO_FEW_VOICED
    assert not m["usable"]
    # Energy and timing are still measured: missingness is per dimension.
    assert np.isfinite(m["mean_intensity"]) and np.isfinite(m["articulation_rate"])


def test_boundary_window_needs_its_own_voiced_frames():
    """Voicing early in the turn must not be borrowed for the final-300 ms slope."""
    _, f0, db = tracks(2.0, lambda t: np.where(t < 1.0, 180.0, 0.0), lambda t: t * 0 + 50)
    m = run([[0.0, 1.6]], "voiced then unvoiced end", f0, db, 0.0, 1.6)
    assert np.isfinite(m["mean_log_f0"])
    assert m["fail_log_f0_slope_final_300ms"] == FAIL_BOUNDARY_TOO_FEW_VOICED
    assert not m["usable"]


def test_frames_in_internal_pauses_do_not_enter_pitch_or_energy():
    """A loud, voiced stretch inside a pause must not move the speech measures."""
    _, f0, db = tracks(3.0, lambda t: np.where((t > 1.0) & (t < 2.0), 400.0, 200.0),
                       lambda t: np.where((t > 1.0) & (t < 2.0), 90.0, 60.0))
    m = run([[0.0, 1.0], [2.0, 3.0]], "one two", f0, db, 0.0, 3.0)
    assert m["mean_log_f0"] == pytest.approx(np.log(200.0), abs=1e-5)
    assert m["mean_intensity"] == pytest.approx(60.0, abs=1e-4)


# -- energy -----------------------------------------------------------------


def test_mean_intensity_uses_praat_energy_averaging():
    """Half the frames at 60 dB, half at 70 dB: energy mean is 10*log10((1e6+1e7)/2)."""
    _, f0, db = tracks(1.0, lambda t: np.full_like(t, 200.0),
                       lambda t: np.where(t < 0.5, 60.0, 70.0))
    m = run([[0.0, 1.0]], "x", f0, db, 0.0, 1.0)
    assert m["mean_intensity"] == pytest.approx(10 * np.log10((1e6 + 1e7) / 2), abs=1e-3)
    m_db = run([[0.0, 1.0]], "x", f0, db, 0.0, 1.0,
               replace(S, intensity_averaging="db"))
    assert m_db["mean_intensity"] == pytest.approx(65.0, abs=1e-4)


def test_intensity_range_is_p95_minus_p5():
    _, f0, db = tracks(1.0, lambda t: np.full_like(t, 200.0), lambda t: 50.0 + 20.0 * t)
    m = run([[0.0, 1.0]], "x", f0, db, 0.0, 1.0)
    vals = 50.0 + 20.0 * (T0 + np.arange(100) * STEP)
    expected = np.percentile(vals, 95) - np.percentile(vals, 5)
    assert m["intensity_range"] == pytest.approx(expected, abs=1e-3)


# -- timing -----------------------------------------------------------------


def test_timing_dimensions_by_hand():
    """Two IPUs (0-1 s, 1.5-2.5 s) and a 0.5 s pause in a 2.5 s turn."""
    _, f0, db = tracks(3.0, lambda t: np.full_like(t, 200.0), lambda t: t * 0 + 60)
    text = "conversation table"                      # 4 + 2 syllables
    m = run([[0.0, 0.4], [0.42, 1.0], [1.5, 2.5]], text, f0, db, 0.0, 2.5)
    assert m["phonation_s"] == pytest.approx(2.0)
    assert m["n_syllables"] == 6
    assert m["articulation_rate"] == pytest.approx(6 / 2.0)
    assert m["pause_ratio"] == pytest.approx(0.5 / 2.5)
    assert m["longest_internal_pause"] == pytest.approx(0.5)


def test_single_ipu_turn_has_a_measured_zero_pause_not_a_missing_one():
    _, f0, db = tracks(1.0, lambda t: np.full_like(t, 200.0), lambda t: t * 0 + 60)
    m = run([[0.0, 0.5]], "yes", f0, db, 0.0, 0.5)
    assert m["longest_internal_pause"] == 0.0 and m["pause_ratio"] == 0.0
    assert m["fail_longest_internal_pause"] is None


def test_turn_without_words_is_missing_everywhere_with_a_reason():
    _, f0, db = tracks(1.0, lambda t: np.full_like(t, 200.0), lambda t: t * 0 + 60)
    m = run(np.zeros((0, 2)), "", f0, db, 0.0, 0.5)
    assert not m["usable"] and m["fail_mean_intensity"] == FAIL_NO_WORDS


# -- overlap handling ------------------------------------------------


def test_exclude_overlap_removes_partner_overlapped_frames_and_pauses():
    """The partner speaks over 0.5-1.0 s (quiet frames) and over the 1.0-1.5 s pause."""
    _, f0, db = tracks(3.0, lambda t: np.full_like(t, 200.0),
                       lambda t: np.where((t > 0.5) & (t < 1.0), 40.0, 60.0))
    words = [[0.0, 1.0], [1.5, 2.5]]
    partner = np.array([[0.5, 1.4]])
    allm = run(words, "a b", f0, db, 0.0, 2.5, S, partner)
    excl = run(words, "a b", f0, db, 0.0, 2.5, replace(S, overlap_handling="exclude_overlap"),
               partner)
    assert allm["intensity_range"] > 15          # quiet overlapped frames widen the range
    assert excl["intensity_range"] == pytest.approx(0.0, abs=1e-6)
    # 0.4 s of the 0.5 s pause is overlapped: 0.1 s remains, in a 2.1 s turn.
    assert excl["pause_ratio"] == pytest.approx(0.1 / 2.1)
    assert allm["pause_ratio"] == pytest.approx(0.5 / 2.5)



def test_exclude_overlap_energy_changes_intensity_only():
    """Energy-only overlap option: intensity as under exclusion, pauses as under 'all'."""
    _, f0, db = tracks(3.0, lambda t: np.full_like(t, 200.0),
                       lambda t: np.where((t > 0.5) & (t < 1.0), 40.0, 60.0))
    words = [[0.0, 1.0], [1.5, 2.5]]
    partner = np.array([[0.5, 1.4]])
    allm = run(words, "a b", f0, db, 0.0, 2.5, S, partner)
    en = run(words, "a b", f0, db, 0.0, 2.5,
             replace(S, overlap_handling="exclude_overlap_energy"), partner)
    assert en["intensity_range"] == pytest.approx(0.0, abs=1e-6)
    for d in ("pause_ratio", "longest_internal_pause", "articulation_rate", "mean_log_f0"):
        assert en[d] == pytest.approx(allm[d])


def test_unknown_overlap_option_is_rejected():
    with pytest.raises(ValueError, match="overlap handling"):
        replace(S, overlap_handling="energy").check()

# -- pitch range -----------------------------------------------------


def test_speaker_adaptive_pitch_range_cannot_be_computed_from_stored_tracks():
    _, f0, db = tracks(1.0, lambda t: np.full_like(t, 200.0), lambda t: t * 0 + 60)
    with pytest.raises(MeasurementNotAvailable, match="new audio"):
        run([[0.0, 1.0]], "x", f0, db, 0.0, 1.0, replace(S, pitch_range="speaker_adaptive"))


# -- syllables ------------------------------------------------------


@pytest.mark.parametrize("word,n", [("conversation", 4), ("make", 1), ("table", 2), ("I", 1),
                                    ("42", 0), ("the", 1), ("liked", 1), ("wanted", 2)])
def test_rule_based_syllables(word, n):
    assert rule_based_syllables(word) == n


def test_dictionary_syllables_with_rule_fallback():
    pytest.importorskip("cmudict")
    from src.features.syllables import dictionary_syllables, fallback_share

    assert dictionary_syllables("every")[0] in (2, 3)        # first listed pronunciation
    assert dictionary_syllables("conversation") == (4, False)
    assert dictionary_syllables("Hello,") == (2, False)
    n, fb = dictionary_syllables("zzxqvtronic")              # not in the dictionary
    assert fb and n == rule_based_syllables("zzxqvtronic") and n >= 1
    assert dictionary_syllables("42") == (0, False)
    assert count_syllables("the conversation", "dictionary") == 5
    assert fallback_share(["the conversation", "zzxqvtronic"]) == pytest.approx(1 / 3)


def test_every_lexical_token_gets_at_least_one_syllable_under_both_methods():
    words = ["I", "a", "hmm", "mm", "shh", "rhythm", "psst", "x", "brr"]
    assert all(rule_based_syllables(w) >= 1 for w in words)
    if __import__("importlib").util.find_spec("cmudict"):
        from src.features.syllables import dictionary_syllables

        assert all(dictionary_syllables(w)[0] >= 1 for w in words)


def test_unknown_syllable_method_is_rejected():
    with pytest.raises(ValueError):
        count_syllables("hello", "neural")


# -- batch computation ------------------------------------------------------


def test_compute_turn_prosody_measures_each_turn_from_its_own_file():
    n = 400
    f0 = np.full(n, 200.0, np.float32)
    db = np.full(n, 60.0, np.float32)
    trk = {"V0_S1_I1_P1": Tracks(STEP, T0, f0, f0 / 200, T0, db, db, 4.0),
           "V0_S1_I1_P2": Tracks(STEP, T0, f0 * 1.5, f0 / 200, T0, db + 5, db, 4.0)}
    words = {"V0_S1_I1_P1": pd.DataFrame({"start": [0.0], "end": [1.0], "word": ["hello"]}),
             "V0_S1_I1_P2": pd.DataFrame({"start": [1.5], "end": [2.5], "word": ["hi"]})}
    turns = pd.DataFrame({
        "turn_id": ["a", "b"], "session_key": ["V0_S1"] * 2,
        "interaction_key": ["V0_S1_I1"] * 2, "speaker": ["V0_P1", "V0_P2"],
        "embedded": [False, False], "start": [0.0, 1.5], "end": [1.0, 2.5]})
    out = compute_turn_prosody(turns, lambda sk, fid: trk.get(fid), words.get, S)
    out = out.set_index("turn_id")
    assert out.loc["a", "mean_log_f0"] == pytest.approx(np.log(200.0), abs=1e-5)
    assert out.loc["b", "mean_log_f0"] == pytest.approx(np.log(300.0), abs=1e-5)
    assert out.loc["b", "mean_intensity"] == pytest.approx(65.0, abs=1e-4)
    table = droppage_by_dimension(out.reset_index())
    assert table.iloc[-1]["missing"] == 0


def test_missing_tracks_give_a_reason_not_a_value():
    turns = pd.DataFrame({"turn_id": ["a"], "session_key": ["V0_S1"],
                          "interaction_key": ["V0_S1_I1"], "speaker": ["V0_P1"],
                          "embedded": [False], "start": [0.0], "end": [1.0]})
    out = compute_turn_prosody(turns, lambda sk, fid: None, lambda fid: None, S)
    assert not out["usable"].iloc[0] and out["fail_mean_log_f0"].iloc[0] == "no_tracks"


def test_measurement_is_deterministic():
    _, f0, db = tracks(2.0, lambda t: 150 + 30 * np.sin(5 * t), lambda t: 55 + 5 * np.cos(3 * t))
    a = run([[0.0, 0.7], [0.9, 1.8]], "some words here", f0, db, 0.0, 1.8)
    b = run([[0.0, 0.7], [0.9, 1.8]], "some words here", f0, db, 0.0, 1.8)
    assert a == b


def test_loudness_offset_is_the_median_intensity_over_voiced_frames():
    f0 = np.array([0, 200, 200, 200, 0], np.float32)
    db = np.array([10, 50, 60, 70, 99], np.float32)
    assert file_loudness_offset(Tracks(STEP, T0, f0, f0, T0, db, db, 0.05)) == 60.0


def test_prosody_settings_are_read_from_the_study_settings():
    s = ProsodySettings.from_config()
    d = Settings()
    assert s.overlap_handling == d.overlap_handling
    assert s.pitch_range == d.pitch_range
    assert s.boundary_min_voiced_frames == d.turn_measurement["boundary_min_voiced_frames"]
