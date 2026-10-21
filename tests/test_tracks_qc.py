"""Frame tracks and channel QC on synthetic signals with known answers."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("parselmouth")

from src.features.tracks import Tracks, extract_tracks, frame_dbfs  # noqa: E402
from src.preprocessing.channel_qc import (  # noqa: E402
    channel_measures,
    flat_top_runs,
    g0_verdict,
    objective_failure,
    vad_mask,
)

SR = 48000


def tone(f0, secs, amp=0.3):
    t = np.arange(int(SR * secs)) / SR
    # a harmonic-rich periodic signal, closer to voice than a pure sine
    return amp * sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))


def test_f0_of_a_periodic_signal_is_recovered():
    tr = extract_tracks(tone(200.0, 1.0), SR)
    voiced = tr.f0_hz[tr.f0_hz > 0]
    assert voiced.size > 50
    assert abs(np.median(voiced) - 200.0) < 2.0


def test_a_loud_transient_does_not_silence_pitch_tracking():
    """Regression. One clipped transient sets the file maximum, and Praat's
    silence threshold is relative to it: whole-file analysis loses the voice,
    per-segment analysis keeps it."""
    quiet = tone(180.0, 2.0, amp=0.01)
    x = np.concatenate([quiet, np.zeros(SR // 2), quiet])
    x[int(2.2 * SR): int(2.2 * SR) + 400] = 1.0          # a clipped bump, not speech
    segs = np.array([[0.0, 2.0], [2.5, 4.5]])
    whole = extract_tracks(x, SR)
    local = extract_tracks(x, SR, speech_segments=segs)
    assert (whole.f0_hz > 0).mean() < 0.2
    assert (local.f0_hz > 0).mean() > 0.7
    assert abs(np.median(local.f0_hz[local.f0_hz > 0]) - 180.0) < 2.0


def test_segment_pitch_is_unvoiced_outside_speech():
    tr = extract_tracks(tone(200.0, 3.0), SR, speech_segments=np.array([[0.5, 1.5]]))
    k = np.arange(tr.f0_hz.size) * tr.step_s + tr.f0_t0
    assert (tr.f0_hz[(k < 0.4) | (k > 1.6)] == 0).all()
    assert (tr.f0_hz[(k > 0.6) & (k < 1.4)] > 0).mean() > 0.9


def test_tracks_are_deterministic_and_round_trip_through_npz(tmp_path):
    x = tone(150.0, 0.6)
    a, b = extract_tracks(x, SR), extract_tracks(x, SR)
    assert np.array_equal(a.f0_hz, b.f0_hz) and np.array_equal(a.intensity_db, b.intensity_db)
    np.savez_compressed(tmp_path / "t.npz", **a.to_npz_dict("F1"))
    c = Tracks.from_npz(np.load(tmp_path / "t.npz"), "F1")
    assert np.array_equal(a.f0_hz, c.f0_hz) and c.duration_s == pytest.approx(0.6)


def test_frame_level_of_full_scale_sine_is_minus_3_dbfs():
    t = np.arange(SR) / SR
    lv = frame_dbfs(np.sin(2 * np.pi * 500 * t), SR, 0.01)  # whole cycles per frame
    assert np.allclose(lv, -3.01, atol=0.05)


def test_flat_top_runs_detect_clipping_after_normalisation():
    x = np.sin(np.linspace(0, 20, 2000))
    clipped = np.clip(x, -0.8, 0.8) / 0.8 * 0.5
    assert flat_top_runs(x) == 0
    assert flat_top_runs(clipped) > 0


def test_vad_mask_marks_frames():
    m = vad_mask(pd.DataFrame({"start": [0.1], "end": [0.3]}), 50, 0.01)
    assert m.sum() == 20 and m[10] and not m[9]


def test_overlap_attenuation_is_measured():
    """Own voice at -10 dB during double talk shows as a -10 dB overlap drop."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1e-4, SR * 3)
    x[: SR] += tone(180.0, 1.0)                            # own only
    x[SR: 2 * SR] += tone(180.0, 1.0) * 10 ** (-10 / 20)   # both talking
    tr = extract_tracks(x, SR)
    own = pd.DataFrame({"start": [0.0], "end": [2.0]})
    partner = pd.DataFrame({"start": [1.0], "end": [3.0]})
    w = pd.DataFrame({"start": [0.1], "end": [0.5], "word": ["hi"]})
    q = channel_measures(x, SR, tr, own, partner, w)
    assert q["overlap_drop_db"] == pytest.approx(-10.0, abs=0.5)
    assert q["bleed_db"] < -40
    assert q["voiced_frac_own_speech"] > 0.8


def test_objective_failures_in_priority_order():
    ok = dict(exists=True, readable=True, sample_rate=48000, channels=1, duration_s=10.0,
              partner_duration_s=10.1, vad_speech_s=3.0, n_words_timed=5,
              expected_sr=48000, max_mismatch_s=0.5)
    assert objective_failure(**ok) is None
    assert objective_failure(**{**ok, "exists": False}) == "missing"
    assert objective_failure(**{**ok, "sample_rate": 16000}) == "wrong_sample_rate"
    assert objective_failure(**{**ok, "partner_duration_s": 11.0}) == "duration_mismatch"
    assert objective_failure(**{**ok, "n_words_timed": 0}) == "no_timed_words"


def test_g0_uses_the_specified_thresholds():
    qc = pd.DataFrame({"failure": [None] * 8 + ["missing"] * 2})
    assert g0_verdict(qc, 0.10, 0.80, 0.15)["passed"] is True
    assert g0_verdict(qc, 0.15, 0.80, 0.15)["passed"] is False
    qc2 = pd.DataFrame({"failure": [None] * 7 + ["missing"] * 3})
    assert g0_verdict(qc2, 0.0, 0.80, 0.15)["passed"] is False


# -- pitch is analysed over voice activity AND transcript ------------


def test_merge_intervals_unions_and_merges_within_the_gap():
    from src.features.tracks import merge_intervals, total_seconds

    a = np.array([[0.0, 1.0], [5.0, 6.0]])
    b = np.array([[0.9, 2.0], [3.0, 3.5]])
    m = merge_intervals(a, b, gap_s=0.0)
    assert m.tolist() == [[0.0, 2.0], [3.0, 3.5], [5.0, 6.0]]
    # a 0.6 s gap closes only when the merge gap reaches it
    assert len(merge_intervals(np.array([[0.0, 1.0], [1.6, 2.0]]), gap_s=0.05)) == 2
    assert len(merge_intervals(np.array([[0.0, 1.0], [1.04, 2.0]]), gap_s=0.05)) == 1
    assert total_seconds(m) == pytest.approx(2.0 + 0.5 + 1.0)
    assert merge_intervals(None, gap_s=0.0).shape == (0, 2)


def test_union_can_only_widen_the_analysed_region():
    """The repair is monotone: adding the transcript never removes coverage."""
    from dataclasses import replace

    from src.config import FeatureConfig
    from src.features.tracks import speech_regions, total_seconds

    vad = np.array([[1.0, 2.0], [8.0, 9.0]])
    words = np.array([[1.5, 2.5], [4.0, 5.0]])          # 4–5 s missed by the detector
    old = replace(FeatureConfig(), pitch_segment_source="vad")
    new = replace(FeatureConfig(), pitch_segment_source="vad_plus_transcript")
    a, b = speech_regions(vad, words, old), speech_regions(vad, words, new)
    assert total_seconds(b) > total_seconds(a)
    for s, e in a:                                       # every old region still covered
        assert any(s >= bs - 1e-9 and e <= be + 1e-9 for bs, be in b)


def test_transcript_recovers_speech_the_detector_missed():
    """The pilot's failure mode: a long turn with no voice-activity segment."""
    from dataclasses import replace

    from src.config import FeatureConfig
    from src.features.tracks import extract_tracks, speech_regions

    x = np.concatenate([np.zeros(SR), tone(180.0, 2.0), np.zeros(SR)])
    vad = np.zeros((0, 2))                               # detector found nothing
    words = np.array([[1.0, 3.0]])                       # transcript did
    old = replace(FeatureConfig(), pitch_segment_source="vad")
    new = replace(FeatureConfig(), pitch_segment_source="vad_plus_transcript")
    lost = extract_tracks(x, SR, old, speech_segments=speech_regions(vad, words, old))
    kept = extract_tracks(x, SR, new, speech_segments=speech_regions(vad, words, new))
    assert (lost.f0_hz > 0).sum() == 0
    assert (kept.f0_hz > 0).sum() > 150
    assert abs(np.median(kept.f0_hz[kept.f0_hz > 0]) - 180.0) < 2.0


def test_the_transient_fix_survives_the_repair():
    """Per-segment tracking must still hold: a loud bump must not silence the file."""
    from src.features.tracks import extract_tracks, speech_regions

    quiet = tone(180.0, 2.0, amp=0.01)
    x = np.concatenate([quiet, np.zeros(SR // 2), quiet])
    x[int(2.2 * SR): int(2.2 * SR) + 400] = 1.0
    words = np.array([[0.0, 2.0], [2.5, 4.5]])
    segs = speech_regions(np.zeros((0, 2)), words)
    assert (extract_tracks(x, SR, speech_segments=segs).f0_hz > 0).mean() > 0.7
    assert (extract_tracks(x, SR).f0_hz > 0).mean() < 0.2      # whole-file still fails


def test_a_session_measured_under_another_source_is_reprocessed(tmp_path):
    import json

    from src.preprocessing.audio_pass import done_variant

    p = tmp_path / "DONE.json"
    p.write_text(json.dumps({"pitch_segment_source": "vad"}), encoding="utf-8")
    assert done_variant(p) == "vad"
    p.write_text(json.dumps({"session_key": "x"}), encoding="utf-8")
    assert done_variant(p) == "vad"          # older records are the old variant
