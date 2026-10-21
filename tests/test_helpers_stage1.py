"""Helpers used by the acquisition and audit notebooks."""

import numpy as np
import pandas as pd

from src.data.acquire import interval_overlap_s
from src.data.seamless import speaker_components
from src.features.prosody import feature_droppage, frames_in, turn_file_id, turn_voicing
from src.features.tracks import Tracks
from src.paths import Paths
from src.preprocessing.channel_qc import usable_interactions, usable_sessions


def test_interval_overlap():
    a = pd.DataFrame({"start": [0.0, 5.0], "end": [2.0, 6.0]})
    b = pd.DataFrame({"start": [1.0, 5.5], "end": [5.2, 9.0]})
    assert abs(interval_overlap_s(a, b) - (1.0 + 0.2 + 0.5)) < 1e-9


def test_speaker_components_link_sessions_sharing_a_participant():
    f = pd.DataFrame({
        "session_key": ["s1", "s1", "s2", "s2", "s3", "s3"],
        "participant_key": ["a", "b", "b", "c", "d", "e"],
    })
    comp = speaker_components(f)
    assert comp["s1"] == comp["s2"] != comp["s3"]


def test_usability_needs_both_channels_of_an_interaction():
    qc = pd.DataFrame({
        "interaction_key": ["V0_S1_I1", "V0_S1_I1", "V0_S1_I2", "V0_S1_I2",
                            "V0_S2_I1", "V0_S2_I1"],
        "failure": [None, None, None, "missing", "missing", None],
    })
    inter = usable_interactions(qc)
    assert inter.to_dict() == {"V0_S1_I1": True, "V0_S1_I2": False, "V0_S2_I1": False}
    assert usable_sessions(qc).to_dict() == {"V0_S1": True, "V0_S2": False}


def test_turn_file_id_handles_suffixed_participants():
    assert turn_file_id("V00_S0692_I00000536", "V00_P0844A") == "V00_S0692_I00000536_P0844A"


def test_frames_in_selects_frame_centres_inside_the_turn():
    sl = frames_in(0.005, 0.01, 100, 0.10, 0.20)
    assert (sl.start, sl.stop) == (10, 20)


def test_turn_voicing_and_droppage(tmp_path):
    p = Paths(tmp_path)
    (p.audio_pass / "V0_S1").mkdir(parents=True)
    f0 = np.zeros(300, np.float32)
    f0[10:40] = 150.0          # turn A: 30 voiced frames
    f0[205:208] = 150.0        # turn C: 3 voiced frames
    tr = Tracks(0.01, 0.005, f0, f0 / 150, 0.02, f0, f0, 3.0)
    np.savez_compressed(p.audio_pass / "V0_S1" / "tracks.npz", **tr.to_npz_dict("V0_S1_I1_P1"))
    turns = pd.DataFrame({
        "turn_id": ["A", "C", "X"], "session_key": ["V0_S1"] * 3,
        "interaction_key": ["V0_S1_I1"] * 3, "speaker": ["V0_P1", "V0_P1", "V0_P9"],
        "start": [0.1, 2.0, 0.0], "end": [0.4, 2.2, 1.0],
    })
    v = turn_voicing(turns, p).set_index("turn_id")
    assert v.loc["A", "voiced_frames"] == 30 and v.loc["C", "voiced_frames"] == 3
    assert not v.loc["X", "has_tracks"]
    pairs = pd.DataFrame({"partner_turn_id": ["A"], "reply_turn_id": ["C"]})
    d = feature_droppage(pairs, v.reset_index(), min_voiced_frames=10)
    assert d["turns_dropped"] == 1 and d["turn_droppage"] == 0.5 and d["pair_loss"] == 1.0


def test_octave_errors_show_as_mass_near_plus_minus_one(tmp_path):
    from src.preprocessing.channel_qc import relative_log2_f0

    p = Paths(tmp_path)
    (p.audio_pass / "V0_S1").mkdir(parents=True)
    f0 = np.full(1000, 200.0, np.float32)
    f0[:100] = 100.0                      # 10% halving errors
    tr = Tracks(0.01, 0.005, f0, f0 / 200, 0.02, f0, f0, 10.0)
    np.savez_compressed(p.audio_pass / "V0_S1" / "tracks.npz", **tr.to_npz_dict("V0_S1_I1_P1"))
    qc = pd.DataFrame({"session_key": ["V0_S1"], "file_id": ["V0_S1_I1_P1"], "failure": [None]})
    counts, bins, per = relative_log2_f0(p, qc)
    assert counts.sum() == 1000
    assert per.loc[0, "share_below_minus_0_75"] == 0.1
    centre_of_mass_low = bins[:-1][counts > 0].min()
    assert -1.05 <= centre_of_mass_low <= -0.95


def test_talk_share_sums_to_one_per_interaction():
    from src.analysis.eda import talk_share

    t = pd.DataFrame({"session_key": "S", "interaction_key": ["I1"] * 3 + ["I2"],
                      "speaker": ["a", "b", "a", "a"], "duration": [3.0, 1.0, 0.0, 2.0]})
    s = talk_share(t)
    assert s.groupby("interaction_key")["share"].sum().round(9).tolist() == [1.0, 1.0]
    assert s.set_index(["interaction_key", "speaker"]).loc[("I1", "a"), "share"] == 0.75


def test_between_group_share_is_one_for_pure_group_differences():
    from src.analysis.eda import between_group_share

    df = pd.DataFrame({"v": [1.0, 1.0, 3.0, 3.0], "g": ["x", "x", "y", "y"]})
    assert between_group_share(df, "v", "g") == 1.0
    df2 = pd.DataFrame({"v": [1.0, 3.0, 1.0, 3.0], "g": ["x", "x", "y", "y"]})
    assert between_group_share(df2, "v", "g") == 0.0


def test_file_prosody_summary_reads_voiced_frames_only(tmp_path):
    from src.analysis.eda import file_prosody_summary

    p = Paths(tmp_path)
    (p.audio_pass / "V0_S1").mkdir(parents=True)
    f0 = np.zeros(200, np.float32)
    f0[50:150] = 200.0
    db = np.full(200, 40.0, np.float32)
    db[50:150] = 70.0
    tr = Tracks(0.01, 0.005, f0, f0 / 200, 0.005, db, db, 2.0)
    np.savez_compressed(p.audio_pass / "V0_S1" / "tracks.npz", **tr.to_npz_dict("V0_S1_I1_P1"))
    qc = pd.DataFrame({"session_key": ["V0_S1"], "file_id": ["V0_S1_I1_P1"], "failure": [None],
                       "interaction_key": ["V0_S1_I1"], "participant_key": ["V0_P1"],
                       "vendor_id": ["0"]})
    s = file_prosody_summary(p, qc).iloc[0]
    assert s["median_f0_hz"] == 200.0 and s["median_intensity_db"] == 70.0
    assert s["voiced_frames"] == 100


def test_sample_choice_is_deterministic_and_bounded():
    from src.analysis.samples import choose_samples, cut_stereo

    rows = []
    specs = [("0", 100, 1, 0), ("0", 500, 9, 0), ("1", 120, 5, 2)]
    for k, (site, dur, ov, flat) in enumerate(specs):
        for p in "ab":
            rows.append({"interaction_key": f"I{k}", "file_id": f"I{k}{p}", "failure": None,
                         "duration_s": dur, "overlap_s": ov, "flat_top_runs": flat,
                         "vendor_id": site})
    qc = pd.DataFrame(rows)
    turns = pd.DataFrame({"interaction_key": ["I0", "I0", "I2"], "start": [3.0, 4.0, 50.0],
                          "duration": [1.0, 2.0, 30.0], "embedded": [False, False, False]})
    a = choose_samples(qc, turns, seed=1)
    assert a.equals(choose_samples(qc, turns, seed=1))
    assert "I1" not in set(a["interaction_key"])          # over the length limit
    assert "I2" in set(a["interaction_key"])              # picked once, never twice
    assert a["interaction_key"].is_unique
    clip = cut_stereo(np.arange(100.0), np.arange(100.0), 10, 2.0, 3.0)
    assert clip.shape == (30, 2)
    # the long-turn clip starts on the longest turn, not on the busiest window
    long_row = a[a["label"] == "longest turns"]
    if len(long_row):
        assert long_row["interaction_key"].iloc[0] == "I2"
        assert long_row["start_s"].iloc[0] == 50.0
