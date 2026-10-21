"""Corpus handling: identifiers, sampling frame, selection and manifest.

The selection rule is where a silent error would change which people are in the
study, so each property the preregistration states is asserted here.
"""

import json

import pandas as pd
import pytest

from src.data.seamless import (
    FileKey,
    apply_frame,
    build_manifest,
    file_url,
    manner_instructing,
    read_vad,
    read_words,
    sessions_to_process,
    speaker_unique_order,
)

BASE = "https://example.invalid/seamless_interaction"
FRAME_CFG = {
    "relationship": "stranger",
    "interaction_types": ["ipc_conversation"],
    "exclude_two_sided_role_prompts": True,
    "exclude_prompt_text_regex": r"\btone\b|\bspeak (?:in|with)\b|\brespond with a\b",
}


def test_file_id_accepts_participant_suffix():
    k = FileKey.parse("V00_S0692_I00000536_P0844A")
    assert k.participant == "0844A"
    assert k.session_key == "V00_S0692"
    assert k.interaction_key == "V00_S0692_I00000536"
    assert k.participant_key == "V00_P0844A"


def test_participant_key_is_vendor_qualified():
    """Participant numbers are per vendor; two vendors can reuse a number."""
    assert FileKey.parse("V00_S1_I1_P0001").participant_key != \
        FileKey.parse("V03_S1_I1_P0001").participant_key


def test_malformed_file_id_is_rejected():
    with pytest.raises(ValueError):
        FileKey.parse("V00_S01_P0001")


def test_video_can_never_be_requested():
    assert file_url(BASE, "train", "audio", "V00_S1_I1_P1").endswith(
        "/naturalistic/train/audio/V00_S1_I1_P1.wav")
    with pytest.raises(ValueError):
        file_url(BASE, "train", "video", "V00_S1_I1_P1")


def _frame(rows):
    df = pd.DataFrame(rows)
    keys = df["file_id"].map(FileKey.parse)
    df["session_key"] = keys.map(lambda k: k.session_key)
    df["interaction_key"] = keys.map(lambda k: k.interaction_key)
    df["participant_key"] = keys.map(lambda k: k.participant_key)
    df["vendor_id"] = keys.map(lambda k: k.vendor)
    df["session_id"] = keys.map(lambda k: k.session)
    df["interaction_id"] = keys.map(lambda k: k.interaction)
    df["participant_id"] = keys.map(lambda k: k.participant)
    for col, val in [("split", "train"), ("batch_idx", "0"), ("archive_idx", "0"),
                     ("relationship_detail", "stranger"), ("prompt_id_unique", "x"),
                     ("participant_b_prompt_text", None), ("ipc_b", None)]:
        if col not in df:
            df[col] = val
    return df


def _two(sess, inter, a, b, **kw):
    base = dict(relationship="stranger", interaction_type="ipc_conversation",
                participant_a_prompt_text="Tell your partner about a trip.")
    base.update(kw)
    return [dict(file_id=f"V00_S{sess}_I{inter}_P{a}", **base),
            dict(file_id=f"V00_S{sess}_I{inter}_P{b}", **base)]


def test_manner_instructing_prompts_are_flagged():
    df = pd.DataFrame({
        "participant_a_prompt_text": [
            "Speak in a firm, confrontational tone.", "Tell your partner about a trip.",
            "Ask: would you rather speak in rhyme?", "Tell a story."],
        "participant_b_prompt_text": [None, None, None, "Respond."],
        "ipc_b": [None, None, None, "APCM"],
    })
    flags = manner_instructing(df, FRAME_CFG["exclude_prompt_text_regex"], two_sided=True)
    assert flags.tolist() == [True, False, True, True]


def test_frame_applies_every_preregistered_exclusion_and_counts_it():
    rows = (
        _two("0001", "1", "0001", "0002")
        + _two("0002", "1", "0003", "0004", relationship="familiar")
        + _two("0003", "1", "0005", "0006", interaction_type="grounded_gesture")
        + _two("0004", "1", "0007", "0008",
               participant_a_prompt_text="Respond with a measured tone.")
        + [dict(file_id="V00_S0005_I1_P0009", relationship="stranger",
                interaction_type="ipc_conversation", participant_a_prompt_text="Hi")]
    )
    eligible, funnel = apply_frame(_frame(rows), FRAME_CFG)
    assert set(eligible["session_key"]) == {"V00_S0001"}
    assert funnel["sessions"].tolist() == [5, 4, 3, 2, 1]


def test_selection_is_speaker_unique_and_deterministic():
    rows = (_two("0001", "1", "0001", "0002") + _two("0002", "1", "0001", "0003")
            + _two("0003", "1", "0004", "0005") + _two("0004", "1", "0005", "0006")
            + _two("0005", "1", "0007", "0008"))
    eligible, _ = apply_frame(_frame(rows), FRAME_CFG)
    a = speaker_unique_order(eligible, seed=20260912)
    b = speaker_unique_order(eligible, seed=20260912)
    assert a == b
    members = eligible.groupby("session_key")["participant_key"].agg(set)
    seen = set()
    for sk in a:
        assert not (members[sk] & seen), "a speaker appears in two accepted sessions"
        seen |= members[sk]
    # V00_S0005 shares no one, so it is always accepted.
    assert "V00_S0005" in a


def test_manifest_roles_follow_rank_and_carry_no_video():
    rows = [r for i in range(1, 6) for r in _two(f"{i:04d}", "1", f"{2*i:04d}", f"{2*i+1:04d}")]
    eligible, _ = apply_frame(_frame(rows), FRAME_CFG)
    order = speaker_unique_order(eligible, seed=1)
    m = build_manifest(eligible, order, n_pilot=2, n_sample=4, base_url=BASE)
    roles = m.drop_duplicates("session_key").set_index("selection_rank")["role"].to_dict()
    assert roles == {0: "pilot", 1: "pilot", 2: "sample", 3: "sample", 4: "reserve"}
    urls = pd.concat([m["audio_url"], m["transcript_url"], m["vad_url"]])
    assert not urls.str.contains("video|mp4|smplh|movement").any()


def test_failed_sessions_are_replaced_in_rank_order():
    m = pd.DataFrame({"session_key": ["a", "b", "c", "d"], "selection_rank": [0, 1, 2, 3]})
    assert sessions_to_process(m, 2) == ["a", "b"]
    assert sessions_to_process(m, 2, failed_sessions=["a"]) == ["b", "c"]


def test_transcript_reader_handles_both_upstream_formats(tmp_path):
    seg = {"start": 0, "end": 1, "transcript": "hi there",
           "words": [{"word": "hi", "start": 0.0, "end": 0.3},
                     {"word": "there", "start": 0.4, "end": 0.8},
                     {"word": "um"}]}
    plain = tmp_path / "a.jsonl"
    plain.write_text(json.dumps(seg) + "\n", encoding="utf-8")
    packed = tmp_path / "b.jsonl"
    packed.write_text(json.dumps([json.dumps(seg)]), encoding="utf-8")
    for p in (plain, packed):
        w = read_words(p)
        assert w["word"].tolist() == ["hi", "there"]
        assert w.attrs["n_words_untimed"] == 1


def test_vad_reader(tmp_path):
    p = tmp_path / "v.jsonl"
    p.write_text('{"start": 1.0, "end": 2.5}\n{"start": 3.0, "end": 3.2}\n', encoding="utf-8")
    assert read_vad(p).to_numpy().tolist() == [[1.0, 2.5], [3.0, 3.2]]
