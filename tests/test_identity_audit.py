"""Participant-identity screen."""

import pandas as pd

from src.data.identity import as_split_links, candidate_links, profile_match_null

U = "Undisclosed"


def _fixture():
    rows = [  # role, session, vendor, pid
        ("pilot", "V00_S1", "00", "0150"), ("pilot", "V00_S1", "00", "0200"),
        ("sample", "V00_S2", "00", "0150A"), ("sample", "V00_S2", "00", "0300"),
        ("sample", "V00_S3", "00", "0400"), ("sample", "V00_S3", "00", "0500"),
        ("sample", "V01_S4", "01", "0150"), ("sample", "V01_S4", "01", "0600"),
        ("reserve", "V00_S9", "00", "0400A"), ("reserve", "V00_S9", "00", "0901"),
    ]
    m = pd.DataFrame(rows, columns=["role", "session_key", "vendor_id", "participant_id"])
    m["participant_key"] = "V" + m["vendor_id"] + "_P" + m["participant_id"]
    prof = {"0200": ("3", "3", "3", "3", "3"), "0400": ("3", "3", "3", "3", "3"),
            "0300": ("2", "4", "1", "5", "2"), "0500": (U,) * 5}
    p = pd.DataFrame([dict(zip(["extraversion_raw", "agreeableness_raw",
                                "conscientiousness_raw", "neuroticism_raw", "openness_raw"],
                               prof.get(pid, ("1", "1", "1", "1", pid[-1])), strict=False),
                           vendor_id=v, participant_id=pid)
                      for v, pid in set(zip(m["vendor_id"], m["participant_id"], strict=False))])
    return m, p


def test_suffix_and_profile_rules_flag_only_cross_session_same_vendor_candidates():
    m, p = _fixture()
    links = candidate_links(m, p)
    got = {(r.rule, frozenset({r.participant_key_a, r.participant_key_b}))
           for r in links.itertuples()}
    assert ("L1_suffix", frozenset({"V00_P0150", "V00_P0150A"})) in got
    assert ("L2_trait_profile", frozenset({"V00_P0200", "V00_P0400"})) in got
    # other vendor's 0150, the reserve 0400A and undisclosed profiles are never linked
    keys = set().union(*[k for _, k in got])
    assert "V01_P0150" not in keys and "V00_P0400A" not in keys and "V00_P0500" not in keys
    assert {(a, b) for a, b, _ in as_split_links(links)} == {("V00_S1", "V00_S2"),
                                                            ("V00_S1", "V00_S3")}


def test_profile_null_counts_chance_matches_and_is_seeded():
    m, p = _fixture()
    a = profile_match_null(m, p, n_perm=50, seed=1)
    b = profile_match_null(m, p, n_perm=50, seed=1)
    assert a == b and a["observed"] == 1 and a["disclosed_participants"] == 7
