"""The notebook-07 split stage end to end on a synthetic project root.

It checks that identity-candidate links from participant metadata are applied
on top of the recorded-ID constraint, that the result is disjoint, and that the
report records the links. No corpus data is read.
"""

import json
import shutil
from pathlib import Path

import pandas as pd

from src.data.splits import assert_disjoint
from src.experiments.stages import splits
from src.paths import Paths
from src.settings import Settings

REPO = Path(__file__).resolve().parents[1]
DOMAINS = ["extraversion_raw", "agreeableness_raw", "conscientiousness_raw",
           "neuroticism_raw", "openness_raw"]


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "coda"
    shutil.copytree(REPO / "configs", root / "configs")
    rows, people, sel = [], [], []
    for i in range(30):
        sk = f"V03_S{i:04d}"
        a, b = f"{2 * i:04d}", f"{2 * i + 1:04d}"
        sel.append({"session_key": sk, "usable": True})
        for pid in (a, b):
            fid = f"{sk}_I00000001_P{pid}"
            rows.append({"role": "pilot", "session_key": sk, "vendor_id": "03",
                         "participant_id": pid, "participant_key": f"V03_P{pid}",
                         "file_id": fid})
            prof = [str(1 + (2 * i + int(pid)) % 7)] * 4 + [pid]   # unique by default
            people.append(dict(zip(DOMAINS, prof, strict=True), vendor_id="03",
                               participant_id=pid))
            t = root / "data" / "raw" / "transcript" / f"{fid}.jsonl"
            t.parent.mkdir(parents=True, exist_ok=True)
            words = [{"start": k, "end": k + 0.5, "word": f"w{i}_{pid}_{k}"} for k in range(30)]
            t.write_text(json.dumps({"words": words}) + "\n", encoding="utf-8")
    # one person under two IDs, linked by the suffix rule: session 0's "0000" and
    # session 7's "0000A"
    rows[14]["participant_id"], rows[14]["participant_key"] = "0000A", "V03_P0000A"
    people[14]["participant_id"] = "0000A"
    (root / "manifests").mkdir(parents=True)
    pd.DataFrame(rows).to_csv(root / "manifests" / "coda_download_manifest.csv", index=False)
    meta = root / "data" / "raw" / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(people).to_csv(meta / "participants.csv", index=False)
    acq = root / "results" / "acquisition"
    acq.mkdir(parents=True)
    pd.DataFrame(sel).to_csv(acq / "selected_sessions_pilot.csv", index=False)
    return root


def test_split_stage_applies_identity_links_and_stays_disjoint(tmp_path):
    root = _root(tmp_path)
    m, rep = splits(Paths(root), "pilot", "speaker_disjoint", Settings())
    assert len(m) == 30 and set(m["split"]) == {"train", "val", "test"}
    links = rep["identity_candidate_links"]
    assert [lk[:2] for lk in links] == [["V03_S0000", "V03_S0007"]]
    part = m.set_index("session_key")["split"]
    assert part["V03_S0000"] == part["V03_S0007"]            # grouped into one partition
    man = pd.read_csv(root / "manifests" / "coda_download_manifest.csv", dtype=str)
    sess = man[["session_key", "participant_key"]].drop_duplicates()
    assert_disjoint(m, sess, [tuple(lk[:2]) + (1.0,) for lk in links])
    assert rep["near_duplicate_links"] == []
    assert (root / "manifests" / "splits").exists()


def test_identity_links_to_sessions_outside_the_stage_do_not_break_the_split(tmp_path):
    """Regression (notebook 07 on the pilot): a link from a pilot session to a sample session
    raised SplitLeakage. Links leaving the stage are dropped for that stage; two stage sessions
    joined through an outside session are still grouped."""
    from src.data.identity import stage_links

    links = [("V03_S0001", "V09_OUT1", 1.0),                 # leaves the stage
             ("V03_S0002", "V09_OUT2", 1.0), ("V09_OUT2", "V03_S0005", 1.0),  # joined outside
             ("V03_S0003", "V03_S0004", 1.0)]                 # inside the stage
    stage = {f"V03_S{i:04d}" for i in range(30)}
    got = stage_links(links, stage)
    assert {frozenset(x[:2]) for x in got} == {frozenset({"V03_S0002", "V03_S0005"}),
                                               frozenset({"V03_S0003", "V03_S0004"})}

    root = _root(tmp_path)
    man = pd.read_csv(root / "manifests" / "coda_download_manifest.csv", dtype=str)
    extra = man[man["session_key"] == "V03_S0003"].copy()
    extra["session_key"] = "V03_S0999"                       # a session outside the stage
    extra["role"] = "sample"
    extra["participant_id"] = ["0006A", "0999"]              # suffix-links to S0003's 0006
    extra["participant_key"] = "V03_P" + extra["participant_id"]
    pd.concat([man, extra]).to_csv(root / "manifests" / "coda_download_manifest.csv",
                                   index=False)
    meta = root / "data" / "raw" / "metadata" / "participants.csv"
    people = pd.read_csv(meta, dtype=str)
    new = people[people["participant_id"].isin(["0006", "0007"])].copy()
    new["participant_id"] = ["0006A", "0999"]
    pd.concat([people, new]).to_csv(meta, index=False)
    m, rep = splits(Paths(root), "pilot", "speaker_disjoint", Settings())   # no raise
    assert len(m) == 30
    assert all(a in stage and b in stage for a, b, _ in rep["identity_candidate_links"])
