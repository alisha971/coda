"""Transcript and VAD acquisition, and turn tables for a stage of the manifest.

Text and VAD files are a few kilobytes each and are kept on Drive under
``data/raw/``; unlike audio they are needed repeatedly (turns, text
embeddings, function labelling). Fetching is threaded because it is dominated
by request latency, and it skips files already present, so reruns are cheap.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd

from src.data.pairs import build_pairs
from src.data.seamless import fetch, read_words
from src.data.turns import build_turns
from src.paths import Paths


def fetch_text_files(rows: pd.DataFrame, paths: Paths, workers: int = 16) -> pd.DataFrame:
    """Fetch transcript and VAD for every file in ``rows``. Returns the fetch log."""
    import requests

    jobs = []
    for r in rows.itertuples(index=False):
        jobs.append((r.transcript_url, paths.raw_transcript / f"{r.file_id}.jsonl",
                     r.file_id, "transcript"))
        jobs.append((r.vad_url, paths.raw_vad / f"{r.file_id}.jsonl", r.file_id, "vad"))

    http = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_maxsize=workers)
    http.mount("https://", adapter)

    def one(job):
        url, dest, fid, kind = job
        log = fetch(url, dest, session=http)
        log.update(file_id=fid, kind=kind,
                   fetched_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        return log

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return pd.DataFrame(list(ex.map(one, jobs)))


def interval_overlap_s(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Total time covered by both interval sets (each: start, end columns)."""
    if a.empty or b.empty:
        return 0.0
    ia = a[["start", "end"]].to_numpy()
    ib = b[["start", "end"]].to_numpy()
    total = 0.0
    j = 0
    ia = ia[ia[:, 0].argsort()]
    ib = ib[ib[:, 0].argsort()]
    for s, e in ia:
        while j < len(ib) and ib[j, 1] <= s:
            j += 1
        k = j
        while k < len(ib) and ib[k, 0] < e:
            total += max(0.0, min(e, ib[k, 1]) - max(s, ib[k, 0]))
            k += 1
    return float(total)


def transcript_vad_agreement(rows: pd.DataFrame, paths: Paths) -> pd.DataFrame:
    """Per file: how much word time falls inside the speaker's own VAD speech.

    Words come from one system (WhisperX) and VAD from another (Silero). Low
    agreement flags a misaligned transcript, which would put turn boundaries,
    and therefore every turn-level measurement, in the wrong place.
    """
    from src.data.seamless import read_vad

    out = []
    for r in rows.itertuples(index=False):
        tp, vp = paths.raw_transcript / f"{r.file_id}.jsonl", paths.raw_vad / f"{r.file_id}.jsonl"
        if not (tp.exists() and vp.exists()):
            continue
        w, v = read_words(tp), read_vad(vp)
        word_s = float((w["end"] - w["start"]).clip(lower=0).sum())
        out.append({"file_id": r.file_id, "session_key": r.session_key,
                    "vendor_id": r.vendor_id, "word_time_s": word_s,
                    "vad_time_s": float((v["end"] - v["start"]).sum()),
                    "words_inside_vad": interval_overlap_s(w, v) / word_s if word_s else None})
    return pd.DataFrame(out)


def build_turn_tables(rows: pd.DataFrame, paths: Paths,
                      n2_min_distance: int = 10) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Turns, candidate pairs and per-file word statistics for ``rows``.

    An interaction is skipped (and reported in the file statistics) when either
    participant's transcript is missing; it cannot yield pairs.
    """
    turns, stats = [], []
    for ik, g in rows.groupby("interaction_key", sort=True):
        words_by_speaker, complete = {}, True
        for r in g.itertuples(index=False):
            path = paths.raw_transcript / f"{r.file_id}.jsonl"
            if not path.exists():
                complete = False
                stats.append({"file_id": r.file_id, "interaction_key": ik,
                              "transcript": "missing", "n_words_timed": 0, "n_words_untimed": 0})
                continue
            w = read_words(path)
            words_by_speaker[r.participant_key] = w
            stats.append({"file_id": r.file_id, "interaction_key": ik, "transcript": "present",
                          "n_words_timed": len(w), "n_words_untimed": w.attrs["n_words_untimed"]})
        if complete and len(words_by_speaker) == 2:
            turns.append(build_turns(words_by_speaker, ik, g["session_key"].iloc[0]))
    turns_df = pd.concat(turns, ignore_index=True) if turns else pd.DataFrame()
    pairs_df = build_pairs(turns_df, n2_min_distance) if len(turns_df) else pd.DataFrame()
    return turns_df, pairs_df, pd.DataFrame(stats)


__all__ = ["fetch_text_files", "interval_overlap_s", "transcript_vad_agreement",
           "build_turn_tables"]
