"""The single audio pass: fetch, measure, discard.

For each session, one interaction at a time:

1. fetch both participants' WAV files to the runtime's local scratch disk;
2. hash them, so any later re-fetch (for example by the SSL stage) can be
   verified to be the same bytes, since the upstream file server is unversioned;
3. extract frame-level tracks (``src/features/tracks.py``) and channel-QC
   measures (``src/preprocessing/channel_qc.py``);
4. delete the audio.

Outputs go to ``data/processed/audio_pass/<session_key>/``:
``tracks.npz``, ``channel_qc.csv``, ``fetch_log.csv`` and ``DONE.json``, the
last written only after everything else succeeded. A session with ``DONE.json``
is skipped on rerun, so a preempted Colab session resumes where it stopped.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FeatureConfig
from src.data.seamless import fetch, read_vad, read_words
from src.features.tracks import extract_tracks, speech_regions, total_seconds
from src.paths import Paths
from src.preprocessing.channel_qc import channel_measures, objective_failure


def _read_audio(path: Path) -> tuple[np.ndarray | None, dict]:
    import soundfile as sf

    try:
        info = sf.info(str(path))
        x, sr = sf.read(str(path), dtype="float64", always_2d=False)
    except Exception as err:  # corrupt or truncated file
        return None, {"readable": False, "read_error": repr(err)}
    return x, {"readable": True, "sample_rate": int(sr), "channels": int(info.channels),
               "subtype": info.subtype}


def process_interaction(rows: list[dict], root: str, scratch: str, cfg: dict) -> dict:
    """Measure one interaction (two files). Runs in a worker process."""
    p = Paths(Path(root))
    fcfg = FeatureConfig()
    qc_cfg = cfg["channel_qc"]
    by_file = {r["file_id"]: r for r in rows}
    fids = sorted(by_file)
    fetched, audio, meta, logs = {}, {}, {}, []
    for fid in fids:
        dest = Path(scratch) / f"{fid}.wav"
        log = fetch(by_file[fid]["audio_url"], dest)
        log.update(file_id=fid, kind="audio",
                   fetched_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        logs.append(log)
        fetched[fid] = log["status"] != "missing"
        if fetched[fid]:
            audio[fid], meta[fid] = _read_audio(dest)
        else:
            audio[fid], meta[fid] = None, {"readable": False}

    words = {f: read_words(p.raw_transcript / f"{f}.jsonl")
             if (p.raw_transcript / f"{f}.jsonl").exists() else pd.DataFrame(
                 columns=["start", "end", "word", "score"]) for f in fids}
    vads = {f: read_vad(p.raw_vad / f"{f}.jsonl")
            if (p.raw_vad / f"{f}.jsonl").exists() else pd.DataFrame(columns=["start", "end"])
            for f in fids}

    qc_rows, npz = [], {}
    durations = {f: (len(audio[f]) / meta[f]["sample_rate"]) if audio[f] is not None
                 and audio[f].ndim == 1 else None for f in fids}
    for f in fids:
        partner = [g for g in fids if g != f]
        partner = partner[0] if partner else None
        row = {"file_id": f, "session_key": by_file[f]["session_key"],
               "interaction_key": by_file[f]["interaction_key"],
               "participant_key": by_file[f]["participant_key"],
               "vendor_id": by_file[f]["vendor_id"], **meta[f]}
        x = audio[f]
        if x is not None and x.ndim == 1 and meta[f].get("sample_rate"):
            vad_iv = vads[f][["start", "end"]].to_numpy()
            word_iv = words[f][["start", "end"]].to_numpy() if len(words[f]) else None
            segs = speech_regions(vad_iv, word_iv, fcfg)
            row["analysed_speech_s"] = total_seconds(segs)
            row["vad_only_speech_s"] = total_seconds(
                speech_regions(vad_iv, None, fcfg))
            tr = extract_tracks(x, meta[f]["sample_rate"], fcfg, speech_segments=segs)
            npz.update(tr.to_npz_dict(f))
            row.update(channel_measures(x, meta[f]["sample_rate"], tr, vads[f],
                                        vads[partner] if partner else vads[f], words[f]))
        else:
            row.update(vad_speech_s=0.0, n_words_timed=int(len(words[f])))
        row["failure"] = objective_failure(
            exists=fetched[f], readable=meta[f].get("readable", False),
            sample_rate=meta[f].get("sample_rate"), channels=meta[f].get("channels"),
            duration_s=durations[f],
            partner_duration_s=durations.get(partner) if partner else None,
            vad_speech_s=row.get("vad_speech_s", 0.0),
            n_words_timed=row.get("n_words_timed", 0),
            expected_sr=qc_cfg["expected_sample_rate_hz"],
            max_mismatch_s=qc_cfg["max_pair_duration_mismatch_s"])
        row["pair_duration_mismatch_s"] = (
            abs(durations[f] - durations[partner])
            if partner and durations[f] is not None and durations[partner] is not None
            else None)
        qc_rows.append(row)

    for f in fids:  # process-and-discard
        (Path(scratch) / f"{f}.wav").unlink(missing_ok=True)
    return {"qc": qc_rows, "npz": npz, "log": logs}


def done_variant(path: Path) -> str | None:
    """The pitch-segment source a finished session was measured under."""
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("pitch_segment_source", "vad")
    except (OSError, ValueError):
        return None


def process_session(session_rows: pd.DataFrame, paths: Paths, cfg: dict,
                    workers: int = 1) -> str:
    """Run the audio pass for one session.

    Returns ``done``, or ``skipped`` when the session is already measured under
    the configured pitch-segment source. A session measured under a different
    source is reprocessed, so a change of measurement never leaves a mixture of
    variants in one table.
    """
    sk = session_rows["session_key"].iloc[0]
    out = paths.audio_pass / sk
    marker = out / "DONE.json"
    if marker.exists() and done_variant(marker) == FeatureConfig().pitch_segment_source:
        return "skipped"
    out.mkdir(parents=True, exist_ok=True)
    scratch = paths.audio_scratch / sk
    scratch.mkdir(parents=True, exist_ok=True)

    groups = [g.to_dict("records") for _, g in session_rows.groupby("interaction_key")]
    args = [(g, str(paths.root), str(scratch), cfg) for g in groups]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(process_interaction, *zip(*args, strict=False)))
    else:
        results = [process_interaction(*a) for a in args]

    qc = pd.DataFrame([r for res in results for r in res["qc"]])
    log = pd.DataFrame([r for res in results for r in res["log"]])
    npz = {k: v for res in results for k, v in res["npz"].items()}
    np.savez_compressed(out / "tracks.npz", **npz)
    qc.to_csv(out / "channel_qc.csv", index=False)
    log.to_csv(out / "fetch_log.csv", index=False)
    (out / "DONE.json").write_text(json.dumps({
        "session_key": sk, "interactions": len(groups), "files": int(len(qc)),
        "failed_files": int(qc["failure"].notna().sum()),
        "pitch_segment_source": FeatureConfig().pitch_segment_source,
        "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    try:
        os.rmdir(scratch)
    except OSError:
        pass
    return "done"


def collect(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate every finished session's QC rows and fetch log."""
    qc, logs = [], []
    for done in sorted(paths.audio_pass.glob("*/DONE.json")):
        qc.append(pd.read_csv(done.parent / "channel_qc.csv"))
        logs.append(pd.read_csv(done.parent / "fetch_log.csv"))
    if not qc:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(qc, ignore_index=True), pd.concat(logs, ignore_index=True)


def load_tracks(paths: Paths, session_key: str):
    """The tracks archive of one session (lazy; use ``Tracks.from_npz``)."""
    return np.load(paths.audio_pass / session_key / "tracks.npz")


__all__ = ["process_interaction", "process_session", "done_variant", "collect",
           "load_tracks"]
