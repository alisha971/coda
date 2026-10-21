"""File-level stage functions the notebooks call (one per notebook stage).

Every function reads the outputs of earlier notebooks under ``data/processed/``
and writes its own, named with the **stage** (``pilot`` or ``sample``) and the
**measurement signature** (``src.settings.measurement_signature``), so the
pilot and the sample, or two measurement variants, can never share or overwrite
a file. Expensive work is skipped when its output already exists for the same
tag; nothing here downloads audio.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import FeatureConfig
from src.paths import Paths
from src.settings import Settings, measurement_signature


def tag(stage: str, settings: Settings | None = None) -> str:
    return f"{stage}__m{measurement_signature(settings)}"


def stage_sessions(paths: Paths, stage: str) -> list[str]:
    """Sessions of a stage that passed channel QC, in selection order."""
    sel = paths.results / "acquisition" / f"selected_sessions_{stage}.csv"
    if not sel.exists():
        raise FileNotFoundError(f"{sel} missing: run notebook 03 for stage {stage!r}")
    s = pd.read_csv(sel)
    if "usable" in s.columns:
        s = s[s["usable"].astype(bool)]
    return s["session_key"].tolist()


def prosody_targets(paths: Paths, stage: str, settings: Settings | None = None,
                    overwrite: bool = False) -> pd.DataFrame:
    """The eight dimensions for every non-embedded turn of the stage's sessions."""
    from src.features.prosody import ProsodySettings, turn_prosody_from_paths

    settings = settings or Settings()
    out = paths.processed / f"turn_prosody__{tag(stage, settings)}.parquet"
    if out.exists() and not overwrite:
        return pd.read_parquet(out)
    turns = pd.read_parquet(paths.processed / "turns.parquet")
    turns = turns[turns["session_key"].isin(stage_sessions(paths, stage))]
    pro = turn_prosody_from_paths(turns, paths,
                                  ProsodySettings.from_config(FeatureConfig(), settings))
    pro.to_parquet(out, index=False)
    return pro


def loudness_offsets(paths: Paths, stage: str) -> pd.Series:
    """Per-file loudness offset from the stored tracks."""
    from src.features.prosody import file_loudness_offset
    from src.features.tracks import Tracks

    out = {}
    for sk in stage_sessions(paths, stage):
        path = paths.audio_pass / sk / "tracks.npz"
        if not path.exists():
            continue
        npz = np.load(path)
        for fid in {k.rsplit("__", 1)[0] for k in npz.files}:
            out[fid] = file_loudness_offset(Tracks.from_npz(npz, fid))
    return pd.Series(out, name="loudness_offset")


def contrast_enabled(settings: Settings | None = None) -> bool:
    """False when the settings drop the dyad-disjoint contrast."""
    return (settings or Settings()).dyad_disjoint_contrast != "drop"


def splits(paths: Paths, stage: str, scheme: str, settings: Settings | None = None,
           ) -> tuple[pd.DataFrame, dict]:
    """The split manifest for a stage (tracked under manifests/splits/)."""
    settings = settings or Settings()
    if scheme == "dyad_disjoint" and not contrast_enabled(settings):
        raise ValueError("the dyad-disjoint contrast is dropped: with one session per "
                         "speaker it is a second random speaker-disjoint partition")
    from src.data.seamless import read_words
    from src.data.splits import make_split, near_duplicate_links, split_report

    cfg = paths.load_config("splits")
    sessions_ok = stage_sessions(paths, stage)
    manifest = pd.read_csv(paths.manifest, dtype=str)
    sess = manifest[manifest["session_key"].isin(sessions_ok)][["session_key", "participant_key"]] \
        .drop_duplicates()
    dup = settings.near_duplicate_rule
    text = {}
    for sk, g in manifest[manifest["session_key"].isin(sessions_ok)].groupby("session_key"):
        words = [read_words(paths.raw_transcript / f"{f}.jsonl")["word"].astype(str)
                 for f in g["file_id"] if (paths.raw_transcript / f"{f}.jsonl").exists()]
        text[sk] = " ".join(" ".join(w) for w in words)
    links = near_duplicate_links(text, int(dup["shingle_words"]), float(dup["jaccard_threshold"]))
    # Candidate same-person links from participant metadata (identity audit
    # protocol): grouped like near-duplicates, nothing excluded. The screen covers
    # the whole analysis set; links are restated for this stage's sessions, so a
    # link to a session outside the stage is dropped here, while stage sessions
    # joined through outside sessions stay grouped.
    from src.data.identity import as_split_links, candidate_links, stage_links

    id_links = stage_links(as_split_links(candidate_links(
        manifest, pd.read_csv(paths.raw_metadata / "participants.csv", dtype=str))),
        sessions_ok)
    links = links + id_links
    block = cfg["primary"] if scheme == "speaker_disjoint" else cfg["contrast"]
    fr = {k: cfg["primary"][k] for k in ("train", "val", "test")}
    m = make_split(sess, fr, int(block["seed"]), scheme, links)
    rep = split_report(m, sess)
    rep["near_duplicate_links"] = [list(map(str, x)) for x in links if x not in id_links]
    rep["identity_candidate_links"] = [list(map(str, x)) for x in id_links]
    out_dir = paths.manifests / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"split__{stage}__{scheme}__{rep['manifest_hash']}.csv"
    if not path.exists():
        m.to_csv(path, index=False)
        (path.with_suffix(".json")).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return m, rep


def text_embeddings(paths: Paths, stage: str, encoder=None, model_info=None,
                    shard_size: int = 2000) -> pd.DataFrame:
    """Frozen MiniLM embeddings for every non-embedded turn (sharded cache on Drive)."""
    from src.features.text_enc import encode_cached, load_minilm

    turns = pd.read_parquet(paths.processed / "turns.parquet")
    turns = turns[(~turns["embedded"]) & turns["session_key"].isin(stage_sessions(paths, stage))]
    if encoder is None:
        fcfg = paths.load_config("features")["text"]
        encoder, model_info = load_minilm(fcfg["encoder"], fcfg.get("revision"))
    return encode_cached(turns["turn_id"].tolist(), turns["text"].fillna("").tolist(), encoder,
                         paths.processed / "text_cache", model_info, shard_size)


def analysis_data(paths: Paths, stage: str, scheme: str = "speaker_disjoint",
                  settings: Settings | None = None, text_emb: pd.DataFrame | None = None):
    """Everything assembled into :class:`AnalysisData` for the Layer-1 drivers."""
    from src.experiments.pipeline import prepare_analysis

    settings = settings or Settings()
    turns = pd.read_parquet(paths.processed / "turns.parquet")
    pairs = pd.read_parquet(paths.processed / "pairs_candidates.parquet")
    keep = set(stage_sessions(paths, stage))
    turns, pairs = turns[turns["session_key"].isin(keep)], pairs[pairs["session_key"].isin(keep)]
    prosody = prosody_targets(paths, stage, settings)
    split, _ = splits(paths, stage, scheme, settings)
    emb = text_emb if text_emb is not None else text_embeddings(paths, stage)
    loud = loudness_offsets(paths, stage)
    data = prepare_analysis(turns, pairs, prosody, split, emb, loud,
                            int(settings.min_turns_for_speaker_stats))
    return data, turns, prosody, split


__all__ = ["tag", "stage_sessions", "prosody_targets", "loudness_offsets", "splits",
           "text_embeddings", "analysis_data"]
