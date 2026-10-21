"""The Seamless Interaction corpus: sampling frame, selection, manifest, fetch.

Everything that depends on how this corpus is organised lives here, so the rest
of the pipeline sees sessions, interactions, participants and turns rather than
file-naming conventions.

Corpus structure, as verified against the pinned sources
(``configs/corpus.yaml``):

- A **session** is one dyad recorded for about an hour. It is the design's
  *conversation*: the unit of analysis, resampling and splitting.
- An **interaction** is one 2-10 minute stretch of talk on one prompt. Its
  identifier is the prompt's hash, not a chronological index, so the order of
  interactions within a session is not recoverable.
- A **file id** ``V<vendor>_S<session>_I<interaction>_P<participant>`` names one
  participant's audio, transcript and VAD for one interaction. Participant
  identifiers can carry a letter suffix (``P0027A``), which the upstream regex
  misses; the pattern here accepts it.

Selection follows ``PREREGISTRATION.md``: stranger sessions,
``ipc_conversation`` interactions, manner-instructing prompts removed, a seeded
random order, and at most one session per speaker.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FILE_ID_RE = re.compile(r"^V(\d+)_S(\d+)_I(\d+)_P(\d+[A-Za-z]*)$")

#: The only modalities CODA fetches. Video and motion features are never requested.
KINDS: dict[str, str] = {
    "audio": "audio/{file_id}.wav",
    "transcript": "metadata/transcript/{file_id}.jsonl",
    "vad": "metadata/vad/{file_id}.jsonl",
}

METADATA_FILES = ("interactions.csv", "participants.csv", "relationships.csv")

MANIFEST_COLUMNS = [
    "selection_rank", "role", "session_key", "interaction_key", "file_id",
    "vendor_id", "session_id", "interaction_id", "participant_id", "participant_key",
    "split", "batch_idx", "archive_idx", "relationship", "relationship_detail",
    "interaction_type", "prompt_id_unique", "audio_url", "transcript_url", "vad_url",
]


# --------------------------------------------------------------------------
# Identifiers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FileKey:
    vendor: str
    session: str
    interaction: str
    participant: str

    @classmethod
    def parse(cls, file_id: str) -> FileKey:
        m = FILE_ID_RE.match(file_id)
        if not m:
            raise ValueError(f"not a Seamless file id: {file_id!r}")
        return cls(*m.groups())

    @property
    def session_key(self) -> str:
        return f"V{self.vendor}_S{self.session}"

    @property
    def interaction_key(self) -> str:
        return f"V{self.vendor}_S{self.session}_I{self.interaction}"

    @property
    def participant_key(self) -> str:
        """Participants are numbered per vendor, so the vendor qualifies them."""
        return f"V{self.vendor}_P{self.participant}"


def file_url(base_url: str, split: str, kind: str, file_id: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"CODA fetches only {sorted(KINDS)}; got {kind!r}")
    return f"{base_url.rstrip('/')}/naturalistic/{split}/{KINDS[kind].format(file_id=file_id)}"


def metadata_urls(source: dict) -> dict[str, str]:
    """Pinned URLs of the file index and the three metadata tables."""
    hf = (f"https://huggingface.co/datasets/{source['hf_repo']}"
          f"/resolve/{source['hf_revision']}/{{name}}")
    gh = (f"https://raw.githubusercontent.com/{source['github_repo']}"
          f"/{source['github_commit']}/assets/filelist.csv")
    return {"filelist.csv": gh, **{n: hf.format(name=n) for n in METADATA_FILES}}


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path, *, retries: int = 4, timeout: float = 120.0,
          session=None) -> dict:
    """Download ``url`` to ``dest`` atomically; skip if ``dest`` already exists.

    Returns a log row: url, path, bytes, sha256, status (``fetched``,
    ``exists`` or ``missing``), attempts. A 403/404 is ``missing`` rather than
    an exception, because upstream marks some transcripts optional and the
    caller decides what a missing file means.
    """
    import requests

    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return {"url": url, "path": str(dest), "bytes": dest.stat().st_size,
                "sha256": sha256_file(dest), "status": "exists", "attempts": 0}
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    http = session or requests
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with http.get(url, stream=True, timeout=timeout) as r:
                if r.status_code in (403, 404):
                    return {"url": url, "path": str(dest), "bytes": 0, "sha256": None,
                            "status": "missing", "attempts": attempt}
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for block in r.iter_content(1 << 20):
                        fh.write(block)
            # Content-Length is the size on the wire; for a compressed response
            # that is not the size of the decoded file, so only check identity.
            expected = r.headers.get("Content-Length")
            encoded = r.headers.get("Content-Encoding", "identity") != "identity"
            if expected is not None and not encoded and int(expected) != tmp.stat().st_size:
                raise OSError(f"truncated download: {tmp.stat().st_size} of {expected} bytes")
            tmp.replace(dest)
            return {"url": url, "path": str(dest), "bytes": dest.stat().st_size,
                    "sha256": sha256_file(dest), "status": "fetched", "attempts": attempt}
        except Exception as err:  # network errors are retried, then re-raised
            last = err
            tmp.unlink(missing_ok=True)
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"failed after {retries} attempts: {url}") from last


def remote_sizes(urls: list[str], workers: int = 16, timeout: float = 30.0) -> pd.DataFrame:
    """HEAD every URL: whether it exists upstream and its size in bytes.

    Used before any audio is fetched, so the download size is a measurement
    rather than an estimate, and a file missing upstream is known in advance.
    """
    from concurrent.futures import ThreadPoolExecutor

    import requests

    http = requests.Session()
    http.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=workers))

    def one(url: str) -> dict:
        for attempt in range(3):
            try:
                r = http.head(url, timeout=timeout, allow_redirects=True)
                size = r.headers.get("Content-Length")
                return {"url": url, "status_code": r.status_code,
                        "exists": r.status_code == 200,
                        "bytes": int(size) if size is not None else None}
            except Exception:  # retried, then reported as unknown
                time.sleep(2 ** attempt)
        return {"url": url, "status_code": None, "exists": None, "bytes": None}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return pd.DataFrame(list(ex.map(one, urls)))


# --------------------------------------------------------------------------
# Sampling frame and selection
# --------------------------------------------------------------------------


def load_frame(meta_dir: Path) -> pd.DataFrame:
    """One row per naturalistic file, joined to prompt and relationship metadata."""
    fl = pd.read_csv(meta_dir / "filelist.csv", dtype=str)
    fl = fl[fl["label"] == "naturalistic"].copy()
    keys = fl["file_id"].map(FileKey.parse)
    fl["vendor_id"] = keys.map(lambda k: k.vendor)
    fl["session_id"] = keys.map(lambda k: k.session)
    fl["interaction_id"] = keys.map(lambda k: k.interaction)
    fl["participant_id"] = keys.map(lambda k: k.participant)
    fl["session_key"] = keys.map(lambda k: k.session_key)
    fl["interaction_key"] = keys.map(lambda k: k.interaction_key)
    fl["participant_key"] = keys.map(lambda k: k.participant_key)

    it = pd.read_csv(meta_dir / "interactions.csv", dtype=str)
    it = it[["prompt_hash", "prompt_id_unique", "participant_a_prompt_text",
             "participant_b_prompt_text", "ipc_a", "ipc_b", "interaction_type"]]
    fl = fl.merge(it, left_on="interaction_id", right_on="prompt_hash", how="left",
                  validate="many_to_one")

    rel = pd.read_csv(meta_dir / "relationships.csv", dtype=str)
    rel["vendor_id"] = rel["vendor_id"].str.lstrip("V")
    rel["session_id"] = rel["session_id"].str.zfill(4)
    rel = rel.drop_duplicates(["vendor_id", "session_id"])
    fl = fl.merge(rel, on=["vendor_id", "session_id"], how="left", validate="many_to_one")
    return fl


def manner_instructing(frame: pd.DataFrame, text_regex: str, two_sided: bool) -> pd.Series:
    """True where the prompt directs how to speak (PREREGISTRATION)."""
    text = (frame["participant_a_prompt_text"].fillna("") + " "
            + frame["participant_b_prompt_text"].fillna(""))
    flag = text.str.contains(text_regex, case=False, regex=True)
    if two_sided:
        flag |= frame["ipc_b"].notna()
    return flag


def apply_frame(frame: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the sampling frame in its preregistered order.

    Returns the eligible files and a funnel table (stage, sessions,
    interactions, files) so every exclusion is counted.
    """
    rows = []

    def record(stage: str, df: pd.DataFrame) -> None:
        rows.append({"stage": stage, "sessions": df["session_key"].nunique(),
                     "interactions": df["interaction_key"].nunique(), "files": len(df)})

    df = frame
    record("naturalistic, all", df)
    df = df[df["relationship"] == cfg["relationship"]]
    record(f"relationship == {cfg['relationship']}", df)
    df = df[df["interaction_type"].isin(cfg["interaction_types"])]
    record("interaction type in " + ", ".join(cfg["interaction_types"]), df)
    df = df[~manner_instructing(df, cfg["exclude_prompt_text_regex"],
                                cfg["exclude_two_sided_role_prompts"])]
    record("manner-instructing prompts removed", df)
    # An interaction is usable only with both participants' files present.
    both = df.groupby("interaction_key")["participant_key"].transform("nunique") == 2
    df = df[both]
    record("both participants present", df)
    return df.copy(), pd.DataFrame(rows)


def speaker_unique_order(eligible: pd.DataFrame, seed: int) -> list[str]:
    """Seeded random order of sessions, keeping a session only if both of its
    participants are new. Each speaker then contributes exactly one dyad, so a
    split by session is automatically speaker-disjoint.
    """
    members = (eligible.groupby("session_key")["participant_key"]
               .agg(lambda s: tuple(sorted(set(s)))))
    bad = members[members.map(len) != 2]
    if len(bad):
        raise ValueError(f"sessions without exactly two participants: {list(bad.index[:5])}")
    order = np.random.default_rng(seed).permutation(sorted(members.index))
    used: set[str] = set()
    accepted: list[str] = []
    for sk in order:
        a, b = members[sk]
        if a in used or b in used:
            continue
        used.update((a, b))
        accepted.append(str(sk))
    return accepted


def speaker_components(files: pd.DataFrame) -> pd.Series:
    """Connected component of each session in the speaker graph.

    Two sessions are linked when they share a participant. Speaker-disjoint
    splitting must keep a whole component in one split, so the size of the
    largest component shows how much the corpus's repeat participation
    constrains splitting before the one-session-per-speaker rule is applied.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    members = files.groupby("session_key")["participant_key"].agg(lambda s: sorted(set(s)))
    for ps in members:
        for q in ps[1:]:
            parent[find(q)] = find(ps[0])
        find(ps[0])
    return members.map(lambda ps: find(ps[0])).rename("component")


def build_manifest(eligible: pd.DataFrame, order: list[str], n_pilot: int,
                   n_sample: int, base_url: str) -> pd.DataFrame:
    """One row per file for every accepted session, in selection order."""
    rank = {sk: r for r, sk in enumerate(order)}
    m = eligible[eligible["session_key"].isin(rank)].copy()
    m["selection_rank"] = m["session_key"].map(rank).astype(int)
    m["role"] = np.select(
        [m["selection_rank"] < n_pilot, m["selection_rank"] < n_sample],
        ["pilot", "sample"], default="reserve",
    )
    for kind in KINDS:
        m[f"{kind}_url"] = [file_url(base_url, s, kind, f)
                            for s, f in zip(m["split"], m["file_id"], strict=False)]
    m = m.sort_values(["selection_rank", "interaction_id", "participant_id"])
    return m[MANIFEST_COLUMNS].reset_index(drop=True)


def select_stage(manifest: pd.DataFrame, stage: str) -> pd.DataFrame:
    """Rows to process for a stage: ``pilot`` or ``sample`` (which includes the pilot).

    The reserve is consumed in rank order only when sessions fail channel QC;
    see :func:`sessions_to_process`.
    """
    if stage == "pilot":
        return manifest[manifest["role"] == "pilot"]
    if stage == "sample":
        return manifest[manifest["role"].isin(["pilot", "sample"])]
    raise ValueError(f"unknown stage {stage!r}")


def sessions_to_process(manifest: pd.DataFrame, target: int,
                        failed_sessions: Iterable[str] = ()) -> list[str]:
    """First ``target`` sessions in rank order, skipping ones that failed QC.

    This is "selected at random among those passing channel QC": walking
    a random order and replacing failures with the next session in that order.
    """
    failed = set(failed_sessions)
    ranked = (manifest.drop_duplicates("session_key")
              .sort_values("selection_rank")["session_key"].tolist())
    return [s for s in ranked if s not in failed][:target]


# --------------------------------------------------------------------------
# Reading transcripts and VAD
# --------------------------------------------------------------------------


def _jsonl_records(path: Path) -> list[dict]:
    """Records from a per-file JSONL, or from the tar-packed variant, which is a
    JSON list of JSONL strings. Both occur upstream."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        blobs = json.loads(text)
        lines = [ln for b in blobs for ln in str(b).splitlines()]
    else:
        lines = text.splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()]


def read_words(path: Path) -> pd.DataFrame:
    """Timed words: start, end (seconds from interaction start), word, score.

    Words without timestamps (about 0.2% upstream) are dropped here and counted
    by the caller from ``n_words_untimed``.
    """
    rows, untimed = [], 0
    for seg in _jsonl_records(path):
        for w in seg.get("words", []) or []:
            s, e = w.get("start"), w.get("end")
            if s is None or e is None:
                untimed += 1
                continue
            rows.append((float(s), float(e), str(w.get("word", "")), w.get("score")))
    df = pd.DataFrame(rows, columns=["start", "end", "word", "score"])
    df = df.sort_values("start", kind="stable").reset_index(drop=True)
    df.attrs["n_words_untimed"] = untimed
    return df


def read_vad(path: Path) -> pd.DataFrame:
    """Speech segments from Silero VAD: start, end in seconds."""
    recs = _jsonl_records(path)
    return pd.DataFrame([(float(r["start"]), float(r["end"])) for r in recs],
                        columns=["start", "end"])


__all__ = [
    "FILE_ID_RE", "KINDS", "METADATA_FILES", "MANIFEST_COLUMNS", "FileKey",
    "file_url", "metadata_urls", "sha256_file", "fetch", "load_frame",
    "remote_sizes", "manner_instructing", "apply_frame", "speaker_components",
    "speaker_unique_order",
    "build_manifest", "select_stage", "sessions_to_process", "read_words", "read_vad",
]
