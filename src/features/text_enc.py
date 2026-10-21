"""Frozen sentence embeddings of turn text, and PCA-64 fitted on training only.

**Encoder.** ``sentence-transformers/all-MiniLM-L6-v2`` (configs/features.yaml),
a MiniLM model (Wang et al. 2020) trained with the Sentence-BERT recipe
(Reimers & Gurevych 2019), giving 384-dimensional embeddings. It is frozen and
never trained: :func:`load_minilm` puts it in inference mode and every call
runs without gradients.

**Cache.** Turn texts are embedded once, in shards of ``shard_size`` turns
written atomically with their turn ids and the model identity. A rerun skips
every completed shard, so a preempted Colab session resumes where it stopped.

**PCA-64.** Reply text and partner text are reduced separately, each
with its own PCA fitted on the training pairs' embeddings only and applied
unchanged to validation and test (representation normalisation). The
fit-index hash is stored and asserted.

**Edge cases.** An empty or whitespace-only text is still encoded (as the
empty string) and flagged; the analysis set excludes pairs whose reply or
partner text is empty.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.normstats import FittedStats

Encoder = Callable[[list[str]], np.ndarray]


def load_minilm(model_name: str, revision: str | None = None, device: str | None = None):
    """The frozen encoder and a record of exactly which model was loaded."""
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, revision=revision, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def encode(texts: list[str]) -> np.ndarray:
        with torch.no_grad():
            return np.asarray(model.encode(texts, batch_size=128, show_progress_bar=False,
                                           convert_to_numpy=True), dtype=np.float32)

    import sentence_transformers

    dim = (model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension")
           else model.get_sentence_embedding_dimension())
    info = {"model": model_name, "revision_requested": revision,
            "library_version": sentence_transformers.__version__,
            "dimension": int(dim),
            "trainable_parameters": int(sum(p.numel() for p in model.parameters()
                                            if p.requires_grad))}
    return encode, info


def _shard_name(k: int) -> str:
    return f"shard_{k:05d}.npz"


def encode_cached(turn_ids: list[str], texts: list[str], encoder: Encoder, cache_dir: Path,
                  model_info: dict, shard_size: int = 2000) -> pd.DataFrame:
    """Embeddings for every turn, read from or written to the sharded cache.

    Order is fixed by sorting turn ids, so shard contents do not depend on the
    order of the request. Returns a frame indexed by turn id with columns
    ``e0..e{d-1}`` and ``text_empty``.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    order = sorted(set(map(str, turn_ids)))
    text_of = dict(zip(map(str, turn_ids), texts, strict=True))
    manifest_path = cache_dir / "manifest.json"
    manifest = {"model_info": model_info, "shard_size": shard_size, "n_turns": len(order),
                "id_hash": hashlib.sha256("\n".join(order).encode()).hexdigest()[:12]}
    old = None
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["model_info"].get("model") != model_info.get("model") or \
                old["shard_size"] != shard_size or old["id_hash"] != manifest["id_hash"]:
            raise RuntimeError(f"{cache_dir} holds a different cache; use a new directory")
    else:
        # Written once, by the run that builds the cache; later readers keep it as the
        # record of which encoder set-up produced the embeddings.
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    built_with = (old or manifest)["model_info"]
    same_setup = all(built_with.get(k) == model_info.get(k)
                     for k in ("model", "revision_requested", "library_version"))
    parts = []
    for k, start in enumerate(range(0, len(order), shard_size)):
        ids = order[start:start + shard_size]
        path = cache_dir / _shard_name(k)
        if path.exists():
            z = np.load(path)
            if list(z["turn_ids"]) != ids:
                raise RuntimeError(f"{path} does not hold the expected turns")
            emb = z["emb"]
        else:
            if not same_setup:
                raise RuntimeError(
                    f"{cache_dir} was built with {built_with}; refusing to add shards encoded "
                    f"with {model_info} (embeddings must come from one encoder set-up)")
            batch = [(text_of[i] or "").strip() for i in ids]
            emb = np.asarray(encoder(batch), dtype=np.float32)
            if emb.shape != (len(ids), emb.shape[1]):
                raise RuntimeError("encoder returned the wrong number of rows")
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, turn_ids=np.asarray(ids), emb=emb)
            tmp.replace(path)
        parts.append(pd.DataFrame(emb, index=ids, columns=[f"e{j}" for j in range(emb.shape[1])]))
    out = pd.concat(parts)
    out["text_empty"] = [not (text_of[i] or "").strip() for i in out.index]
    return out


def fit_pca(emb: np.ndarray, fit_rows: np.ndarray, ids, split_hash: str, k: int,
            name: str) -> FittedStats:
    """PCA by SVD on the fitting rows only."""
    X = emb[fit_rows].astype(np.float64)
    mu = X.mean(axis=0)
    _, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    if Vt.shape[0] < k:
        raise ValueError(f"cannot keep {k} components from {Vt.shape[0]}")
    var = s ** 2 / (len(X) - 1)
    return FittedStats(name, {"mean": mu, "components": Vt[:k],
                              "explained_variance_ratio": (var[:k] / var.sum())},
                       [str(i) for i in np.asarray(ids)[fit_rows]], split_hash,
                       extra={"k": k, "explained_total": float(var[:k].sum() / var.sum())})


def apply_pca(stats: FittedStats, emb: np.ndarray) -> np.ndarray:
    return (emb - stats.params["mean"]) @ stats.params["components"].T


__all__ = ["load_minilm", "encode_cached", "fit_pca", "apply_pca"]
