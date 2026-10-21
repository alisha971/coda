"""Frozen SSL features of partner turns, cached under a storage budget.

Five sources — WavLM layers 3, 6, 9, 12 and HuBERT's final layer — extracted
from the **final 8 seconds** of each turn (≤ 400 frames at 50 Hz), with the true
frame count kept for masking. Turn-final truncation keeps the material
immediately before the reply.

**Leakage properties, enforced here:**

- extraction is a pure function of audio: :func:`extract_turn` takes a waveform
  and an extractor and nothing else — no turn role, label or target (
  SSL-extraction row);
- the encoders are frozen: every parameter has ``requires_grad = False`` and the
  forward pass runs under ``torch.no_grad`` (SSL fine-tuning row);
- the per-source PCA to 128 dimensions is fitted on **training partner turns
  only** (:func:`fit_source_pca` asserts the fit set), then applied everywhere
  (SSL feature PCA row).

**Audio integrity.** The upstream file server is unversioned, so every
re-fetched file is checked against the SHA-256 recorded by the audio pass
(:func:`verify_audio`). A mismatch stops extraction for that file.

**Cache.** fp16, sharded (``shard_turns`` turns per shard), resumable:
completed shards are skipped. About 0.5 MB per turn.

Model checkpoints and revisions are ``Settings.ssl_stack``; the revisions
must be pinned before extraction runs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.features.normstats import FittedStats
from src.models.ssl_layer_mixer import SOURCES

Extractor = Callable[[np.ndarray], dict[str, np.ndarray]]
TARGET_SR = 16000
FRAME_RATE_HZ = 50


class AudioMismatch(RuntimeError):
    """A re-fetched audio file differs from the bytes measured in the audio pass."""


def verify_audio(path: Path, expected_sha256: str) -> None:
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if h != expected_sha256:
        raise AudioMismatch(f"{path}: sha256 {h[:12]} != recorded {expected_sha256[:12]}")


def to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    """Polyphase resampling to 16 kHz (48 kHz → 16 kHz is an exact factor of 3)."""
    from math import gcd

    from scipy.signal import resample_poly

    if sr == TARGET_SR:
        return x.astype(np.float32)
    g = gcd(sr, TARGET_SR)
    return resample_poly(x, TARGET_SR // g, sr // g).astype(np.float32)


def turn_tail(x: np.ndarray, sr: int, start: float, end: float, seconds: float) -> np.ndarray:
    """The final ``seconds`` of the turn [start, end)."""
    a = int(max(start, end - seconds) * sr)
    return x[a:int(end * sr)]


def extract_turn(wave_16k: np.ndarray, extractor: Extractor,
                 max_frames: int = 400) -> tuple[dict[str, np.ndarray], int]:
    """Frame sequences per source, trimmed to the last ``max_frames``; true length."""
    feats = extractor(wave_16k)
    if set(feats) != set(SOURCES):
        raise ValueError(f"extractor returned {sorted(feats)}, expected {list(SOURCES)}")
    T = min(min(v.shape[0] for v in feats.values()), max_frames)
    return {k: v[-T:] for k, v in feats.items()}, int(T)


def load_extractor(ssl_stack: dict, device: str = "cuda", settings=None) -> Extractor:
    """The real frozen WavLM + HuBERT extractor (needs transformers and the weights).

    Refuses while the neural arm is deferred, so no weights are downloaded and no
    real-data SSL features are extracted under the current preregistration.
    """
    from src.settings import Settings

    if (settings or Settings()).neural_arm == "deferred":
        raise RuntimeError("the neural arm is deferred; SSL extraction is not run")
    import torch
    from transformers import AutoFeatureExtractor, HubertModel, WavLMModel

    md27 = ssl_stack
    rev = md27.get("revisions") or {}
    if not rev:
        raise RuntimeError("SSL model revisions must be pinned before extraction")
    wavlm = WavLMModel.from_pretrained(md27["wavlm"], revision=rev["wavlm"]).to(device).eval()
    hubert = HubertModel.from_pretrained(md27["hubert"], revision=rev["hubert"]).to(device).eval()
    for m in (wavlm, hubert):
        for p in m.parameters():
            p.requires_grad_(False)
    # Each checkpoint gets its own feature extractor: input normalisation can
    # differ between checkpoints, and using one model's settings for the other
    # would feed it inputs unlike those it was trained on.
    fe_w = AutoFeatureExtractor.from_pretrained(md27["wavlm"], revision=rev["wavlm"])
    fe_h = AutoFeatureExtractor.from_pretrained(md27["hubert"], revision=rev["hubert"])
    assert_frozen(wavlm)
    assert_frozen(hubert)

    @torch.no_grad()
    def extract(wave: np.ndarray) -> dict[str, np.ndarray]:
        iw = fe_w(wave, sampling_rate=TARGET_SR, return_tensors="pt").input_values.to(device)
        ih = fe_h(wave, sampling_rate=TARGET_SR, return_tensors="pt").input_values.to(device)
        hw = wavlm(iw, output_hidden_states=True).hidden_states
        hh = hubert(ih, output_hidden_states=True).hidden_states
        # hidden_states[0] is the input to the first transformer layer; layer k is index k.
        out = {"wavlm_l3": hw[3], "wavlm_l6": hw[6], "wavlm_l9": hw[9], "wavlm_l12": hw[12],
               "hubert_final": hh[-1]}
        return {k: v[0].float().cpu().numpy() for k, v in out.items()}

    return extract


def assert_frozen(model) -> None:
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if trainable:
        raise AssertionError(f"SSL encoder has trainable parameters: {trainable[:3]}")


def fit_source_pca(frames_by_source: dict[str, np.ndarray], fit_turns: list[str],
                   training_partner_turns: set[str], k: int, split_hash: str,
                   ) -> dict[str, FittedStats]:
    """PCA-128 per source, fitted on frames of training partner turns only."""
    outside = set(fit_turns) - set(training_partner_turns)
    if outside:
        raise AssertionError(f"SSL PCA fit set includes {len(outside)} non-training turns")
    out = {}
    for src, X in frames_by_source.items():
        X = X.astype(np.float64)
        mu = X.mean(axis=0)
        _, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
        var = s ** 2
        out[src] = FittedStats(f"ssl_pca_{src}", {"mean": mu, "components": Vt[:k]},
                               list(fit_turns), split_hash,
                               extra={"k": k, "explained": float(var[:k].sum() / var.sum())})
    return out


def write_shard(path: Path, turn_ids: list[str], feats: list[dict[str, np.ndarray]],
                lengths: list[int], pca: dict[str, FittedStats], max_frames: int = 400) -> None:
    """One fp16 shard: (n, max_frames, 5, 128) frames, right-padded, with true lengths."""
    k = next(iter(pca.values())).params["components"].shape[0]
    arr = np.zeros((len(turn_ids), max_frames, len(SOURCES), k), np.float16)
    for i, f in enumerate(feats):
        for j, src in enumerate(SOURCES):
            st = pca[src]
            z = (f[src] - st.params["mean"]) @ st.params["components"].T
            arr[i, :len(z), j] = z.astype(np.float16)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, turn_ids=np.asarray(turn_ids), frames=arr, lengths=np.asarray(lengths))
    tmp.replace(path)


class SSLCache:
    """Read-only access to the sharded cache by turn id."""

    def __init__(self, cache_dir: Path):
        self.dir = Path(cache_dir)
        index = json.loads((self.dir / "index.json").read_text(encoding="utf-8"))
        self.where = {t: (s, i) for s, ids in index["shards"].items() for i, t in enumerate(ids)}
        self._open: dict[str, np.lib.npyio.NpzFile] = {}

    def get(self, turn_ids) -> tuple[np.ndarray, np.ndarray]:
        frames, lengths = [], []
        for t in turn_ids:
            s, i = self.where[t]
            z = self._open.setdefault(s, np.load(self.dir / s, mmap_mode="r"))
            frames.append(np.asarray(z["frames"][i]))
            lengths.append(int(z["lengths"][i]))
        return np.stack(frames), np.asarray(lengths)


def write_index(cache_dir: Path, shards: dict[str, list[str]], info: dict) -> None:
    (Path(cache_dir) / "index.json").write_text(json.dumps({"shards": shards, **info},
                                                           indent=2), encoding="utf-8")


__all__ = ["AudioMismatch", "verify_audio", "to_16k", "turn_tail", "extract_turn",
           "load_extractor", "assert_frozen", "fit_source_pca", "write_shard", "SSLCache",
           "write_index", "TARGET_SR", "FRAME_RATE_HZ"]
