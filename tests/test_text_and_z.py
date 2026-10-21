"""Text cache, PCA fitted on training rows only, and the conditioning set."""

import numpy as np
import pandas as pd
import pytest

from src.features.conditioning import replier_indicators
from src.features.normstats import ProvenanceError
from src.features.text_enc import apply_pca, encode_cached, fit_pca


class FakeEncoder:
    """Deterministic stand-in for MiniLM: hashes words into 384 dimensions."""

    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        out = np.zeros((len(texts), 384), np.float32)
        for i, t in enumerate(texts):
            for w in t.split():
                out[i, hash(w) % 384] += 1.0
        return out


def test_cache_is_sharded_and_resumes_without_re_encoding(tmp_path):
    ids = [f"t{i:03d}" for i in range(25)]
    texts = [f"word{i} other" for i in range(25)]
    enc = FakeEncoder()
    a = encode_cached(ids, texts, enc, tmp_path, {"model": "fake"}, shard_size=10)
    assert enc.calls == 3 and len(list(tmp_path.glob("shard_*.npz"))) == 3
    (tmp_path / "shard_00001.npz").unlink()                 # a preempted run
    enc2 = FakeEncoder()
    b = encode_cached(list(reversed(ids)), list(reversed(texts)), enc2, tmp_path,
                      {"model": "fake"}, shard_size=10)
    assert enc2.calls == 1
    pd.testing.assert_frame_equal(a, b)


def test_a_different_model_cannot_reuse_the_cache(tmp_path):
    encode_cached(["a"], ["x"], FakeEncoder(), tmp_path, {"model": "fake"}, 10)
    with pytest.raises(RuntimeError, match="different cache"):
        encode_cached(["a"], ["x"], FakeEncoder(), tmp_path, {"model": "other"}, 10)


def test_empty_text_is_encoded_and_flagged(tmp_path):
    out = encode_cached(["a", "b"], ["hello", "   "], FakeEncoder(), tmp_path, {"model": "f"})
    assert out.loc["b", "text_empty"] and not out.loc["a", "text_empty"]


def test_pca_is_fitted_on_training_rows_only():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(500, 384))
    train = np.arange(500) < 400
    emb[~train] += 50.0                     # a shifted held-out distribution
    st = fit_pca(emb, train, np.arange(500), "h", 64, "reply_pca")
    st.assert_training_only(np.arange(400))
    with pytest.raises(ProvenanceError):
        st.assert_training_only(np.arange(100))
    assert np.allclose(st.params["mean"], emb[train].mean(0))
    out = apply_pca(st, emb)
    assert out.shape == (500, 64)
    assert abs(out[train].mean()) < 1e-8 and abs(out[~train].mean()) > 1.0


def test_unseen_repliers_get_all_zero_identity_columns():
    rep = pd.Series(["a", "b", "a", "c", "d"])
    X, names = replier_indicators(rep, np.array([True, True, True, False, False]))
    assert names == ["replier=a", "replier=b"]
    assert X[3:].sum() == 0 and X[0, 0] == 1 and X[1, 1] == 1


def test_cache_manifest_is_kept_and_a_different_encoder_setup_cannot_add_shards(tmp_path):
    """Regression (notebook 10 reloaded the encoder under a newer library and rewrote the
    manifest): the manifest written by the building run is kept, reuse of complete shards
    works under any set-up, and new shards under a different set-up are refused."""
    import json

    import numpy as np

    from src.features.text_enc import encode_cached

    def enc(texts):
        return np.ones((len(texts), 4), np.float32)

    built = {"model": "m", "revision_requested": "r1", "library_version": "3.0.1"}
    other = {**built, "library_version": "5.1.0"}
    ids = [f"t{i}" for i in range(5)]
    encode_cached(ids, ["a"] * 5, enc, tmp_path, built, shard_size=2)
    before = (tmp_path / "manifest.json").read_text()
    out = encode_cached(ids, ["a"] * 5, enc, tmp_path, other, shard_size=2)   # reuse: fine
    assert len(out) == 5 and (tmp_path / "manifest.json").read_text() == before
    assert json.loads(before)["model_info"]["library_version"] == "3.0.1"
    (tmp_path / "shard_00002.npz").unlink()                       # a shard must be recomputed
    with pytest.raises(RuntimeError, match="one encoder set-up"):
        encode_cached(ids, ["a"] * 5, enc, tmp_path, other, shard_size=2)
    encode_cached(ids, ["a"] * 5, enc, tmp_path, built, shard_size=2)        # same set-up: fine
