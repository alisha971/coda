"""Layer-2 architecture, capacity control, masking, training and extraction.

Tiny synthetic tensors only; nothing here trains a real model.
"""

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models.delivery_head import block_loss, block_weights  # noqa: E402
from src.models.neural_context_model import (  # noqa: E402
    Architecture,
    NeuralContextModel,
    assert_identical_capacity,
    parameter_count,
)
from src.models.ssl_layer_mixer import SOURCES  # noqa: E402
from src.representations.probes import interpretable_probe, speaker_probe  # noqa: E402
from src.representations.ssl_extract import (  # noqa: E402
    AudioMismatch,
    extract_turn,
    fit_source_pca,
    to_16k,
    verify_audio,
)
from src.training.config import TrainConfig, assert_same_config, freeze, load_frozen  # noqa: E402
from src.training.trainer import ArrayData, predict, train  # noqa: E402

BLOCKS = [[0, 1], [2], [3, 4], [5, 6, 7]]
SMALL = Architecture(max_frames=20, n_speakers=5, dropout=0.0)


def batch(n=6, T=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"frames": torch.randn(n, T, 5, 128, generator=g),
            "lengths": torch.tensor([20, 15, 7, 1, 20, 3][:n]),
            "partner_vector": torch.randn(n, 8, generator=g),
            "partner_text": torch.randn(n, 64, generator=g),
            "reply_text": torch.randn(n, 64, generator=g),
            "metadata": torch.randn(n, 4, generator=g),
            "speaker": torch.tensor([0, 1, 2, 3, 4, 0][:n])}


def test_full_size_parameter_count_is_inside_the_registered_range():
    n = parameter_count(NeuralContextModel(Architecture(n_speakers=351)))
    assert 1_500_000 <= n <= 2_500_000


def test_parameter_count_is_identical_in_every_partner_mode_and_ablation():
    models = {"real": NeuralContextModel(SMALL), "null": NeuralContextModel(SMALL),
              "vector": NeuralContextModel(SMALL), "frozen": NeuralContextModel(SMALL)}
    models["frozen"].mixer.freeze_to(SOURCES.index("wavlm_l6"))
    n = assert_identical_capacity(models)
    assert n == parameter_count(models["real"], trainable_only=False)
    bigger = NeuralContextModel(replace(SMALL, ffn_dim=512))
    with pytest.raises(AssertionError, match="capacity"):
        assert_identical_capacity({"a": models["real"], "b": bigger})


@pytest.mark.parametrize("mode", ["real", "donor", "null_token",
                                  "delivery_vector_interpretable"])
def test_forward_shapes_in_every_mode(mode):
    m = NeuralContextModel(SMALL).eval()
    out = m(batch(), mode)
    assert out["mean"].shape == (6, 8) and out["log_var"].shape == (6, 8)
    assert out["pooled"].shape == (6, 192)
    assert torch.isfinite(out["mean"]).all()


def test_padded_frames_never_change_the_output():
    m = NeuralContextModel(SMALL).eval()
    b = batch()
    with torch.no_grad():
        a = m(b, "real")["mean"]
        b2 = dict(b)
        f = b["frames"].clone()
        for i, L in enumerate(b["lengths"].tolist()):
            f[i, L:] = 1e3 * torch.randn_like(f[i, L:])
        b2["frames"] = f
        c = m(b2, "real")["mean"]
    torch.testing.assert_close(a, c, atol=1e-5, rtol=1e-5)


def test_null_token_mode_ignores_the_partner_frames():
    m = NeuralContextModel(SMALL).eval()
    b = batch()
    with torch.no_grad():
        a = m(b, "null_token")["mean"]
        c = m({**b, "frames": torch.randn_like(b["frames"])}, "null_token")["mean"]
    torch.testing.assert_close(a, c)


def test_frozen_mixer_is_one_hot_and_receives_no_gradient():
    m = NeuralContextModel(SMALL)
    m.mixer.freeze_to(1)
    a = m.mixer.alpha()
    assert a[1] > 0.999 and not m.mixer.logits.requires_grad
    loss = m(batch(), "real")["mean"].sum()
    loss.backward()
    assert m.mixer.logits.grad is None


def test_alpha_is_a_trained_parameter_not_a_selection():
    m = NeuralContextModel(SMALL)
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert "mixer.logits" in names


def test_overlong_or_empty_sequences_are_refused():
    m = NeuralContextModel(SMALL)
    b = batch()
    with pytest.raises(ValueError):
        m({**b, "lengths": torch.tensor([0, 1, 1, 1, 1, 1])}, "real")
    with pytest.raises(ValueError, match="cap"):
        m({**b, "frames": torch.randn(6, 25, 5, 128)}, "real")


def test_block_weights_give_each_block_one_quarter():
    w = block_weights(BLOCKS, 8)
    assert [float(w[b].sum()) for b in BLOCKS] == pytest.approx([0.25] * 4)
    y, mean = torch.zeros(4, 8), torch.zeros(4, 8)
    mean[:, 5:8] = 1.0                    # error only in the three timing dims
    assert float(block_loss(mean, torch.zeros_like(mean), y, w, False)) == pytest.approx(0.25)


def tiny_data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    arrays = {"frames": rng.normal(size=(n, 20, 5, 128)).astype(np.float32),
              "lengths": rng.integers(1, 21, n), "partner_vector": rng.normal(size=(n, 8)),
              "partner_text": rng.normal(size=(n, 64)).astype(np.float32),
              "reply_text": rng.normal(size=(n, 64)).astype(np.float32),
              "metadata": rng.normal(size=(n, 4)).astype(np.float32),
              "speaker": rng.integers(0, 5, n)}
    arrays["y"] = (0.8 * arrays["partner_vector"] + 0.3 * rng.normal(size=(n, 8))).astype(
        np.float32)
    arrays["partner_vector"] = arrays["partner_vector"].astype(np.float32)
    return ArrayData(arrays)


def test_training_reduces_validation_loss_and_never_reads_test_rows(tmp_path):
    data = tiny_data()
    tr, va, te = np.arange(80), np.arange(80, 100), np.arange(100, 120)
    cfg = TrainConfig(max_epochs=8, patience=3, batch_size=16, learning_rate=3e-3,
                      speaker_dropout=0.2)
    sentinel = dict(data.arrays)
    sentinel["y"] = data.arrays["y"].copy()
    sentinel["y"][te] = 1e6                        # a test leak would explode the loss
    m1 = NeuralContextModel(SMALL)
    h = train(m1, ArrayData(sentinel), tr, va, "delivery_vector_interpretable", cfg, 0, BLOCKS,
              checkpoint_dir=tmp_path / "ck")
    assert h["history"][-1]["val_loss"] < h["history"][0]["val_loss"]
    assert all(np.isfinite(e["train_loss"]) and e["train_loss"] < 100 for e in h["history"])
    out = predict(m1, data, te, "delivery_vector_interpretable",
                  ledger=tmp_path / "reads.jsonl")
    assert out["mean"].shape == (20, 8)
    assert len((tmp_path / "reads.jsonl").read_text().strip().splitlines()) == 1


def test_training_is_reproducible_under_a_seed():
    data = tiny_data(seed=3)
    cfg = TrainConfig(max_epochs=2, patience=5, batch_size=16)
    tr, va = np.arange(80), np.arange(80, 100)
    from src.experiments.run_neural import Job, make_model

    job = Job("no_partner_acoustic", "null_token", 7)
    a = train(make_model(SMALL, job), data, tr, va, "null_token", cfg, 7, BLOCKS)
    b = train(make_model(SMALL, job), data, tr, va, "null_token", cfg, 7, BLOCKS)
    assert a["history"] == b["history"]
    c = train(make_model(SMALL, Job("no_partner_acoustic", "null_token", 8)), data, tr, va,
              "null_token", cfg, 8, BLOCKS)
    assert c["history"] != a["history"]          # the seed matters


def test_frozen_training_config_is_hashed_and_cannot_be_replaced(tmp_path):
    cfg = TrainConfig()
    h = freeze(cfg, tmp_path / "frozen.json", {"grid": "x"})
    assert load_frozen(tmp_path / "frozen.json").hash() == h
    with pytest.raises(FileExistsError):
        freeze(replace(cfg, learning_rate=1e-3), tmp_path / "frozen.json", {})
    assert_same_config({"real": h, "N1": h})
    with pytest.raises(AssertionError):
        assert_same_config({"real": h, "N1": replace(cfg, dropout=0.3).hash()})


# -- SSL extraction ---------------------------------------------------------


def fake_extractor(wave):
    T = len(wave) // 320
    return {s: np.full((T, 768), i, np.float32) + wave[: T * 320].reshape(T, 320)[:, :1]
            for i, s in enumerate(SOURCES)}


def test_extraction_keeps_the_final_frames_and_the_true_length():
    w = np.linspace(0, 1, 16000 * 10, dtype=np.float32)        # 10 s
    feats, T = extract_turn(w, fake_extractor, max_frames=400)
    assert T == 400 and feats["wavlm_l3"].shape == (400, 768)
    assert feats["wavlm_l3"][-1, 0] > 0.99                       # the end of the turn is kept
    short, T2 = extract_turn(w[:16000], fake_extractor)
    assert T2 == 50


def test_extraction_takes_no_label_input():
    import inspect

    assert list(inspect.signature(extract_turn).parameters) == ["wave_16k", "extractor",
                                                                "max_frames"]


def test_ssl_pca_refuses_non_training_turns():
    X = {s: np.random.default_rng(0).normal(size=(300, 768)) for s in SOURCES}
    st = fit_source_pca(X, ["a", "b"], {"a", "b", "c"}, 128, "h")
    assert st["hubert_final"].params["components"].shape == (128, 768)
    with pytest.raises(AssertionError, match="non-training"):
        fit_source_pca(X, ["a", "z"], {"a", "b"}, 128, "h")


def test_resampling_and_audio_hash_check(tmp_path):
    assert to_16k(np.zeros(48000), 48000).shape == (16000,)
    f = tmp_path / "x.wav"
    f.write_bytes(b"abc")
    import hashlib

    verify_audio(f, hashlib.sha256(b"abc").hexdigest())
    with pytest.raises(AudioMismatch):
        verify_audio(f, "0" * 64)


# -- probes -----------------------------------------------------------------


def test_probes_recover_planted_structure():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(600, 32))
    y = np.column_stack([a[:, 0], rng.normal(size=600)])
    r = interpretable_probe(a[:400], y[:400], a[400:], y[400:])
    assert r["per_dimension_r2"][0] > 0.9 and r["per_dimension_r2"][1] < 0.1
    spk = np.repeat(np.arange(10), 40)
    emb = rng.normal(size=(400, 16)) + 3 * np.eye(16)[spk]
    p = speaker_probe(emb, spk)
    assert p["accuracy"] > 0.8 and p["chance"] == pytest.approx(0.1)
