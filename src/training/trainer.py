"""Training loop for the Layer-2 model.

- **Seeded**: Python, NumPy and torch seeds are set from the run seed; data order
  is drawn from a seeded generator.
- **Early stopping on validation loss only**; the test split is never read
  during training or checkpoint selection (neural model selection).
- **Gradient clipping**, **AdamW** with weight decay, dropout in the model.
- **Checkpoint every epoch** (``checkpoints/`` is git-ignored) and resume from
  the last one, so a preempted Colab GPU session loses at most one epoch.
- **AMP off** unless enabled after it is shown to help.
- The test split is predicted exactly once, after training, and that single
  read is recorded.

Device: CUDA when available, else CPU. Expected cost on a T4 with cached
features: about 10-25 minutes per run.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from src.models.delivery_head import block_loss, block_weights
from src.models.neural_context_model import NeuralContextModel, parameter_count
from src.training.config import TrainConfig


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ArrayData:
    """In-memory batches; the SSL cache loader yields the same dictionary keys."""

    def __init__(self, arrays: dict[str, np.ndarray]):
        n = {len(v) for v in arrays.values()}
        if len(n) != 1:
            raise ValueError("all arrays need the same number of rows")
        self.arrays = arrays
        self.n = n.pop()

    def batches(self, idx: np.ndarray, batch_size: int, rng: np.random.Generator | None,
                speaker_dropout: float = 0.0, dev: torch.device | None = None):
        order = rng.permutation(idx) if rng is not None else idx
        for s in range(0, len(order), batch_size):
            rows = np.sort(order[s:s + batch_size])
            b = {k: torch.as_tensor(v[rows]) for k, v in self.arrays.items()}
            if "frames" in b:
                T = int(b["lengths"].max())
                b["frames"] = b["frames"][:, :T]
            if speaker_dropout > 0 and rng is not None:
                drop = torch.as_tensor(rng.random(len(rows)) < speaker_dropout)
                b["speaker"] = torch.where(drop, torch.zeros_like(b["speaker"]), b["speaker"])
            yield {k: v.to(dev) if dev is not None else v for k, v in b.items()}


def train(model: NeuralContextModel, data: ArrayData, train_idx: np.ndarray,
          val_idx: np.ndarray, mode: str, cfg: TrainConfig, seed: int, blocks: list[list[int]],
          checkpoint_dir: Path | None = None) -> dict:
    """Fit on training rows, early-stop on validation loss; returns the history.

    The seed here fixes data order and dropout. Initial weights are fixed by
    seeding before the model is built (``run_neural.make_model``).
    """
    seed_everything(seed)
    dev = device()
    model.to(dev)
    weights = block_weights(blocks, model.arch.n_targets)
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimiser != "adamw":
        raise ValueError(f"unknown optimiser {cfg.optimiser!r}")
    opt = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(seed)
    start_epoch, best, best_state, bad, history = 0, float("inf"), None, 0, []
    ckpt = checkpoint_dir / "last.pt" if checkpoint_dir else None
    if ckpt and ckpt.exists():
        st = torch.load(ckpt, map_location=dev)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start_epoch, best, bad, history = st["epoch"] + 1, st["best"], st["bad"], st["history"]
        best_state = st["best_state"]
        rng = np.random.default_rng([seed, start_epoch])
    for epoch in range(start_epoch, cfg.max_epochs):
        hetero = cfg.heteroscedastic_after_warm_start and epoch >= cfg.warm_start_epochs
        model.train()
        tr_loss = []
        for b in data.batches(train_idx, cfg.batch_size, rng, cfg.speaker_dropout, dev):
            out = model(b, mode)
            loss = block_loss(out["mean"], out["log_var"], b["y"].float(), weights, hetero)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm)
            opt.step()
            tr_loss.append(float(loss))
        val = evaluate_loss(model, data, val_idx, mode, cfg, weights, hetero, dev)
        history.append({"epoch": epoch, "train_loss": float(np.mean(tr_loss)), "val_loss": val})
        if val < best - 1e-6:
            best, bad = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ckpt:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                        "best": best, "bad": bad, "history": history,
                        "best_state": best_state}, ckpt)
        if bad >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_val_loss": best, "epochs_run": len(history),
            "config": asdict(cfg), "config_hash": cfg.hash(), "seed": seed, "mode": mode,
            "trainable_parameters": parameter_count(model),
            "alpha": model.mixer.alpha().detach().cpu().tolist()}


@torch.no_grad()
def evaluate_loss(model, data, idx, mode, cfg, weights, hetero, dev) -> float:
    model.eval()
    losses, n = 0.0, 0
    for b in data.batches(idx, cfg.batch_size, None, 0.0, dev):
        out = model(b, mode)
        losses += float(block_loss(out["mean"], out["log_var"], b["y"].float(), weights,
                                   hetero)) * len(b["y"])
        n += len(b["y"])
    return losses / max(n, 1)


@torch.no_grad()
def predict(model, data: ArrayData, idx: np.ndarray, mode: str, batch_size: int = 256,
            ledger: Path | None = None, split_name: str = "test") -> dict:
    """Predictions for ``idx``. A test-split read is appended to ``ledger``."""
    model.eval()
    dev = next(model.parameters()).device
    means, logv, pooled = [], [], []
    for b in data.batches(idx, batch_size, None, 0.0, dev):
        out = model(b, mode)
        means.append(out["mean"].cpu().numpy())
        logv.append(out["log_var"].cpu().numpy())
        pooled.append(out["pooled"].cpu().numpy())
    if ledger is not None:
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"split": split_name, "rows": int(len(idx)), "mode": mode}) + "\n")
    return {"mean": np.concatenate(means), "log_var": np.concatenate(logv),
            "pooled": np.concatenate(pooled)}


__all__ = ["seed_everything", "device", "ArrayData", "train", "evaluate_loss", "predict"]
