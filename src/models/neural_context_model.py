"""The assembled Layer-2 architecture (Prereg §7).

    partner frames (5 sources × 128, ≤ 400) ─► SSL layer mixer (α, → 192)
                                            ─► temporal transformer ─► frames, a_i
    partner text (64) ─► MLP ─► p_i (96)
    reply text (64) + metadata + speaker ─► MLP ─► c_i (96)
    fusion: cross-attention([c;p] → frames) + a_i ; concat [c ; p ; attended]
    head: 2-layer MLP ─► 8 means (+ 8 log-variances)

**One forward, four partner modes, identical parameters**:

- ``real`` / ``donor`` — the SSL frame sequence of the partner turn or of a
  donor turn (the model cannot tell them apart; only the data differ);
- ``null_token`` — ablation 1: a learned null-token sequence, cut to the true
  length of the partner turn so the length distribution is that of real turns;
- ``delivery_vector_interpretable`` — ablation 3: the eight interpretable
  partner dimensions, projected to 192 and given as a length-1 sequence.

Every module used by any mode is built in every mode, so the trainable
parameter count is bit-identical across conditions; :func:`parameter_count`
reports it exactly and :func:`assert_identical_capacity` enforces it.

**Speaker identity** enters as an embedding with index 0 reserved for
unseen speakers; during training a fraction of speakers is replaced by index 0
(``speaker_dropout``) so the unseen-speaker embedding is learned. All
validation and test speakers use index 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models.delivery_head import DeliveryHead
from src.models.multimodal_fusion import CrossModalFusion
from src.models.ssl_layer_mixer import SOURCES, SSLLayerMixer
from src.models.temporal_transformer import TemporalTransformer

PARTNER_MODES = ("real", "donor", "null_token", "delivery_vector_interpretable")


@dataclass(frozen=True)
class Architecture:
    """Frozen before J1 (Prereg §7); identical in every condition."""

    d_model: int = 192
    n_blocks: int = 3
    n_heads: int = 4
    ffn_dim: int = 384
    max_frames: int = 400
    ssl_dim: int = 128
    n_sources: int = len(SOURCES)
    text_dim: int = 64
    n_metadata: int = 4
    n_speakers: int = 1              # training speakers + 1 for "unseen"
    speaker_dim: int = 16
    branch_dim: int = 96
    n_targets: int = 8
    dropout: float = 0.1


def _mlp(i: int, h: int, o: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, o))


class NeuralContextModel(nn.Module):
    def __init__(self, arch: Architecture):
        super().__init__()
        self.arch = arch
        d = arch.d_model
        self.mixer = SSLLayerMixer(arch.ssl_dim, d, arch.n_sources)
        self.null_tokens = nn.Parameter(torch.randn(arch.max_frames, d) * 0.02)
        self.vector_proj = nn.Linear(arch.n_targets, d)
        self.encoder = TemporalTransformer(d, arch.n_blocks, arch.n_heads, arch.ffn_dim,
                                           arch.max_frames, arch.dropout)
        self.partner_text = _mlp(arch.text_dim, arch.branch_dim, arch.branch_dim, arch.dropout)
        self.speaker = nn.Embedding(arch.n_speakers, arch.speaker_dim)
        self.context = _mlp(arch.text_dim + arch.n_metadata + arch.speaker_dim, arch.branch_dim,
                            arch.branch_dim, arch.dropout)
        if 2 * arch.branch_dim != d:
            raise ValueError("the fusion query [c;p] must have width d_model")
        self.fusion = CrossModalFusion(d, arch.n_heads, arch.dropout)
        self.head = DeliveryHead(2 * arch.branch_dim + d, d, arch.n_targets, arch.dropout)

    def partner_sequence(self, mode: str, frames, lengths, vector):
        if mode in ("real", "donor"):
            return self.mixer(frames.float()), lengths
        if mode == "null_token":
            T = int(lengths.max())
            seq = self.null_tokens[:T][None].expand(lengths.shape[0], -1, -1)
            return seq, lengths
        if mode == "delivery_vector_interpretable":
            seq = self.vector_proj(vector.float())[:, None, :]
            return seq, torch.ones_like(lengths)
        raise ValueError(f"unknown partner mode {mode!r}")

    def forward(self, batch: dict, mode: str) -> dict:
        seq, lengths = self.partner_sequence(mode, batch.get("frames"), batch["lengths"],
                                             batch.get("partner_vector"))
        frames, pooled, mask = self.encoder(seq, lengths)
        p = self.partner_text(batch["partner_text"].float())
        spk = self.speaker(batch["speaker"])
        c = self.context(torch.cat([batch["reply_text"].float(), batch["metadata"].float(), spk],
                                   dim=-1))
        fused = self.fusion(c, p, frames, mask, pooled)
        mean, log_var = self.head(fused)
        return {"mean": mean, "log_var": log_var, "pooled": pooled}


def parameter_count(model: nn.Module, trainable_only: bool = True) -> int:
    return int(sum(p.numel() for p in model.parameters()
                   if p.requires_grad or not trainable_only))


def assert_identical_capacity(models: dict[str, nn.Module]) -> int:
    """Total parameter count identical across conditions; returns it.

    Counted over all parameters, trainable or frozen, because ablation 4
    freezes the mixer logits without removing them.
    """
    counts = {k: parameter_count(m, trainable_only=False) for k, m in models.items()}
    shapes = {k: [tuple(p.shape) for p in m.parameters()] for k, m in models.items()}
    first = next(iter(counts.values()))
    if any(v != first for v in counts.values()) or \
            any(s != next(iter(shapes.values())) for s in shapes.values()):
        raise AssertionError(f"capacity differs across conditions: {counts}")
    return first


__all__ = ["PARTNER_MODES", "Architecture", "NeuralContextModel", "parameter_count",
           "assert_identical_capacity"]
