"""The conditioning set Z.

Z for pair i: reply text embedding (PCA-64), reply word count, reply syllable
count, partner text embedding (PCA-64), replier identity, turn position, and the
per-file loudness offset of the reply's file (the design maps the
"per-conversation" offset to the file, because the release is peak-normalised
per file).

``Z_content`` (the J1 baseline) is the same without the partner text block; the
H1 reference model is Z with partner text.

**Replier identity.** One indicator column per training
replier. Validation and test repliers never appear in training (speaker-disjoint
split), so their indicator columns are all zero: after the column centring done
inside the ridge, that is the training-population average. The indicators let
the model absorb residual per-speaker offsets among training speakers, so A is
not credited with them; they cannot carry identity information at test time.

Every column is standardised inside the estimator using fitting-row statistics
only, so no held-out row influences how Z is scaled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ZBlocks:
    """The conditioning set as named column blocks, rows aligned with the analysis set."""

    reply_text: np.ndarray
    partner_text: np.ndarray
    metadata: np.ndarray
    metadata_names: list[str]
    replier: np.ndarray
    replier_names: list[str]

    def content(self) -> np.ndarray:
        """Z without partner text: the ``Z_content`` condition."""
        return np.hstack([self.reply_text, self.metadata, self.replier])

    def full(self) -> np.ndarray:
        """Z with partner text: the H1 reference model."""
        return np.hstack([self.reply_text, self.partner_text, self.metadata, self.replier])

    def column_names(self, with_partner_text: bool = True) -> list[str]:
        r = [f"reply_pc{j}" for j in range(self.reply_text.shape[1])]
        p = [f"partner_pc{j}" for j in range(self.partner_text.shape[1])]
        return r + (p if with_partner_text else []) + self.metadata_names + self.replier_names


def replier_indicators(repliers: pd.Series, train_mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """One column per training replier; zero rows for unseen repliers."""
    train_ids = sorted(repliers[train_mask].unique())
    col = {s: j for j, s in enumerate(train_ids)}
    X = np.zeros((len(repliers), len(train_ids)), dtype=np.float32)
    for i, s in enumerate(repliers):
        j = col.get(s)
        if j is not None:
            X[i, j] = 1.0
    return X, [f"replier={s}" for s in train_ids]


def build_z(analysis: pd.DataFrame, turns: pd.DataFrame, prosody: pd.DataFrame,
            reply_pcs: np.ndarray, partner_pcs: np.ndarray,
            loudness_by_file: pd.Series, train_mask: np.ndarray) -> ZBlocks:
    """Assemble Z for the rows of ``analysis`` (already restricted to included pairs)."""
    t = turns.set_index("turn_id")
    pr = prosody.set_index("turn_id")
    reply = analysis["reply_turn_id"]
    meta = pd.DataFrame({
        "reply_word_count": t.loc[reply, "n_words"].to_numpy(float),
        "reply_syllable_count": pr.loc[reply, "n_syllables"].to_numpy(float),
        "turn_position": analysis["reply_turn_position"].to_numpy(float),
        "loudness_offset": pr.loc[reply, "file_id"].map(loudness_by_file).to_numpy(float),
    })
    if meta.isna().any().any():
        bad = meta.columns[meta.isna().any()].tolist()
        raise ValueError(f"Z has missing metadata in {bad}")
    rep, rep_names = replier_indicators(analysis["replier"].reset_index(drop=True), train_mask)
    return ZBlocks(reply_pcs, partner_pcs, meta.to_numpy(), list(meta.columns), rep, rep_names)


__all__ = ["ZBlocks", "replier_indicators", "build_z"]
