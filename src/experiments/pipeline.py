"""From processed tables to the arrays every Layer-1 and Layer-2 analysis uses.

:func:`prepare_analysis` takes the turn, pair, prosody and split tables plus the
text embeddings and per-file loudness offsets, and returns an
:class:`AnalysisData` holding, for one common set of pairs:

- **Y** — reply delivery, per-speaker standardised (registered mode per row),
  then within-block orthogonalised with a transform fitted on training rows;
- **A** — partner delivery (Pathway A), standardised the same way;
- **S** — the replier's self-previous delivery, standardised the same way;
- **Z** — the conditioning set (``src.features.conditioning``) with PCA-64 text
  blocks fitted on training rows;
- session groups, partition masks, and every fitted transform with its
  provenance record.

Every exclusion is kept with its reason, and the standardisation check
is computed on the held-out targets. Fitted transforms are asserted to have
been fitted on training rows only before anything is returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import TARGET_BLOCKS, TARGET_DIMENSIONS
from src.data.dataset import build_analysis_set, exclusion_table
from src.data.splits import manifest_hash
from src.features.conditioning import ZBlocks, build_z
from src.features.normstats import speaker_stats
from src.features.text_enc import apply_pca, fit_pca
from src.targets.normalise import Standardiser, standardisation_check, standardise_pair_roles
from src.targets.orthogonalise import BlockOrthogonaliser

DIMS = list(TARGET_DIMENSIONS)


def block_indices() -> list[np.ndarray]:
    out, k = [], 0
    for dims in TARGET_BLOCKS.values():
        out.append(np.arange(k, k + len(dims)))
        k += len(dims)
    return out


@dataclass
class AnalysisData:
    pairs: pd.DataFrame                 # included pairs, row-aligned with every array
    Y: np.ndarray
    Y_standardised: np.ndarray          # before orthogonalisation
    A: np.ndarray
    S: np.ndarray
    Z: ZBlocks
    groups: np.ndarray
    split: np.ndarray
    blocks: list[np.ndarray]
    standardiser: Standardiser
    orthogonaliser: BlockOrthogonaliser
    fitted: dict                        # name -> FittedStats
    candidates: pd.DataFrame            # every candidate pair with its exclusion reason
    checks: dict = field(default_factory=dict)
    split_hash: str = ""

    @property
    def train(self) -> np.ndarray:
        return self.split == "train"

    @property
    def val(self) -> np.ndarray:
        return self.split == "val"

    @property
    def test(self) -> np.ndarray:
        return self.split == "test"

    def provenance(self) -> dict:
        return {"split_hash": self.split_hash, "n_pairs": int(len(self.pairs)),
                "n_sessions": int(len(np.unique(self.groups))),
                "pairs_by_split": pd.Series(self.split).value_counts().to_dict(),
                "fitted": {k: v.record() for k, v in self.fitted.items()},
                "checks": self.checks}


def prepare_analysis(turns: pd.DataFrame, pairs: pd.DataFrame, prosody: pd.DataFrame,
                     split_manifest: pd.DataFrame, text_emb: pd.DataFrame,
                     loudness_by_file: pd.Series, min_turns: int,
                     text_pca_dim: int = 64) -> AnalysisData:
    split_hash = manifest_hash(split_manifest)
    cand = build_analysis_set(pairs, turns, prosody, split_manifest)
    inc = cand[cand["included"]].reset_index(drop=True)

    part = split_manifest.set_index("session_key")["split"]
    train_turns = prosody.loc[prosody["session_key"].map(part) == "train", "turn_id"]
    stats = speaker_stats(prosody, DIMS, train_turns, split_hash)
    stats.assert_training_only(train_turns)
    std = Standardiser(prosody, DIMS, stats, min_turns)
    roles = standardise_pair_roles(inc, std)

    failed = np.zeros(len(inc), bool)
    reason = pd.Series(None, index=inc.index, dtype=object)
    for role, res in roles.items():
        bad = res["standardisation_failure"].notna().to_numpy() & ~failed
        reason[bad] = f"standardisation_{role}:" + res.loc[bad, "standardisation_failure"]
        failed |= bad
    if failed.any():
        drop_ids = set(inc.loc[failed, "pair_id"])
        cand.loc[cand["pair_id"].isin(drop_ids), "included"] = False
        cand.loc[cand["pair_id"].isin(drop_ids), "exclusion"] = \
            cand.loc[cand["pair_id"].isin(drop_ids), "pair_id"].map(
                dict(zip(inc["pair_id"], reason, strict=True)))
    keep = ~failed
    inc = inc[keep].reset_index(drop=True)
    Ys = roles["reply"].loc[keep, DIMS].to_numpy()
    A = roles["partner"].loc[keep, DIMS].to_numpy()
    S = roles["self_previous"].loc[keep, DIMS].to_numpy()
    split = inc["split"].to_numpy()
    train = split == "train"

    orth = BlockOrthogonaliser().fit(Ys[train], inc.loc[train, "pair_id"], split_hash)
    orth.fitted.assert_training_only(inc.loc[train, "pair_id"])
    Y = orth.transform(Ys)

    emb_cols = [c for c in text_emb.columns if c.startswith("e")]
    reply_emb = text_emb.loc[inc["reply_turn_id"], emb_cols].to_numpy(np.float64)
    partner_emb = text_emb.loc[inc["partner_turn_id"], emb_cols].to_numpy(np.float64)
    pca_r = fit_pca(reply_emb, train, inc["pair_id"], split_hash, text_pca_dim, "reply_text_pca")
    pca_p = fit_pca(partner_emb, train, inc["pair_id"], split_hash, text_pca_dim,
                    "partner_text_pca")
    for st in (pca_r, pca_p):
        st.assert_training_only(inc.loc[train, "pair_id"])
    Z = build_z(inc, turns, prosody, apply_pca(pca_r, reply_emb), apply_pca(pca_p, partner_emb),
                loudness_by_file, train)

    held = ~train
    checks = {"standardisation_heldout_targets": standardisation_check(Ys[held]),
              "exclusions": exclusion_table(cand).to_dict()}
    if checks["standardisation_heldout_targets"]["exactly_zero_one"]:
        raise AssertionError("held-out targets are exactly 0/1: their statistics include them")
    return AnalysisData(pairs=inc, Y=Y, Y_standardised=Ys, A=A, S=S, Z=Z,
                        groups=inc["session_key"].to_numpy(), split=split,
                        blocks=block_indices(), standardiser=std, orthogonaliser=orth,
                        fitted={"speaker_stats": stats, "orthogonaliser": orth.fitted,
                                "reply_text_pca": pca_r, "partner_text_pca": pca_p},
                        candidates=cand, checks=checks, split_hash=split_hash)


__all__ = ["AnalysisData", "prepare_analysis", "block_indices", "DIMS"]
