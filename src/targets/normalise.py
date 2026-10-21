"""Per-speaker standardisation in the two registered modes.

Per-speaker z-scoring follows Levitan and Hirschberg's norming procedure (
Prereg §4). It removes each speaker's habitual level so that what remains is
how this turn differs from the speaker's own norm. It is called per-speaker
standardisation, nothing more.

**Training-speaker mode** (pairs in the training split). Statistics are that
speaker's mean and SD over their usable training turns (``normstats``).

**Held-out-speaker mode** (validation and test pairs, whose speakers never
appear in training). For pair *i*, statistics come from the speaker's other
usable turns **in the same session**, excluding every turn in
E_i = {reply_i, partner_i, self_previous_i} and the turn being standardised.
The rule is applied to every vector the pair uses: the reply (target), the
partner turn (A), the self-previous turn and any donor turn, so a donor is
processed exactly like the partner it replaces.

The mode that produced each row is recorded. SDs use ddof = 1. A row whose
speaker has fewer than ``min_turns`` remaining turns, or zero spread,
is returned as missing with a reason rather than guessed.

The per-(speaker, session) sums are built once (:class:`Standardiser`); each
batch of rows, including the leave-out subtraction, is then vectorised, so the
500 donor draws of a null cost 500 array operations rather than 500 rebuilds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.normstats import FittedStats

MODE_TRAIN = "training_speaker"
MODE_HELD_OUT = "held_out_speaker"


def _missing(e) -> bool:
    return e is None or (isinstance(e, float) and np.isnan(e))


class Standardiser:
    """Per-(speaker, session) sums over usable turns, built once and reused."""

    def __init__(self, prosody: pd.DataFrame, dims: list[str],
                 train_stats: FittedStats | None, min_turns: int):
        p = prosody.reset_index(drop=True)
        self.dims, self.train_stats, self.min_turns = list(dims), train_stats, min_turns
        self.turn_pos = pd.Series(np.arange(len(p)), index=p["turn_id"].astype(str))
        self.values = p[self.dims].to_numpy(np.float64)
        self.usable = (p["usable"].astype(bool).to_numpy()
                       & np.isfinite(self.values).all(axis=1))
        self.speaker = p["speaker"].astype(str).to_numpy()
        key = (p["speaker"].astype(str) + "|" + p["session_key"].astype(str)).to_numpy()
        keys, self.key_code = np.unique(key, return_inverse=True)
        v = np.where(self.usable[:, None], self.values, 0.0)
        self.s1 = np.zeros((len(keys), len(self.dims)))
        self.s2 = np.zeros((len(keys), len(self.dims)))
        self.n = np.zeros(len(keys))
        np.add.at(self.s1, self.key_code, v)
        np.add.at(self.s2, self.key_code, v ** 2)
        np.add.at(self.n, self.key_code, self.usable.astype(float))
        if train_stats is not None:
            spk = pd.Series(self.speaker)
            self.tr_mean = train_stats.params["mean"].reindex(spk).to_numpy()
            self.tr_sd = train_stats.params["sd"].reindex(spk).to_numpy()
            self.tr_n = train_stats.params["n"].reindex(spk).fillna(0).to_numpy().astype(int)

    def positions(self, turn_ids) -> np.ndarray:
        pos = self.turn_pos.reindex([str(t) for t in turn_ids]).to_numpy()
        if np.isnan(pos.astype(float)).any():
            raise KeyError("a requested turn has no prosody row")
        return pos.astype(int)

    def _exclusion_positions(self, exclude: np.ndarray, rows: np.ndarray) -> list[np.ndarray]:
        out = []
        ex = np.asarray(exclude, dtype=object)[rows]
        for j in range(ex.shape[1]):
            ids = ex[:, j]
            present = np.array([not _missing(e) for e in ids], dtype=bool)
            q = np.full(len(rows), -1)
            if present.any():
                q[present] = (self.turn_pos.reindex([str(e) for e in ids[present]])
                              .fillna(-1).to_numpy().astype(int))
            out.append(q)
        return out

    def rows(self, turn_ids, modes, exclude: np.ndarray | None) -> pd.DataFrame:
        """z-scores for rows; ``exclude`` is an (n × k) array of turn ids (or None)."""
        turn_ids = [str(t) for t in turn_ids]
        pos = self.positions(turn_ids)
        x = self.values[pos]
        modes = np.asarray(modes, dtype=object)
        out = np.full_like(x, np.nan)
        n_used = np.zeros(len(pos), dtype=int)
        reason = np.full(len(pos), None, dtype=object)

        tr = modes == MODE_TRAIN
        if tr.any():
            if self.train_stats is None:
                raise ValueError("training-speaker rows need fitted training statistics")
            q = pos[tr]
            out[tr] = (x[tr] - self.tr_mean[q]) / self.tr_sd[q]
            n_used[tr] = self.tr_n[q]

        ho = modes == MODE_HELD_OUT
        if ho.any():
            idx = np.flatnonzero(ho)
            kc = self.key_code[pos[idx]]
            s1, s2, nn = self.s1[kc].copy(), self.s2[kc].copy(), self.n[kc].copy()
            # The row's own turn first, then each listed exclusion; each removed once.
            cols = [pos[idx]]
            if exclude is not None:
                cols += self._exclusion_positions(exclude, idx)
            seen: list[np.ndarray] = []
            for q in cols:
                ok = q >= 0
                ok[ok] = (self.key_code[q[ok]] == kc[ok]) & self.usable[q[ok]]
                for prev in seen:
                    ok &= q != prev
                seen.append(q)
                if ok.any():
                    s1[ok] -= self.values[q[ok]]
                    s2[ok] -= self.values[q[ok]] ** 2
                    nn[ok] -= 1
            with np.errstate(invalid="ignore", divide="ignore"):
                mu = s1 / nn[:, None]
                var = (s2 - nn[:, None] * mu ** 2) / (nn[:, None] - 1)
                out[idx] = (x[idx] - mu) / np.sqrt(np.clip(var, 0, None))
            n_used[idx] = nn.astype(int)

        reason[n_used < self.min_turns] = "too_few_turns_for_stats"
        bad = ~np.isfinite(out).all(axis=1) & np.array([r is None for r in reason])
        reason[bad] = "zero_spread_or_missing"
        failed = np.array([r is not None for r in reason])
        out[failed] = np.nan
        res = pd.DataFrame(out, columns=self.dims)
        res["turn_id"] = turn_ids
        res["mode"] = modes
        res["n_stats_turns"] = n_used
        res["standardisation_failure"] = reason
        return res


def standardise_rows(turn_ids, modes, exclude: list[list], prosody: pd.DataFrame,
                     dims: list[str], train_stats: FittedStats | None,
                     min_turns: int) -> pd.DataFrame:
    """One-off convenience wrapper around :class:`Standardiser`."""
    width = max((len(e) for e in exclude), default=0)
    ex = np.full((len(exclude), max(width, 1)), None, dtype=object)
    for i, e in enumerate(exclude):
        for j, t in enumerate(e):
            ex[i, j] = t
    return Standardiser(prosody, dims, train_stats, min_turns).rows(turn_ids, modes, ex)


def mode_for_split(split: pd.Series) -> np.ndarray:
    """Training split -> training-speaker mode; validation and test -> held-out mode."""
    return np.where(np.asarray(split) == "train", MODE_TRAIN, MODE_HELD_OUT)


def exclusion_matrix(analysis: pd.DataFrame) -> np.ndarray:
    """E_i = {reply_i, partner_i, self_previous_i} per pair, as an (n × 3) array."""
    cols = ["reply_turn_id", "partner_turn_id", "self_previous_turn_id"]
    return analysis[cols].to_numpy(dtype=object)


def standardise_pair_roles(analysis: pd.DataFrame, std: Standardiser,
                           roles=("reply", "partner", "self_previous")) -> dict[str, pd.DataFrame]:
    """Standardised vectors for each role of every pair in ``analysis``."""
    modes = mode_for_split(analysis["split"])
    ex = exclusion_matrix(analysis)
    col = {"reply": "reply_turn_id", "partner": "partner_turn_id",
           "self_previous": "self_previous_turn_id"}
    out = {}
    for role in roles:
        res = std.rows(analysis[col[role]].to_numpy(), modes, ex)
        res.index = analysis.index
        out[role] = res
    return out


def standardise_donors(analysis: pd.DataFrame, donor_turns: np.ndarray,
                       std: Standardiser) -> np.ndarray:
    """Standardised donor vectors, shape (n_pairs, n_draws, n_dims).

    Each donor is processed exactly like the partner it replaces: the same mode,
    and the pair's exclusion set E_i plus the donor turn itself.
    """
    modes = mode_for_split(analysis["split"])
    ex = exclusion_matrix(analysis)
    n, b = donor_turns.shape
    out = np.empty((n, b, len(std.dims)))
    for k in range(b):
        out[:, k, :] = std.rows(donor_turns[:, k], modes, ex)[std.dims].to_numpy()
    return out


def standardisation_check(values: np.ndarray) -> dict:
    """Held-out standardised values should be near, not exactly, 0 mean / 1 SD.

    Exactly 0 and 1 to machine precision would mean the test rows contributed
    to their own statistics.
    """
    v = values[np.isfinite(values).all(axis=1)]
    mean, sd = v.mean(axis=0), v.std(axis=0, ddof=1)
    exact = bool(np.allclose(mean, 0, atol=1e-12) and np.allclose(sd, 1, atol=1e-12))
    return {"mean": mean.tolist(), "sd": sd.tolist(), "exactly_zero_one": exact,
            "n": int(len(v))}


__all__ = ["MODE_TRAIN", "MODE_HELD_OUT", "Standardiser", "standardise_rows",
           "mode_for_split", "exclusion_matrix", "standardise_pair_roles", "standardise_donors",
           "standardisation_check"]
