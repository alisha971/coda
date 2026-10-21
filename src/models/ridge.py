"""Layer 1: ridge regression with grouped nested cross-validation.

**Model.** For pair rows, Y (n × m) is predicted from X = [Z A], where Z is the
conditioning set and A the extra block a condition adds (partner delivery,
partner text, self-previous delivery, or a donor vector). Every column of X is
standardised on the fitting rows and Y is centred there, so the intercept is
unpenalised. The ridge solution is β = (XᵀX + λI)⁻¹XᵀY (Hoerl & Kennard 1970).

**Why the solver is organised around a fixed Z.** Across the 500 donor draws
of a null, Z and Y never change; only A does. The eigendecomposition
ZᵀZ = V diag(e) Vᵀ is therefore computed once per fold, and each draw costs
only the A-dependent pieces. With K = ZᵀZ + λI and the Schur complement

    S = AᵀA + λI − AᵀZ K⁻¹ ZᵀA,
    β_A = S⁻¹ (AᵀY − AᵀZ K⁻¹ ZᵀY),     β_Z = K⁻¹ (ZᵀY − ZᵀA β_A),

which is exactly the joint ridge solution (block matrix inversion). This is the
"decomposition of the fixed Z part" that the design requires.

**λ selection.** Nested, grouped by session: K folds of whole sessions inside
the fitting rows, λ chosen per target block (or per arbitrary column group) as
the value minimising out-of-fold SSE, then a final fit on all fitting rows.
Evaluation rows (the test split) never enter any of this. The grid is relative
to the number of fitting rows; a choice at either end of the grid is
flagged.

**Effective degrees of freedom** (reported): Σ eᵢ/(eᵢ + λ) over the
eigenvalues of the standardised XᵀX, the trace of the ridge hat matrix
(Hastie, Tibshirani & Friedman, *Elements of Statistical Learning*).

The solver is deterministic: no seeds, other than the seeded assignment of
sessions to inner folds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def grouped_folds(groups: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Fold number per row; whole groups (sessions) per fold, balanced by size.

    Groups are visited in a seeded random order and each is given to the fold
    holding the fewest rows so far.
    """
    groups = np.asarray(groups)
    uniq, counts = np.unique(groups, return_counts=True)
    if len(uniq) < k:
        raise ValueError(f"{len(uniq)} groups cannot form {k} folds")
    order = np.random.default_rng(seed).permutation(len(uniq))
    load = np.zeros(k)
    fold_of = {}
    for i in order:
        f = int(np.argmin(load))
        fold_of[uniq[i]] = f
        load[f] += counts[i]
    return np.array([fold_of[g] for g in groups])


@dataclass
class _Scaled:
    mean: np.ndarray
    sd: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> _Scaled:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        return cls(mu, np.where(sd > 0, sd, 1.0))

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.sd


@dataclass
class _FoldZ:
    """Everything about one fold that depends on Z and Y only."""

    fit: np.ndarray
    ev: np.ndarray
    zs: _Scaled
    Zf: np.ndarray            # scaled Z, fitting rows
    V: np.ndarray             # eigenvectors of ZfᵀZf
    e: np.ndarray             # eigenvalues
    ZeV: np.ndarray           # scaled Z on evaluation rows, times V
    ymean: np.ndarray
    VtZtY: np.ndarray         # Vᵀ Zfᵀ (Y - ymean)
    Yf: np.ndarray            # centred Y on fitting rows
    Ye: np.ndarray            # raw Y on evaluation rows (may be NaN for test placeholders)
    G: np.ndarray | None = None      # ZeVᵀ ZeV: Gram matrix of evaluation rows (Z eigenbasis)
    rr: np.ndarray | None = None     # Σ (y − ȳ_fit)² per column on evaluation rows
    ZtR: np.ndarray | None = None    # ZeVᵀ (y − ȳ_fit) on evaluation rows


class ZCache:
    """Z-only precomputation for the inner folds and the final fit.

    ``fit_mask`` marks the rows models are fitted on (the training split);
    ``eval_mask`` marks the rows the final model predicts (the test split).
    """

    def __init__(self, Z: np.ndarray, groups: np.ndarray, fit_mask: np.ndarray,
                 eval_mask: np.ndarray, n_folds: int = 5, seed: int = 20260912):
        Z = np.asarray(Z, dtype=np.float64)
        if not np.isfinite(Z).all():
            raise ValueError("Z contains missing values")
        fit_mask = np.asarray(fit_mask, bool)
        eval_mask = np.asarray(eval_mask, bool)
        if (fit_mask & eval_mask).any():
            raise ValueError("fitting and evaluation rows overlap")
        self.Z, self.groups = Z, np.asarray(groups)
        self.fit_idx = np.flatnonzero(fit_mask)
        self.eval_idx = np.flatnonzero(eval_mask)
        fit_groups = self.groups[self.fit_idx]
        if set(fit_groups) & set(self.groups[self.eval_idx]):
            raise ValueError("a session appears in both fitting and evaluation rows")
        self.n_folds = n_folds
        # n_folds = 0 builds the final fit only, for refits whose λ is fixed
        # (bootstrap and speaker-fold stability of the coupling map).
        self.fold = grouped_folds(fit_groups, n_folds, seed) if n_folds else None
        # (fit rows, evaluation rows) for each inner fold, then the final fit.
        self.splits = [(self.fit_idx[self.fold != f], self.fit_idx[self.fold == f])
                       for f in range(n_folds)] + [(self.fit_idx, self.eval_idx)]
        self._z = []
        for fi, ev in self.splits:
            zs = _Scaled.fit(Z[fi])
            Zf = zs(Z[fi])
            e, V = np.linalg.eigh(Zf.T @ Zf)
            e = np.clip(e, 0, None)
            ZeV = zs(Z[ev]) @ V
            self._z.append((fi, ev, zs, Zf, V, e, ZeV, ZeV.T @ ZeV))

    def folds(self, Y: np.ndarray) -> list[_FoldZ]:
        Y = np.asarray(Y, dtype=np.float64)
        out = []
        for fi, ev, zs, Zf, V, e, ZeV, G in self._z:
            yf = Y[fi]
            if not np.isfinite(yf).all():
                raise ValueError("Y has missing values on fitting rows")
            ym = yf.mean(axis=0)
            Yc = yf - ym
            R = Y[ev] - ym
            ok = bool(np.isfinite(R).all())
            out.append(_FoldZ(fi, ev, zs, Zf, V, e, ZeV, ym, V.T @ (Zf.T @ Yc), Yc, Y[ev], G,
                              (R ** 2).sum(axis=0) if ok else None,
                              ZeV.T @ R if ok else None))
        return out


@dataclass
class RidgeResult:
    """Output of one condition: held-out predictions plus selection diagnostics."""

    lambdas: np.ndarray               # chosen λ (absolute) per column group
    lambda_rel: np.ndarray            # chosen λ / n_fit
    at_grid_edge: np.ndarray          # bool per column group
    cv_sse: np.ndarray                # (n_lambda, n_groups) out-of-fold SSE
    pred_eval: np.ndarray             # (n_eval, m)
    oof_fit: np.ndarray | None        # (n_fit, m) out-of-fold predictions at the chosen λ
    beta_A: np.ndarray | None         # (q, m) final A-block coefficients (standardised A)
    edf: np.ndarray | None            # effective df per column group, final fit
    extra: dict = field(default_factory=dict)


class RidgeEngine:
    """Fits one target matrix Y for many choices of the extra block A."""

    def __init__(self, zcache: ZCache, Y: np.ndarray, col_groups: list[np.ndarray],
                 lambda_rel_grid=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)):
        self.zc = zcache
        self.Y = np.asarray(Y, dtype=np.float64)
        self.col_groups = [np.asarray(g) for g in col_groups]
        self.grid_rel = np.asarray(lambda_rel_grid, dtype=np.float64)
        self.folds = zcache.folds(self.Y)

    # -- core solve ------------------------------------------------------

    @staticmethod
    def _a_pieces(fz: _FoldZ, A: np.ndarray | None):
        if A is None:
            return None
        Af_raw = A[fz.fit]
        if not np.isfinite(Af_raw).all():
            raise ValueError("A has missing values on fitting rows")
        a_s = _Scaled.fit(Af_raw)
        Af = a_s(Af_raw)
        Ae = a_s(A[fz.ev])
        VtZtA = fz.V.T @ (fz.Zf.T @ Af)
        out = {"Af": Af, "Ae": Ae, "VtZtA": VtZtA, "AtA": Af.T @ Af, "AtY": Af.T @ fz.Yf,
               "ZeVtAe": fz.ZeV.T @ Ae, "AeAe": Ae.T @ Ae}
        if fz.rr is not None:
            out["AeR"] = Ae.T @ (fz.Ye - fz.ymean)
        return out

    @staticmethod
    def _sse(fz: _FoldZ, ap, gZ, bA, cols=None) -> np.ndarray:
        """Evaluation-row SSE per column from cross-products, without predictions.

        With r = y − ȳ_fit and ŷ − ȳ_fit = ZeV γ + Ae β_A:
        SSE = rᵀr − 2(γᵀZeVᵀr + β_AᵀAeᵀr) + γᵀGγ + 2γᵀ(ZeVᵀAe)β_A + β_Aᵀ(AeᵀAe)β_A,
        column by column. The cost per λ is O(p²·m) instead of O(n_eval·p·m).
        """
        if fz.rr is None:
            raise ValueError("evaluation targets are missing; SSE cannot be computed")
        c = slice(None) if cols is None else cols
        sse = fz.rr[c] - 2 * (gZ * fz.ZtR[:, c]).sum(0) + (gZ * (fz.G @ gZ)).sum(0)
        if bA is not None:
            sse = (sse - 2 * (bA * ap["AeR"][:, c]).sum(0)
                   + 2 * (gZ * (ap["ZeVtAe"] @ bA)).sum(0)
                   + (bA * (ap["AeAe"] @ bA)).sum(0))
        return sse

    @staticmethod
    def _solve(fz: _FoldZ, ap, lam: float):
        """β in the Z eigenbasis (γ_Z = Vᵀβ_Z) and β_A for one λ."""
        inv = 1.0 / (fz.e + lam)
        if ap is None:
            return inv[:, None] * fz.VtZtY, None, None
        KiZtA = inv[:, None] * ap["VtZtA"]                       # Vᵀ K⁻¹ ZᵀA
        S = ap["AtA"] + lam * np.eye(ap["AtA"].shape[0]) - ap["VtZtA"].T @ KiZtA
        rhs = ap["AtY"] - ap["VtZtA"].T @ (inv[:, None] * fz.VtZtY)
        bA = np.linalg.solve(S, rhs)
        gZ = inv[:, None] * (fz.VtZtY - ap["VtZtA"] @ bA)
        return gZ, bA, S

    @staticmethod
    def _predict(fz: _FoldZ, ap, gZ, bA, rows=None, cols=None):
        ZeV = fz.ZeV if rows is None else fz.ZeV[rows]
        pred = ZeV @ gZ + (fz.ymean if cols is None else fz.ymean[cols])
        if bA is not None:
            Ae = ap["Ae"] if rows is None else ap["Ae"][rows]
            pred = pred + Ae @ bA
        return pred

    # -- condition fit ---------------------------------------------------

    def run(self, A: np.ndarray | None = None, return_oof: bool = False,
            return_edf: bool = False, eval_sse_only: bool = False) -> RidgeResult:
        """Nested CV over λ, then a final fit predicting the evaluation rows.

        ``eval_sse_only`` returns the evaluation SSE per column (``extra["eval_sse"]``)
        instead of predictions; donor nulls and calibration need nothing else.
        Inner-fold SSE always uses the cross-product form unless out-of-fold
        predictions are requested.
        """
        if A is not None:
            A = np.asarray(A, dtype=np.float64)
        if self.zc.n_folds == 0 and len(self.grid_rel) > 1:
            raise ValueError("λ selection needs inner folds; this cache has none")
        m = self.Y.shape[1]
        L = len(self.grid_rel)
        sse = np.zeros((L, m))
        inner = self.folds[:-1]
        n_fit = len(self.zc.fit_idx)
        oof_all = np.zeros((L, n_fit, m)) if return_oof else None
        pos = {r: i for i, r in enumerate(self.zc.fit_idx)} if return_oof else None
        for fz in inner:
            ap = self._a_pieces(fz, A)
            n_f = len(fz.fit)
            for li, g in enumerate(self.grid_rel):
                gZ, bA, _ = self._solve(fz, ap, g * n_f)
                if return_oof:
                    pred = self._predict(fz, ap, gZ, bA)
                    sse[li] += ((fz.Ye - pred) ** 2).sum(axis=0)
                    oof_all[li, [pos[r] for r in fz.ev]] = pred
                else:
                    sse[li] += self._sse(fz, ap, gZ, bA)
        group_sse = np.stack([sse[:, g].sum(axis=1) for g in self.col_groups], axis=1)
        best = group_sse.argmin(axis=0)
        final = self.folds[-1]
        ap = self._a_pieces(final, A)
        n_f = len(final.fit)
        pred_eval = None if eval_sse_only else np.empty((len(final.ev), m))
        eval_sse = np.empty(m) if eval_sse_only else None
        beta_A = np.empty((A.shape[1], m)) if A is not None else None
        edf = np.empty(len(self.col_groups)) if return_edf else None
        if return_edf:
            ev_x = self._x_eigs(final, ap)
        for li in np.unique(best):
            cols = np.concatenate([g for g, b in zip(self.col_groups, best, strict=True)
                                   if b == li])
            gZ, bA, _ = self._solve(final, ap, self.grid_rel[li] * n_f)
            if eval_sse_only:
                eval_sse[cols] = self._sse(final, ap, gZ[:, cols],
                                           None if bA is None else bA[:, cols], cols=cols)
            else:
                pred_eval[:, cols] = self._predict(final, ap, gZ[:, cols],
                                                   None if bA is None else bA[:, cols],
                                                   cols=cols)
            if beta_A is not None:
                beta_A[:, cols] = bA[:, cols]
        if return_edf:
            for gi, li in enumerate(best):
                lam = self.grid_rel[li] * n_f
                edf[gi] = float((ev_x / (ev_x + lam)).sum())
        oof = None
        if return_oof:
            oof = np.empty((n_fit, m))
            for g, li in zip(self.col_groups, best, strict=True):
                oof[:, g] = oof_all[li][:, g]
        return RidgeResult(lambdas=self.grid_rel[best] * n_f, lambda_rel=self.grid_rel[best],
                           at_grid_edge=(best == 0) | (best == L - 1), cv_sse=group_sse,
                           pred_eval=pred_eval, oof_fit=oof, beta_A=beta_A, edf=edf,
                           extra={"eval_sse": eval_sse} if eval_sse_only else {})

    @staticmethod
    def _x_eigs(fz: _FoldZ, ap) -> np.ndarray:
        """Eigenvalues of the standardised XᵀX for the effective-df formula."""
        if ap is None:
            return fz.e
        ZtZ = (fz.V * fz.e) @ fz.V.T
        ZtA = fz.V @ ap["VtZtA"]
        XtX = np.block([[ZtZ, ZtA], [ZtA.T, ap["AtA"]]])
        return np.clip(np.linalg.eigvalsh(XtX), 0, None)

    # -- reduced-rank ridge (used by src/models/rrr.py) ------------------

    def rank_constrained(self, fz: _FoldZ, ap, lam: float, ranks):
        """Predictions on the fold's evaluation rows for each rank, plus factors.

        With Z profiled out, the A-block objective is ||S^{1/2}(β_A − β̂_A)||²
        + const, so the best rank-r β_A is β̂_A Q_r Q_rᵀ, where Q holds the right
        singular vectors of S^{1/2} β̂_A (Mukherjee & Zhu 2011, restricted to the
        A block after partialling out Z). β_Z is then re-solved given β_A.
        """
        inv = 1.0 / (fz.e + lam)
        gZ_full, bA, S = self._solve(fz, ap, lam)
        w, U = np.linalg.eigh(S)
        S_half = (U * np.sqrt(np.clip(w, 0, None))) @ U.T
        P, d, Qt = np.linalg.svd(S_half @ bA, full_matrices=False)
        out = {}
        for r in ranks:
            Q = Qt[:r].T
            bA_r = bA @ Q @ Q.T
            gZ = inv[:, None] * (fz.VtZtY - ap["VtZtA"] @ bA_r)
            out[r] = self._predict(fz, ap, gZ, bA_r)
        return out, {"beta_A": bA, "d": d, "Q": Qt.T, "S_half": S_half}


def block_col_groups(n_dims_per_block: list[int], n_replicates: int = 1) -> list[np.ndarray]:
    """Column groups for stacked targets: one group per (replicate, block)."""
    groups, start = [], 0
    width = sum(n_dims_per_block)
    for r in range(n_replicates):
        start = r * width
        for k in n_dims_per_block:
            groups.append(np.arange(start, start + k))
            start += k
    return groups


__all__ = ["grouped_folds", "ZCache", "RidgeEngine", "RidgeResult", "block_col_groups"]
