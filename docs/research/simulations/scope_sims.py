"""Design-validation simulations (synthetic data only).

See SIMULATION_PROTOCOL.md in this directory, written before this script was run.
These are not registered experiments and say nothing about conversation; they
establish what the registered Layer-1 design can distinguish and at what sample
size. Closed-form OLS stands in for nested-CV ridge: with ~20k test pairs and a
handful of predictors, shrinkage is immaterial to the comparisons made here.

Usage:  python docs/research/simulations/scope_sims.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261016
N_INTER, N_TURNS, N_TEXT = 11, 36, 8
GAMMA = np.array([0.30, 0.20, 0.15, 0, 0, 0, 0, 0])
OUT = Path(__file__).resolve().parent / "results"


# --------------------------------------------------------------------------- data
def simulate(rng, n_sessions, beta=0.0, c=0.0, rho=0.9, strata=None, delta=0.0):
    """Turn-level table for n_sessions dyads; vectorised over interactions.

    strata: probabilities (question, backchannel) per pair; the question stratum's
    coupling is beta * (1 + delta).
    """
    n_int = n_sessions * N_INTER
    sess = np.repeat(np.arange(n_sessions), N_INTER)
    first = rng.integers(0, 2, n_int)                      # who opens each interaction
    mu = rng.normal(0, 1, (n_sessions, 2))                 # speaker means (removed by z-scoring)
    sd = rng.uniform(0.7, 1.4, (n_sessions, 2))            # speaker scales
    x = rng.normal(0, 1, (n_int, N_TURNS, N_TEXT))
    e = rng.normal(0, 1, (n_int, N_TURNS))
    q = np.zeros((n_int, N_TURNS), bool)
    bc = np.zeros((n_int, N_TURNS), bool)
    if strata is not None:
        u = rng.random((n_int, N_TURNS))
        q, bc = u < strata[0], (u >= strata[0]) & (u < strata[0] + strata[1])
    d = np.zeros((n_int, N_TURNS))
    d[:, 0] = rng.normal(0, 1, n_int)
    for t in range(1, N_TURNS):
        d[:, t] = rho * d[:, t - 1] + np.sqrt(1 - rho ** 2) * rng.normal(0, 1, n_int)
    latent = np.zeros((n_int, N_TURNS))                    # speaker-free delivery component
    for t in range(N_TURNS):
        prev = latent[:, t - 1] if t > 0 else 0.0
        b_t = beta * (1 + delta * q[:, t])
        latent[:, t] = c * d[:, t] + b_t * prev + x[:, t] @ GAMMA + e[:, t]
    who = (first[:, None] + np.arange(N_TURNS)[None, :]) % 2
    y = mu[sess[:, None], who] + sd[sess[:, None], who] * latent
    # per-speaker z-score within session (held-out-speaker rule, approximated over all turns)
    ys = np.empty_like(y)
    for s in range(2):
        m = who == s
        flat = np.where(m, y, np.nan).reshape(n_sessions, -1)
        mean = np.nanmean(flat, 1)[sess]
        std = np.nanstd(flat, 1)[sess]
        ys[m] = ((y - mean[:, None]) / std[:, None])[m]
    return dict(sess=sess, y=ys, x=x, q=q, bc=bc, n_sessions=n_sessions)


def pairs(sim, t_min=2):
    """Pairs (partner t-1, reply t) with t >= 2 so the replier's own previous turn exists."""
    n_int = sim["y"].shape[0]
    ii, tt = np.meshgrid(np.arange(n_int), np.arange(t_min, N_TURNS), indexing="ij")
    ii, tt = ii.ravel(), tt.ravel()
    return dict(i=ii, t=tt, sess=sim["sess"][ii], y=sim["y"][ii, tt], A=sim["y"][ii, tt - 1],
                S=sim["y"][ii, tt - 2], X=sim["x"][ii, tt], XP=sim["x"][ii, tt - 1],
                q=sim["q"][ii, tt], bc=sim["bc"][ii, tt])


def split_sessions(rng, n_sessions):
    order = rng.permutation(n_sessions)
    n_tr, n_va = int(round(0.7 * n_sessions)), int(round(0.1 * n_sessions))
    lab = np.empty(n_sessions, "<U5")
    lab[order[:n_tr]] = "train"
    lab[order[n_tr:n_tr + n_va]] = "val"
    lab[order[n_tr + n_va:]] = "test"
    return lab


# ------------------------------------------------------------------------ donors
def n1_donor(rng, P, split_of_pair):
    """Partner value of a random pair from a different session in the same split."""
    A = P["A"].copy()
    for lab in ("train", "test"):
        idx = np.flatnonzero(split_of_pair == lab)
        pick = rng.choice(idx, idx.size)
        bad = P["sess"][pick] == P["sess"][idx]
        while bad.any():
            pick[bad] = rng.choice(idx, bad.sum())
            bad = P["sess"][pick] == P["sess"][idx]
        A[idx] = P["A"][pick]
    return A


def n2_donor(rng, P, y_turns, min_dist=10):
    """Same interaction, same partner speaker, >= min_dist turns from the partner turn."""
    p = P["t"] - 1
    j = np.empty_like(p)
    todo = np.ones(p.size, bool)
    while todo.any():
        # same parity = same speaker
        k = rng.integers(0, N_TURNS // 2, todo.sum()) * 2 + (p[todo] % 2)
        ok = np.abs(k - p[todo]) >= min_dist
        sel = np.flatnonzero(todo)[ok]
        j[sel] = k[ok]
        todo[sel] = False
    return y_turns[P["i"], j]


# --------------------------------------------------------------------- estimator
class Increment:
    """Held-out partial R2 of A over a fixed conditioning design (FWL, closed form)."""

    def __init__(self, Z, y, tr, te):
        self.Ztr, self.Zte, self.ytr, self.yte = Z[tr], Z[te], y[tr], y[te]
        self.tr, self.te = tr, te
        self.Pz = np.linalg.pinv(self.Ztr)
        cz = self.Pz @ self.ytr
        self.ry = self.ytr - self.Ztr @ cz
        self.sse0 = np.sum((self.yte - self.Zte @ cz) ** 2)

    def __call__(self, A):
        a_tr, a_te = A[self.tr], A[self.te]
        ra = a_tr - self.Ztr @ (self.Pz @ a_tr)
        b = (ra @ self.ry) / (ra @ ra)
        cz = self.Pz @ (self.ytr - b * a_tr)
        sse1 = np.sum((self.yte - self.Zte @ cz - b * a_te) ** 2)
        return 1 - sse1 / self.sse0


def design(P, with_self=False):
    cols = [np.ones(P["y"].size), *P["X"].T, *P["XP"].T]
    if with_self:
        cols.append(P["S"])
    return np.column_stack(cols)


# ------------------------------------------------------------------- experiments
def s1(rng, reps, draws, n_sessions=250):
    cells = [("null", 0.0, 0.0, 0.9), ("drift_rho0.90", 0.0, 0.6, 0.90),
             ("drift_rho0.97", 0.0, 0.6, 0.97), ("local", 0.10, 0.0, 0.9),
             ("both_rho0.90", 0.10, 0.6, 0.90)]
    rows = []
    for name, beta, c, rho in cells:
        for r in range(reps):
            sim = simulate(rng, n_sessions, beta=beta, c=c, rho=rho)
            P = pairs(sim)
            lab = split_sessions(rng, n_sessions)[P["sess"]]
            tr, te = lab == "train", lab == "test"
            n1 = [n1_donor(rng, P, lab) for _ in range(draws)]
            n2 = [n2_donor(rng, P, sim["y"]) for _ in range(draws)]
            for cond, with_self in (("Z", False), ("Z+self_previous", True)):
                inc = Increment(design(P, with_self), P["y"], tr, te)
                obs = inc(P["A"])
                null1 = np.array([inc(a) for a in n1])
                null2 = np.array([inc(a) for a in n2])
                p1 = np.mean(null1 >= obs)
                p2 = np.mean(null2 >= obs)
                n2_above_n1 = np.median(null2) > np.quantile(null1, 0.95)   # default rule
                reading = classify(p1, p2, n2_above_n1)
                rows.append(dict(cell=name, beta=beta, c=c, rho=rho, rep=r, condition=cond,
                                 observed_pr2=obs, n1_median=np.median(null1),
                                 n2_median=np.median(null2), p_vs_n1=p1, p_vs_n2=p2,
                                 n2_above_n1=n2_above_n1, reading=reading))
        print(f"S1 {name} done", flush=True)
    return pd.DataFrame(rows)


def classify(p1, p2, n2_above_n1, alpha=0.05):
    """The three registered readings of the N1/N2 ordering."""
    if p1 > alpha:
        return "real≈N1 (nothing)"
    if p2 <= alpha and n2_above_n1:
        return "real>N2>N1 (local adaptation)"
    if p2 > alpha and n2_above_n1:
        return "real≈N2>N1 (conversation-level style)"
    if p2 <= alpha and not n2_above_n1:
        return "real>N2≈N1 (unclassified)"
    return "real≈N2≈N1 with real>N1 (unclassified)"


def s2(rng, reps, draws):
    rows = []
    for n_sessions in (40, 100, 250):
        for beta in (0.0, 0.03, 0.045, 0.07, 0.10):
            for r in range(reps):
                sim = simulate(rng, n_sessions, beta=beta)
                P = pairs(sim)
                lab = split_sessions(rng, n_sessions)[P["sess"]]
                tr, te = lab == "train", lab == "test"
                inc = Increment(design(P), P["y"], tr, te)
                obs = inc(P["A"])
                null1 = np.array([inc(n1_donor(rng, P, lab)) for _ in range(draws)])
                rows.append(dict(n_sessions=n_sessions, n_test_pairs=int(te.sum()), beta=beta,
                                 rep=r, observed_pr2=obs, p_vs_n1=np.mean(null1 >= obs)))
            print(f"S2 S={n_sessions} beta={beta} done", flush=True)
    return pd.DataFrame(rows)


def cr1_wald(X, y, groups, j):
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    G = np.unique(groups, return_inverse=True)[1]
    scores = np.zeros((G.max() + 1, X.shape[1]))
    np.add.at(scores, G, X * e[:, None])
    n, k, g = X.shape[0], X.shape[1], scores.shape[0]
    V = XtX_inv @ (scores.T @ scores) @ XtX_inv * (g / (g - 1)) * ((n - 1) / (n - k))
    return b[j], np.sqrt(V[j, j])


def s3(rng, reps, n_sessions=250, alpha=0.05 / 3):
    from math import erf, sqrt
    rows = []
    for delta in (0.0, 0.5, 1.0):
        for r in range(reps):
            sim = simulate(rng, n_sessions, beta=0.07, strata=(0.06, 0.35), delta=delta)
            P = pairs(sim)
            A, q, bc = P["A"], P["q"].astype(float), P["bc"].astype(float)
            X = np.column_stack([design(P), A, q, bc, A * q, A * bc])
            j = X.shape[1] - 2
            est, se = cr1_wald(X, P["y"], P["sess"], j)
            z = est / se
            p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
            rows.append(dict(delta=delta, rep=r, n_question_pairs=int(q.sum()), estimate=est,
                             se=se, p=p, reject=p <= alpha))
        print(f"S3 delta={delta} done", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny run for a smoke test")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    k = dict(s1=(2, 9), s2=(3, 9), s3=10) if a.quick else dict(s1=(20, 100), s2=(100, 99), s3=200)
    d1 = s1(rng, *k["s1"])
    d2 = s2(rng, *k["s2"])
    d3 = s3(rng, k["s3"])
    tag = "_quick" if a.quick else ""
    d1.to_csv(OUT / f"S1{tag}.csv", index=False)
    d2.to_csv(OUT / f"S2{tag}.csv", index=False)
    d3.to_csv(OUT / f"S3{tag}.csv", index=False)
    rec = dict(seed=SEED, quick=a.quick, wall_seconds=round(time.time() - t0, 1),
               python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
               settings=k, created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (OUT / f"run_record{tag}.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
