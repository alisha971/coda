"""Bootstrap interval comparison (BOOTSTRAP_COMPARISON_PROTOCOL.md). Synthetic only.

Usage: python docs/research/simulations/boot_compare.py [n_reps] [n_boot]
Writes docs/research/simulations/results/bootstrap_comparison.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.evaluation.bootstrap import interval_methods  # noqa: E402

SEED = 20261016
BLOCKS = [np.arange(0, 2), np.arange(2, 3), np.arange(3, 5), np.arange(5, 8)]
SCENARIOS = {
    "B1": dict(n_sess=50, per=300, frac=0.01, sd_sess=0.02, unequal=False),
    "B2": dict(n_sess=50, per=300, frac=0.01, sd_sess=0.05, unequal=False),
    "B3": dict(n_sess=25, per=300, frac=0.01, sd_sess=0.02, unequal=False),
    "B4": dict(n_sess=25, per=300, frac=0.01, sd_sess=0.05, unequal=False),
    "B5": dict(n_sess=50, per=300, frac=0.00, sd_sess=0.02, unequal=False),
    "B6": dict(n_sess=50, per=300, frac=0.01, sd_sess=0.05, unequal=True),
}
METHODS = ("percentile", "bca", "bootstrap_t")
ORDER = {m: i for i, m in enumerate(METHODS)}          # simpler first


def draw(rng, n_sess, per, frac, sd_sess, unequal):
    sizes = (np.maximum(20, np.round(per * rng.lognormal(-0.125, 0.5, n_sess))).astype(int)
             if unequal else np.full(n_sess, per))
    g = np.repeat(np.arange(n_sess), sizes)
    f = np.clip(frac + rng.normal(0, sd_sess, n_sess), -0.2, 0.5)[g]
    base = rng.gamma(2.0, 0.5, size=(g.size, 8)) * rng.gamma(4, 0.25, n_sess)[g][:, None]
    ef = base * (1 - f[:, None]) * rng.lognormal(0, 0.05, base.shape)
    return ef, base, g


def exact_ci(k, n):
    c = binomtest(k, n).proportion_ci(0.95, "exact")
    return [round(c.low, 4), round(c.high, 4)]


def main(n_reps=1000, n_boot=1000):
    rng = np.random.default_rng(SEED)
    res = {"protocol": "BOOTSTRAP_COMPARISON_PROTOCOL.md", "seed": SEED, "n_reps": n_reps,
           "n_boot": n_boot, "scenarios": {}}
    t0 = time.time()
    for sid, sc in SCENARIOS.items():
        pool = dict(sc, n_sess=max(20000 // sc["per"] * 50, 1000))
        ef, er, _ = draw(rng, **pool)
        truth = float(np.mean([1 - ef[:, b].sum() / er[:, b].sum() for b in BLOCKS]))
        hits = {m: 0 for m in METHODS}
        widths = {m: [] for m in METHODS}
        for r in range(n_reps):
            ef, er, g = draw(rng, **sc)
            o = interval_methods(ef, er, g, BLOCKS, n_boot=n_boot, seed=SEED + r)
            for m in METHODS:
                lo, hi = o[m]
                hits[m] += lo <= truth <= hi
                widths[m].append(hi - lo)
        res["scenarios"][sid] = {
            "settings": sc, "truth": truth,
            "methods": {m: {"coverage": hits[m] / n_reps, "exact_ci": exact_ci(hits[m], n_reps),
                            "median_width": float(np.median(widths[m]))} for m in METHODS}}
        print(sid, {m: res["scenarios"][sid]["methods"][m]["coverage"] for m in METHODS},
              flush=True)
    # selection rule, as fixed in the protocol
    sc = res["scenarios"].values()
    eligible = [m for m in METHODS if all(s["methods"][m]["exact_ci"][1] >= 0.95 for s in sc)]
    worst = {m: max(abs(s["methods"][m]["coverage"] - 0.95) for s in sc) for m in METHODS}
    if eligible:
        best = min(worst[m] for m in eligible)
        choice = min((m for m in eligible if worst[m] - best < 0.01), key=ORDER.get)
        rule = "eligible; smallest worst-case deviation (ties under 0.01 to the simpler)"
    else:
        lowest = {m: min(s["methods"][m]["coverage"] for s in sc) for m in METHODS}
        choice = max(METHODS, key=lambda m: (lowest[m], -ORDER[m]))
        rule = "no method eligible; highest worst-case coverage, reported as a limitation"
    res["selection"] = {"eligible": eligible, "worst_case_deviation": worst,
                        "recommended": choice, "rule_applied": rule}
    res["seconds"] = time.time() - t0
    out = ROOT / "docs" / "research" / "simulations" / "results" / "bootstrap_comparison.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(res["selection"])


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:3]))
