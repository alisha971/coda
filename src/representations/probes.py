"""Linear probes on the pooled neural partner representation.

**Interpretable probe.** A ridge from the pooled representation a_i to the
partner's eight interpretable dimensions, fitted on training speakers and scored
on held-out speakers; per-dimension R². It says what the learned
representation encodes in the terms of the statistical instrument.

**Speaker-identity probe.** Reported beside every neural Δ (speaker
shortcut row): a multinomial logistic regression predicting speaker identity
from a_i, on held-out speakers' turns split within each speaker, against the
chance rate. A high value means a_i carries identity; the speaker-disjoint test
split is what makes that shortcut unhelpful for the target.

Linear probes, not saliency or attention weights (attention weights are
not explanations; Jain & Wallace 2019).
"""

from __future__ import annotations

import numpy as np


def interpretable_probe(a_train: np.ndarray, y_train: np.ndarray, a_test: np.ndarray,
                        y_test: np.ndarray, alphas=(0.1, 1.0, 10.0, 100.0)) -> dict:
    from sklearn.linear_model import RidgeCV

    m = RidgeCV(alphas=alphas).fit(a_train, y_train)
    pred = m.predict(a_test)
    ss_res = ((y_test - pred) ** 2).sum(axis=0)
    ss_tot = ((y_test - y_test.mean(axis=0)) ** 2).sum(axis=0)
    return {"per_dimension_r2": (1 - ss_res / ss_tot).tolist(), "alpha": float(m.alpha_)}


def speaker_probe(a: np.ndarray, speakers: np.ndarray, seed: int = 0) -> dict:
    """Identity decodability among held-out speakers (half of each speaker's turns)."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    tr = np.zeros(len(speakers), bool)
    for s in np.unique(speakers):
        idx = np.flatnonzero(speakers == s)
        tr[rng.permutation(idx)[: len(idx) // 2]] = True
    clf = LogisticRegression(max_iter=2000).fit(a[tr], speakers[tr])
    acc = float((clf.predict(a[~tr]) == speakers[~tr]).mean())
    return {"accuracy": acc, "chance": 1.0 / len(np.unique(speakers)),
            "n_speakers": int(len(np.unique(speakers)))}


__all__ = ["interpretable_probe", "speaker_probe"]
