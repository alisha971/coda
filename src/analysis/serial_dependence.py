"""Within-speaker serial dependence of standardised delivery (diagnostic D-AC).

Descriptive only. It places the real data on the sensitivity table of the power
projection (``src/experiments/power_projection.py``), where the projected MDE runs from
about 0.0005 (turns independent) to 0.0044 (lag-1 correlation 0.9 in both
partner delivery and reply residuals).

For each target dimension, on training rows only, it correlates the reply's
standardised value with the same speaker's own previous turn (the
self-previous vector S). It reports the raw correlation and the correlation
after both are residualised on Z_content, which is the residual J3 resamples.
No partner-delivery quantity is used, so the partner-to-reply relationship
under test is not looked at.

Because every speaker's turns enter as replies, the same figure describes the
serial dependence of partner delivery, which comes from the same speakers' turn
sequences.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.prosody import TARGET_DIMS


def _residualise(x: np.ndarray, Z: np.ndarray, rows: np.ndarray, ridge: float = 1e-3):
    zr = Z[rows]
    mu_z, mu_x = zr.mean(0), x[rows].mean(0)
    zc = zr - mu_z
    G = zc.T @ zc
    G[np.diag_indices_from(G)] += ridge * len(zr)
    b = np.linalg.solve(G, zc.T @ (x[rows] - mu_x))
    return (x - mu_x) - (Z - mu_z) @ b


def within_speaker_lag1(data, dims=TARGET_DIMS) -> pd.DataFrame:
    """Lag-1 correlation of each speaker's standardised delivery with their own previous turn."""
    rows = data.train & np.isfinite(data.S).all(1) & np.isfinite(data.Y_standardised).all(1)
    y, s = data.Y_standardised, data.S
    Zc = data.Z.content()
    ry, rs = _residualise(y, Zc, rows), _residualise(s, Zc, rows)
    out = []
    for j, d in enumerate(dims):
        out.append({"dimension": d, "n_pairs": int(rows.sum()),
                    "r_raw": float(np.corrcoef(y[rows, j], s[rows, j])[0, 1]),
                    "r_residual_on_Z_content": float(np.corrcoef(ry[rows, j], rs[rows, j])[0, 1])})
    return pd.DataFrame(out)


__all__ = ["within_speaker_lag1"]
