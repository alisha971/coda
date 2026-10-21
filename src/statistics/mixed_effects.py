"""The mixed-effects re-check and the J4 interaction.

``delivery ~ partner delivery + Z + (1|speaker) + (1|dyad)`` — the entrainment
literature's own tool, fitted with statsmodels ``MixedLM``.

**Random-effects structure.** In this sample each speaker belongs to exactly
one session, so speakers are nested in sessions. The model uses
session as the grouping factor with a random intercept, and the replier as a
variance component inside the session — the nested form of the registered
crossed structure, which is what (1|speaker) + (1|dyad) reduces to when no
speaker spans two dyads.

**Never simplified.** If the fit does not converge, that is recorded and
reported as a convergence limitation (Prereg §8); the random-effects structure
is not reduced to force convergence.

**Tests.** The partner-delivery block is tested by a Wald test of its
coefficients; the J4 interaction by a Wald test of the function × partner
terms. Each target dimension is fitted separately. How the per-dimension
interaction tests combine into one H3 p-value is ``Settings.h3_aggregation``
(Holm-adjusted minimum over dimensions).

Barr, Levy, Scheepers & Tily (2013) argue for the maximal random-effects
structure the design justifies in confirmatory tests. Random slopes for partner
delivery by speaker are not part of the registered formula and are not added.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def _wald(res, names: list[str]) -> dict:
    params = list(res.params.index)     # fixed effects followed by variance parameters
    R = np.zeros((len(names), len(params)))
    for i, n in enumerate(names):
        R[i, params.index(n)] = 1.0
    w = res.wald_test(R, scalar=True)
    return {"statistic": float(np.asarray(w.statistic).squeeze()),
            "df": len(names), "p": float(np.asarray(w.pvalue).squeeze())}


def _fit(formula: str, data: pd.DataFrame, group: str, speaker: str):
    import statsmodels.formula.api as smf

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = smf.mixedlm(formula, data, groups=data[group], re_formula="1",
                            vc_formula={"speaker": f"0 + C({speaker})"})
        res = model.fit(reml=True, method=["lbfgs"])
    msgs = sorted({str(w.message) for w in caught})
    return res, {"converged": bool(getattr(res, "converged", False)), "warnings": msgs}


def _failed(target: str, err: Exception) -> dict:
    """A fit that could not be completed is a reported limitation, not a crash."""
    return {"target": target, "fit_failed": True, "converged": False,
            "error": f"{type(err).__name__}: {err}",
            "note": "reported as a fitting limitation; the random-effects structure is not "
                    "simplified (Prereg §8)"}


def partner_model(data: pd.DataFrame, target: str, partner_cols: list[str],
                  z_cols: list[str], group: str = "session_key",
                  speaker: str = "replier") -> dict:
    """Fit the re-check for one target dimension; Wald test of the partner block."""
    formula = f"{target} ~ " + " + ".join(partner_cols + z_cols)
    try:
        res, conv = _fit(formula, data, group, speaker)
    except (np.linalg.LinAlgError, ValueError) as err:
        return _failed(target, err)
    return {"fit_failed": False, "target": target, "n": int(res.nobs),
            "groups": int(data[group].nunique()),
            **conv, "partner_coefficients": res.fe_params[partner_cols].to_dict(),
            "partner_se": res.bse_fe[partner_cols].to_dict(),
            "wald_partner_block": _wald(res, partner_cols)}


def interaction_model(data: pd.DataFrame, target: str, partner_cols: list[str],
                      z_cols: list[str], function_col: str = "function",
                      group: str = "session_key", speaker: str = "replier") -> dict:
    """J4: function × partner-delivery interaction for one target dimension."""
    levels = sorted(data[function_col].unique())
    if len(levels) < 2:
        raise ValueError("the interaction needs at least two function categories")
    d = data.copy()
    ref = levels[0]
    inter = []
    for lv in levels[1:]:
        dummy = f"fn_{lv}"
        d[dummy] = (d[function_col] == lv).astype(float)
        for pc in partner_cols:
            name = f"{pc}_x_{lv}"
            d[name] = d[pc] * d[dummy]
            inter.append(name)
    dummies = [f"fn_{lv}" for lv in levels[1:]]
    formula = f"{target} ~ " + " + ".join(partner_cols + dummies + inter + z_cols)
    try:
        res, conv = _fit(formula, d, group, speaker)
    except (np.linalg.LinAlgError, ValueError) as err:
        return {**_failed(target, err), "reference_category": ref}
    return {"fit_failed": False, "target": target, "reference_category": ref,
            "n": int(res.nobs), **conv,
            "wald_interaction": _wald(res, inter),
            "interaction_coefficients": res.fe_params[inter].to_dict()}


def aggregate_h3(per_dimension_p: dict[str, float | None], rule: str) -> float | None:
    """One H3 p-value from per-dimension interaction tests.

    If any dimension's fit failed, no aggregate is formed (None): dropping the
    failed dimensions would change the registered test without saying so.
    """
    if rule != "holm_min_over_dimensions":
        raise ValueError(f"unknown H3 aggregation {rule!r}")
    if any(v is None for v in per_dimension_p.values()):
        return None
    p = np.sort(np.asarray(list(per_dimension_p.values()), float))
    m = len(p)
    return float(min(1.0, np.min([(m - i) * p[i] for i in range(m)])))


__all__ = ["partner_model", "interaction_model", "aggregate_h3"]
