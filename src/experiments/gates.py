"""Decision gates and the stop rule, as executable rules (Prereg §10).

Each gate returns a :class:`GateResult` with one of five statuses:

- ``PASS`` / ``FAIL`` — evaluated with the registered criteria;
- ``PROVISIONAL`` — evaluated under development settings (``confirmatory=False``,
  as in synthetic dry runs); the provisional outcome is kept in ``evidence`` and
  cannot license a confirmatory claim;
- ``BLOCKED`` — cannot be evaluated: a needed value has no source (for G2, the
  plausible effect range);
- ``NOT_RUN`` — its inputs do not exist yet.

:func:`require` turns gate states into permission: G1 FAIL stops everything
downstream, G2 must PASS before J1 is confirmatory, and T1/T2 can never
block a Layer-1 step. Thresholds come from configuration or from
:class:`src.settings.Settings`; none is chosen here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

STATUSES = ("PASS", "FAIL", "PROVISIONAL", "BLOCKED", "NOT_RUN", "NOT_EVALUATED")
LAYER1_GATES = ("G0", "G1", "G2", "G3", "G4", "G5")
NEURAL_GATES = ("T1", "T2")


class GateBlocked(RuntimeError):
    """A step was requested whose prerequisite gate does not permit it."""


@dataclass
class GateResult:
    gate: str
    status: str
    criterion: str
    source: str
    evidence: dict = field(default_factory=dict)
    settings: list[str] = field(default_factory=list)
    consequence: str = ""

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"unknown gate status {self.status!r}")

    def as_dict(self) -> dict:
        return asdict(self)


def _status(passed: bool, confirmatory: bool) -> str:
    if not confirmatory:
        return "PROVISIONAL"
    return "PASS" if passed else "FAIL"


def mc_test_size(alpha: float, n_draws: int) -> float:
    """Exact size of the donor test with ``n_draws`` draws under exchangeability.

    p = (number of null draws >= observed) / B and the test rejects when p <= alpha,
    so it rejects when the observed value ranks in the top floor(alpha*B) + 1 of
    B + 1 exchangeable values: size (floor(alpha*B) + 1) / (B + 1). With B = 500
    and alpha = 0.05 this is 26/501 = 0.0519, slightly above alpha by design.
    """
    return (np.floor(alpha * n_draws + 1e-12) + 1) / (n_draws + 1)


def _cp_upper(k: int, n: int, conf: float) -> float:
    from scipy.stats import beta

    return 1.0 if k >= n else float(beta.ppf(conf, k + 1, n - k))


def g1(rates: dict[str, dict], settings, n_draws: int | None = None) -> GateResult:
    """β = 0 rejection rate for N1 and N2 at nominal level (Layer 1).

    ``Settings.g1_rule["acceptance"]``:

    - ``bradley_liberal_upper_bound``: pass when, for each
      scheme, the one-sided 95% Clopper-Pearson upper bound of the rejection rate
      is at most 1.5 alpha (Bradley's liberal criterion). At R0 = 2,000 a valid
      test (true rate 0.0519) passes both schemes with probability 0.99; a rate of
      0.075 passes with probability 0.002.
    - ``rejection_rate_within_binomial_95ci_of_alpha``: the original provisional
      rule, kept for comparison.

    A rate well below alpha is reported as conservatism, not as a failure.
    """
    rule = settings.g1_rule
    alpha = float(rule["alpha"])
    acc = rule["acceptance"]
    if acc == "bradley_liberal_upper_bound":
        bound = 1.5 * alpha
        upper = {k: _cp_upper(int(r["rejections"]), int(r["replicates"]), 0.95)
                 for k, r in rates.items()}
        ok = {k: upper[k] <= bound for k in rates}
        extra = {"upper_bound_95": upper, "liberal_bound": bound}
        crit = "one-sided 95% upper bound of the β = 0 rejection rate ≤ 1.5α, N1 and N2"
    elif acc == "rejection_rate_within_binomial_95ci_of_alpha":
        ok = {k: r["ci95"][0] <= alpha <= r["ci95"][1] for k, r in rates.items()}
        extra = {}
        crit = "β = 0 rejection rate at nominal level for N1 and N2"
    else:
        raise ValueError(f"unknown G1 acceptance rule {acc!r}")
    passed = set(rates) >= {"N1", "N2"} and all(ok.values())
    evidence = {"alpha": alpha, "rates": rates, "passes": ok, "acceptance": acc,
                "provisional_outcome": "PASS" if passed else "FAIL", **extra}
    if n_draws:
        evidence["mc_test_exact_size"] = float(mc_test_size(alpha, n_draws))
        evidence["n_draws"] = int(n_draws)
    return GateResult("G1", _status(passed, settings.confirmatory), crit, "Prereg §10",
                      evidence, ["g1_rule"],
                      "stop and debug; nothing downstream runs" if not passed else "")


#: ``Settings.g2_plausible_range`` value recording the registered G2 rule: G2 is a
#: reporting requirement.
G2_REPORTING_ONLY = "reporting_only"


def g2(mde: dict, settings) -> GateResult:
    """MDE at achieved n below the plausible effect range (Layer 1).

    Under the registered G2 rule (``g2_plausible_range = "reporting_only"``) G2 is
    a reporting requirement: it returns ``NOT_EVALUATED`` with the MDE and
    whether it was stated, and it can never return PASS.
    """
    rng = settings.g2_plausible_range
    if rng == G2_REPORTING_ONLY:
        stated = mde.get("aggregate") is not None
        return GateResult(
            "G2", "NOT_EVALUATED",
            "reporting requirement (registered rule): MDE and power stated; no effect-size "
            "threshold is established, so no pass/fail result exists",
            "Prereg §10; the registered G2 reporting rule replaces the pass criterion",
            {"mde": mde, "mde_stated": stated, "reporting_rule_in_force": settings.confirmatory},
            ["mde_power", "g2_plausible_range"],
            "" if stated else "J3 did not reach power 0.80 on the grid: report power at each "
            "grid value and state that the MDE exceeds the largest value tested")
    if rng is None:
        return GateResult("G2", "BLOCKED", "MDE below the plausible effect range from the "
                          "entrainment literature", "Prereg §10",
                          {"mde": mde}, ["g2_plausible_range", "mde_power"],
                          "no document states the plausible range numerically; it must be "
                          "supplied with its literature source before G2 can be evaluated")
    bound = float(rng)
    aggregate = mde.get("aggregate")
    passed = aggregate is not None and aggregate < bound
    return GateResult("G2", _status(passed, settings.confirmatory), f"aggregate MDE < {bound}",
                      "Prereg §10", {"mde": mde, "bound": bound},
                      ["mde_power", "g2_plausible_range"],
                      "" if passed else "add conversations, or reduce the target to the three "
                      "best-measured blocks; J1 is not confirmatory until this passes")


#: What G3 may say. The registered reading "real > N2 > N1 → local adaptation"
#: is never issued: a synthetic corpus with shared drift and no local coupling
#: produces real > N1 and real > N2, so the N2
#: contrast cannot identify a local mechanism. The ordering is reported as
#: predictive dependence, and the mechanism as not identified.
G3_PATTERNS = {
    "adjacent": "real > N1 and real > N2: the adjacent partner turn predicts the reply beyond "
                "distant same-speaker turns. Local adaptation and shared dependence that changes "
                "within about ten turns are NOT distinguished (H4 inconclusive about mechanism)",
    "distant": "real > N1, real ≈ N2: distant same-speaker turns predict as well as the adjacent "
               "one; consistent with conversation-level style",
    "none": "real ≈ N1: no predictive dependence on the genuine pairing detected",
    "inconsistent": "real ≈ N1 but real > N2: internally inconsistent; reported as observed",
}


def g2_sensitivity(levels: list[dict], candidates=(0.0025, 0.005, 0.01),
                   power_level: float = 0.80) -> dict:
    """Detectability at candidate effect sizes. A report, never a gate verdict.

    ``levels`` are J3 (or projection) levels with ``target_partial_r2`` and a
    rejection rate. For each candidate value the measured power is given (when
    the value is on the grid) or linearly interpolated, together with whether
    power reaches ``power_level``. None of the candidates is a registered or
    literature-derived threshold; G2 itself
    stays a reporting requirement.
    """
    pts = sorted((x["target_partial_r2"], x["rejection"]["rate"]) for x in levels
                 if x["target_partial_r2"] > 0)
    t = np.array([a for a, _ in pts])
    pw = np.array([b for _, b in pts])
    rows = []
    for c in candidates:
        measured = bool(np.isclose(t, c).any())
        est = float(np.interp(c, t, pw)) if len(t) else float("nan")
        rows.append({"candidate_partial_r2": c, "power": est,
                     "measured_on_grid": measured,
                     "detectable_at_power_level": bool(est >= power_level)})
    return {"status": "SENSITIVITY ONLY — G2 not evaluated", "power_level": power_level,
            "candidates": rows,
            "note": "candidate values are not literature-derived thresholds"}


def g3(p_n1: float, p_n2: float, null_n1: np.ndarray, null_n2: np.ndarray,
       settings) -> GateResult:
    """The N1/N2 ordering, reported without a mechanism claim.

    The registered readings name a mechanism ("local adaptation"). The design
    identifies only predictive dependence on the adjacent turn relative to
    distant turns, so the mechanism field is always ``not identified``.
    """
    rule = settings.g3_rule
    alpha = float(rule["alpha"])
    if rule["n2_vs_n1"] != "n2_null_median_above_n1_null_q95":
        raise ValueError(f"unknown N2-vs-N1 rule {rule['n2_vs_n1']!r}")
    real_gt_n1 = p_n1 <= alpha
    real_gt_n2 = p_n2 <= alpha
    n2_gt_n1 = float(np.median(null_n2)) > float(np.quantile(null_n1, 0.95))
    if real_gt_n1 and real_gt_n2:
        key = "adjacent"
    elif real_gt_n1:
        key = "distant"
    elif real_gt_n2:
        key = "inconsistent"
    else:
        key = "none"
    return GateResult("G3", "PROVISIONAL", "N1/N2 ordering reported as predictive dependence",
                      "Prereg §10 (H4 identification)",
                      {"p_vs_N1": p_n1, "p_vs_N2": p_n2, "real_gt_N1": real_gt_n1,
                       "real_gt_N2": real_gt_n2, "N2_gt_N1": n2_gt_n1, "pattern": key,
                       "reading": G3_PATTERNS[key], "mechanism": "not identified"},
                      ["g3_rule"],
                      "H4 cannot separate local adaptation from shared short-timescale drift; "
                      "no local mechanism may be claimed")


def g5(h3_p_holm: float | None, alpha: float, power_at_mde: dict | None,
       strata: list[str] | None = None) -> GateResult:
    """Interaction significant after correction, or credibly null with stated power.

    A null counts as credible only when ``power_at_mde["per_stratum_mde"]`` gives
    a finite MDE for every confirmatory stratum. Whole-sample power is
    not a substitute, and no scaling rule from the whole sample is assumed.
    """
    if h3_p_holm is None:
        return GateResult("G5", "NOT_RUN", "J4 interaction", "Prereg §10", {}, ["h3_aggregation"])
    sig = h3_p_holm <= alpha
    per = (power_at_mde or {}).get("per_stratum_mde") or {}
    stated = bool(strata) and all(per.get(c) is not None and np.isfinite(per[c])
                                  for c in strata)
    credible_null = (not sig) and stated
    return GateResult("G5", "PROVISIONAL", "significant after correction, or credibly null "
                      "with stated power", "Prereg §10",
                      {"p_holm": h3_p_holm, "significant": sig, "power": power_at_mde,
                       "provisional_outcome": "PASS" if (sig or credible_null) else "FAIL"},
                      ["h3_aggregation"], "" if sig else "report as null with per-stratum MDE")


def t1(p_value: float | None, settings) -> GateResult:
    """Neural trainability: recovery of the injected effect at partial R² ≈ 0.03."""
    if p_value is None:
        return GateResult("T1", "NOT_RUN", "recovery at partial R² ≈ 0.03", "Prereg §10", {},
                          ["neural_gate_rules"])
    rule = settings.neural_gate_rules
    passed = p_value <= float(rule["alpha"])
    return GateResult("T1", _status(passed, settings.confirmatory),
                      "neural Δ at the injected effect exceeds its 30-draw N1 null",
                      "Prereg §10", {"p": p_value}, ["neural_gate_rules"],
                      "" if passed else "debug once; then report a negative technical result")


def t2(rate: dict | None, settings) -> GateResult:
    if rate is None:
        return GateResult("T2", "NOT_RUN", "β = 0 rejection rate of the 30-draw null", "Prereg §10",
                          {}, ["neural_gate_rules"])
    alpha = float(settings.neural_gate_rules["alpha"])
    passed = rate["ci95"][0] <= alpha <= rate["ci95"][1]
    return GateResult("T2", _status(passed, settings.confirmatory),
                      "at nominal level within Monte-Carlo error", "Prereg §10", {"rate": rate},
                      ["neural_gate_rules"],
                      "" if passed else "neural Δ reported descriptively, no p-value")


def require(gates: dict[str, GateResult], step: str) -> str:
    """Permission for a step, or :class:`GateBlocked`.

    Returns the label the step's outputs must carry: ``confirmatory`` or
    ``not_confirmatory: <reason>``.
    """
    def st(g):
        return gates[g].status if g in gates else "NOT_RUN"

    if step in ("j1", "j2", "h2", "j4", "neural"):
        if st("G1") == "FAIL":
            raise GateBlocked("G1 failed: stop and debug; nothing downstream runs")
    def g2_ok():
        # Registered G2 rule: G2 is satisfied as a reporting requirement once the J3 MDE
        # has been computed and stated (G2 = NOT_EVALUATED under the registered
        # reporting rule); it is never a PASS.
        if st("G2") == "PASS":
            return True
        g = gates.get("G2")
        return (st("G2") == "NOT_EVALUATED" and g is not None
                and bool(g.evidence.get("reporting_rule_in_force"))
                and "mde" in g.evidence)

    if step in ("j1", "j2", "h2"):
        if st("G1") == "PASS" and g2_ok():
            return "confirmatory"
        return f"not_confirmatory: G1={st('G1')}, G2={st('G2')}"
    if step == "j4":
        if st("G4") == "FAIL":
            return "exploratory: G4 failed"
        if st("G1") == "PASS" and g2_ok() and st("G4") == "PASS":
            return "confirmatory"
        return f"not_confirmatory: G1={st('G1')}, G2={st('G2')}, G4={st('G4')}"
    if step == "neural":
        return "corroborating (no confirmatory α)"
    raise ValueError(f"unknown step {step!r}")


def stop_rule(state: dict) -> dict:
    """The stop rule, evaluated on the recorded state of the study.

    ``state`` keys: ``delta_n1``, ``delta_n2`` (J1/J2 done on the speaker-disjoint
    split), ``mde_stated`` (J3 done), ``g4`` and ``g5`` statuses, and for the
    neural arm ``t1``, ``t2`` statuses and ``ablations_run``.
    """
    part1 = bool(state.get("delta_n1") and state.get("delta_n2") and state.get("mde_stated"))
    g4, g5_ = state.get("g4"), state.get("g5")
    if g4 == "FAIL" and state.get("g4_no_category"):
        part2, how = True, "no category reliable: shipped at Level D"
    else:
        part2 = g5_ in ("PASS", "FAIL", "PROVISIONAL") and g5_ is not None
        how = "function-dependence answered by J4" if part2 else "J4 not yet answered"
    neural_done = (state.get("t1") == "PASS" and state.get("t2") == "PASS"
                   and bool(state.get("ablations_run")))
    g2_state = state.get("g2")
    licensed = state.get("g1") == "PASS" and (
        g2_state == "PASS" or (g2_state == "NOT_EVALUATED" and bool(state.get("mde_stated"))))
    return {"complete": part1 and part2, "part_i_delta_under_N1_N2_with_MDE": part1,
            "part_ii_function_dependence": part2, "part_ii_how": how,
            "claims_licensed": licensed,
            "claims_note": ("G2 not evaluated (registered rule): a rejection licenses "
                            "conditional predictive dependence only, with no claim that its "
                            "size matters; a non-rejection is 'no measurable contribution "
                            "above the stated MDE'") if licensed and g2_state == "NOT_EVALUATED"
            else "" if licensed else
            "estimates exist but G1 has not passed, or G2 is neither passed nor reported "
            "under the registered rule; no confirmatory claim may be made from them",
            "neural_arm_finished": neural_done,
            "note": "no experiment is added after completion"}


__all__ = ["GateResult", "GateBlocked", "STATUSES", "G3_PATTERNS", "mc_test_size", "g1", "g2",
           "g2_sensitivity", "G2_REPORTING_ONLY", "g3", "g5", "t1", "t2", "require",
           "stop_rule"]
