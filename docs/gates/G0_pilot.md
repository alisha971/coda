# Gate G0 — corpus and target viability (pilot)

**Verdict: FAIL.**

| Criterion | Required | Observed |
|---|---|---|
| Channels usable | ≥ 80% | 96.7% of 916 |
| Feature droppage | < 15% | 17.6% |

- Sample: 40 sessions, stage `pilot`, selection seed 20260912.
- Droppage is the share of candidate-pair turns with < 10 voiced frames; pairs lost because either turn is unmeasurable: 31.3%.
- On failure the specification switches the primary corpus to AMI two-party stretches; a configuration change, not a redesign.
- The expectation for this gate was recorded before it was run (2026-09-23).
- Evidence: `results/audit/` and `figures/04_audit/`, produced by `notebooks/04_audit.ipynb`.
