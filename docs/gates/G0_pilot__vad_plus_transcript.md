# Gate G0 — corpus and target viability (pilot)

**Verdict: PASS.**

| Criterion | Required | Observed |
|---|---|---|
| Channels usable | ≥ 80% | 96.7% of 916 |
| Feature droppage | < 15% | 6.1% |

- Sample: 40 sessions, stage `pilot`, selection seed 20260912.
- Speech regions for pitch: `vad_plus_transcript`.
- Droppage is the share of candidate-pair turns with < 10 voiced frames; pairs lost because either turn is unmeasurable: 11.5%.
- On failure the specification switches the primary corpus to AMI two-party stretches; a configuration change, not a redesign.
- The expectation for this gate was recorded before it was run (2026-09-23).
- Evidence: `results/audit/` and `figures/04_audit/`, produced by `notebooks/04_audit.ipynb`.
