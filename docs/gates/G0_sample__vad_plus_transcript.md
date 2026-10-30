# Gate G0 — corpus and target viability (sample)

**Verdict: PASS.**

| Criterion | Required | Observed |
|---|---|---|
| Channels usable | ≥ 80% | 97.2% of 5764 |
| Feature droppage | < 15% | 5.8% |

- Sample: 250 sessions, stage `sample`, selection seed 20260912.
- Speech regions for pitch: `vad_plus_transcript`.
- Droppage is the share of candidate-pair turns with < 10 voiced frames; pairs lost because either turn is unmeasurable: 10.9%.
- On failure the specification switches the primary corpus to AMI two-party stretches; a configuration change, not a redesign.
- The expectation for this gate was recorded before it was run (2026-09-23).
- Evidence: `results/audit/` and `figures/04_audit/`, produced by `notebooks/04_audit.ipynb`.
