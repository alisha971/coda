# Design-validation simulations — results

**Run:** 2026-10-16, `python docs/research/simulations/scope_sims.py`, seed 20261016, 2,163 s on a local CPU (Python 3.11.1, NumPy 1.26.4, pandas 2.0.2). Raw rows: `results/S1.csv`, `results/S2.csv`, `results/S3.csv`; run record: `results/run_record.json`. Protocol and expectations, written before the run: `SIMULATION_PROTOCOL.md`.

**Status of these numbers.** Synthetic data from a stated generating model. They describe the behaviour of the registered **design and decision rules**, not conversation. Magnitudes depend on the chosen generating parameters; the qualitative conclusions are what carry forward.

**Reading note.** pandas reads the literal cell label `null` as missing; read `S1.csv` with `keep_default_na=False`.

---

## S1 — Does N2 separate local adaptation from slowly drifting shared state?

250 sessions, 20 replicates per cell, 100 donor draws per control. Rejection at α = 0.05; reading assigned with the registered orderings and the MD-20 rule for N2 > N1.

| Generating model | Conditioning | Observed partial R² (mean, SD) | N2 null median | Reject vs N1 | Reject vs N2 | Registered reading obtained |
|---|---|---|---|---|---|---|
| null (c = 0, β = 0) | Z | −0.0000 | −0.0000 | 0/20 | 0/20 | real ≈ N1 ("nothing"), 20/20 |
| null | Z + self-previous | −0.0000 | −0.0000 | 0/20 | 0/20 | real ≈ N1, 20/20 |
| **drift only**, ρ = 0.90 | Z | 0.053 (0.004) | 0.001 | 20/20 | 20/20 | **"local adaptation", 20/20** |
| drift only, ρ = 0.90 | Z + self-previous | 0.039 (0.003) | 0.001 | 20/20 | 20/20 | **"local adaptation", 20/20** |
| **drift only**, ρ = 0.97 | Z | 0.057 (0.006) | 0.017 | 20/20 | 20/20 | **"local adaptation", 20/20** |
| drift only, ρ = 0.97 | Z + self-previous | 0.040 (0.004) | 0.012 | 20/20 | 20/20 | **"local adaptation", 20/20** |
| local only (β = 0.10) | Z | 0.010 (0.002) | −0.0000 | 20/20 | 20/20 | **unclassified (real > N2 ≈ N1, finding A-2), 20/20** |
| local only | Z + self-previous | 0.010 (0.002) | −0.0000 | 20/20 | 20/20 | unclassified, 20/20 |
| both, ρ = 0.90 | Z | 0.113 (0.006) | 0.002 | 20/20 | 20/20 | "local adaptation", 20/20 |
| both, ρ = 0.90 | Z + self-previous | 0.081 (0.004) | 0.001 | 20/20 | 20/20 | "local adaptation", 20/20 |

**Against the expectations recorded before the run.**

1. *Null:* as expected; 0/40 rejections (mean p against N1 = 0.63, slightly conservative, consistent with the registered `≥` p-value and 20 replicates).
2. *Drift only:* real > N1 always, as expected. Real > N2 occurred in **every** replicate, including ρ = 0.97, where a turn ten apart retains about 74% of the drift correlation. This is stronger than expected: with ~19,000 test pairs, even a small loss of shared state between lag 1 and lag ≥ 10 is detected. **The expectation that conditioning on the replier's own previous turn would remove most of the spurious increment was wrong.** It removed 27–30%, and the reading did not change. The partner turn is a second noisy measurement of the same shared state, so it stays informative after the replier's own previous turn is added.
3. *Local only:* as expected; real > N1 and real > N2 with N2 ≈ N1, which none of the three registered readings covers.
4. *Both:* as expected; the increment over Z + self-previous is smaller but clearly non-zero.

**What this shows about the registered design.**

- Under the registered decision rules, **the "local adaptation" reading is produced by slowly drifting shared state with no local coupling at all**, provided the shared state decorrelates at all over ten turns. A real-data result of real > N2 > N1 therefore cannot, on its own, be written as local adaptation.
- **Genuine local coupling without conversation-level state is unclassifiable** under the registered orderings.
- The planned self-previous condition does not repair this.

**What the design can still claim.** Real > N2 identifies dependence on the partner turn that is **stronger at adjacency than at a distance of ten or more turns inside the same interaction**. That includes local adaptation and fast-changing shared state, and the design cannot separate them. This is a statement about interpretation, which can be fixed before J1.

---

## S2 — Detectability at the registered sample size (local coupling only)

100 replicates per cell, 99 N1 draws, α = 0.05. "Power" is the share of replicates with p ≤ 0.05 against N1.

| Sessions | Test pairs | β | Realised partial R² (mean) | Power |
|---|---|---|---|---|
| 40 | 2,992 | 0 | −0.0001 | 0.02 |
| 40 | 2,992 | 0.030 | 0.0006 | 0.51 |
| 40 | 2,992 | 0.045 | 0.0021 | 0.82 |
| 40 | 2,992 | 0.070 | 0.0051 | 0.98 |
| 100 | 7,480 | 0 | −0.0000 | 0.09 |
| 100 | 7,480 | 0.030 | 0.0009 | 0.91 |
| 100 | 7,480 | 0.045 | 0.0016 | 0.90 |
| 100 | 7,480 | 0.070 | 0.0050 | 1.00 |
| 250 | 18,700 | 0 | −0.0000 | 0.07 |
| 250 | 18,700 | 0.030 | 0.0009 | 0.97 |
| 250 | 18,700 | 0.045 | 0.0021 | 1.00 |
| 250 | 18,700 | 0.070 | 0.0049 | 1.00 |

(β = 0.10, partial R² ≈ 0.0099, gives power ≥ 0.99 at every size.) Monte-Carlo standard error of a rejection rate near 0.05 with 100 replicates is about 0.022, so the β = 0 rates (0.02, 0.09, 0.07) are compatible with nominal size. The value at 100 sessions is about two standard errors high and is worth re-checking in J3.

**Against expectations.** Better than expected. Under this generating model, a single-dimension partial R² of about **0.001** is detected with power 0.97 at 250 sessions, and about 0.002 with power 0.82 even on the 40-session pilot.

**Caveats that make real-data MDEs larger.** There is no between-speaker heterogeneity in coupling, no drift (which adds within-interaction clustering), no measurement error beyond Gaussian noise, ordinary least squares rather than nested-CV ridge, and one dimension standing in for a block. The registered J3 on real residuals is the binding test (G1, G2). The conclusion is only that **250 sessions is not obviously underpowered** for the 0.0025–0.005 range considered in the prior-work review.

---

## S3 — Power for the H3 function interaction

250 sessions; question → answer stratum ≈ 6% of pairs (≈ 5,600 pairs); base coupling β = 0.07 (partial R² ≈ 0.005); question-stratum coupling × (1 + δ); session-clustered (CR1) Wald test at α = 0.05/3; 200 replicates.

| δ | Question pairs (mean) | Interaction estimate (mean) | Clustered SE (mean) | Power |
|---|---|---|---|---|
| 0 | 5,608 | −0.001 | 0.013 | 0.01 (size) |
| 0.5 | 5,608 | 0.034 | 0.013 | 0.58 |
| 1.0 | 5,604 | 0.071 | 0.013 | 1.00 |

**Against expectations.** Better than expected for δ = 1.0 (expected "moderate"); δ = 0.5 is moderate (0.58; expected "low").

**Caveats specific to H3 that the simulation does not include.** (a) Labels come from a text classifier with κ ≥ 0.6 at best, and misclassification attenuates the interaction towards zero. (b) 5.9% of partner turns ending in "?" is a lower bound on the question share. (c) Pitch is missing for 35% of turns under 0.25 s, so the backchannel stratum loses data selectively, by a criterion that depends on the stratum. (d) Function is strongly associated with reply length, which is in Z. (e) Structural dependence (different dimensions per function) needs per-stratum reduced-rank fits and is less powerful than this magnitude test.

**Conclusion.** A magnitude difference of about +50% or more in the question stratum is within reach of 250 sessions in the idealised case. H3 is therefore worth running at Layer 1, with "underpowered, power stated" as a legitimate outcome if label noise or measurability erodes this.

---

## Limitations of these simulations

The data are synthetic and come from a stated generating model. The model is simple. Ordinary least squares stands in for the nested cross-validated ridge, and one dimension stands in for each target block. The simulations include no between-speaker heterogeneity in coupling and no measurement error beyond Gaussian noise. They describe how the registered design and decision rules behave under the stated generating parameters. They carry no evidential weight about conversation, and the registered calibration on real residuals (J3) is the binding test.
