# Design-validation simulations — protocol and expectations

**Status.** Written 2026-10-16, **before** `scope_sims.py` was run. Synthetic data only. No corpus audio, transcripts, prosody tracks or targets are read. These are design-validation simulations. They are **not** registered experiments, they do not replace J3 (the registered calibration), and they carry no evidential weight about conversation. Their only job is to establish what the registered design *can* and *cannot* distinguish, and at what sample size.

**Instrument.** A closed-form linear approximation of the Layer-1 instrument: ordinary least squares on a train split, block partial R² `1 − SSE(Z + A)/SSE(Z)` on a held-out, session-disjoint test split, referenced to donor refits in which A is replaced in both train and test. One target dimension stands in for one block. Sessions are split 70/10/20 as registered; the validation split is unused, as in MD-07. Speakers occur in one session only, so session-disjoint is speaker-disjoint.

**Synthetic conversation.** Each session has 11 interactions of 36 alternating turns (≈ 385 pairs per session, close to the pilot's ≈ 408 usable pairs). Per turn *t* by speaker *s* in interaction *k*:

    y_t = μ_s + c · d_{k,t} + β · y_{t−1} + γ · x_t + e_t
    d_{k,t} = ρ · d_{k,t−1} + sqrt(1 − ρ²) · u_t      (reset at each interaction)

`y_{t−1}` is the partner's immediately preceding turn (local coupling, strength β); `d` is a slowly varying state shared by both speakers inside one interaction (topic arousal, prompt-induced energy; strength c, persistence ρ per turn); `x_t` stands for the reply's own text features (8 columns, of which 3 carry signal). Targets are z-scored per speaker within session, as in the held-out-speaker rule.

**Donors.** N1: the partner turn of a random pair from a different session in the same split. N2: a turn by the same partner speaker in the same interaction at least 10 turns away .

## S1 — Can the N2 control separate local adaptation from slow shared drift?

Four generating scenarios: **null** (c = 0, β = 0); **drift only** (c = 0.6, β = 0, ρ ∈ {0.90, 0.97}); **local only** (c = 0, β = 0.10); **both**. Two conditioning sets: the registered H1 set (Z) and Z plus the replier's own previous turn (the J1 `+self_previous` information, used here as a conditioning variable rather than as a competitor). 250 sessions, 20 replicates per cell, 100 donor draws per control.

**Expected before running.**
1. *Null:* rejection near 5% against both N1 and N2.
2. *Drift only:* real > N1 almost always (drift is shared within a dyad, absent across dyads). Real > N2 often, because a turn ten turns away shares less drift than the adjacent turn: at ρ = 0.90, ρ¹⁰ ≈ 0.35; at ρ = 0.97, ρ¹⁰ ≈ 0.74. The registered ordering would then read **"local adaptation" with no local coupling present**, more often at the lower ρ. Conditioning on the replier's own previous turn should remove most of this spurious increment, because the self-previous turn sits only one turn further from the drift than the partner turn.
3. *Local only:* real > N1 and real > N2 at high rates under both conditioning sets; N2 ≈ N1 (this is the ordering finding A-2 says the three registered readings do not cover).
4. *Both:* real > N2 under both sets; the increment over Z + self-previous is smaller than over Z but non-zero.

**What would change the conclusion.** If drift-only produced real > N2 rarely (< 20%) at ρ = 0.90, the N2 control would be adequate as registered and no change would be proposed.

## S2 — Detectability at the registered sample size

Local-only scenario, β chosen so that the true held-out block partial R² is ≈ 0, 0.001, 0.002, 0.005, 0.01; S ∈ {40, 100, 250} sessions; 100 replicates; 99 N1 draws; α = 0.05.

**Expected before running.** With ≈ 20,000 test pairs at S = 250, partial R² of 0.005 should be detected with power near 1, and 0.002 with moderate-to-high power; the 40-session pilot should only detect ≥ 0.01. If power at 0.002 and S = 250 is below 0.5, the registered sample is underpowered for the smallest effect sizes the literature makes plausible (see `docs/literature/PRIOR_WORK_AND_NOVELTY_REVIEW.md`).

## S3 — Power for the function interaction (H3)

Local-only scenario with three strata assigned to the pair: question → answer (6% of pairs, the pilot's share of partner turns ending in "?"), backchannel-adjacent (35%), other (59%). Base coupling gives partial R² ≈ 0.005 in "other"; the question stratum's coupling is multiplied by (1 + δ), δ ∈ {0, 0.5, 1.0}. Test: OLS on all pairs with a stratum × partner-delivery interaction, session-clustered (CR1) standard error, α = 0.05/3 (the first Holm step for a family of three). 250 sessions, 200 replicates.

**Expected before running.** Power for δ = 1.0 (coupling twice as strong in questions) to be moderate; for δ = 0.5 to be low. The question stratum is the binding constraint, not the backchannel stratum.

## Reproducibility

`python docs/research/simulations/scope_sims.py` writes `results/S1.csv`, `results/S2.csv`, `results/S3.csv` and `results/run_record.json` (seed, versions, wall time) beside this file. Seed 20261016.
