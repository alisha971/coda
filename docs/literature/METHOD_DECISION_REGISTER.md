# Method decisions

This note gives the reasons for each method choice that the preregistration (`PREREGISTRATION.md`) leaves to the implementation. Every entry states the value used, the evidence behind it, the alternatives considered, and the consequence for the analysis. The values themselves are fields of `Settings` in `src/settings.py`, and the preregistration lists them in its tables. References are in `REFERENCE_LIBRARY.bib` and `DECISION_REFERENCES.bib`, and the wider method review is in `LITERATURE_REVIEW.md`.

Identifiers (MD-nn) match the preregistration tables.

---

## A. Measurement

### MD-01 — Overlapping speech in the energy measures
- **Value.** `exclude_overlap_energy`. Frames during the partner's speech leave mean intensity and intensity range. Pause measures use the whole turn.
- **Problem.** The provider's echo suppression may attenuate a speaker during double talk, and that attenuation could depend on the partner. A partner-dependent change in level would create coupling in the energy block that is a property of the recording and not of the speakers.
- **Evidence.** In the pilot, the median overlap attenuation is −1.0 dB (IQR −3.8 to +1.0 dB, 5th percentile −13.4 dB), and double talk is 15.2% of own speech. About 43% of replies begin in overlap.
- **Reasoning.** Pause measures come from word timing, which echo suppression does not affect, and an overlapped gap is still a gap in the replier's own speech. Intensity is the quantity that suppression can change.
- **Alternatives.** Keep all frames. Exclude overlap from every measure. Exclude only channels below an attenuation cut, which was rejected because the cut would be chosen after seeing the distribution.
- **Consequence.** Missingness rises for the two intensity values, and the increase is reported per dimension. Test: `test_exclude_overlap_energy_changes_intensity_only`.

### MD-02 — Pitch range
- **Value.** `fixed`, 75 to 500 Hz as applied at extraction.
- **Evidence.** In the octave profile of the pilot, on average 2.3% of voiced frames lie more than 0.75 octave below the channel median and 3.7% lie above. In 151 of 886 channels more than 5% of frames lie on the low side.
- **Reasoning.** The preregistration fixes floor and ceiling in configuration. A speaker-adaptive range changes the Praat floor and ceiling, which were applied at extraction, so it needs a new audio pass. The code raises `MeasurementNotAvailable` for that option.
- **Alternatives.** A two-pass adaptive range. A verifiable primary source for a specific two-pass procedure was not found, so none is cited.

### MD-03 — Syllable counting
- **Value.** `dictionary`. The CMU Pronouncing Dictionary (`cmudict` 1.0.32, first listed pronunciation) counts stress-marked vowel phones. A vowel-group rule counts unlisted words. Every lexical token receives at least one syllable.
- **Evidence.** The rule agreed with the dictionary on 86.5% of word types in a check of 20,000 words. Systematic differences occur for words such as going, doing, being and maybe.
- **Alternatives.** The rule alone. Acoustic syllable nuclei (de Jong and Wempe, 2009) from the stored tracks, which would make articulation rate partly an acoustic measure.
- **Consequence.** Count noise enters reply syllable count in Z and articulation rate in Y in every condition alike. It can reduce power and cannot create a difference between conditions. The share of tokens counted by the rule is reported.

### MD-04 — Turn-level operational details
- Speech frames are the frames inside the turn's inter-pausal units.
- Mean intensity uses Praat's energy averaging, which is the Praat default.
- Intensity range is the 95th minus the 5th percentile, matching the log-F0 range.
- Boundary slope is the least-squares slope of ln F0 on time over voiced frames in the final 300 ms and needs at least 10 such frames.
- A single-unit turn has an internal pause of exactly 0.

The boundary minimum drops more turns than the 10-voiced-frame criterion of G0 counts, so the loss is reported per dimension before any fit.

### MD-05 — Minimum turns for held-out speaker statistics
Value 10. Speakers average about 245 turns per session in the pilot, so the rule rarely binds. Failures are dropped and counted.

### MD-22 — Replier identity in Z
One-hot over training repliers, with zeros for unseen repliers. Identity cannot carry information at test time under a speaker-disjoint split.

### MD-23 — Loudness offset
The median intensity over the voiced frames of the reply's file. It absorbs the per-file peak normalisation.

---

## B. Splits and the analysis set

### MD-25 — Near-duplicate detection
Five-word shingles of the transcript with a Jaccard threshold of 0.5. Linked sessions are placed in the same split.

### MD-26 — Dyad-disjoint contrast
Value `drop`. With one session per speaker, a dyad-disjoint split is also speaker-disjoint, so the contrast cannot test generalisation to seen speakers. `stages.splits` refuses to build it and notebook 07 skips it. The alternative was to keep it as a second random partition.

### MD-37 — One common analysis set
Every condition and donor scheme uses the pairs whose reply, partner and self-previous turns are usable, whose texts are non-empty, and that have an N2 candidate. This makes droppage identical across conditions by construction. The first pair of each interaction has no self-previous turn and is excluded from every condition.

### MD-07 — Layer 1 fits on training rows only
Validation rows are reserved for the neural layer's early stopping. Layer 1 nests its cross-validation inside training. Training plus validation would add 10% of the data and would make the training sets of the two layers differ.

---

## C. Estimators

### MD-06 — Within-block orthogonalisation by ZCA-cor
Kessy, Lewin and Strimmer (2018) show that ZCA-cor keeps each whitened variable closest to its original, so each output keeps the name of its dimension.

### MD-39 — Orthogonalisation of targets only
The partner and self-previous vectors are predictors, where ridge handles collinearity.

### MD-08 — Penalty grid, folds and selection
The penalty is g times the number of fitting rows, with g on a grid of 15 values from 1e-5 to 100. Five inner folds use whole sessions. The penalty is chosen per target block by minimum out-of-fold squared error. The reduced-rank fit chooses penalty and rank by block-weighted out-of-fold squared error. A choice at either end of the grid is flagged.

### MD-09 — Effective degrees of freedom
The trace of the ridge hat matrix (Hastie, Tibshirani and Friedman, section 3.4.1), checked against an explicit hat matrix in `tests/test_ridge_rrr.py`.

### MD-10 — Conditional variance model
Ridge on squared out-of-fold residuals, floored at 1% of the mean squared residual. The share of floored values is reported. A log-scale variance model would be positive by construction but is not the registered form.

### MD-14 — Reduced-rank ridge on the partner block
The projection of Mukherjee and Zhu (2011) applied after profiling out Z. The restriction to one coefficient block is a derivation made for this study. It is checked numerically: rank 8 equals ridge, and a planted rank-1 coupling is recovered.

---

## D. Inference, calibration and gates

### MD-11 — Bootstrap
Cluster bootstrap-t over 2,000 session resamples, with intervals at 68% and 95%, and percentile and BCa intervals reported beside it. A comparison of the three methods was specified before it was run. No method stayed within Monte-Carlo error of 0.95 in all six scenarios. The worst-case coverage was 0.934 for the bootstrap-t, 0.916 for the percentile interval and 0.907 for BCa. The worst case is stated beside every interval (`docs/research/simulations/BOOTSTRAP_COMPARISON_PROTOCOL.md`).

### MD-12 — N1 matching
Duration deciles within the split, then exact word count, otherwise the nearest word count within the decile. The exact-match share is reported.

### MD-13 — Donor p-value
The share of null draws at or above the observed value, as registered. A p-value of 0 is reported as below 1/B (Phipson and Smyth, 2010).

### MD-15 — H2 stability rule
No source gives numbers for what counts as stable under the bootstrap and across speaker folds. Per-loading sign agreement fails whenever a loading is near zero, because its sign flips at random while the direction is unchanged. The rule therefore uses subspace similarity (mean squared cosine of the principal angles) for the reply and partner loadings. The bootstrap mean and the speaker-fold minimum must each exceed the 95th percentile of the same statistic refitted with N1 donor partners (19 draws, 50 bootstrap refits each), and the leading-strength donor p-value must be at most 0.05. In a synthetic study, a conventional fixed congruence of 0.85 was exceeded by the speaker-fold statistic in about one null data set in ten, which is why no fixed cut is used.

### MD-16 — Injection details
Fifty replicates per β. Sessions are resampled as contiguous runs in their original order (`session_block_ordered`), which keeps serial dependence between turns as well as the covariance across dimensions. Row-wise resampling understated the minimum detectable effect by about 2.2 times in simulation. β comes from training variances. The cross-dimension variant is a fixed permutation in which each reply dimension is driven by a partner dimension from another block, validated as a cross-block permutation. Replicates run in chunks of 250 and each replicate is seeded by its level seed and index, so chunking changes no result.

### MD-35 — Injection in standardised space
The design says the entire pipeline runs on the injected targets. Injection happens after per-speaker standardisation because the fitted prediction and the residuals are defined there. Orthogonalisation, nested cross-validation and the donor nulls are re-run on the injected targets. Re-standardising them would be an affine near-identity.

### MD-38 — Scale of the minimum detectable effect
In the synthetic dry run, the realised injected effect matched the target at 0.02 (0.0214) and fell short at 0.05 (0.0375). The stated minimum detectable effect is the larger of the target-scale and realised-scale values, and both are reported.

### MD-17 — G1 acceptance rule
α = 0.05 with 2,000 replicates at β = 0 per scheme. A scheme passes when the one-sided 95% Clopper–Pearson upper bound of its rejection rate is at most 0.075, which is Bradley's liberal criterion. With 500 draws the exact size of the donor test is 26/501 = 0.0519. A valid test passes both schemes with probability 0.991 at 2,000 replicates and 0.832 at 1,000, so 2,000 is used. A test with a true rate of 0.075 passes with probability 0.002. A rate well below α is reported as conservatism.

### MD-18 — Power for the minimum detectable effect
0.80 (Bloom, 1995).

### MD-19 — G2 as a reporting requirement
No source gives a plausible effect range in block partial R² for conditional, speaker-disjoint, forward prediction. Published entrainment effects are unconditioned correlations on other units. G2 therefore reports the minimum detectable effect with its assumptions and power at block partial R² of 0.0025, 0.005 and 0.01, and it is never recorded as passed. These three values are a sensitivity grid and not thresholds.

### MD-20 — G3 ordering rule
Real beats N1 and real beats N2 when the corresponding donor p-value is at most 0.05. N2 exceeds N1 when the median N2 null statistic exceeds the 95th percentile of the N1 null. G3 reports one of four patterns (adjacent, distant, none, inconsistent) and never names a mechanism.

### MD-41 — What H4 identifies
The within-conversation donor holds fixed what is constant within an interaction. It does not hold fixed shared state that changes within about ten turns. In synthetic data with such state and no turn-to-turn coupling, the contrast produced real > N2 > N1 in 40 of 40 replicates. A positive H4 result is reported as predictive dependence concentrated at the adjacent turn, with the mechanism not identified.

### MD-21 — One H3 p-value
The Holm-adjusted minimum over the eight per-dimension interaction tests. If any fit fails, no aggregate is formed.

### MD-33 — Mixed-model structure
Groups are sessions with a speaker variance component, because speakers are nested in sessions here. All included pairs are used. Barr and colleagues (2013) would favour random slopes, but the registered formula has random intercepts only and is kept. A convergence failure is reported and the structure is not simplified.

### MD-34 — The self-previous condition
Z plus the replier's previous turn, without partner delivery, as in `configs/j1_conditions.yaml`. It is compared with the partner-text condition.

---

## E. Neural layer and situation analysis

The neural layer is deferred. The code and its synthetic tests are kept. `load_extractor` and `run_neural` refuse real data while `Settings.neural_arm` is `deferred`, so no weights are downloaded and no real-data neural result is produced (MD-40). Test: `test_deferred_neural_arm_refuses_real_data_and_allows_synthetic`.

| ID | Decision |
|---|---|
| MD-27 | WavLM Base+ and HuBERT Base LS960 with model revisions that must be pinned before extraction, 16 kHz polyphase resampling, PCA fitted on at most 20,000 training partner turns, 500 turns per shard |
| MD-28 | AdamW, learning rate 3e-4, weight decay 0.01, dropout 0.1, gradient clip 1.0, batch 64, at most 40 epochs, patience 5, speaker embedding 16 with dropout 0.2, and a grid over learning rate and dropout searched once |
| MD-29 | Ablation 4 frozen to WavLM layer 6 |
| MD-30 | T1: the neural Δ at partial R² 0.03 has p ≤ 0.05 against its 30-draw N1 null. T2: 3 replicates at β = 0, limited by the run budget, so its Monte-Carlo error is large |
| MD-31 | Text-only function classifier and prompt. Not set |
| MD-32 | 300 random pairs, seed 20260912, labelled blind to the automatic labels |
| MD-36 | Partner, self-previous and donor vectors standardised per pair with the same leave-out rule as the target |
