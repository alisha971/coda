# CODA — preregistration

**Committed before any confirmatory experiment runs.** The git timestamp on this file is the evidence that the hypotheses, the primary test, the architecture and the analysis plan were fixed before the results were seen.

Nothing in this document may be revised after a confirmatory result has been inspected. If a legitimate methodological problem is found later, it is recorded as an amendment with its own date and reasoning, and the original text stays visible.

---

## 1. Hypotheses

Stated before any confirmatory result is inspected.

**H1 — primary, confirmatory.** Conditional on Z = {reply text, reply length, partner text, replier identity}, the partner turn's delivery is predictive of the reply's delivery: Δ exceeds the matched-donor null, on a speaker-disjoint test set.
*Null:* the observed statistic falls inside the donor-null distribution.

**H2 — secondary, confirmatory.** The relationship is structured: there exist reproducible partner-dimension to reply-dimension directions rather than diffuse association. Reproducibility requires all of: reduced-rank structure; stability of loading sign and magnitude across dyad-level bootstrap resamples; stability across held-out-speaker folds; concentration of explained covariance in the leading directions; and comparison against a random-rotation null.
*Null:* no direction survives the bootstrap, the speaker-fold and the rotation null simultaneously.

**H3 — bridge to Situational Style Cloning.** The coupling is function-dependent: magnitude, dimensional structure, or both differ systematically across conversational functions of the turn pair.
*Null:* no function-by-coupling interaction after correction.

**H4 — competing explanation, mandatory.** The relationship reflects local adaptation, not merely shared dyad or session state: it is weaker when the partner turn is replaced by a distant turn from the same speaker in the same conversation.
*Null:* real context and within-dyad distant context perform equivalently, in which case the phenomenon is conversation-level style and H1 and H3 are reinterpreted accordingly rather than discarded.

---

## 2. Estimand and reported units

For turn pair *i*: **Y** is the reply's delivery vector, speaker-standardised; **A** is the partner turn's delivery evidence; **Z** is the conditioning set — reply text embedding, reply word count, reply syllable count, partner text embedding, replier identity, turn position, per-conversation loudness offset.

> **Δ = the out-of-sample increase in predictive accuracy of Y from adding A to Z, referenced to the distribution of that increase under conditionally matched replacement of A.**

**Two pre-specified units.** Partial R² per target block is **primary**. The out-of-fold predictive log-likelihood increment is the second unit. If they disagree, partial R² governs and the disagreement is reported, never averaged away.

**Unit of analysis is the dyad.** All resampling, all intervals and all random effects are at conversation level. The effective sample size quoted is the number of conversations, not the number of turn pairs.

**Language.** Δ is a conditional predictive dependence. It is never described as causal, never as mutual information, and never as the acoustic channel's information content without the qualifier "predictive, conditional on Z, under this model class".

---

## 3. Primary test and multiplicity

**One primary test, declared here in advance:** H1, aggregate across blocks, speaker-disjoint split, **Layer 1** (ridge and reduced-rank ridge), with N1 and N2 as reference nulls.

**H2, H3 and H4 are Holm-corrected within a family of three.**

**The neural layer carries no confirmatory significance level.** Its results are corroborating, do not enter the correction family, and can neither rescue a null nor overturn a positive from Layer 1.

---

## 4. Target vector

**Four blocks, eight dimensions.**

| Block | Dimensions | Count |
|---|---|---|
| Pitch level and span | mean log-F0; log-F0 range (p95 − p5) | 2 |
| Pitch boundary | slope of log-F0 over the final 300 ms | 1 |
| Energy | mean intensity; intensity range | 2 |
| Timing | articulation rate; pause ratio; longest internal pause | 3 |
| | | **8** |

**Confirmatory inference is at block level: four tests, one per block, equal weight.**

> **Count correction, recorded before J1.** Earlier project drafts state the target as *nine* dimensions in their prose while enumerating *eight* in the table above; the same table appears in the predecessor document marked unchanged, and no draft names a ninth dimension. The enumeration is the operative definition and has always been eight. Adopting the prose count would have required inventing a prosodic dimension: every plausible candidate is either near-collinear with a dimension already present, and would be removed by within-block orthogonalisation while retaining its measurement noise, or is contour-derived and therefore listed as optional and outside the required target. It would also raise a block's detectable-effect floor for no gain at block level, where inference happens. Recorded here so the correction is timestamped before any confirmatory result.

**Standardisation.** Per-speaker z-scoring following Levitan and Hirschberg's norming procedure, cited to them. For unseen test speakers, statistics come from that speaker's other turns in the same conversation, **excluding the target turn and every turn used as its context**.

**Within-block orthogonalisation** is fitted on training data and applied everywhere.

**Not part of the required target:** any contour basis beyond the final-300 ms boundary slope; voice-quality dimensions.

**Measurement instrument.** Praat, via the parselmouth binding, for F0 and intensity, matching the source of the cited feature set. Pitch floor, ceiling, frame step and voicing threshold are fixed in the tracked feature configuration and pinned identically in both execution environments.

---

## 5. Data and splits

**Corpus.** CANDOR, the single required corpus. AMI, restricted to two-party stretches, is the pre-committed fallback if corpus viability fails, and is not a required replication.

**Sample.** 250 conversations, the floor of the specified 250–350 range, selected at random among those passing channel quality control, with the **selection seed recorded**. Audio is process-and-discard.

**Splits.** Dyad-level, approximately 70/10/20, under a **speaker-disjointness constraint** so that no participant appears in more than one split. Near-duplicate conversations are clustered and assigned whole to a single split. A **dyad-disjoint-only split** is fitted as a contrast. The split seed is recorded.

**Development subset.** A 40-conversation pilot, used for all routine iteration. **No confirmatory result is reported from the pilot.**

---

## 6. Conditions and controls

**J1 conditions:** `Z_content`, `+partner_text`, `+partner_delivery`, `+self_previous`, each on both splits.

**N1 — cross-dyad matched donor.** Drawn from another dyad in the same split, matched on partner-turn duration decile and word count. Preserves dimensionality, scale, marginals, length structure and capacity; destroys the pairing.

**N2 — within-dyad distant donor.** Same dyad, same partner speaker, at least **10 turns away**. Preserves dyad, speaker, session, channel, register and broad topic; destroys local adjacency only.

**Donor draws:** **500 per control for Layer 1**, **30 per control for Layer 2**, pre-registered as such. The asymmetry is stated in the write-up. A 30-draw null is sufficient to detect the failure mode in which the real condition is indistinguishable from N1, and is **not** sufficient for a precise p-value; none is claimed from it.

Donor draws are seeded and saved so that nulls are exactly reproducible.

**Pre-committed reading of the ordering:**

- `real > N2 > N1` → local adaptation;
- `real ≈ N2 > N1` → conversation-level style;
- `real ≈ N1` → nothing.

**Positive control.** Semi-synthetic injection, run and reported **before any real result**: fitted content-only prediction, plus a coupling term scaled by β, plus resampled real residuals preserving their covariance. The coupling is identity on matched dimensions plus one cross-dimension variant. β is swept to span partial R² from **0.002 to 0.05**. The entire pipeline runs on the injected targets.

---

## 7. Neural architecture, frozen before J1

Recorded here so that architecture selection cannot follow the result.

Frozen WavLM layers 3, 6, 9 and 12 plus HuBERT's final layer, combined by a learned softmax mixture over per-source projections into **d = 192**. Partner acoustic encoder: learned positional encoding, **3 pre-norm transformer blocks, 4 heads, FFN 384**, key-padding mask from true frame lengths, attention pooling with a learned query. Fusion: **one** cross-attention block with the context and partner-text representation as query and the masked partner frames as key and value, a residual pooled path, and concatenation. Head: 2-layer MLP to **8 means**, optionally 8 log-variances, with a block-aware warm start.

**Trainable parameter target:** approximately 1.5 to 2.5 million, **reported exactly**, alongside the dyad-level effective sample size.

**Objective:** block-aware multi-target regression, equal weight per block.

**Capacity control.** Identical architecture in every condition including donor conditions and the ablation, where the partner-acoustic pathway is fed a learned null token sequence rather than removed, so parameter count is identical across conditions. Identical optimiser, schedule, batch size, epoch budget, early stopping and patience. Identical splits. The hyperparameter budget is searched **once** on the real condition's validation set, then frozen and reused unchanged in every donor condition and every ablation. No test-set model selection. **Three seeds per condition**, with seed variance reported separately from donor-null variance.

**Four architectural ablations, and no more:** no partner acoustic pathway; the full learned pathway; the pathway fed the interpretable delivery vector as a length-one sequence; the pathway with the mixer frozen to a single source.

---

## 8. Analysis plan

Model selection is by nested cross-validation inside training for Layer 1 and validation-based early stopping for Layer 2. **The test set is touched once per condition.**

Δ is computed per block and aggregated with equal weight per block. p-values are the proportion of donor-null draws at or above the observed statistic. Intervals come from a **paired dyad-level cluster bootstrap** and are reported alongside the permutation p-values. Interval coverage is reported per block at 68% and 95%.

**Effect sizes are always reported with the block-level minimum detectable effect overlaid**, so that a small estimate is read as a bound rather than an absence.

**Confirmatory re-check:** a mixed-effects model with crossed speaker and dyad random effects, fitted in statsmodels. If it fails to converge, that is reported as a convergence limitation; the random-effects structure is not simplified to force convergence.

**Specified in advance.** Turns with missing prosody are dropped, the count is logged, and droppage is checked not to be condition-dependent. **A negative Δ is interpreted only against the null, never as evidence that acoustic context hurts.**

**Deliberately not used:** conditional mutual information estimation, because there is no accepted protocol at this sample size and dimensionality; likelihood-ratio tests, because they are invalid for regularised cross-validated fits.

---

## 9. Situation analysis

**Categories, assigned to the pair:** question to answer; assessment or opinion to response; narrative or inform to continuation; backchannel-adjacent; other, which is **excluded from analysis and not modelled**.

Interrogative syntax is taken from text. The remaining categories come from a classifier over the **turn-pair transcript only — never audio, never delivery features**.

**Reliability gate, fixed in advance.** 300 randomly sampled pairs are hand-labelled, Cohen's κ is reported per category, and **any category with κ below 0.6 is dropped from confirmatory analysis**. A stratum with fewer than **200 pairs** is descriptive only.

The interaction is tested **directly as a function-by-partner-delivery term in the mixed-effects model**, not by comparing subgroup estimates by eye. Per-stratum Δ with intervals is the descriptive companion. Any neural view of function-dependent structure is exploratory and labelled as such.

---

## 10. Gates, with pass criteria fixed before the numbers

| Gate | Test | Pass | On failure |
|---|---|---|---|
| **G0** | 40-conversation pilot: channel quality, F0 tracking confidence, feature droppage | ≥ 80% channels usable, droppage < 15% | switch primary corpus to AMI two-party stretches |
| **G1** | rejection rate at β = 0 for N1 and N2, Layer 1 | at nominal level | stop and debug; nothing downstream runs |
| **G2** | minimum detectable effect at achieved n, Layer 1 | below the plausible effect range from the entrainment literature | add conversations, or reduce the target to the three best-measured blocks; J1 is not confirmatory until this passes |
| **G3** | J1 and J2, speaker-disjoint, Layer 1 | one of the three pre-committed readings in §6 | — |
| **G4** | κ per function category on 300 hand-labelled pairs | κ ≥ 0.6 for at least 2 categories | situation arm becomes exploratory; if no category passes, ship at Level D |
| **G5** | J4 interaction | significant after correction, or credibly null with stated power | report as null with per-stratum detectable-effect floor |
| **T1** *(technical only)* | does the architecture recover an injected effect at partial R² ≈ 0.03 | yes | debug once; if it still fails, report the neural arm as a negative technical result and ship the scientific study unchanged |
| **T2** *(technical only)* | β = 0 rejection rate for the 30-draw neural null | at nominal level within Monte-Carlo error | neural Δ reported descriptively without a p-value |

**T1 and T2 gate only the neural layer's interpretability. Neither can block, delay or extend the scientific study, and failing either changes no scientific conclusion.**

---

## 11. Pre-committed outcome interpretations

Four, fixed in advance. **No fifth outcome is created.**

1. **Coupling present, surviving both controls, structured, function-dependent** — supports modelling expressive conversational speech as a situation-conditioned delivery policy.
2. **Coupling present, not function-dependent** — supports context-conditioned modelling; provides no evidence that situation conditioning is necessary, leaving that assumption untested rather than supported.
3. **Coupling present but equivalent under N2** — conversation-level style alignment rather than turn-level adaptation; favours a running conversation-style estimate over last-turn conditioning.
4. **Calibrated null** — with a stated detectable-effect floor on the achieved sample and unseen speakers, coupling is weak or not reliably present under controlled evaluation.

A neural positive under a Layer-1 null is reported as **representation-dependence**, which qualifies outcome 4 rather than creating a fifth.

---

## 12. Stop rule

> CODA is complete when the conditional contribution of partner delivery has been estimated under N1 and N2 with speaker-disjoint evaluation and a stated minimum detectable effect, **and** the function-dependence question has been answered — positively, negatively, or as underpowered with the power stated.

No experiment is added because it would improve publication potential, because another corpus exists, because another representation could be tested, or because a reviewer might ask. Once T1 and T2 pass and the four ablations are run, the neural layer is finished.

**Drop first if time runs short:** the dyad-disjoint contrast split; the mixed-model re-check, keeping the bootstrap; the frozen single-source mixer ablation; stability shading on the coupling map figure.

**Never drop:** G1; G2; N1; N2; speaker-disjoint evaluation; the positive control; identical-capacity discipline in the neural arm.

---

## 13. Resolutions fixed before J1

Details the governing specification leaves to the implementer, resolved here so the resolutions are timestamped rather than chosen after seeing a result.

| Item | Resolution |
|---|---|
| Target vector size | Eight dimensions in four blocks (§4) |
| F0 and intensity extraction | Praat via parselmouth; parameters fixed in the tracked feature configuration |
| Mixed-effects software | statsmodels; convergence limitations reported rather than worked around |
| Module layout | All application modules rooted at `src/`, preserving specified names |
| Sample size | 250 conversations, the floor of the specified range |
| Development subset | The 40-conversation G0 pilot; no confirmatory result reported from it |

Remaining unspecified details are resolved during implementation, documented at the time, and added here as amendments if they are settled before J1.

---

## Amendments

*None. Amendments appear below this line with their own date and reasoning; earlier text is never edited.*
