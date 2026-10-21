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

**H3 — function dependence.** The coupling is function-dependent: magnitude, dimensional structure, or both differ systematically across conversational functions of the turn pair.
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

> **Target size.** The target holds eight dimensions, enumerated in the table above. A ninth dimension is not part of the required target. Every plausible candidate is either near-collinear with a dimension already present, and would be removed by within-block orthogonalisation while retaining its measurement noise, or is contour-derived and therefore listed as optional. A ninth dimension would also raise a block's detectable-effect floor for no gain at block level, where inference happens.

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

1. **Coupling present, surviving both controls, structured, function-dependent** — supports describing conversational delivery as depending on the situation.
2. **Coupling present, not function-dependent** — supports context-conditioned modelling and provides no evidence that conditioning on the situation is necessary, which stays untested.
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

*Amendments appear below this line with their own date and reasoning; earlier text is never edited.*

### Amendment 1 — 2026-09-23 — Corpus: Seamless Interaction, Naturalistic partition

Recorded before any audio has been processed and before J1.

**Change.** §5 names CANDOR as the single corpus. The corpus is replaced by the **Seamless Interaction dataset, Naturalistic partition** (Agrawal et al., 2025, arXiv:2506.22554; `huggingface.co/datasets/facebook/seamless-interaction`, revision `ba9e212ab927ba05bfd80778f53bf9de69f65e3b`; file index from `github.com/facebookresearch/seamless_interaction`, commit `b61b0324461f05221659cc38a1130f854fdd2855`; licence CC BY-NC 4.0).

**Reason.** CANDOR is released through an access process (an application through BetterUp). Seamless Naturalistic is openly downloadable and preserves every property the design depends on: dyadic, unscripted speech; a separate audio file per participant, so no diarisation error enters target or predictor; word-level timestamps; persistent participant identifiers, so speaker-disjoint evaluation is possible; and enough dyads for the planned sample with a large reserve.

**Unchanged.** Hypotheses, estimand, reported units, the primary test and its multiplicity control, the eight target dimensions and four blocks, the conditioning set, both donor controls, the positive control, the neural architecture, every gate and pass criterion, the outcome interpretations and the stop rule. AMI, two-party stretches, remains the fallback if G0 fails.

**Mapping of the design onto the corpus.** A Seamless *session* is one dyad for roughly an hour, released as a set of *interactions*, each a 2–10 minute stretch of talk on one prompt. The time between interactions is not released, and interaction identifiers are prompt identifiers rather than chronological indices, so the order of interactions within a session is not recoverable.

| Design element | Mapped to |
|---|---|
| Conversation / dyad (unit of analysis, resampling, random effects, splits) | Session |
| Speaker identity | Vendor-qualified participant identifier |
| Turn adjacency, turn position, turn distance | Within one interaction only; pairs never cross an interaction boundary |
| N2 donor ("same dyad, same partner speaker, ≥ 10 turns away") | Same interaction, same partner speaker, ≥ 10 turns away. Turn distance is undefined across interactions because their order is not released; a donor inside the interaction satisfies "same dyad" |
| Held-out-speaker standardisation ("that speaker's other turns in the same conversation") | That speaker's other turns in the same session, excluding the target turn and every turn used as its context |
| Per-conversation loudness offset | Per audio file (participant × interaction), because the released audio is peak-normalised per file |
| Channel QC exclusion ("a conversation is excluded if either channel fails") | An interaction is excluded if either participant's file fails; a session is excluded if no interaction survives |

**Sampling frame.** Naturalistic sessions whose relationship is recorded as `stranger`, matching the population the design was drawn up for. Interactions of type `ipc_conversation` only: the other types are games (a gesture-acting game built on emphasising a written word, charades, and sentence-alternating storytelling), where delivery or turn-taking is imposed by the task. Interactions whose prompt instructs the manner of speaking are excluded: two-sided role-play prompts, and prompts whose text directs tone or speaking style. Instructed delivery would be a shared external driver of the very variable being measured.

**Selection.** Eligible sessions are placed in random order (seed 20260912). A session is accepted only if neither participant already appears in an accepted session, so each speaker contributes one dyad. This mirrors the source design, where speakers rarely recur, and makes speaker-disjoint splitting exact rather than dependent on how participants chain across sessions. Sessions are processed in that order; the first 40 accepted form the G0 pilot, and processing continues down the same order until 250 sessions pass channel QC, which is the specified random selection among sessions passing QC. The ordered list is the tracked file `manifests/coda_download_manifest.csv`.

**Turn construction** (the corpus ships words, not turns). Words are grouped per speaker into inter-pausal units separated by silences of at least 50 ms, following the inter-pausal-unit definition of the work the target feature set is drawn from. A turn is a maximal run of one speaker's inter-pausal units with no speech from the partner between them. A partner unit lying entirely inside the other speaker's turn does not end that turn; it is kept, flagged as embedded, and does not form a pair. A pair is two consecutive non-embedded turns by different speakers in the same interaction.

**Known properties of the released audio, handled rather than ignored.** The audio is peak-normalised per file and processed with an echo-suppression algorithm to remove the partner's voice. Absolute level is therefore not comparable across files, which the per-file loudness offset addresses, and suppression may attenuate a speaker during overlapping speech. Channel QC measures the own-voice level drop during overlap and the resulting silence, and reports both before any target is used. Roughly 10% of interactions carry timestamp or prompt-order errors according to the dataset documentation; the interaction-type label of an affected interaction can be wrong.

**Population statement for the write-up.** Speech is unscripted but prompted, recorded in person with a moderator present, and prompts change every few minutes.


### Amendment 2 — 2026-09-24 — Where pitch is measured

Recorded before any confirmatory experiment and before any target dimension has been computed.

**Change.** Speech regions for pitch analysis are the union of the released voice-activity segments and the regions the released transcript places words in, merged where they overlap or lie within 50 ms (`pitch_segment_source: vad_plus_transcript` in `configs/features.yaml`). Both sources ship with the corpus and neither is a label or a target, so extraction remains a pure function of the audio and its metadata. No Praat parameter changes. The 10-voiced-frame minimum and the definition of droppage are unchanged. Because the change is a union, the analysed region can only widen and droppage can only fall or stay level.

**Reason.** Voice-activity segments alone miss speech that the transcript documents. At the largest recording site the mean share of transcribed word time falling inside those segments is 0.73, with a 5th percentile of 0.15 and a minimum of 0.00. Where the detector missed documented speech, a turn received no pitch regardless of its length.

**Reporting, fixed in advance.**

1. Both pitch-region settings are evaluated at G0 on the 40-session pilot. Outputs of each setting carry the measurement variant in their identifier, and both verdicts are reported.
2. Recovery is expected to concentrate in turns of a second and longer. Recovery spread evenly across all turn lengths, including the shortest, would indicate pitch being reported where there is no speech to measure, and would be treated as a measurement defect.
3. Should droppage be at or above 15 per cent, the specification's response applies and the primary corpus switches to AMI two-party stretches.

**Sampling is not adjusted in response to this change.** The sample remains the seeded selection of Amendment 1. In particular it is not re-drawn towards conversations with longer turns. Turn length is related both to whether a turn can be measured and, plausibly, to the strength of the relationship under test, so selecting on it would choose the sample partly on the outcome. The corpus already supplies extended speech without any such selection: turns of two seconds and longer are 36 per cent of turns and about 90 per cent of all speech time, and turns over five seconds carry about 77 per cent. Where turn length matters to interpretation it is handled inside the analysis as designed, through reply word count in the conditioning set, N1 donors matched on partner-turn duration decile and word count, and the situation analysis separating backchannel-adjacent pairs from narrative and inform pairs.

**Unchanged.** Hypotheses, estimand, reported units, the primary test and its multiplicity control, the eight target dimensions and four blocks, the conditioning set, both donor controls, the positive control, the neural architecture, every gate and its pass criteria, the outcome interpretations, the stop rule, the corpus, the sampling frame and the selection.

### Amendment 3 — 2026-10-20 — Population wording, scope, H4 interpretation, G2 reporting, and measurement details fixed before J1

Recorded before J1. No confirmatory result has been computed.

**1. Population description (correction to Amendment 1).**

The release paper (Agrawal et al., 2025, arXiv:2506.22554) states that audio was "recorded at 48khz / 16bit using worn lapel mics". It also states that each dyadic pair of audio files was post-processed "to remove speaker bleed using Beryl AEC", followed by peak normalisation. The dataset card on the Hugging Face page above describes the files as "speaker-bleed denoised" WAV at "48kHz, 16-bit". The files read in the pilot are 32-bit floating-point WAV at 48 kHz, one participant per file. The card's bit depth therefore describes the recording, not the released files.

The same paper states that:
- "a moderator is always present in a recording session";
- "over 95% percent of the interactions took place in a common-room";
- "less than five percent of sessions were recorded in separate-rooms, with life-size monitors placed in front of participants";
- each session is about an hour, divided into 5–20 interactions of 2–10 minutes;
- about half of the Naturalistic dyads were recruited as familiar pairs.

This study keeps only dyads recorded as `stranger` (Amendment 1).

The population statement for the write-up becomes:

> Unscripted but prompted conversation between strangers, with a moderator present. In over 95% of interactions corpus-wide, both participants were in the same room; the rest used separate rooms with life-size monitors. Prompts change every 2–10 minutes. Each participant is recorded with a worn lapel microphone. The provider removed speaker bleed with acoustic echo cancellation and peak-normalised each file. The released files are 32-bit floating point at 48 kHz.

The documentation also reports participant-identifier errors in both directions: one identifier assigned to two people, and, "rarely", one person mapped to two identifiers. The second can place one person on both sides of a speaker-disjoint split. A metadata audit (protocol fixed before it ran) found that real-person identity cannot be established from the available metadata. Its handling is in section 6, and the resulting claim is stated in section 7.

**Reason for the corpus change.** CANDOR is released through an access process: registration and application through BetterUp, and for the TalkBank copy, a TalkBank account and an emailed request after BetterUp's approval (Reece et al., 2023, data availability statement). Access was not requested. The study uses an openly downloadable corpus with every property the design depends on. No CANDOR data was obtained or used.

The corpus, the sampling frame and the selection rule of Amendment 1 are unchanged.

**2. Contribution (narrowed).**

Prior work has reported that the interlocutor's preceding speech improves prediction of a speaker's next-turn energy, F0 range and speech rate (Fuscone, Favre & Prévot, 2020). That work did not condition on either speaker's words, use a pairing-destroying null, or hold out speakers. This study's contribution is the **controlled measurement** of that dependence, which:
- conditions on reply and partner words and on replier identity;
- evaluates on held-out speakers;
- refers the result to matched-donor nulls;
- states a minimum detectable effect.

It is not presented as the first evidence that partner delivery predicts reply delivery.

**3. Neural layer deferred (MD-40).**

The neural layer (Layer 2) is not run under this preregistration. The following are deferred: the frozen WavLM and HuBERT feature extraction, the training of the architecture in §7 on the real data, its four ablations and its 30-draw N1 and N2 donor conditions, gates T1 and T2, and the linear probes on the learned partner representation, including the speaker-identity probe. Jobs J1 to J4, hypotheses H1 to H4 and gates G0 to G5 are run in full at Layer 1.

**Reason.** The neural layer carried no confirmatory significance level (§3). It could neither rescue a null nor overturn a positive from Layer 1, so no conclusion of the study depends on it. It also carries most of the compute cost, about 20 to 35 GPU-hours in total.

The code is kept and tested on synthetic data only. No hypothesis, test, correction family or outcome interpretation changes. The "representation-dependence" reading of §11 cannot arise and is not reported. Any later neural analysis is a separate, separately registered study.

**4. H4: an ordering of predictive dependence, inconclusive about mechanism (MD-41).**

The N2 donor holds fixed what is constant within an interaction. It does not hold fixed shared state that changes within about ten turns. In synthetic data (section 11) with such state and no turn-to-turn coupling, the registered contrast produced "real > N2 > N1" in every replicate (40 of 40, per-turn persistence 0.80–0.99). Conditioning on the partner's earlier turns did not remove it.

H4 is therefore read as follows:
- **A positive H4** is reported as: *predictive dependence is concentrated at the partner's adjacent turn, relative to the same partner's turns at least ten turns away in the same interaction; this design cannot distinguish local, turn-to-turn adaptation from shared state varying over a few turns.* The words "local adaptation" are not used for this result.
- **A negative H4** (real ≈ N2 > N1) is reported as outcome 3 of §11, as registered. The synthetic study supports this reading: interaction-level structure produced it in 9 of 10 replicates.

No reading of H4 names a mechanism. The test, donors, statistic, α and Holm family are unchanged. The synthetic counterexamples are kept in the repository as a regression test.

**5. G2: a reporting requirement, not a go/no-go gate (MD-19).**

No published source gives an effect size or plausible range on this study's scale: held-out block partial R² of partner delivery, conditioned on words, after per-speaker standardisation. Published entrainment effects are unconditioned correlations on other units and cannot be converted. G2's pass criterion therefore cannot be applied without inventing a threshold. G2 becomes a reporting requirement:

1. J3 reports the minimum detectable effect at power 0.80, aggregate and per block, on the target and realised scales (the larger is the stated MDE). It also reports power with its Monte-Carlo interval at block partial R² 0.0025, 0.005 and 0.01. **These three values are a sensitivity grid chosen to span small effects. They are not thresholds and not plausible effect sizes.**
2. G2 is recorded as "not evaluated: no established effect-size threshold". It is never recorded as passed. Its on-failure actions (adding conversations; reducing the target to three blocks) are not triggered.
3. A J1 or J2 result may be labelled confirmatory when G1 has passed under its registered rule **and** item 1 has been reported. G1 is unchanged by this amendment and must pass first.
4. Interpretation is limited accordingly:
   - a rejection establishes conditional predictive dependence at the stated α, without any claim that its size matters;
   - a non-rejection is reported only as "no measurable contribution above [MDE] under this design and sample size";
   - neither is described as negligible, meaningful or important.
5. The MDE is stated with the assumptions of the calibration that produced it, including how residuals were resampled (section 6).

G2 remains on the never-drop list in this sense: the MDE is always computed and reported.

**6. Measurement, split and inference details fixed before J1.**

- **Overlap (MD-01):** frames during the partner's speech are excluded from mean intensity and intensity range. Pause measures use the whole turn.
- **Syllables (MD-03):** CMU Pronouncing Dictionary (`cmudict` 1.0.32, first listed pronunciation), with the vowel-group rule for unlisted words. The fallback share is reported.
- **H2 stability (MD-15):** subspace similarity of the reply and partner loadings, computed as the bootstrap mean and the speaker-fold minimum. Each must exceed the 95th percentile of the same statistic refitted with N1 donor partners, and the leading-strength donor p must be ≤ 0.05. This replaces per-loading sign agreement.
- **Dyad-disjoint contrast (MD-26):** dropped. Under one session per speaker it is a second random speaker-disjoint partition.
- **Residual resampling in J3 (MD-16):** whole sessions are resampled as contiguous, order-preserving runs. This keeps serial dependence between turns as well as the covariance across dimensions (§11.2). Row-wise resampling understated the MDE by about 2.2× in simulation (section 11). Registered replicate counts are kept. J3 runs in checkpointed chunks, and each replicate is seeded by (level seed, replicate index), so neither chunking nor resumption changes any p-value or decision.
- **Bootstrap intervals (MD-11):** cluster bootstrap-t. Its worst simulated coverage, 0.93 at 25 test sessions, is stated with every interval.
- **G1 (MD-17):** pass when, for N1 and for N2, the one-sided 95% Clopper–Pearson upper bound of the β = 0 rejection rate is ≤ 0.075 (Bradley's liberal criterion), with **2,000** β = 0 replicates. Validation: with 500 draws the donor test's exact size is 26/501 = 0.0519. At 1,000 replicates a valid calibration would fail G1 about 17% of the time; at 2,000 it passes 99% of the time, while a true rate of 0.075 passes 0.2% of the time. A rate well below α is reported as conservatism. The registered p-value definition and 500 draws are unchanged.
- **Participant identifiers:** a metadata screen, specified before it ran, flags pairs of IDs in different sessions of the same vendor when one ID equals the other plus the suffix "A" (rule L1), or when both disclosed BFI-2 profiles are identical (rule L2). Flagged sessions are placed in one split group, exactly as near-duplicate transcripts are. Nothing is excluded.
  - **Result:** rule L1 flagged nothing. All 51 flagged pairs come from rule L2, all at vendor 02. They join 22 of the 250 sessions into five groups: one of 14 sessions (47 links through 10 distinct profiles) and four of 2. In the pilot stage alone, the 5 pilot sessions of the 14-session group form one group.
  - **Groups are conservative constraints, not identities.** Vendor 02 has many repeated, near-identical profiles. Vendors 00, 01 and 03 have no repeats among 1,850 disclosed profiles. Within the 14-session group, two sessions have both conversation partners sharing one profile, and two partners in one conversation are not one person. The groups are therefore not evidence that any two IDs belong to one person. They only keep the flagged sessions together, which can reduce, and never add, cross-split leakage.
  - **Partition sizes:** in a preview of the 250-session split without near-duplicate transcript links, the partition sizes stayed 175/25/50.

**7. What "speaker-disjoint" means in reporting.** Evaluation is disjoint by **recorded participant ID**, with metadata-flagged candidate duplicates grouped. It is not verified unseen-person generalisation. A person registered under two IDs whose profiles are undisclosed or differ cannot be detected. Every report states this, and that any such leakage would make the held-out increment optimistic.

**8. H3 dependency.** H3 requires a function classifier (MD-31). The recommended open-weights model needs separately authorised weight downloads. If that authorisation is not given before J4, H3 is not run: the deviation is reported, H1, H2 and H4 proceed, and the stop rule's part (ii) is recorded as unanswered.

**9. Resolutions of unspecified details, fixed before J1.** The specification leaves these details to the implementation. They are recorded here so that they are timestamped before any confirmatory result exists. Values are exactly those in `src/settings.py` at adoption; reasons and alternatives are in `docs/literature/METHOD_DECISION_REGISTER.md`. None changes a hypothesis, the estimand, a control or a gate criterion.

| Decision | What it fixes | Value |
|---|---|---|
| MD-02 | Pitch range for F0 extraction | `"fixed"` |
| MD-04 | Turn measurement details (intensity averaging and range; voiced frames at the boundary) | `{"intensity_averaging": "energy", "intensity_range": "p95_p5", "boundary_min_voiced_frames": 10}` |
| MD-05 | Minimum other turns for a speaker's held-out statistics | `10` |
| MD-07 | Rows used to fit Layer 1 | `"train_only_validation_reserved_for_layer2"` |
| MD-08 | Ridge penalty grid (relative to fitting rows), folds and rank grid | `{"lambda_relative_to_n_fit": [1e-05, 3e-05, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0], "nested_cv_folds": 5, "fold_unit": "session", "selection": "per_target_block_min_oof_sse", "rrr_selection": "min_block_weighted_oof_sse_over_lambda_and_rank"}` |
| MD-10 | Variance model for the log-likelihood unit | `{"target": "squared_out_of_fold_residuals", "floor_fraction_of_mean_squared_residual": 0.01}` |
| MD-12 | N1 donor matching | `{"duration_deciles_within_split": true, "word_count": "exact_then_nearest_within_decile"}` |
| MD-18 | Power level defining the MDE | `0.8` |
| MD-20 | G3 ordering rule (reported as a pattern only; section 4) | `{"alpha": 0.05, "n2_vs_n1": "n2_null_median_above_n1_null_q95"}` |
| MD-21 | Combining the eight per-dimension H3 tests | `"holm_min_over_dimensions"` |
| MD-22 | Replier identity in Z | `"one_hot_training_repliers_zero_for_unseen"` |
| MD-23 | Per-file loudness offset | `"file_median_intensity_over_voiced_frames"` |
| MD-25 | Near-duplicate transcript rule for split grouping | `{"shingle_words": 5, "jaccard_threshold": 0.5}` |
| MD-32 | Hand-labelling protocol for G4 | `{"n_pairs": 300, "seed": 20260912, "labeller": "study_author", "blind_to_classifier": true}` |
| MD-33 | Mixed-model structure for the re-check | `"session_groups_with_speaker_variance_component_all_included_pairs"` |
| MD-34 | The self-previous condition | `"z_plus_self_previous_without_partner_delivery"` |
| MD-35 | Injection space for J3 | `"standardised_target_space_orthogonalisation_refitted"` |
| MD-37 | One common analysis set for all conditions | `"reply_partner_self_previous_usable_texts_nonempty_n2_available"` |
| MD-38 | Scale on which the MDE is stated | `"max_of_target_and_realised_scale"` |

The G2 reporting rule of section 5 is recorded as MD-19 (`reporting_only_amendment_3`). MD-31 (the H3 function classifier) is **not** resolved by this amendment (section 8).

**10. Decision values adopted with sections 3–6.** The decisions described in prose above are adopted with exactly these values, as recorded in `src/settings.py` at adoption. Together with section 9, this lists every decision this amendment adopts. MD-27 to MD-30 (the deferred neural layer) and MD-31 are not adopted by it.

| Decision | Section | Value |
|---|---|---|
| MD-01 | §6 | `"exclude_overlap_energy"` |
| MD-03 | §6 | `"dictionary"` |
| MD-11 | §6 | `{"resamples": 2000, "unit": "session", "interval": "bootstrap_t", "simulated_worst_case_coverage": 0.934, "levels": [0.68, 0.95]}` |
| MD-15 | §6 | `{"rule": "donor_calibrated_subspace_stability", "null_draws": 19, "n_boot": 50}` |
| MD-16 | §6 | `{"replicates_per_beta": 50, "residual_resampling": "session_block_ordered", "chunk_replicates": 250, "cross_variant_permutation": {"mean_log_f0": "articulation_rate", "log_f0_range_p95_p5": "intensity_range", "log_f0_slope_final_300ms": "pause_ratio", "mean_intensity": "mean_log_f0", "intensity_range": "longest_internal_pause", "articulation_rate": "log_f0_range_p95_p5", "pause_ratio": "log_f0_slope_final_300ms", "longest_internal_pause": "mean_intensity"}}` |
| MD-17 | §6 | `{"alpha": 0.05, "replicates_beta0": 2000, "acceptance": "bradley_liberal_upper_bound"}` |
| MD-19 | §5 | `"reporting_only_amendment_3"` |
| MD-26 | §6 | `"drop"` |
| MD-40 | §3 | `"deferred"` |
| MD-41 | §4 | `"ordering_only_inconclusive_about_local_versus_longer_term"` |

**11. Synthetic data behind the numbers in sections 4 to 6.** Five synthetic studies support statements in this amendment. None reads corpus audio, transcripts, targets or model weights. None is a registered experiment. Each describes how the registered design and decision rules behave under a stated generating model. Magnitudes depend on the generating parameters, and the qualitative conclusions are what carry forward. They carry no evidential weight about conversation, and the registered calibration on real residuals (J3) remains the binding test.

| Study | Question | Generating model and size | What it showed |
|---|---|---|---|
| **A. H4 identification** (section 4) | Can the N1 and N2 contrast tell turn-to-turn coupling from shared state that changes within a few turns? | Per-turn latent delivery = coupling × partner's previous turn + self-persistence × own previous turn + N(0, 1) + a shared AR(1) drift + an interaction offset. 40 sessions of 80 turns (about 3,080 pairs), 10 seeds per scenario, 39 N1 and 39 N2 draws, and the real Layer-1 ridge and donor code. Nine scenarios: local coupling only, shared drift only (persistence 0.80, 0.95 and 0.99), local coupling with drift, neither, interaction-level offsets only, self-persistence only, and self-persistence with drift. | Drift with no coupling gave real > N2 > N1 in 40 of 40 replicates. Local coupling gave it in 20 of 20. Interaction offsets alone gave real ≈ N2 > N1 in 9 of 10. With neither, N1 rejected in 5 of 60 and N2 in 3 of 60. |
| **B. Power under serial dependence** (sections 5 and 6) | What block partial R² can J3 detect, and how does serial dependence in the turns change it? | A known coupling is injected into synthetic data of the planned shape: 250 sessions, 150 pairs per session, 482 conditioning columns, 40 replicates per level, 39 donor draws, with the real ridge estimator and donor null. Serial dependence (AR(1) with ρ 0.6 or 0.9) is added to partner delivery and to residuals. | The projected MDE at power 0.80 is about 0.0005 with independent turns, 0.0015 at ρ 0.6 and 0.0044 at ρ 0.9. With ρ 0.6, resampling residuals row by row gave 0.0007 and order-preserving session blocks gave 0.0016, so row-wise resampling understates the MDE by about 2.2 times. |
| **C. Bootstrap interval comparison** (section 6) | Which session-bootstrap interval comes closest to 95% coverage at this study's number of test sessions? | Per-pair squared errors for a reduced and a full model, eight dimensions in blocks of 2, 1, 2 and 3, with gamma errors and a session scale factor. Six scenarios: 25 or 50 test sessions, about 300 pairs per session (one scenario with unequal sizes), mean fraction of error removed 0.00 or 0.01, and session SD 0.02 or 0.05. 1,000 data sets per scenario and 1,000 resamples per interval. The selection rule was fixed before the run. | Worst-case coverage was 0.934 for the bootstrap-t, 0.916 for the percentile interval and 0.907 for BCa. No method stayed within Monte-Carlo error of 0.95 in all scenarios. |
| **D. Design-validation simulations** (sections 4 and 5) | The same questions as A and B with a closed-form stand-in for the estimator. | Ordinary least squares in place of the nested cross-validated ridge, and one dimension in place of each block. 250 sessions of 11 interactions with 36 alternating turns each, a slowly varying shared state, and local coupling. Three parts: the N2 control (S1), detectability at 40, 100 and 250 sessions (S2), and power for the function interaction (S3). | The same conclusion as A for the N2 control. A single-dimension partial R² of about 0.001 is detected with power 0.97 at 250 sessions under idealised data. A 50% larger coupling in the question stratum is detected with power 0.58 at 250 sessions. |
| **E. Software check** (no number in this amendment comes from it) | Is the pipeline connected end to end? | A small synthetic corpus (24 sessions) with a planted coupling and with none, run through J1 to J4 with tiny replicate counts. | It recovers the planted effect and finds none without it. Gates report PROVISIONAL because the replicate counts are too small to issue verdicts. |

The generator is `make_corpus` in `src/experiments/dry_run.py`. The scenario runner for study A is `src/experiments/h4_identification.py`, and the projection for study B is `src/experiments/power_projection.py`. The synthetic counterexamples of study A are kept as a regression test in `tests/test_h4_identification_and_g2.py`. Studies B, C and D, with their protocols written before the runs and their raw results, are in `docs/research/simulations/`.

**Unchanged.**
- H1, H2 and H3 and their nulls;
- the estimand and both units;
- the primary test;
- the Holm family of three;
- the eight dimensions and four blocks;
- the fixed 75–500 Hz pitch range;
- the conditioning set;
- N1 and N2, with 500 draws each;
- the positive control;
- G0, G1, G3, G4 and G5 and their failure actions;
- the four outcome interpretations;
- the stop rule.

---

## References

- Agrawal, V. et al. (2025). Seamless Interaction: dyadic audiovisual motion modeling and large-scale dataset. arXiv:2506.22554.
- Fuscone, S., Favre, B., and Prévot, L. (2020). Neural representations of dialogical history for improving upcoming turn acoustic parameters prediction. *Interspeech 2020*. doi:10.21437/Interspeech.2020-2785
- Levitan, R., and Hirschberg, J. (2011). Measuring acoustic-prosodic entrainment with respect to multiple levels and dimensions. *Interspeech 2011*.
- Reece, A., Cooney, G., Bull, P., et al. (2023). The CANDOR corpus: insights from a large multimodal dataset of naturalistic conversation. *Science Advances*. doi:10.1126/sciadv.adf3197
