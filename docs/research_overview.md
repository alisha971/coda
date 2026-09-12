# CODA — research overview

**Conversational Delivery Adaptation: a foundational empirical study for Situational Style Cloning.**

This document is the research-facing single source of truth for what CODA is scientifically. It explains the question, the design, and why the major components exist. It is not an implementation plan, a schedule, or a status report.

---

## 1. Project identity

CODA measures whether, in spontaneous dyadic conversation, the way a partner speaks predicts the way a reply is delivered, once the reply's own words and the partner's transcript are already accounted for.

**Relationship to Situational Style Cloning (SSC).** SSC is a larger research programme that treats expressive conversational delivery as a *situation-conditioned policy* rather than one global speaking style per person. That design choice rests on an empirical premise nobody has tested directly: that the relationship between conversational context and delivery is stable, generalises to unseen speakers, and varies with the situation. CODA tests that premise. It does not build SSC, and a positive result does not prove SSC will work — it establishes whether attempting it is warranted.

**Scope boundary.** CODA generates no audio. It is a measurement study.

---

## 2. The scientific problem

Once a conversational speech system has decided *what* to say, it still has to decide *how* to say it. In practice that decision is made almost entirely from text: the transcript so far, plus the reply's own words. The open question is whether the *sound* of the partner's turn carries delivery-relevant information that the transcript does not.

The direction of this question is not open. Conversational speech synthesis has used acoustic context encoders since around 2022, and prosodic entrainment between conversational partners has been documented for decades. What is missing is a **quantity**.

Existing evidence comes in two forms, and neither answers the measurement question:

- **System-level comparisons** in speech synthesis, where adding an acoustic context encoder changes modality, parameter count and optimisation together, and the result is read out as listener preference over synthesised speech.
- **Correlational similarity indices** in the entrainment literature, which measure how alike two speakers are with both turns in hand, rather than how predictable one is from the other with the reply's own content held fixed.

So the field knows the effect exists and acts on it, but has not reported how much delivery information conversational context carries, where in the delivery it lives, whether it holds for speakers the model has never heard, or whether it depends on what the turn pair is doing conversationally.

---

## 3. Scientific framing

**Conversational delivery** here means the measurable prosodic manner of a turn: its pitch level and span, its final pitch movement, its loudness, and its timing. Not its words, and not voice identity.

**Partner acoustic delivery** is the same set of measurements taken on the turn immediately preceding the reply, spoken by the other participant.

**Conditional predictive dependence** is the relationship CODA estimates. It is explicitly not causation, not mutual information, and not "the information content of the acoustic channel". It is the out-of-sample gain in predicting the reply's delivery from adding the partner's delivery to a model that already has the reply's words, the partner's words, and who is speaking.

**Conversational function** is what the turn pair is doing: a question and its answer, an assessment and its response, a narrative and its continuation, or a backchannel-adjacent exchange.

**Speaker generalisation** means evaluation on participants who appear in no training conversation. The claim is about the relationship, not about fitting particular voices.

---

## 4. Hypotheses

**H1 — primary, confirmatory.** Conditional on the reply's text and length, the partner's text, and replier identity, the partner turn's delivery is predictive of the reply's delivery, on a speaker-disjoint test set, beyond a matched-donor null.

**H2 — secondary, confirmatory.** The relationship is *structured*: there are reproducible partner-dimension to reply-dimension directions rather than diffuse association. "Reproducible" is operationalised as a battery, not a single number — reduced-rank structure, stability of loadings across dyad-level bootstrap resamples, stability across held-out-speaker folds, concentration of explained covariance, comparison against a random-rotation null, and consistency with what the learned representation encodes. The null is that no direction survives the bootstrap, the speaker-fold and the rotation null simultaneously.

**H3 — the SSC bridge.** The coupling is function-dependent: its magnitude, its dimensional structure, or both differ systematically across conversational functions.

**H4 — competing explanation, mandatory.** The relationship reflects local adaptation rather than merely shared conversation-level state: it weakens when the partner turn is replaced by a distant turn from the same speaker in the same conversation.

H4 exists because the strongest objection to this project is mundane and serious: two people at a funeral both speak quietly, and that is not adaptation. H4 is how that objection gets tested rather than argued about.

---

## 5. The estimand

For turn pair *i*:

- **Y** — the reply's delivery vector, speaker-standardised.
- **A** — the partner turn's delivery evidence.
- **Z** — the conditioning set: reply text embedding, reply word count, reply syllable count, partner text embedding, replier identity, turn position, and a per-conversation loudness offset.

> **Δ is the out-of-sample increase in predictive accuracy of Y from adding A to Z, referenced to the distribution of that increase under conditionally matched replacement of A.**

Two pre-specified units: **partial R² per target block**, which is primary, and the **out-of-fold predictive log-likelihood increment**. If the two disagree, partial R² governs and the disagreement is reported rather than averaged away.

**Why a donor null rather than a plain comparison.** Δ is compared against a distribution generated by refitting with *donor* partner vectors matched on Z. This puts dimensionality, marginals, feature scale, optimisation difficulty and the regularisation path *inside* the null, rather than adjusting for them afterwards. It is what turns Δ from a statement about input width into a statement about information.

**Unit of analysis: the dyad.** All resampling, all intervals and all random effects are at conversation level. The effective sample size is the number of conversations, not the number of turn pairs, which is far larger and far from independent.

---

## 6. Evidence architecture

CODA has two layers with distinct jobs, and keeping them distinct is the point.

| | **Layer 1 — statistical instrument** | **Layer 2 — neural modelling layer** |
|---|---|---|
| What | ridge and reduced-rank ridge on interpretable features | one compact multimodal transformer over frozen self-supervised speech representations |
| Job | the primary interpretable, capacity-controlled Δ under both controls; the coupling map; block-level effect sizes; the detectability floor | whether a modern learned representation recovers the same structure, and whether temporal and nonlinear structure the linear model cannot express is present |
| Status | **primary scientific evidence** | **corroborating technical evidence** |

If the two disagree, the disagreement is investigated and reported, the statistical result remains primary, and the disagreement itself becomes a finding about representations. If the neural layer fails to train stably, the scientific claim is unaffected and that failure is reported as a technical result.

The hierarchy is deliberate: **scientifically identified first, technically sophisticated second.** A sophisticated model without matched-donor controls would be weaker evidence than a ridge with them.

> Model complexity is not evidence of scientific validity. The scientific claim is identified by the same matched-donor controls and speaker-disjoint evaluation whether the model is linear or neural.

---

## 7. Representations

### Target: reply delivery — 4 blocks, 8 dimensions

| Block | Dimensions |
|---|---|
| Pitch level and span | mean log-F0; log-F0 range (p95 − p5) |
| Pitch boundary | slope of log-F0 over the final 300 ms |
| Energy | mean intensity; intensity range |
| Timing | articulation rate; pause ratio; longest internal pause |

Measured with Praat, matching the source of the feature set.

**Standardisation** is per-speaker z-scoring, following Levitan and Hirschberg's norming procedure and cited to them. For unseen test speakers the statistics come from that speaker's *other* turns in the same conversation, excluding the target turn and every turn used as its context — otherwise the normalisation would leak the answer.

This is called per-speaker standardisation, never "disentanglement" and never "speaker-invariant", because nothing here establishes that speaker identity has been removed.

**Within-block orthogonalisation** is fitted on training data and applied everywhere, so that aggregating to block level does not double-count correlated dimensions.

**Confirmatory inference is at block level: four tests, not one per dimension.**

### Partner context: two pathways

Both are core; neither replaces the other.

**Pathway A — the interpretable delivery vector.** The partner turn's own dimensions, computed and standardised identically to the target. It produces the primary Δ, the coupling map, the dimension-to-dimension interpretation, and the donor construction. Two properties make it the primary instrument: it contains no lexical content by construction, so "beyond the transcript" holds at the level of the representation rather than only because partner text sits in Z; and at 8 dimensions against text's 64 it is *narrower* than the lexical channel, so input width runs against a positive result rather than manufacturing one.

**Pathway B — the learned representation.** Five frame-level self-supervised representations of the same partner turn — WavLM layers 3, 6, 9 and 12, plus HuBERT's final layer — combined by a learned, softmax-normalised mixture over per-source projections into a shared space.

Mixing rather than choosing is deliberate. Selecting a layer on validation performance would turn representation choice into a hyperparameter search with a tuning path into the answer. Mixing makes it a learned parameter inside a fixed architecture. The learned weights are reported, and their interpretation is a *modelling* statement about which level of pretrained abstraction carries useful evidence — never a research finding in itself. There is no table crowning a winning layer.

### Text and metadata

A frozen sentence encoder produces embeddings for reply and partner text separately, each reduced to 64 dimensions. The encoder is never trained. Metadata entering Z: reply word count, reply syllable count, turn position, per-conversation loudness offset, and replier identity. The replier's own previous turn's delivery vector is held separately as the predictor H4 needs.

---

## 8. Model architecture

The neural layer is one compact architecture, not a family.

```
partner audio -> frozen WavLM L3/L6/L9/L12 + HuBERT   (no gradients)
                        |
              learned layer mixture -> shared space (d = 192)
                        |
        partner acoustic encoder
          learned positional encoding
          3 pre-norm transformer blocks, 4 heads, FFN 384
          key-padding mask from true frame lengths
          attention pooling with a learned query
                        |
partner text -> MLP ----+                    context -> MLP
                        |                            |
              cross-modal fusion: one cross-attention block
              (query = context and partner text, key/value = masked
               partner frames), residual pooled path, concatenation
                        |
              delivery head: 2-layer MLP -> 8 means
                                  (+ 8 log-variances, optional)
```

**Why each element is present.** The sequence model exists because delivery structure is temporal and the model should be able to weight the parts of the partner turn that matter, plausibly the turn-final material. The single cross-attention block exists because it matches the question: given what I am about to say and who I am, which parts of how you spoke matter. It generates nothing — there is no decoder, no token generation, no audio generation.

**Modality separation is explicit and auditable.** Partner acoustics, partner text, reply text and metadata are encoded separately and fused only afterwards, never concatenated as raw vectors at the input. This is what makes the "beyond lexical information" question answerable inside the architecture: the partner-acoustic pathway can be ablated without touching anything else.

**Capacity philosophy.** The self-supervised encoders are frozen; only the mixer, projections, transformer, fusion and head are trained, targeting roughly 1.5 to 2.5 million trainable parameters, reported exactly alongside the dyad-level sample size. Against a few hundred conversations, the model is deliberately small. The training objective weights the four blocks equally, so that a three-dimension block cannot dominate a one-dimension block through count or through scale.

**The neural layer is not claimed as a contribution.** It is an instrument.

---

## 9. Experimental logic

Four jobs. The neural layer runs the same conditions and adds no new experiments.

| Job | What it answers |
|---|---|
| **J1 — condition sweep** | How predictable is delivery at all; what does the partner's transcript add; what does the partner's delivery add; does it hold for unseen speakers |
| **J2 — null family** | Does the result depend on genuine pairing; is it local or conversation-level |
| **J3 — calibration** | Can the instrument detect an effect of plausible size at this sample size |
| **J4 — function interaction** | Does the coupling depend on conversational function |

J3 runs and is reported **before any real result is inspected**. Establishing what the study could detect, before knowing what it found, is what allows a small or null estimate to be read as a bound rather than an absence.

---

## 10. Controls

**N1 — cross-dyad matched donor.** The partner turn is replaced by one from a different conversation in the same split, matched on duration decile and word count. This preserves dimensionality, scale, marginals, length structure and model capacity, and destroys the pairing entirely. It is the control that makes Δ an information statement rather than a statement about input width or capacity.

**N2 — within-dyad distant donor.** The partner turn is replaced by one from the *same* conversation and the *same* speaker, at least ten turns away. This preserves the dyad, the speaker, the session, the channel, the register and the broad topic, and destroys only local adjacency. It is the test of H4 and the answer to the funeral objection.

Both controls bind on both layers, with architecture, preprocessing, training and evaluation held fixed. Only the partner-context assignment changes. Donor draws are seeded and saved so nulls are exactly reproducible.

**The reading is in the ordering:**

- `real > N2 > N1` — local adaptation;
- `real ≈ N2 > N1` — conversation-level style;
- `real ≈ N1` — nothing.

**Positive control.** A semi-synthetic signal of known size is injected into the targets and the entire pipeline is run on it, sweeping the injected effect across the plausible range. The model must recover what was put in. An architecture that cannot recover a known effect cannot be trusted to report its absence — this is what makes the neural arm evidence rather than decoration.

**Capacity control for the neural layer.** Identical architecture in every condition, including the ablation, where the partner-acoustic pathway is not removed but fed a learned null sequence, so parameter count is identical across conditions. Identical optimiser, schedule, batch size, epoch budget and early stopping. The hyperparameter budget is searched once on the real condition's validation set and then frozen and reused unchanged everywhere. The comparison is *same model class, same capacity, same optimisation, different information available.*

---

## 11. Statistical framework

**Splits** are at dyad level, roughly 70/10/20, under a speaker-disjointness constraint so no participant appears in more than one split. A dyad-disjoint-only split is fitted as a contrast.

**Model selection** happens inside training, by nested cross-validation for the statistical layer and validation-based early stopping for the neural layer. The test set is touched once per condition.

**Effect sizes** are per block, aggregated with equal weight per block, and always reported with the block-level minimum detectable effect overlaid, so a small estimate reads as a bound rather than an absence.

**Uncertainty** comes from a paired dyad-level cluster bootstrap, reported alongside permutation p-values from the donor nulls. Sampling uncertainty and seed-to-seed variability are reported separately, never pooled.

**Multiplicity.** One primary test is declared in advance: H1, aggregate, speaker-disjoint, Layer 1, with both controls as reference nulls. H2 through H4 are Holm-corrected within a family of three. The neural layer's results are corroborating, carry no confirmatory significance level of their own, do not enter the correction family, and can neither rescue a null nor overturn a positive from the statistical layer.

**A confirmatory re-check** fits a mixed-effects model with crossed speaker and dyad random effects — the entrainment literature's own tool, and the read-out most legible to the audience best placed to judge it.

**Deliberately not used.** Conditional mutual information estimation, because there is no accepted protocol at this sample size and dimensionality. Likelihood-ratio tests, because they are invalid for regularised cross-validated fits. Both exclusions are stated rather than hidden.

**Specified in advance.** Turns with missing prosody are dropped, the count is logged, and droppage is checked not to differ by condition. A negative Δ is interpreted only against the null, never as evidence that acoustic context hurts.

---

## 12. Validity and leakage

Leakage prevention is part of the experimental design, not a cleanup step. Every route below has named evidence — a test, an assertion, a schema constraint, or a timestamp — and the integrity suite passes before any fit runs.

**The information firewall is per pair, not global.** A pair may never receive its own reply's audio, its own reply's learned embedding, or its own reply's true duration. The same turn *must* be served normally when it is the context of a different pair. A global blacklist would be both wrong and easy, which is why the rule is enforced and tested in both directions.

**Standardisation cannot see the answer.** For a held-out speaker, the statistics come from that speaker's other turns excluding the target turn and all of its context turns.

**Train-only statistics.** Every normalisation and dimensionality reduction, including both text and self-supervised feature reductions, is fitted on training data and applied everywhere. Standardised test targets should come out *near but not exactly* mean zero and unit spread; exactly zero and one would indicate peeking.

**Speaker contamination** is prevented by construction and asserted: a participant appears in one split only. Near-duplicate conversations are clustered and assigned whole to a single split.

**Representation extraction is label-blind.** Self-supervised features are a pure function of audio; the extraction takes no label input. The encoders are frozen, asserted, and no gradient reaches them for any speaker.

**A speaker-identity probe** on the pooled learned partner representation is reported beside every neural Δ, so that a shortcut through speaker identity would be visible rather than assumed absent.

**Function labels are text-only.** The situation classifier receives the turn-pair transcript and never audio or delivery features.

**Hypotheses precede results.** The preregistration is committed before the confirmatory experiments run, and its timestamp is the evidence.

---

## 13. Situation analysis

This is the component that speaks directly to SSC. Its premise is that expressive delivery is situation-conditioned rather than one global style. The minimal test is not "can a policy be learned" but "does the context-to-delivery relationship itself change with the situation?" If it does not, a situation-conditioned policy is over-parameterised for the phenomenon and SSC's central design choice is unmotivated.

**Function categories**, assigned to the pair: question to answer; assessment or opinion to response; narrative or inform to continuation; backchannel-adjacent; and other, which is excluded from analysis rather than modelled.

**Reliability gates the claim.** Interrogative syntax comes from text; the remaining categories from a text-only classifier. Then 300 randomly sampled pairs are hand-labelled, agreement is reported per category, and any category falling below the agreement threshold is dropped from confirmatory analysis. Strata below a minimum size are descriptive only. A finding built on unreliable labels is not a finding.

**Two forms of dependence are distinguished**, and they imply different things:

- **Magnitude dependence** — the coupling is stronger in some functions. A policy should weight context differently by situation.
- **Structural dependence** — the coupling loads on *different dimensions* in different functions. A policy needs a situation-conditioned mapping, not merely a situation-conditioned gain. This is the stronger finding, and the reduced-rank factors are what can show it.

The interaction is tested directly as a term in the mixed-effects model, not by eyeballing subgroup estimates. Any neural view of function-dependent structure is exploratory and labelled as such.

---

## 14. Decision logic

The study is built in two levels, with gates between them. A gate is a pre-committed decision point with a stated pass criterion and a stated consequence for failing, fixed before the number is seen.

| | **Level D** | **Level E** |
|---|---|---|
| Claim | partner delivery is, or is not, conditionally predictive of reply delivery on unseen speakers, with a stated detectability floor | additionally, that relationship is, or is not, systematically function-dependent |
| Hypotheses | H1, H2, H4 | plus H3 |
| Jobs | J1, J2, J3 | plus J4 |

The gates cover corpus viability, null calibration, detectability, the core test, label reliability and the situation interaction. Two further checkpoints gate only the neural layer's interpretability; neither can block, delay or extend the scientific study, and failing either changes no scientific conclusion.

**The most consequential gate is calibration.** If the procedure rejects at the nominal rate when there is nothing to find, and can detect an effect of plausible size at the achieved sample, then the study can answer its question. If it cannot, nothing downstream runs. That check happens before the answer is known.

**Stop rule.** CODA is complete when the conditional contribution of partner delivery has been estimated under both controls with speaker-disjoint evaluation and a stated detectability floor, and the function-dependence question has been answered — positively, negatively, or as underpowered with the power stated. No experiment is added because it would improve publication prospects, because another corpus exists, or because a reviewer might ask.

---

## 15. Scope boundaries

**Not in CODA, by design:** any speech generation or text-to-speech; style cloning or transfer; style and identity disentanglement; a retrieval-based or language-model-based style policy; an end-to-end speech agent; a human appropriateness benchmark or listening test; a complete situation-to-style policy learner; delivery ranking or any downstream usefulness task; and causal claims of any kind.

**One architecture.** Separate models per self-supervised family, multiple architectures as independent experiments, size sweeps and architecture search are all explicitly rejected. There is no leaderboard and no dashboard of model comparisons.

**Relationship to SSC.** CODA is the measurement that says whether the next stage is worth building. That stage — learning a mapping from situational context to a delivery representation, conditioning expressive synthesis on it, and evaluating whether listeners judge the result appropriate — needs style and identity separation, multi-speaker data and a human evaluation protocol, all deliberately outside CODA.

---

## 16. Expected interpretations

Four outcomes are committed to in advance. No fifth is created after the fact.

**1. Coupling present, surviving both controls, structured, and function-dependent.** Reply delivery is conditionally predictive from partner delivery beyond lexical context on unseen speakers; the relationship concentrates in identifiable dimensions; and it differs systematically by conversational function. This supports modelling expressive conversational speech as a situation-conditioned delivery policy.

**2. Coupling present, not function-dependent.** Reliable contextual coupling that does not vary detectably with function, with the per-stratum detectability floor stated. This supports context-conditioned delivery modelling but provides no evidence that situation conditioning is necessary. SSC's situation-conditioning assumption is then untested rather than supported.

**3. Coupling present but equivalent under the within-dyad control.** The relationship reflects conversation-level style alignment rather than turn-level adaptation. For agent design this favours a running conversation-style estimate over last-turn conditioning.

**4. Calibrated null.** With a stated detectable-effect floor on the achieved sample and unseen speakers, delivery-to-delivery coupling is weak or not reliably present under controlled evaluation. This places limits on the premise that conversational context provides a stable basis for a general delivery policy.

**Where the neural layer enters.** It can strengthen or weaken confidence in the representation-level result *within* any of these four. It cannot create a fifth. A neural positive under a statistical null is reported as representation-dependence — the relationship is not recoverable from pooled prosodic scalars but is recoverable from a learned representation at this sample size — which qualifies outcome 4 rather than replacing it, and is exactly the kind of finding the two-layer design exists to catch.

**Success is a credible answer, not a positive one.** All four outcomes are writable and all four lead somewhere.

---

## 17. Literature positioning

**Established, and not claimed here.** That dialogue context improves generated delivery, including acoustic context. That conversational partners' prosody is measurably related. That coordination varies by dialogue act, correlationally. The shuffled-context control, which is the sham-conversation device that has defined local entrainment measurement since at least Nasir and colleagues (2018). The target feature families and per-speaker normalisation, which are Levitan and Hirschberg's (2011) and are cited to them.

**Unresolved, and what CODA addresses.** The size of the partner-delivery contribution as a quantity with an interval on interpretable delivery dimensions. The detectability floor, without which small or null results in this area cannot be read as bounds. A predictive dimension-to-dimension coupling map with stability testing. Function-dependence of a conditional predictive increment with reply content held fixed. And the separation of local adjacency from conversation-level state, which standard sham controls conflate because they draw donors from different conversations entirely.

**Contribution class, stated without inflation:** a measurement study that converts an assumption the field already acts on into a quantity with an interval, together with the first function-conditional version of that quantity.

**Three claims, and no more:**

1. An identified, capacity-neutral effect size for a conditioning channel the field already deploys, where existing evidence is system-level and confounded.
2. Speaker-generalising evidence for the predictive claim, which the entrainment literature supplies only correlationally and the synthesis literature typically supplies on one or two speakers.
3. A function-conditional characterisation of the predictive relationship, extending correlational act-dependence findings into a predictive frame.

**Explicitly not claimed:** discovery of the phenomenon; a novel method; a novel architecture; novelty of the matched-pairing control; novelty of the feature set or of per-speaker normalisation.

Stated as a discovery, this work would be dismissed by anyone who knows the area. Stated as the identified version of a question the field has answered indirectly, it is a narrow empirical contribution whose null result would be as informative as its positive one.
