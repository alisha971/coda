# Prior work and novelty review

**Date:** 2026-10-16. **Scope:** the scientific questions of CODA (H1–H4), not the implementation methods (those are in `LITERATURE_REVIEW.md`). **Companion table:** `PRIOR_WORK_EVIDENCE_TABLE.csv` (40 rows, one per study or study group, with verification level, design, controls, speaker hold-out and what remains open). Row identifiers `E01`–`E40` are cited below.

---

## 0. Search procedure and its limits

**Sources searched.** The ISCA Archive, ACL Anthology, arXiv, PubMed Central, publisher landing pages, university repositories (HAL, NSF PAR, Columbia, Stuttgart IMS, Brown VIVO) and corpus distribution pages (TalkBank, ConvoKit, BetterUp, Hugging Face).

**Query families** : prosodic/acoustic-prosodic entrainment, accommodation, alignment, convergence, synchrony + review / methods comparison; "predict" + next / upcoming turn + prosody / acoustic parameters + interlocutor; speaker-independent / held-out speakers + entrainment or response prediction; dialog act / backchannel / question + entrainment; local vs global entrainment; difference-in-difference convergence; surrogate / sham / shuffled nulls for synchrony; conversational TTS + acoustic context; spoken LLM + response style / paralinguistics; SSL + prosody; "CANDOR corpus" + entrainment / prosody; "Seamless Interaction" + prosody / entrainment.

**Citation chasing.** Backward: the reference lists of Fuscone et al. 2020 (E18), Levitan & Hirschberg 2011 (E03), Reichel et al. 2018 (E11) and Nasir et al. 2018 (E13) were read and their most relevant entries followed. Forward: searches for 2024–2026 work naming the two candidate corpora, and for work citing the closest predictive study (E18) by title.

**Date range covered.** 2006–2026, with earlier foundational work only where cited by included papers.

**Reading depth.** Each row of the evidence table states it: **FULL TEXT** (methods and results read, 3 rows), **PARTIAL TEXT** (the sections named were read, 11), **ABSTRACT** (20), **SECONDARY** (the claim rests on another source's report, 6). A finding marked ABSTRACT or SECONDARY is a reported claim, not a verified methodology. Two full texts that bear directly on CODA could not be retrieved (Kruyt et al. 2023, restricted access; Cohen Priva & Sanker 2019, not located) and are flagged wherever they are used.

**Limits.** No bibliographic database with controlled indexing (Web of Science, Scopus, Google Scholar citation graphs) was available, so forward citation chasing relied on title searches and is incomplete. Non-English literature was not searched. Theses and conference posters are under-represented. **Not finding a study is not evidence that it does not exist**; every "open" or "potentially novel" classification below is conditional on this search.

---

## 1. Conversational prosodic entrainment: the phenomenon (ESTABLISHED)

Partners in conversation become measurably similar in intensity, pitch, speaking rate and voice quality, both over a conversation and at turn exchanges. The standard apparatus — proximity, convergence and synchrony at global and local levels, with partner-versus-non-partner comparisons and shuffled baselines — dates to Edlund et al. (2009, E06) and Levitan & Hirschberg (2011, E03), and has been reproduced across corpora, languages and populations (E08, E21, E22, E36). Two recent reviews (Wynn & Borrie 2022, E02; Kruyt et al. 2023, E01) document that terminology and measures vary widely, and Kruyt et al. report that twelve methods applied to one corpus disagree about whether, and how, entrainment is present.

Three properties of this literature matter for CODA.

1. **Magnitudes are unconditioned.** Levitan & Hirschberg's turn-level synchrony correlations (E03, full text) are r = 0.47–0.50 for intensity, 0.28 for mean pitch and 0.15 for speaking rate, on 12 sessions, with features z-scored **by gender, not by speaker**, and turns pooled as if independent. These correlations include between-speaker, between-session and shared-channel variance. They are upper bounds on, not estimates of, a conditional increment after per-speaker standardisation and lexical conditioning.
2. **Alignment is feature-specific and loosely structured.** Ostrand & Chodroff (2021, E10) find alignment on temporal features and holistic profile but not on several spectral features; Weise & Levitan (2018, E09) find no shared structure among entrainment measures. Weise et al. (2019, E08) find large individual differences in the strength and sign of local entrainment.
3. **Naive nulls over-detect.** Surrogate-data work outside speech (Moulder et al. 2018, E23) shows that common synchrony measures register synchrony between people who never interacted; nulls must preserve autocorrelation.

**Correction to the specification's citation.** §5 and §8.1 cite Levitan & Hirschberg (2011) as the source of both the feature set and **per-speaker** z-scoring. The full text shows gender-level z-scoring. Per-speaker z-scoring is used in later work from the same group (Weise et al. 2019, E08, verified in partial text). The feature families are correctly attributed; the normalisation attribution should be corrected in the write-up.

---

## 2. Predicting reply delivery from partner delivery (PARTIALLY ANSWERED)

This is the question that matters most for novelty, and the existing survey (`CODA_literature_survey.md`) missed its closest prior study.

**Fuscone, Favre & Prévot (2020, E18; full text read).** On 642 Switchboard conversations, an LSTM predicts the upcoming turn's mean energy, F0 range and speech rate from the previous 1–10 turns. Adding the **interlocutor's** turns to the upcoming speaker's own history lowers L1 error for energy (N = 1: 0.381 vs 0.416, p < 10⁻⁸) and F0 range (0.371 vs 0.390, p < 0.05), but not for speech rate. A journal version (E19, abstract only, via a secondary summary) reports about 7% average error reduction from dialogical history. This is the CODA question in its unconditioned form: forward prediction of a reply's prosody from partner prosody, over and above the replier's own history, in spontaneous conversation.

What the study does **not** do, verified in the full text:
- it does not condition on the reply's words or the partner's words (only a three-class dialog act and turn duration enter);
- it normalises energy by the speaker's conversation mean, which includes the target turn;
- hyperparameters are searched separately for each setup, so capacity is not held fixed across the compared conditions;
- folds are not described as grouped by speaker, and Switchboard speakers recur across conversations;
- there is no pairing-destroying null, no effect size in variance units, and no detectability floor.

**Other forward-regression evidence.** Schweitzer & Lewandowski (2013, E21, abstract) regress local articulation rate on the interlocutor's preceding rate in 46 German dialogues and find an effect moderated by liking. Cohen Priva et al. (2017, E16), reproduced by Fuscone et al. (2021, E17, partial text), find that a speaker's conversation-level speech rate depends on the interlocutor's baseline rate measured in **other** conversations, with a standardised coefficient of 0.046–0.060 against 0.78 for the speaker's own baseline. Nasir et al. (2018, E13, full text) train a network to predict the next turn's acoustic functionals from the current turn and use its embedding as an entrainment distance, validated against turn-shuffled sessions.

**Classification.** That partner prosody adds forward predictive information about the next turn's prosody, beyond the replier's own history, is **partially answered** for energy and F0 range (E18) and for rate at conversation level (E16/E17). It is **open** whether this survives conditioning on the reply's own words and the partner's transcript, evaluation on unseen speakers, and a null that holds model capacity fixed.

---

## 3. Separating partner effects from lexical content, context, speaker identity and self-history (OPEN as a combination)

| Confound | How prior work handles it | Evidence rows |
|---|---|---|
| Speaker identity / habitual level | Per-speaker z-scoring (E08); speaker baselines from other conversations (E16); gender z-scoring only (E03) | E03, E08, E16 |
| Speaker's own previous delivery | Speaker-history baseline (E18); own baseline (E16) | E16, E18 |
| Lexical content of the reply | Lexically normalised speech rate (E16) is the only instance found; no study conditions reply prosody on reply text embeddings | E16 |
| Partner transcript | Not found in the entrainment literature; present as an input, not as a conditioning set, in spoken LLMs (E28, E29) | E28, E29 |
| Dyad / session-level shared state | Cross-conversation sham pairs (E03, E13, E33); interlocutor baseline from other conversations (E16). Dyad-level variance is large (E24) | E13, E24, E33 |

No study found combines all five. **Conditioning reply prosody on the reply's own words and the partner's transcript, then asking what partner delivery adds, is open within the limits of this search.** It is a new combination of known controls applied to a known question, which is a measurement contribution, not a new phenomenon.

---

## 4. Generalisation to unseen speakers (PARTIALLY ANSWERED, outside the target)

Speaker-disjoint evaluation exists for partner-to-response prediction of **sentiment** (ParalinGPT, E29: 385/86/72 train/dev/test speakers, verified), for dyad-structure discrimination on CANDOR (E33, robust to a fully speaker-disjoint split), and for the Seamless corpus's own behaviour models (E31, participant-disjoint splits). Turn-taking prediction uses conversation-level splits (E37). **For forward prediction of continuous reply prosody from partner prosody, no speaker-disjoint evaluation was found** (E18 does not state one; Switchboard speakers recur). The existing survey's statement that speaker-disjoint evaluation is "rare in the predictive work" should be sharpened: it exists in adjacent tasks, not for this target.

---

## 5. Local versus longer-term entrainment (PARTIALLY ANSWERED; registered control under-identifies it)

Levitan & Hirschberg (E03) find speakers "even more similar to each other at turn exchanges" than over the session. Fuscone et al. (E18) find energy predicted mainly by the most recent turns and F0 range and rate improving with longer history. Cohen Priva et al. (E16) document convergence to a partner's baseline, a global effect. Kruyt et al. (E01) group methods by timescale and find they disagree. Cavalcanti & Skantze (2025, E24) find a dyad-specific component of turn timing larger than the speaker-specific component.

The registered N2 control replaces the partner turn with a turn by the same partner at least ten turns away in the same interaction. This removes adjacency but keeps dyad, session, prompt and topic. Its interpretive premise is that shared state is constant within an interaction, so that anything N2 removes is local. **No reviewed study establishes that the shared state of a 2–10-minute prompted interaction is constant over ten turns.** If shared state drifts, N2 also removes part of it, and real > N2 does not by itself identify local adaptation. The scoping simulation S1 tests this directly (`docs/research/simulations/`).

---

## 6. Conversational function (PARTIALLY ANSWERED correlationally; predictive version OPEN)

- **Backchannels.** Backchannel pitch is closer to the immediately preceding interlocutor speech than non-backchannel pitch (Heldner et al. 2010, E05, partial text); partners entrain on backchannel-preceding cues (E04). Function-dependent **local** similarity is therefore already documented for one of CODA's four strata.
- **Dialog acts.** Reichel, Mády & Cole (2018, E11, partial text) find entrainment differs by dialog act in game dialogues, but on 16 dialogues and as **global** entrainment between same-act pairs, not adjacent call–response pairs.
- **Qualitative grounding.** Conversation analysis treats prosodic matching in responses as an interactional resource (Szczepek Reed 2006, E25, secondary).
- **As an input.** Fuscone et al. (E18) use a three-class dialog act as an input feature and find it improves prediction, but do not test whether the partner effect differs by act.

**Classification.** Function dependence of prosodic coordination is **partially answered** correlationally. Whether the **conditional predictive increment** differs by function, in magnitude or in dimensional structure, is **open**, and potentially novel within the limits of the search. Feasibility risks are specific (§9).

---

## 7. Statistics and evaluation for dependent conversational data (ESTABLISHED tools; application gaps)

Mixed-effects models with speaker and dyad random effects are the entrainment literature's standard (E11, E16, E17, E21). Blocked cross-validation, cluster bootstrap and few-cluster corrections are established (E40; `LITERATURE_REVIEW.md`). Conditional randomisation and permutation tests give the theory behind matched-donor nulls, with validity resting on the conditional law of A given Z (E39). Surrogate nulls must preserve autocorrelation (E23). Two documented weaknesses recur in prior work: pooling turns as independent (E03) and normalising with statistics that include the target turn (E18). CODA's design already guards against both.

---

## 8. Studies on the candidate corpora

- **Seamless Interaction** (Agrawal et al. 2025, E31): no prosody or partner-dependence analysis in the release paper. 2026 work uses it for prosody reference ranges in agent evaluation (E34, abstract) and possibly for emotional-entrainment detection (E35, abstract; source corpus not confirmed). **No study estimating partner-to-reply prosodic dependence on Seamless was found.**
- **CANDOR** (Reece et al. 2023, E32): a 2026 preprint shows dyad-specific acoustic structure via speaker switching, robust to speaker-disjoint splits (E33, partial text). No turn-level conditional estimate was found.

---

## 9. Conversational speech synthesis and spoken LLMs: what they do and do not establish

The existing survey states that acoustic dialogue context is "established" as useful for generated reply delivery. The verified texts support a weaker statement.

- **Nishimura et al. (2022, E26):** on three actors, *simply adding prosodic context did not improve quality*; one training method helped in some comparisons, with MOS differences of about 0.1. The survey's description of this paper as the closest demonstration of usefulness should be corrected.
- **M2-CTTS (2023, E27):** two speakers (DailyTalk), CMOS on 20 utterances.
- **StyleTalk / Spoken-LLM (2024, E28):** the speech is **synthesised by a commercial TTS** from LLM-written dialogues; the style–response relationships are those the generator was told to produce. It carries no evidence about human conversation.
- **ParalinGPT (2024, E29):** real Switchboard speech, speaker-disjoint split, categorical sentiment target.

These systems show that acoustic context can be engineered into generation and sometimes helps listener ratings. They do not show how much partner delivery information human replies carry. That strengthens CODA's motivation, but the claim that "the direction is already answered in conversational-TTS work" (spec §5, §21) overstates what that literature shows. The direction is better supported by the entrainment literature (§1) and by E18.

---

## 10. Effect-size anchors for gate G2

G2 requires the block MDE to fall "below the plausible effect range from the entrainment literature". No reviewed study reports CODA's quantity: block partial R² of a conditional, speaker-disjoint, forward increment. The nearest anchors, converted under stated assumptions:

| Source | Reported | Converted to an approximate block partial R² | Direction of bias for CODA |
|---|---|---|---|
| E16/E17 (conversation-level rate, partner baseline from other conversations) | standardised β ≈ 0.05 | β² ≈ 0.0025, if the partner baseline is roughly orthogonal to the other predictors | turn-level noise lowers it; local adjacency may raise it |
| E18 (turn-level energy, F0 range; own history as baseline; no lexical conditioning) | L1 error −4.9% (F0 range) to −8.4% (energy) | not reliably convertible; a relative L1 reduction of 5–8% would correspond to a much larger R² change if errors were Gaussian, but energy normalisation includes the target turn | inflated relative to CODA (no lexical conditioning, leakage-prone normalisation) |
| E03 (adjacent-IPU synchrony, gender-normalised) | r = 0.15–0.50 | r² = 0.02–0.25 | heavily inflated (between-speaker and session variance included) |

**Reading.** Under conditioning on reply text and per-speaker standardisation, the plausible range for a block is of order **0.002–0.02**, with the lower end anchored by E16/E17 and the upper end an extrapolation from unconditioned sources. A G2 criterion of **block MDE ≤ 0.005** would sit inside this range; a criterion of 0.0025 would sit at its anchored lower end. **These values are not literature-derived thresholds and serve only as a sensitivity grid** (MD-19). Simulation S2 reports whether 250 sessions can meet either criterion.

---

## 11. Classification of CODA's questions

| Question | Classification | Basis |
|---|---|---|
| Partners' prosody is related across turns (the phenomenon) | **ESTABLISHED** | §1 |
| Partner prosody adds forward predictive information about the next turn beyond the replier's own history | **PARTIALLY ANSWERED** | E18 (energy, F0 range; not rate), E16/E17 (rate, global), E21 |
| **H1** — the increment survives conditioning on reply words, reply length, partner transcript and replier identity, on unseen speakers, against a matched-donor null with capacity held fixed, with a stated MDE | **POTENTIALLY NOVEL (narrow; measurement contribution)** | No study combines these (§2–4); the components are all known |
| **H2** — reproducible cross-dimension coupling structure | **OPEN, weakly motivated** | Matched-feature entrainment is documented; cross-feature structure is not, and E09/E10 give a weak or negative prior |
| **H3** — function dependence of the conditional increment | **PARTIALLY ANSWERED** correlationally (E05, E11); **POTENTIALLY NOVEL** in predictive form | §6; feasibility risks in the audit |
| **H4** — local adaptation versus dyad-level state | **PARTIALLY ANSWERED** (E03, E18, E16); registered control **under-identifies** it unless shared state is constant within an interaction | §5; S1 |
| Neural arm — whether a learned SSL representation recovers the same structure | **NOT JUSTIFIED as a scientific contribution** (the specification already says so); that SSL features encode prosody is **ESTABLISHED** (E30) | §5 of spec; E30 |

**New question, new dataset, or new combination?** CODA's primary question is a familiar one (E18 asks it without controls), tested with a new combination of established controls (lexical conditioning, per-speaker standardisation, matched-donor nulls, speaker-disjoint evaluation, a stated MDE) on a corpus not previously used for it. That is a **measurement contribution with a narrow but genuine gap**, not a new phenomenon and not a new method. H3 in predictive form is the only part that could reasonably be called a new question, and its value depends on whether it is adequately powered.

---

## 12. Corrections and additions the review requires

1. Add Fuscone et al. (2020, E18) as the closest prior study, and narrow the contribution claim: the forward-predictive version of the question has been asked; what is new is the identified, conditioned, speaker-disjoint, capacity-neutral increment with an MDE.
2. Correct the per-speaker normalisation citation (E03 uses gender z-scoring; cite E08 or the procedure itself).
3. Correct the description of Nishimura et al. (2022, E26) and of StyleTalk (E28: synthesised speech).
4. Replace "the direction is answered in conversational-TTS work" with "the direction is supported by the entrainment literature and by forward-prediction work without lexical conditioning".
5. Record that speaker-disjoint evaluation exists in adjacent tasks (E29, E33), so the novelty is its application to this target.
6. Note that H2's prior is weak (E09, E10), so a null on H2 is the expected outcome and should be written as such in advance.

Items 1–6 change framing and citations, not the design. The write-up's contribution statement follows the narrowed wording.
