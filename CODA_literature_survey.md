# Literature survey

## Partner delivery and reply delivery in spontaneous conversation: what is established, what is inferred, and what remains unmeasured

---

# 1. Research problem and motivation

## 1.1 The question

This survey positions a foundational empirical study, **Conversational Delivery Adaptation (CODA)**, whose scientific question is:

> In spontaneous dyadic conversation, and conditional on the reply's own words and the partner's transcript, is the reply's prosodic delivery predictively related to the partner's delivery — for speakers never seen in training — and does the strength or structure of that relationship depend on the conversational function of the turn pair?

Before any shorthand is used, the components of that question are worth stating in ordinary research language.

**What is being predicted.** The *delivery* of a conversational reply: how it was spoken rather than what was said. Delivery is described here by eight interpretable prosodic dimensions organised into four families — pitch level and span (mean log-F0, log-F0 range), pitch boundary behaviour (log-F0 slope over the final 300 ms), energy (mean intensity, intensity range), and timing (articulation rate, pause ratio, longest internal pause). These are continuous descriptors of a single turn, not categories, and not a learned embedding.

**Whose delivery is the target.** The person who replies. The prediction target is that person's delivery on the turn immediately following the partner's turn.

**Whose delivery is the contextual predictor.** The conversational partner, on the immediately preceding turn.

**Why the reply's own words matter.** How a sentence is delivered is substantially constrained by what the sentence is. Long sentences are spoken differently from short ones; questions are spoken differently from assertions; emphatic content invites emphatic delivery. If the reply's own lexical content is not held fixed, any apparent relationship between partner delivery and reply delivery may simply reflect the fact that both are correlated with what the conversation is about. The reply's text embedding, word count and syllable count are therefore part of the conditioning set, not part of the effect being measured.

**Why the partner's transcript is conditioned on.** The question is not whether the partner's turn is informative — it plainly is — but whether the *acoustic form* of that turn adds predictive information beyond its *words*. A system that has already read the partner's transcript knows the topic, the speech act, and much of the affective content. The scientific quantity of interest is what remains once that is accounted for.

**What "predictively related" means here.** It means an out-of-sample increase in predictive accuracy, estimated on held-out data and referenced against a null distribution built by replacing the partner's turn with a conditionally matched donor turn. It is a **conditional predictive dependence**, not a causal claim. Nothing in this design licenses the statement that a partner's delivery *causes* a change in the reply's delivery; two speakers in the same conversation share topic, affect, register and physical setting, and any of these can produce dependence without adaptation.

**Why unseen-speaker generalisation matters.** Individual speakers have stable prosodic habits and stable relationships with their own conversational partners. A model evaluated on speakers it has already seen can exploit those habits. Evaluating on speakers absent from training is what separates a claim about *conversational behaviour* from a claim about *particular people*.

**Why conversational function matters.** A question invites an answer; an assessment invites agreement or disagreement; a narrative turn invites continuation. If the relationship between partner delivery and reply delivery is the same in all of these, then delivery behaviour can be treated as a single global mapping. If it differs by function — either in magnitude or in which dimensions are involved — then delivery is situationally structured, which is a different and stronger empirical statement with different engineering consequences.

## 1.2 Why this is not already answered

Three substantial literatures bear on this question, and each answers a neighbouring version of it:

- **Conversational speech synthesis** has established that conditioning generated speech on the *acoustic* form of the dialogue history, not only its transcript, improves listener-rated output.
- **Acoustic-prosodic entrainment research** has established that conversational partners' prosody becomes measurably related over the course of an interaction, and has developed the standard measurement and control apparatus for studying this.
- **Speech-conditioned conversational modelling** has established that models given the partner's speech predict properties of the response — its style, its sentiment, its timing — better than models given only text.

What none of these has produced is an *isolated, quantified predictive increment*: a number, with an interval and a stated detectability floor, describing how much of a reply's interpretable prosodic delivery is predictable from the partner's delivery once the reply's own content and the partner's transcript are accounted for, under matched model capacity, with a control that destroys the original context–reply pairing while preserving everything else, evaluated on unseen speakers, and broken down by conversational function.

The remainder of this survey establishes that claim by examining each literature in turn.

---

# 2. What conversational speech research already knows

## 2.1 Dialogue context improves generated delivery — first with text, then with audio

The line begins with text-only dialogue context. Guo et al. (SLT 2021) built a conversational end-to-end text-to-speech system for voice agents in which a context encoder consumed BERT embeddings of the preceding chat history together with speaker identity, and showed that conditioning on this history improved the prosody of the generated response relative to a system without it.

The acoustic extension followed quickly and is now standard practice rather than a frontier claim:

- **Nishimura, Saito, Takamichi, Tachibana and Saruwatari (Interspeech 2022)** proposed acoustic modelling for end-to-end empathetic dialogue speech synthesis using **both linguistic and prosodic contexts of the dialogue history**, explicitly framed as an extension of conventional linguistic-feature-based history modelling. Their method includes a style-guided training component in which a prosody embedding of the current utterance is predicted from the context embedding. Of all the work surveyed here, this is the closest in *direction* to the present question: it predicts a representation of the upcoming utterance's delivery from a multimodal representation of what came before.
- **Xue et al., M2-CTTS (ICASSP 2023)** built an end-to-end conversational TTS system with a **textual context module and an acoustic context module** operating at coarse and fine granularity, on the explicit argument that textual features alone are insufficient because acoustic features carry prosodic information the transcript does not.
- **Li et al. (2022)** inferred speaking styles for a response from multimodal conversational context, using graph-based multi-scale context modelling; the target there is a learned style representation rather than interpretable prosodic scalars.
- **Xin et al. (ICASSP 2023)**, working on audiobook rather than dialogue speech, compared an acoustic context encoder against a combined acoustic-and-textual context encoder for predicting the prosody of the next sentence, and additionally swept the amount of textual context, reporting that a small window of preceding sentences was most useful.
- **Deng et al., CONCSS**, is notable for a different reason: its motivation is that a context encoder trained only through the synthesis reconstruction objective may not produce a latent that is genuinely indicative of context variation, and it introduces a contrastive pretext task to make the context representation more discriminative. This is an admission from inside the field that end-to-end training does not by itself demonstrate that a context representation carries context information.
- A broader family of systems — including emotion-aware conversational synthesis systems, large-language-model-coupled conversational speech generators, and emphasis-oriented systems built on annotated derivatives of dialogue corpora — extends the same architecture pattern with richer context encoders and additional annotation. These typically report ablation tables in which removing an acoustic context component degrades perceived quality.

**Established by this literature.** Acoustic dialogue context is useful for generating appropriate reply delivery, across multiple architectures, corpora and languages. The field does not treat "does acoustic context help?" as an open question.

**Synthesis, not a claim made by any single paper.** These results are *system-level*. Each compares a model that has an acoustic context pathway against a model that does not, where the two models differ simultaneously in input modality, module structure, parameter count, optimisation dynamics, and — because the read-out is synthesised speech — in what the decoder does downstream. The comparisons demonstrate usefulness convincingly. They are not designed to, and do not, isolate the size of the acoustic channel's contribution.

**Open for the present study.** How large is that contribution when measured directly on the reply's delivery, under matched capacity, without a synthesiser in the read-out path?

## 2.2 What the read-out is, and why it matters

Evaluation in this literature is predominantly mean-opinion-score or preference testing on synthesised audio, sometimes supplemented by objective distances between predicted and reference acoustic features. Both read-outs are appropriate for the question those papers ask, which is whether the generated speech is better.

Neither read-out yields a quantity that is comparable across studies, decomposable by prosodic dimension, or interpretable as an effect size. A preference gain of a given magnitude between two systems does not translate into a statement about how much delivery variance the acoustic channel explains, because the decoder sits between the context representation and the measurement.

---

# 3. What entrainment and accommodation research already knows

## 3.1 The phenomenon and its standard measurement

Conversational partners become more similar to each other over the course of an interaction. This has been documented at many linguistic levels and, most extensively for quantitative work, at the prosodic level; the review *Measuring Prosodic Entrainment in Conversation: A Review and Comparison of Different Methods* (Journal of Speech, Language, and Hearing Research, 2023) surveys the terminological variety — entrainment, accommodation, alignment, convergence, synchrony, imitation — and notes that the choice of term often reflects theoretical perspective rather than a different phenomenon.

**Levitan and Hirschberg (Interspeech 2011)** established much of the standard apparatus: a feature set covering pitch (mean, maximum, minimum), intensity (mean, maximum), speaking rate and voice quality measures, together with a **per-speaker z-score norming procedure** applied feature by feature, and a distinction between several measurement targets — proximity, convergence and synchrony — that are conceptually separate and need not co-occur. Gravano and colleagues extended measurement to categorical intonational (ToBI-based) descriptions. Reichel's *Entrainment profiles* work (Speech Communication, 2018) extended the feature space to register, pitch-accent shape and rhythmic descriptors derived from a superpositional intonation stylisation.

**Directly relevant consequence for the present study.** The interpretable eight-dimensional delivery description used here, and the per-speaker standardisation applied to it, are drawn from this tradition and are cited to it. They are not presented as design contributions. Their use is a decision to measure delivery in the terms the relevant scientific literature already uses, which makes results comparable and makes the dimension-level analysis interpretable to the audience best placed to evaluate it.

## 3.2 The sham-conversation control

An important methodological precedent concerns how entrainment is distinguished from coincidence. **Nasir et al. (Interspeech 2018)**, presenting a discriminative acoustic-prosodic approach to measuring local entrainment, operationalise the phenomenon by requiring that turn-level acoustic-prosodic feature differences be maximally different between **real conversations** and **sham conversations constructed by randomly mixing speaking turns from the original partners**; they project onto a discriminative subspace that maximises this separation, and relate the resulting entrainment measure to conversational success. That paper also credits earlier use of randomly generated dialogues assembled from partners in *different* conversations as a validation device.

**Established.** Randomised or sham pairing is the accepted way to establish that a measured partner-to-partner relationship depends on the actual pairing rather than on marginal properties of the two speakers' speech. It is a foundation of this literature, not a novel device.

**Where the present study's use differs, stated carefully.** In the entrainment tradition, the sham construction serves to define or validate a *similarity measure*: the question is whether feature differences discriminate real from sham. In the measurement framework considered here, matched-donor replacement serves a different inferential job: it generates a **null distribution for an out-of-sample predictive increment**, so that input dimensionality, feature scale, marginal distributions, optimisation difficulty and model capacity are all held inside the null rather than being subtracted afterwards. The device is inherited; the quantity it calibrates is different. Two donor schemes are used — one drawing from a different dyad in the same split, matched on partner-turn duration decile and word count, and one drawing from the *same* dyad and *same* partner speaker at a distance of at least ten turns.

## 3.3 What entrainment measurement can and cannot answer here

**Can answer.** Whether two conversational partners' prosody is related; in which features; how that relationship evolves; how it relates to task success, perceived trust, rapport, and clinical populations (the disordered-speech entrainment work of Borrie and colleagues shows reduced entrainment in dialogues involving speakers with communication disorders, with a positive relationship between entrainment indices and communicative success).

**Cannot answer, and this is the central limitation for the present question.** Entrainment measures are typically computed with both turns already in hand and describe *similarity* or *co-variation*. They are not forward predictions made under the information constraint that a system generating a reply would face, and they do not condition on the reply's own lexical content. A proximity measure between adjacent turns will register a relationship that is entirely attributable to both speakers discussing the same thing in the same register; conditioning on the reply's words and the partner's words is what separates that from a delivery-specific relationship.

**Statistical framing.** The entrainment literature standardly uses correlation-based indices and linear mixed-effects models with speaker and dyad random effects. That is a stronger and more efficient treatment of the clustering problem than much of the machine-learning literature applies, and the present study's confirmatory re-check with a mixed-effects model including speaker and dyad random effects follows that practice rather than departing from it.

---

# 4. Predicting responses from partner acoustic behaviour

A separate line has asked directly whether interlocutor acoustics predict *something about what comes next*. The targets differ from reply prosody, but the prediction direction is the same.

**Language modelling.** Ward and Walker (Interspeech 2009) estimated the potential of signal-track and interlocutor-track information for language modelling. Ward, Vega and Baumann (Speech Communication, 2011) showed that local prosodic and temporal features — recent speaking rate, volume, pitch, time into the utterance — reduced next-word perplexity by approximately 8.4% on conversational telephone speech, with a small relative word-error-rate improvement. This is a genuine *incremental information from acoustics* result under forward prediction, established more than a decade ago, for a lexical target.

**Turn-taking.** The voice-activity-projection line (Ekstedt and Skantze) predicts future voice activity from partner audio, and has been applied to the automatic evaluation of turn-taking cues in synthesised conversational speech. The target is timing of the floor rather than the delivery of the reply's content, but it demonstrates that partner acoustics support forward prediction of interactional behaviour and provides an existing objective evaluator for a related property.

**Response style and sentiment.** ParalinGPT conditions a language model on self-supervised speech embeddings of the dialogue history, and predicts, in a serialised chain, the current turn's sentiment, then the **response's** sentiment, then the response text, on conversational telephone speech, with text-only baselines for comparison. Spoken-LLM / StyleTalk (Lin, Chiang and Lee, ACL 2024) is built on precisely the premise underlying the present question: that two current utterances with the same content but different speaking styles warrant different responses. It predicts the **response speaking style** from the partner's speech and reports gains over text-only language-model baselines. Style-Talker (COLM 2024) feeds both the transcript and a style embedding of preceding turns into a dialogue model, keeping the prosodic and semantic aspects of the context separate.

**Established.** Partner acoustic information supports forward prediction of several response properties beyond what the transcript supports: next words, upcoming voice activity, response sentiment, and categorical response speaking style.

**What these results do not establish for the present question.** The targets are lexical, temporal, categorical or learned-embedding-valued. None is a continuous, interpretable, speaker-normalised description of the reply's prosody, and so none supports a dimension-level statement about which aspects of delivery are involved. In addition, StyleTalk's evidence comes from a constructed dataset assembled for the purpose rather than from naturally occurring conversation, which bears on how far the result generalises to spontaneous speech; this is a property of the resource, noted rather than a criticism of the work.

---

# 5. Acoustic versus textual/contextual information

Two literatures have made the acoustic-versus-textual comparison directly.

**In synthesis.** The with-and-without-acoustic-context ablations described in §2 are, structurally, comparisons of {reply text + history text} against {reply text + history text + history audio}. It is worth being explicit about a point that is easy to miss: in any text-to-speech setting the target text is an input by construction. Conditioning on the reply's own content is therefore *automatic* in this literature and is not a distinguishing feature of any design. Xin et al.'s acoustic-context-encoder versus acoustic-plus-textual-context-encoder comparison is the cleanest explicit instance, though it is set in read audiobook rather than conversational speech.

**In entrainment.** Work in this tradition has compared text-derived against speech-derived feature representations by their ability to discriminate real from sham conversations, reporting advantages for acoustic-based representations. The comparison is between feature families on a *discriminative* task, not an incremental predictive contribution on a forward task.

**Established.** Acoustic and textual context have been compared, in two literatures, under two read-outs.

**Synthesis.** Neither comparison yields a conditional predictive increment. In the synthesis case, the two arms differ in capacity as well as in information, and the read-out passes through a decoder. In the entrainment case, the task is discrimination of real from sham rather than prediction of the reply, and the reply's own content is not conditioned on.

---

# 6. The measurement problem

This section states the central methodological theme of the survey.

A comparison of the form

> system with textual context < system with textual and acoustic context

is strong evidence that acoustic context is **useful**. It is weaker evidence about **how much information the acoustic channel contains**, for reasons that are structural rather than a matter of care or rigour in any particular paper:

1. **Modality and capacity move together.** Adding an acoustic context module adds parameters. A model with more parameters and a wider input may perform differently for reasons unrelated to the information carried by the new input — in either direction, since a wide, weakly informative input can also hurt under a fixed regularisation regime. None of the conversational-synthesis ablations surveyed here reports a capacity-matched comparison.
2. **Optimisation moves too.** Different input widths and module structures induce different optimisation problems. "The information is absent" and "the optimiser could not extract it under this training budget" are not separable in an end-to-end comparison.
3. **The read-out is downstream of a decoder.** When the measured quantity is listener preference over synthesised speech, the decoder can absorb, amplify or mask differences in the context representation. CONCSS's own motivation — that a context latent trained through a reconstruction objective may not be sufficiently indicative of context variation — is an argument from inside the field that system performance is an indirect indicator of what the context representation encodes.
4. **The result is not an effect size.** A preference score difference cannot be compared across corpora, decomposed across prosodic dimensions, or used to decide whether the channel is worth the engineering cost.

**The appropriate framing, and the one adopted here.** Prior work provides strong evidence that acoustic conversational context is useful. The remaining question is whether its contribution can be isolated as a controlled predictive quantity, rather than inferred indirectly from end-to-end system improvement.

**What an identified measurement requires, derived from the above.** Capacity held neutral across the compared conditions; a null distribution that preserves every nuisance property of the acoustic input and destroys only its correspondence with the reply; conditioning on the reply's own content and the partner's transcript; a read-out taken directly on interpretable properties of the reply rather than through a generative model; evaluation on unseen speakers; and a stated minimum detectable effect, so that a small estimate can be read as a bound rather than as an absence.

---

# 7. Conversational situation and function

## 7.1 What has been shown

Evidence that conversational prosodic coordination is not uniform across the conversation already exists, in the entrainment literature and in correlational form.

- **Reichel, Mády and Cole (2018), *Prosodic entrainment in dialog acts*.** Entrainment was examined separately for several dialogue acts in cooperative and competitive game dialogues, using intonation features from a superpositional stylisation and rhythm features. Differences were related both to the cooperative or competitive nature of the game and to properties of the dialogue act — its intrinsic authority, its supportiveness, and its distributional characteristics. In cooperative games, dialogue acts with high authority arising from knowledge, and with high frequency, showed the most entrainment.
- **Gauder, Reartes, Gálvez, Beňuš and Gravano (Speech Prosody 2018)** implemented entrainment and disentrainment policies **at the dialogue-act level** in a spoken dialogue system and reported a pilot study, motivated by the observation that dialogue acts correlate with acoustic-prosodic features and that entraining or disentraining across acts may introduce misleading artefacts.
- Subsequent work citing both has treated stronger entrainment at the dialogue-act level as an indication that entrainment is not a purely automatic process but is at least partly actively controlled.
- Relatedly, dialogue-act-level coordination has been documented at the level of the acts themselves, and the 2023 methods review notes entrainment across linguistic levels including dialogue acts.

**Established.** Prosodic coordination between conversational partners varies by dialogue act, and this variation has been observed in more than one corpus and more than one measurement framework. Conversational function is not a novel conditioning variable.

## 7.2 What has not been shown

**Synthesis.** The existing function-dependence results are (a) correlational rather than forward-predictive, (b) computed without conditioning on the reply's own lexical content, (c) reported as differences in an entrainment index rather than as differences in a predictive increment, and (d) obtained on task-oriented or game-based dialogue rather than on unstructured spontaneous conversation. In the synthesis literature, function or act information is sometimes used as an input feature, but the *dependence of the context-to-delivery relationship on function* is not itself a reported quantity.

**Open for the present study.** Whether the **conditional predictive increment** from partner delivery to reply delivery differs by conversational function, and whether the difference is one of **magnitude** (the same dimensions couple more strongly in some functions) or of **structure** (different dimensions couple in different functions). The second is the stronger empirical statement and is the one with direct consequences for whether delivery behaviour should be modelled as situationally conditioned or as a single global mapping.

**A methodological caution taken from this literature.** Reichel et al. and Gauder et al. both rely on annotated dialogue acts. Where function labels are derived automatically, their reliability has to be established before the interaction is interpreted, rather than assumed — hence the present study's requirement that automatically assigned function categories be validated against a hand-labelled sample and dropped from confirmatory analysis below a stated agreement threshold.

---

# 8. Closest prior studies

The table compares the studies nearest to the present question. "Capacity controlled?" asks whether the compared conditions were matched in trainable capacity; "Controls" asks whether any procedure destroyed the context–reply pairing while preserving nuisance structure.

| Study | Context | Acoustic input | Text input | Target | Prediction direction | Evaluation | Controls | Capacity controlled? | Function examined? | Relevance to the present question |
|---|---|---|---|---|---|---|---|---|---|---|
| Nishimura et al., Interspeech 2022 | Japanese empathetic dialogue (STUDIES) | SSL features of history speech | history transcripts | reply speech; plus a prosody embedding of the current utterance predicted from context | forward | synthesis quality; subjective | none reported | no | no | **Closest in direction.** Explicitly adds prosodic context to linguistic-only history modelling, and predicts a delivery representation from context. Does not isolate the increment or report an effect size |
| Xue et al., M2-CTTS, ICASSP 2023 | conversational corpus | wav2vec 2.0 over history audio, coarse + fine | BERT over history text | reply speech prosody | forward | MOS / ablation | none reported | no | no | The canonical acoustic-versus-textual context ablation. Establishes usefulness; modality and capacity change together |
| Li et al., 2022 | conversational corpus | multimodal context graph | multimodal context graph | reply **style embedding** | forward | synthesis quality | none reported | no | no | Same prediction arrow; learned rather than interpretable target, so no dimension-level statement |
| Xin et al., ICASSP 2023 | Japanese audiobook | acoustic context encoder | bilateral textual context encoder | next-sentence prosody | forward | preference; context-length sweep | none reported | no | no | Cleanest explicit acoustic-vs-textual context comparison, but monologue rather than dialogue |
| Deng et al., CONCSS | dialogue corpus | multimodal history | multimodal history | reply prosody | forward | synthesis quality; contrastive pretext | contrastive discriminability objective | no | no | Argues from inside the field that context latents trained end-to-end may not encode context; motivates direct measurement |
| Lin, Chiang & Lee, Spoken-LLM / StyleTalk, ACL 2024 | constructed speech-to-speech dataset | speech of the current turn via a speech encoder | dialogue text | **response speaking style** (categorical) + response text | forward | style F1; human preference | none reported | no | no | Publishes the affirmative headline under different vocabulary; constructed data, categorical target |
| ParalinGPT | Switchboard | SSL speech embeddings of history | history transcripts | current sentiment → **response sentiment** → response text | forward | classification metrics; text-only ablation | none reported | no | no | Partner acoustics predict a response property beyond text; target is a coarse label, not delivery |
| Guo et al., SLT 2021 | agent chat corpus | none (text-only context) | BERT history + speaker ID | reply prosody | forward | MOS | none reported | no | no | The text-only-context reference point the acoustic extensions were built against |
| Nasir et al., Interspeech 2018 | task-oriented dyads | turn-level acoustic-prosodic features | — | entrainment index → conversational success | not forward (both turns in hand) | discrimination of real vs **sham**; downstream success classification | **sham conversations** | not applicable | no | Source of the matched-pairing control logic; different inferential job |
| Levitan & Hirschberg, Interspeech 2011 | Columbia Games | adjacent-IPU prosodic features | — | proximity / convergence / synchrony | not forward | correlational | — | not applicable | no | Source of the feature families and per-speaker norming used as the delivery description |
| Reichel, Mády & Cole, 2018 | cooperative/competitive game dialogues | intonation + rhythm features | — | entrainment by dialogue act | not forward | correlational, by act | — | not applicable | **yes** | Best existing evidence that coordination varies by conversational function; correlational, no reply-content conditioning |
| Gauder et al., Speech Prosody 2018 | spoken dialogue system, Argentine Spanish | system-side prosody manipulation | — | user behaviour | interventional on the system side | pilot study | act-level entrainment/disentrainment policies | not applicable | **yes** | Treats dialogue act as the unit at which coordination policy should operate |
| Ward, Vega & Baumann, Speech Communication 2011 | Switchboard | local prosodic and temporal context | n-gram history | **next words** | forward | perplexity (≈8.4% reduction), WER | — | not applicable | no | Established precedent for incremental information from interlocutor-track acoustics under forward prediction; lexical target |
| Ekstedt & Skantze (VAP line) | dialogue corpora | partner audio | — | future voice activity | forward | objective | — | no | no | Forward prediction from partner acoustics for interactional timing; different target |

Two observations follow from the table and are worth stating plainly rather than being left implicit.

**First, several prior studies do things the present study does not.** They generate speech, run listening tests, cover more languages, use gold dialogue-act annotation, and in the case of the turn-taking line provide a reusable automatic evaluator. The present study produces no audio and runs no listening test.

**Second, the "Controls" and "Capacity controlled?" columns are nearly empty on the synthesis and speech-conditioned-modelling rows, and the "Function examined?" column is populated only on entrainment rows that are not forward-predictive.** No single row combines forward prediction of interpretable reply delivery, conditioning on reply content and partner transcript, a pairing-destroying control, capacity matching, unseen-speaker evaluation, and function conditioning.

---

# 9. Novelty and gap audit

Each dimension is classified as **established** (prior work has it), **measured differently** (closely related but with a different read-out or inferential target), **uncommon** (rare but present), or **unresolved** (not answered by existing evidence).

| # | Dimension | Status | Reasoning |
|---|---|---|---|
| 1 | The research question in yes/no form | **Established** | The explicit motivation of at least five conversational-synthesis papers since 2022, and the founding premise of StyleTalk |
| 2 | Partner acoustic delivery as predictive context | **Established** | Acoustic context encoders are standard; forward prediction from interlocutor acoustics dates to at least 2009 |
| 3 | Partner transcript conditioning | **Established** | Present in every multimodal dialogue-context system |
| 4 | Reply-content conditioning | **Established, and automatic** | The target text is an input by construction in any TTS setting. This is not a distinguishing design feature and should not be presented as one |
| 5 | Interpretable reply prosody as the target | **Uncommon** | The entrainment literature uses these feature families but not as a forward-prediction target; the synthesis literature predicts speech or learned style embeddings, not interpretable scalars |
| 6 | Speaker-disjoint generalisation for the predictive claim | **Uncommon** | Standard in entrainment work (which is correlational); rare in the predictive conversational-synthesis work, which is often one- or two-speaker |
| 7 | Matched-donor pairing control | **Established as a device, measured differently here** | Sham conversations define local entrainment measurement (Nasir et al. 2018 and earlier). Using matched donors to generate a null distribution for an out-of-sample predictive increment is a different inferential use of an inherited device |
| 8 | Acoustic-versus-textual decomposition | **Established** | Present in synthesis ablations and in text-versus-acoustic entrainment model comparisons |
| 9 | Effect-size measurement of the acoustic contribution | **Unresolved** | No surveyed study reports the contribution as a quantity with an interval on interpretable delivery dimensions |
| 10 | Stated precision / detectability floor | **Unresolved** | No surveyed study reports a minimum detectable effect, so small or null results in this area cannot currently be read as bounds |
| 11 | Structured partner-dimension → reply-dimension coupling | **Unresolved as a predictive map** | Entrainment work reports per-feature convergence between partners; synthesis work reports aggregate gains. A predictive dimension-to-dimension map with stability testing is not reported |
| 12 | Conversational-function dependence | **Established correlationally; unresolved predictively** | Reichel et al. (2018) and Gauder et al. (2018) establish act-dependence of coordination. Function-dependence of a conditional predictive increment, with reply content conditioned, is not reported |
| 13 | Local versus dyad/session-level explanation | **Unresolved** | Sham controls in the entrainment literature typically draw from different conversations, which destroys dyad and topic along with adjacency. A donor drawn from the same dyad and same speaker at distance, which isolates adjacency alone, is not standard |
| 14 | Learned speech representations for this prediction | **Established** | SSL speech representations are used throughout the synthesis and speech-conditioned-modelling literatures |
| 15 | The combination as a measurement framework | **Uncommon rather than novel** | Every component exists somewhere. The assembly — forward prediction of interpretable delivery, reply content conditioned, capacity neutral, matched-donor null, unseen speakers, function-stratified — does not appear in any surveyed paper |

**What is actually new about the scientific evidence, stated without inflation.** Not the phenomenon; not the question; not the control device; not the feature set; not the use of self-supervised speech representations. What would be new is **the quantity**: an identified, capacity-neutral, interval-bearing estimate of the partner-delivery contribution to interpretable reply delivery, on unseen speakers, accompanied by the smallest effect the procedure could have detected, and reported separately by prosodic dimension and by conversational function. Items 9, 10, 11, 12 (in its predictive form) and 13 are the parts of the table that are genuinely unresolved, and they are the parts the present study is designed to address.

---

# 10. Defensible contribution

The narrowest formulation the literature supports:

> Existing conversational speech research establishes that acoustic dialogue context improves generated response delivery, and entrainment research establishes that conversational partners' prosody is measurably related and that this relationship varies by dialogue act. Both literatures evaluate the phenomenon through system-level improvement, listener preference, or correlational similarity indices. What has not been established is the size of the partner's delivery contribution to the reply's interpretable prosodic delivery, conditional on the reply's own words and the partner's transcript, under matched capacity, with a control that destroys the context–reply pairing while preserving its nuisance structure, on speakers absent from training — nor whether that contribution varies by conversational function.

Three specific claims follow, and no more:

1. **An identified, capacity-neutral effect size** for a conditioning channel the field already deploys, where existing evidence is system-level and confounded.
2. **Speaker-generalising evidence for the predictive claim**, which the entrainment literature supplies only correlationally and the synthesis literature typically supplies on one or two speakers.
3. **A function-conditional characterisation of the predictive relationship**, extending correlational act-dependence findings into a predictive frame.

Explicitly **not** claimed: discovery of the phenomenon; a novel method; a novel architecture; novelty of the sham/matched-pairing control; novelty of the prosodic feature families or of per-speaker normalisation. The last two are cited to Levitan and Hirschberg (2011) and the first to Nasir et al. (2018) and its predecessors.

Contribution class, stated honestly: **a measurement study that converts an assumption the field already acts on into a quantity with an interval, together with the first function-conditional version of that quantity.** Modest, defensible, and appropriate as a foundational empirical study.

---

# 11. Threats to novelty

The three papers most likely to prompt a reviewer to say the work has already been done, and the honest response to each.

**Nishimura et al. (Interspeech 2022).** The strongest threat, because it explicitly extends linguistic-only dialogue-history conditioning with prosodic context and includes a style-guided objective predicting a prosody embedding of the current utterance from the context embedding. The direction, the nesting, and part of the target are shared. *Response:* the evaluation is synthesis quality with no capacity matching and no pairing control, the target is a learned prosody embedding rather than interpretable dimensions, and no contribution size is reported. The work establishes usefulness; it does not isolate the increment. This paper must be read in full and cited prominently.

**Xue et al., M2-CTTS (ICASSP 2023).** The paper a reviewer is most likely to name, because its textual-versus-acoustic module ablation is structurally the comparison at issue. *Response:* the same capacity and read-out arguments apply, and the ablation removes a module rather than replacing its input under matched capacity. The distinction is real but narrow, and it must be stated precisely rather than overstated.

**Lin, Chiang and Lee, Spoken-LLM / StyleTalk (ACL 2024).** Publishes an affirmative version of the headline under different vocabulary — response *speaking style* predicted from partner speech, beating text-only baselines. *Response:* the evidence comes from a constructed dataset, the target is categorical style rather than continuous delivery, and the evaluation is style classification and human preference rather than an identified increment. This makes the present study complementary rather than redundant, which is a narrowing, not a rebuttal.

**A fourth threat of a different kind: Nasir et al. (Interspeech 2018).** It does not threaten the research question but it owns the control device. Failing to cite it would be a substantive error, and citing it strengthens the design by anchoring the control in accepted method.

---

# 12. Final literature verdict

The direction of the question is answered; its magnitude, its structure, and its dependence on conversational function are not.

Three literatures converge on the present problem and each stops short of it. Conversational speech synthesis has shown that acoustic dialogue context improves generated delivery, using end-to-end comparisons in which information, capacity and optimisation change together and the read-out passes through a decoder. Entrainment research has established that partners' prosody is related, has supplied the feature families, the per-speaker normalisation and the sham-pairing control that this kind of work requires, and has shown correlationally that coordination differs by dialogue act — but measures similarity with both turns in hand rather than forward predictability with the reply's own content held fixed. Speech-conditioned conversational modelling has shown that partner acoustics predict response properties beyond text, for lexical, temporal, sentiment and categorical-style targets.

What remains open is a measurement question rather than a discovery question: whether the partner's delivery contributes a quantitatively identifiable predictive increment to the reply's interpretable prosodic delivery, beyond the textual and contextual information already available, under matched capacity and a control that destroys only the original context–reply pairing; how that increment is distributed across prosodic dimensions; whether it survives evaluation on unseen speakers; whether it survives replacement of the adjacent partner turn by a distant turn from the same speaker in the same conversation; and whether it depends on the conversational function of the turn pair.

The position of the present study in this literature is therefore **incremental and defensible rather than novel**: the core idea exists, and the contribution is a controlled measurement and a structured read-out of a relationship the field already assumes. Stated as a discovery, the work would be dismissed by anyone who knows the area. Stated as the identified version of a question the field has answered indirectly — with the effect size, the detectability floor, the dimension-level structure and the function-dependence that indirect evidence cannot supply — it is a narrow empirical contribution that is worth testing and whose null result would be as informative as its positive one.

---

---

# 13. Note on corpus style

Much of the conversational-synthesis literature against which this study positions itself works on read or acted dialogue, whereas the coordination phenomenon is documented mainly on spontaneous dyads. Adaptation is generally expected to be weaker in read speech, so results obtained on spontaneous conversation and results obtained on read corpora are not directly comparable in magnitude. Magnitude comparisons across corpora are therefore avoided in the write-up.
