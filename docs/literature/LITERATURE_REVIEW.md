# Literature review for the implementation methods

**Scope and honesty about coverage.** This review supports the *methods* the implementation had to choose where the governing documents leave a detail open. It is not a review of the scientific question; that is `CODA_literature_survey.md`, whose framing and citations are not repeated here. It is not exhaustive.

**Search procedure.** Date: 2026-10-15. Sources: publisher pages, proceedings archives (ISCA, ACL Anthology, NeurIPS), DOI records, RePEc/IDEAS, PubMed Central, and official software documentation (Praat manual, statsmodels). Inclusion: the original method paper or an authoritative evaluation or critique of it, plus official documentation for software behaviour. Exclusion: blog posts and forum answers as evidence. **Reading depth:** for every source, the bibliographic record and the abstract or a methods summary were read, and full texts were not read end to end. Where a decision would need more than an abstract supports, the decision register says so.

Evidence types are distinguished: **T** theoretical justification, **E** empirical evaluation, **C** convention, **D** implementation documentation.

---

## 1. Measurement of the eight dimensions

| Source | Contribution (type) | Bearing on CODA | Limitation |
|---|---|---|---|
| Levitan & Hirschberg (2011) | Feature families (pitch, intensity, rate, voice quality) and per-speaker z-score norming for entrainment; turn-level versus conversation-level entrainment (E, C) | The registered target families and norming are cited to this work (§5, §8.1). Supports per-speaker standardisation (MD-05 context). | Their measures are similarity indices with both turns in hand; CODA's forward-prediction use is different (survey §3.3). Exact operational definitions (frame sets, averaging) not checked in full text. |
| Boersma (1993); Praat manual | Autocorrelation F0 tracking underlying Praat (T, D) | The pinned pitch instrument. | — |
| Praat manual, *Intensity: Get mean* | "energy", "sones" and "dB" averaging; energy averages power before converting to dB (D) | Default for mean intensity (MD-04): Praat's own default, so a Praat-based feature set means the same thing here. | The dB alternative is what many scripts use. |
| de Jong & Wempe (2009) | Syllable-nucleus detection from intensity peaks and voicing, validated against human counts on Dutch (E) | An alternative syllable counter for articulation rate (MD-03) that needs no transcript and could run on the stored tracks. | Validated on Dutch read and spontaneous speech, not this corpus; adds a detection threshold. |

**Speaker-adaptive pitch range (MD-02).** The audit notebook mentions a two-pass floor/ceiling method and speaker-sex-specific ranges. A search for the two-pass formulas (Hirst; De Looze) did **not** return a verifiable primary source, so no formula is cited or implemented. The fixed range follows `PREREGISTRATION.md` §4 ("pitch floor, ceiling … fixed in the tracked feature configuration"). The octave profile in `results/audit/` (2.3% of voiced frames more than 0.75 octave below and 3.7% above the channel median) is the evidence the choice rests on.

## 2. Splits and leakage

| Source | Contribution | Bearing | Limitation |
|---|---|---|---|
| Roberts et al. (2017) | Random CV under dependence underestimates predictive error; blocked CV recommended whenever dependence exists (E, T) | Inner CV folds are whole sessions; the outer split is by session (MD-08). | Ecological examples; the principle transfers, the magnitudes do not. |
| Kapoor & Narayanan (2023) | Taxonomy of eight leakage types across ML-based science (E) | The leakage register and its tests cover the types relevant here: duplicates across splits, preprocessing on test data, temporal leakage, test-set selection. | Taxonomy, not a method. |
| Varma & Simon (2006) | CV error after tuning on the same folds is optimistically biased; nested CV corrects it (E) | λ and rank are selected inside training; the test split is read once per condition. | Classification setting. |
| Bates, Hastie & Tibshirani (2024) | CV estimates average error over training sets, not the error of the fitted model; naive CV intervals undercover (T, E) | CODA's held-out Δ comes from a single fit on training and one read of test, with uncertainty from a session bootstrap of the test set; CV is used only for selection. | — |

## 3. Estimators

| Source | Contribution | Bearing | Limitation |
|---|---|---|---|
| Hoerl & Kennard (1970) | Ridge regression (T) | Layer-1 estimator (§9.1). | — |
| Izenman (1975) | Reduced-rank regression for the multivariate linear model (T) | The coupling map's reduced-rank structure (§9.1, H2). | — |
| Mukherjee & Zhu (2011) | Reduced-rank ridge: ridge penalty plus rank constraint, solved by projecting the ridge fit (T, E) | RRR here applies the same projection to the partner block after profiling out Z. | The restriction to one coefficient block is a derivation made for this study, checked numerically (rank 8 equals ridge), not taken from the paper. |
| Hastie, Tibshirani & Friedman, *Elements of Statistical Learning* §3.4.1 | Effective degrees of freedom of ridge as the trace of the hat matrix (T) | Reported df (MD-09). Not re-verified online; the formula is checked numerically by a test. | — |
| Kessy, Lewin & Strimmer (2018) | Whitening is not unique; ZCA-cor keeps whitened variables closest to the originals (T) | Within-block orthogonalisation (MD-06), so each output keeps the name of its dimension. | Optimality is with respect to cross-correlation; other criteria favour other whitenings. |

## 4. Nulls, p-values and multiplicity

| Source | Contribution | Bearing | Limitation |
|---|---|---|---|
| Candès, Fan, Janson & Lv (2018) | Model-X framework; conditional randomisation test valid when X's distribution given the other covariates is known (T) | The registered donor null is a matched-donor approximation of a conditional randomisation test (§6 says so). | Exact validity needs the conditional law of A given Z; matching on duration decile and word count only approximates it. That is why G1 is an empirical calibration, not an assumption. |
| Berrett, Wang, Barber & Samworth (2020) | Conditional permutation test; Type I error inflation no worse than the CRT when the conditional law is estimated (T) | Supports refitting under conditionally matched replacements; frames the N1 matching as the key approximation. | — |
| Nasir et al. (2018) | Real versus shuffled-turn sessions to validate entrainment measures (E) | The device the N1 control inherits (§5). | Used there to validate a similarity measure, not as a null for a predictive increment. |
| Phipson & Smyth (2010) | Monte-Carlo permutation p-values understate p by about 1/m and should not be reported as zero (T) | The registered p = share of draws ≥ observed is computed as written; p = 0 is reported as "< 1/B". | The registered formula differs from the (b+1)/(m+1) correction; the difference is stated rather than changed. |
| Holm (1979) | Step-down Bonferroni controlling FWER (T) | H2-H4 family (§12.5). | — |

## 5. Uncertainty and detectability

| Source | Contribution | Bearing | Limitation |
|---|---|---|---|
| Field & Welsh (2007) | Cluster bootstrap is consistent under both random-effect and transformation models of clustering (T) | Session-level bootstrap for all intervals (§12.6). | Full text restricted; abstract only. |
| Cameron, Gelbach & Miller (2008) | With few clusters, cluster-robust inference over-rejects; bootstrap-t refinements help (E) | About 50 test sessions is moderate. The self-test measured 95% percentile coverage of 0.92 at 50 clusters (within Monte-Carlo error, slightly low); a bootstrap-t variant is the documented alternative if coverage proves short. | Their setting is regression coefficients with 5-30 clusters. |
| Bloom (1995) | MDE: smallest true effect with a stated chance of a significant estimate at a stated level (T, C) | The MDE definition used in J3 (MD-18 power level, conventionally 0.80). | — |

## 6. Mixed models and labels

| Source | Contribution | Bearing | Limitation |
|---|---|---|---|
| Barr, Levy, Scheepers & Tily (2013) | Confirmatory LMMs should use the maximal random-effects structure the design justifies (E, T) | Read as an argument against simplifying the registered structure when convergence fails (Prereg §8). Random slopes are not added because the registered formula does not include them. | The registered formula has random intercepts only; this source would argue for slopes, a tension recorded in the decision register (MD-33). |
| statsmodels documentation | MixedLM groups are nested; crossed effects via variance components with one group (D) | Speakers are nested in sessions here (one session per speaker), so groups = session with a speaker variance component is the registered structure. | Robustness for crossed designs is weaker than lme4. |
| Cohen (1960) | κ (T) | G4 (§13). | κ depends on prevalence; per-category one-vs-rest κ is the registered form. |

## 7. Self-supervised representations and probes

| Source | Contribution | Bearing | Limitation |
|---|---|---|---|
| Chen et al. (2022), WavLM | The WavLM model family (E) | Sources L3/6/9/12 (§8.2). | Checkpoint (Base or Base+) is MD-27, open. |
| Hsu et al. (2021), HuBERT | HuBERT (E) | The fifth source. | Journal record not confirmed; arXiv cited. |
| Yang et al. (2021), SUPERB | Frozen upstream with a learned weighting of layers as a standard evaluation protocol (E, C) | Precedent for a learned layer mixture rather than layer selection (§8.2). | Weighted-sum details known from a secondary summary only. |
| Pasad, Chou & Livescu (2021) | Information changes non-monotonically across SSL layers (E) | Supports mixing several depths rather than choosing one. | wav2vec 2.0, not WavLM/HuBERT. |
| Reimers & Gurevych (2019); Wang et al. (2020) | Sentence-BERT training and MiniLM distillation (E) | The frozen text encoder (§8.3). | — |
| Hewitt & Liang (2019) | Probe selectivity via control tasks (E) | Recorded as a known limitation of linear probes. Control tasks are **not** added: they are not registered (§18). | — |
| Jain & Wallace (2019) | Attention weights are not reliable explanations (E) | Why attention weights are never interpreted (§12.1). | Text classification setting. |

---

## Decision-level synthesis

Each decision, its supporting and cautionary evidence, and its remaining uncertainty are in `METHOD_DECISION_REGISTER.md`. The central conclusions:

1. **The donor null is an approximation whose validity must be measured, not assumed.** CRT theory (Candès et al. 2018; Berrett et al. 2020) makes the null exact only under the true conditional law of A given Z. N1 matches two variables. G1's β = 0 calibration is therefore the evidence that the null holds, which is why G1 is on the never-drop list and why its acceptance rule (MD-17) must be fixed before it is run.
2. **Every resampling and validation unit is the session.** Supported by blocked-CV evidence (Roberts et al. 2017), cluster-bootstrap theory (Field & Welsh 2007) and the leakage taxonomy (Kapoor & Narayanan 2023). The bootstrap self-test demonstrates the cost of getting this wrong: a pair-level bootstrap understated the SE about sevenfold.
3. **Small-cluster coverage is the main statistical limitation.** With about 50 test sessions, the percentile bootstrap measured 0.92 coverage at a nominal 0.95 (Cameron, Gelbach & Miller 2008 describe the same direction of error). Intervals are reported with this caveat; the alternative (bootstrap-t) is documented, not silently substituted.
4. **G2 cannot be evaluated from the literature reviewed here.** None of the sources gives a "plausible effect range" in block partial R² for conditional, speaker-disjoint, forward prediction; the entrainment literature reports similarity indices, not this quantity (survey §9, items 9-10). G2 is therefore a reporting requirement (MD-19).
