# CODA research journal

What was expected, what was observed, what was learned, and what was decided — in the order it happened.

Append entries; do not rewrite history. If later evidence changes an interpretation, add a correction as a new entry rather than editing the old one. The expectation is written **before** the corresponding result is inspected, so that hindsight bias stays visible.

This journal is distinct from `docs/session_log.md`, which records what was executed, and from the decision log, which indexes choices.

---

## 2026-09-12 — Project setup, and a discrepancy in the target vector

**Question.** Before any data is touched: is the governing specification internally consistent enough to implement without guessing?

**Ran.** Repository foundation. Configuration objects and run-identifier hashing, with the first slice of the integrity suite (20 tests). Research overview, preregistration, pinned environment.

**Expected.** Setup to be mechanical. The specification had been read in full and appeared complete.

**Observed.** Two problems, neither anticipated.

First, the repository's ignore rules used unanchored directory patterns, so three of the source directories the specification names — the information firewall, target extraction, and the entire estimator and neural layer — would have been silently untracked. Found by checking each planned path against the ignore rules rather than by assuming.

Second, and more consequential: the specification states the target vector is **nine** dimensions in thirteen places, while the table that names the dimensions enumerates **eight** (2 + 1 + 2 + 3). This surfaced from a test asserting the dimension count, written for the configuration module before any prosody code existed. The predecessor document carries the identical table, marked unchanged, under the same nine-dimension heading. No document names a ninth dimension.

**Surprised by.** That a test written for configuration hashing caught a scientific specification error. The check cost nothing and found something no amount of re-reading had.

**Interpretation.** The count is the error, not the table. Two independently drafted documents enumerate the same eight dimensions; only a repeated count disagrees with them. A number carried by copy is more likely stale than a table that was rewritten and re-checked.

Adopting nine would have required inventing a prosodic dimension. Every plausible candidate is either near-collinear with a dimension already present — speaking rate is close to articulation rate times one minus pause ratio; pitch and intensity maxima are largely mean plus range — and would be removed by within-block orthogonalisation while keeping its measurement noise; or it is contour-derived, which the specification lists as optional and outside the required target. Either way it would raise a block's detectable-effect floor, working against the calibration gate that decides whether this study can answer its question at all, while buying nothing at block level where inference actually happens.

**Decided.** Eight dimensions in four blocks, with the correction recorded in the preregistration before any confirmatory result. Rejected: a turn-final energy slope, which is the only ninth defensible on its merits, and a turn-initial pitch reset, both on the contour-derived and detectability grounds above. Would reconsider only if a source naming a ninth dimension surfaced.

Also decided: Praat via parselmouth for F0 and intensity, because the specification cites Levitan and Hirschberg for both the feature set and the normalisation and that work is Praat-based. Matching the measurement tool keeps the comparison with the entrainment literature honest; a different tracker would make the feature set nominally the same and numerically different.

**Open questions.** Whether the corpus ships word-level timestamps, which the literature survey flags as unresolved and which prosody extraction depends on. Whether prosody extraction is affordable at the planned sample on the available compute; this is estimated, not measured, and the pilot measures it.

**Provenance.** `src/config.py`, `tests/test_config.py`, `PREREGISTRATION.md` §4.
