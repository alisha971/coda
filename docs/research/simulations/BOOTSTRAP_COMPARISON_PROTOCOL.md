# Bootstrap interval comparison — protocol (decision MD-11)

**Written 2026-10-16, before any comparison run.** Synthetic data only.

## Question

Which session-bootstrap interval for the held-out aggregate block partial R² comes closest to nominal 95% coverage with the number of test sessions this study will have: percentile, BCa, or cluster bootstrap-t?

## Methods compared

All three use the same resamples of whole test sessions, with models not refitted (`src/evaluation/bootstrap.py`).

| Method | Interval |
|---|---|
| Percentile | 2.5th and 97.5th percentiles of the bootstrap statistic |
| BCa (Efron 1987) | Percentiles adjusted by the bias correction z₀ (share of bootstrap statistics below the estimate) and the acceleration a, from the leave-one-session-out jackknife |
| Cluster bootstrap-t (studentized) | T − q₀.₉₇₅·SE and T − q₀.₀₂₅·SE. Here q is a quantile of (T* − T)/SE*, and SE and SE* are leave-one-session jackknife standard errors in the sample and in each resample (Cameron, Gelbach & Miller 2008) |

## Generator

Per-pair squared errors for a reduced and a full model, eight dimensions in blocks [2, 1, 2, 3]. Errors are gamma with a session scale factor. The full model removes a fraction `frac` of the error, and that fraction varies across sessions with SD `sd_sess`. The truth is the aggregate partial R² on a pool of 20,000 / pairs-per-session × 50 sessions. This generator is extended with unequal session sizes.

## Scenarios (fixed before the run)

| ID | Test sessions | Pairs per session | Mean `frac` | `sd_sess` |
|---|---|---|---|---|
| B1 | 50 | 300 | 0.01 | 0.02 |
| B2 | 50 | 300 | 0.01 | 0.05 |
| B3 | 25 | 300 | 0.01 | 0.02 |
| B4 | 25 | 300 | 0.01 | 0.05 |
| B5 | 50 | 300 | 0.00 | 0.02 |
| B6 | 50 | lognormal around 300 (σ 0.5) | 0.01 | 0.05 |

The study's 20% test split of 250 sessions gives 50 test sessions. 25 is the lower bound of the plausible range if sessions are lost.

1,000 data sets per scenario and 1,000 resamples per interval. Seed 20261016. Monte-Carlo SE of a coverage near 0.95 is about 0.007.

## Selection rule (fixed before the run)

1. For each method and scenario, record coverage with its exact 95% binomial interval, and the median interval width.
2. A method is **eligible** if, in every scenario, its coverage interval contains 0.95 or lies above it (it is never significantly below nominal).
3. Among eligible methods, choose the one with the smallest worst-case |coverage − 0.95|. Differences under 0.01 count as ties and go to the simpler method (percentile, then BCa, then bootstrap-t).
4. If no method is eligible, recommend the method with the highest worst-case coverage, and report that worst case beside every interval as a stated limitation.

The result informs MD-11. Nothing is adopted by this comparison.
