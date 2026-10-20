# Conversational Delivery Adaptation (CODA)

**Conversational Delivery Adaptation: a foundational empirical study for Situational Style Cloning.**

In spontaneous dyadic conversation, and conditional on the reply's own words and the partner's transcript, is the reply's prosodic delivery predictively related to the partner's delivery — for speakers never seen in training — and does the strength or structure of that relationship depend on the conversational function of the turn pair?

**Start here: [`docs/research_overview.md`](docs/research_overview.md).** It explains the question, the design, and why each component exists, without assuming familiarity with the project.

## What this is, and is not

CODA is a measurement study. It estimates how much the partner's delivery contributes to predicting a reply's delivery, beyond the text both speakers produce, evaluated on speakers held out entirely.

It generates no audio. There is no text-to-speech, no style cloning, no end-to-end agent, and no causal claim.

## Repository

| Path | What |
|---|---|
| [`docs/research_overview.md`](docs/research_overview.md) | The scientific design and its rationale |
| [`PREREGISTRATION.md`](PREREGISTRATION.md) | Hypotheses, primary test and analysis plan, committed before the confirmatory experiments |
| [`docs/research_journal.md`](docs/research_journal.md) | What was expected, observed and decided, in order |
| [`CODA_literature_survey.md`](CODA_literature_survey.md) | What the surrounding literature establishes, and what it leaves open |
| `src/` | Source: data handling, features, targets, estimators, controls, statistics, analysis |
| `tests/` | The integrity suite, including the leakage checks. Passes before any fit runs |
| `notebooks/` | Fifteen notebooks in execution order, each a reading surface for one stage |
| `results/` | One directory per run, keyed by a hash of its configuration |

## Reproducing

```bash
python -m pip install -e .
python -m pytest tests/ -q
```

The statistical core runs without a GPU. Stages that touch audio, and the self-supervised feature extraction, run on a cloud GPU runtime; see the reproducibility section of the report for corpus acquisition and environment details.

## Data

CANDOR, a corpus of unscripted dyadic conversations, free on registration. Audio is processed and discarded: features are extracted, then the audio is deleted. No audio or corpus-derived cache is included in this repository.

## Status

Under active development. Results are not yet available.
