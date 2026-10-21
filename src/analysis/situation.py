"""Situation analysis: function labels, their reliability, and strata (J4).

**Categories, assigned to the pair**: question → answer;
assessment/opinion → response; narrative/inform → continuation;
backchannel-adjacent; other (excluded, not modelled).

**Labelling.** Interrogative syntax is taken from text (:func:`interrogative`):
a pair whose partner turn is interrogative is ``question_to_answer``. The
remaining categories come from a text-only classifier over the turn-pair
transcript. **Which classifier, and its prompt, is not set**: no LLM is chosen here.
:class:`TextOnlyClassifier` is the interface any chosen classifier must satisfy;
it receives exactly two strings and nothing else, which is the input-schema half of
the function-label leakage row.

**Reliability gate (G4).** 300 seeded pairs are hand-labelled blind to the
classifier; Cohen's κ (1960) is computed per category, one category
against the rest; a category with κ < 0.6 leaves confirmatory analysis, and G4
passes when at least two categories reach 0.6. A stratum with fewer than 200
pairs is descriptive only.
"""

from __future__ import annotations

import re
from typing import Protocol

import numpy as np
import pandas as pd

from src.data.dataset import LeakageError

CATEGORIES = ("question_to_answer", "assessment_opinion_to_response",
              "narrative_inform_to_continuation", "backchannel_adjacent", "other")
MODELLED = CATEGORIES[:-1]

_WH = r"(what|why|how|when|where|who|whom|whose|which)"
_AUX = (r"(do|does|did|is|are|was|were|am|can|could|will|would|shall|should|may|might|"
        r"have|has|had|must)")
_INTERROGATIVE = re.compile(rf"^\s*(?:{_WH}\b|{_AUX}\s+(?:i|you|we|they|he|she|it|there|that)\b)",
                            re.IGNORECASE)


def interrogative(text: str) -> bool:
    """Interrogative syntax from text: a question mark, or a clause opening with a
    wh-word or subject-auxiliary inversion."""
    t = (text or "").strip()
    return t.endswith("?") or bool(_INTERROGATIVE.match(t))


class TextOnlyClassifier(Protocol):
    """A function classifier sees the pair's two transcripts and nothing else."""

    def __call__(self, partner_text: str, reply_text: str) -> str: ...


ALLOWED_INPUTS = ("pair_id", "partner_text", "reply_text")


def classifier_inputs(pairs: pd.DataFrame, turns: pd.DataFrame) -> pd.DataFrame:
    """The only table a classifier may receive: pair id and the two texts."""
    t = turns.set_index("turn_id")["text"].fillna("")
    return pd.DataFrame({"pair_id": pairs["pair_id"].to_numpy(),
                         "partner_text": t.loc[pairs["partner_turn_id"]].to_numpy(),
                         "reply_text": t.loc[pairs["reply_turn_id"]].to_numpy()})


def label_pairs(inputs: pd.DataFrame, classifier: TextOnlyClassifier | None) -> pd.DataFrame:
    """Function label per pair. Raises if the input table carries anything but text."""
    extra = set(inputs.columns) - set(ALLOWED_INPUTS)
    if extra:
        raise LeakageError(f"the function classifier may see text only; got {sorted(extra)}")
    q = inputs["partner_text"].map(interrogative)
    labels = pd.Series(np.where(q, "question_to_answer", None), index=inputs.index,
                       dtype=object)
    if classifier is None:
        if (~q).any():
            labels[~q] = "unlabelled_classifier_blocked"
    else:
        rest = inputs[~q]
        out = [classifier(p, r) for p, r in zip(rest["partner_text"], rest["reply_text"],
                                                  strict=True)]
        bad = set(out) - set(CATEGORIES[1:])
        if bad:
            raise ValueError(f"classifier returned unknown categories {sorted(bad)}")
        labels[~q] = out
    return pd.DataFrame({"pair_id": inputs["pair_id"].to_numpy(), "function": labels.to_numpy(),
                         "interrogative_rule": q.to_numpy()})


def hand_label_sample(inputs: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Seeded sample for hand labelling, blind to any automatic label."""
    pick = np.random.default_rng(seed).choice(len(inputs), size=min(n, len(inputs)),
                                              replace=False)
    s = inputs.iloc[np.sort(pick)][list(ALLOWED_INPUTS)].copy()
    s["hand_label"] = ""
    return s.reset_index(drop=True)


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's κ for two label vectors (Cohen 1960)."""
    a, b = np.asarray(a), np.asarray(b)
    cats = np.union1d(a, b)
    po = float((a == b).mean())
    pe = float(sum((a == c).mean() * (b == c).mean() for c in cats))
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def kappa_per_category(hand: pd.Series, auto: pd.Series,
                       categories=MODELLED) -> dict[str, float]:
    """One-vs-rest κ for each category."""
    return {c: cohen_kappa((hand == c).to_numpy(), (auto == c).to_numpy()) for c in categories}


def g4_verdict(kappas: dict[str, float], min_kappa: float = 0.6,
               min_categories: int = 2) -> dict:
    passing = [c for c, k in kappas.items() if np.isfinite(k) and k >= min_kappa]
    if len(passing) >= min_categories:
        status, consequence = "PASS", "categories with κ ≥ 0.6 enter confirmatory J4"
    elif passing:
        status, consequence = "FAIL", "situation arm becomes exploratory"
    else:
        status, consequence = "FAIL", "no category passes: ship at Level D"
    return {"gate": "G4", "status": status, "passing_categories": passing,
            "kappas": kappas, "criterion": f"κ ≥ {min_kappa} for ≥ {min_categories} categories",
            "consequence": consequence}


def confirmatory_strata(labels: pd.Series, kappas: dict[str, float], min_kappa: float = 0.6,
                        min_pairs: int = 200) -> pd.DataFrame:
    """Per category: count, κ, and whether it is confirmatory or descriptive."""
    counts = labels.value_counts()
    rows = []
    for c in MODELLED:
        n = int(counts.get(c, 0))
        k = kappas.get(c, float("nan"))
        role = ("confirmatory" if np.isfinite(k) and k >= min_kappa and n >= min_pairs else
                "descriptive")
        rows.append({"category": c, "pairs": n, "kappa": k, "role": role,
                     "why": None if role == "confirmatory" else
                     ("kappa below threshold" if not (np.isfinite(k) and k >= min_kappa)
                      else f"fewer than {min_pairs} pairs")})
    rows.append({"category": "other", "pairs": int(counts.get("other", 0)), "kappa": None,
                 "role": "excluded", "why": "excluded from analysis, not modelled"})
    return pd.DataFrame(rows)


__all__ = ["CATEGORIES", "MODELLED", "interrogative", "TextOnlyClassifier", "ALLOWED_INPUTS",
           "classifier_inputs", "label_pairs", "hand_label_sample", "cohen_kappa",
           "kappa_per_category", "g4_verdict", "confirmatory_strata"]
