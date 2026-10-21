"""Syllable counts from transcript words.

Two quantities depend on this: the reply syllable count in the conditioning set
Z, and the numerator of articulation rate. The method is an open decision:

- ``rule_based`` (the implemented default): count groups of consecutive vowel
  letters, discount a silent final "e", and give every word with letters at
  least one syllable. No dependency, deterministic, and noisy on irregular
  English spellings.
- ``dictionary``: CMU Pronouncing Dictionary lookup (package ``cmudict``,
  pinned in the requirements) with the rule as fallback for words the
  dictionary does not list. A word's count is the number of vowel phones
  (those carrying a stress digit) in its first listed pronunciation. Every
  word with letters gets at least one syllable under either path.

Whatever the method, the same counter is applied to every turn in every
condition, so its noise can lower power but cannot create a difference between
conditions.
"""

from __future__ import annotations

import re
from functools import lru_cache

_VOWEL_GROUP = re.compile(r"[aeiouy]+")
_LETTERS = re.compile(r"[a-z]")


def rule_based_syllables(word: str) -> int:
    """Syllables in one word by vowel-group counting.

    ``"conversation"`` -> 4, ``"make"`` -> 1, ``"table"`` -> 2, ``"I"`` -> 1,
    ``"42"`` -> 0.
    """
    w = re.sub(r"[^a-z]", "", word.lower())
    if not _LETTERS.search(w):
        return 0
    n = len(_VOWEL_GROUP.findall(w))
    # A final "e" is usually silent ("make"), except in "-le" after a consonant
    # ("table") and when it is the only vowel ("the", "be").
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and n > 1:
        n -= 1
    elif w.endswith("le") and len(w) > 2 and w[-3] in "aeiouy" and n > 1:
        n -= 1
    # "-es"/"-ed" after most consonants add no syllable ("makes", "liked").
    if re.search(r"[^aeiouytd]e[sd]$", w) and n > 1 and not w.endswith(("ses", "zes", "ces",
                                                                          "ges", "xes")):
        n -= 1
    return max(n, 1)


@lru_cache(maxsize=1)
def _cmu() -> dict:
    try:
        import cmudict
    except ImportError as e:            # pragma: no cover - depends on the environment
        raise ImportError("the 'dictionary' syllable method needs the pinned "
                          "'cmudict' package") from e
    return cmudict.dict()


def dictionary_syllables(word: str) -> tuple[int, bool]:
    """Syllables in one word from the CMU dictionary, and whether the rule was used."""
    w = re.sub(r"[^a-z']", "", word.lower()).strip("'")
    if not _LETTERS.search(w):
        return 0, False
    prons = _cmu().get(w)
    if not prons:
        return rule_based_syllables(w), True
    return max(sum(ph[-1].isdigit() for ph in prons[0]), 1), False


def count_syllables(text: str, method: str = "rule_based") -> int:
    """Syllables in a transcript string under the configured method."""
    if method == "rule_based":
        return sum(rule_based_syllables(t) for t in text.split())
    if method == "dictionary":
        return sum(dictionary_syllables(t)[0] for t in text.split())
    raise ValueError(f"unknown syllable method {method!r}")


def fallback_share(texts, method: str = "dictionary") -> float:
    """Share of lexical tokens counted by the rule because the dictionary lacks them."""
    if method != "dictionary":
        return 0.0
    n = fb = 0
    for text in texts:
        for t in str(text).split():
            if _LETTERS.search(t.lower()):
                n += 1
                fb += dictionary_syllables(t)[1]
    return fb / n if n else 0.0


__all__ = ["rule_based_syllables", "dictionary_syllables", "count_syllables",
           "fallback_share"]
