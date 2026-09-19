"""Looks up a single word's sentiment valence from the bundled VADER lexicon.

Source: the lexicon data file from vaderSentiment 3.3.2 (MIT License, C.J. Hutto &
E. Gilbert, https://github.com/cjhutto/vaderSentiment). Only the lexicon *data file*
is bundled here, not the library: VADER's own polarity_scores() runs sentence-level
heuristics (negation, booster words, punctuation emphasis) that don't apply to scoring
one isolated token, and a plain dict lookup is fast enough to run on every simulated
tick without measurably slowing down the reading loop.
"""

import functools
import pathlib

LEXICON_PATH = pathlib.Path(__file__).parent / "vader_lexicon.txt"
LEXICON_SCORE_MAGNITUDE = 4.0  # VADER's raw per-word mean scores range roughly -4..+4


@functools.lru_cache(maxsize=1)
def _load_lexicon() -> dict[str, float]:
    """Parse the bundled lexicon file into a {word: mean_valence_score} dict. Cached
    after the first call since the file is fixed at ~7500 lines and never changes."""
    lexicon: dict[str, float] = {}
    for line in LEXICON_PATH.read_text(encoding="utf-8").splitlines():
        word, mean_score, _std_dev, _raw_ratings = line.split("\t")
        lexicon[word] = float(mean_score)
    return lexicon


def word_valence(word: str) -> float:
    """Return word's sentiment valence normalized to [-1.0, 1.0] (negative = negative
    sentiment, positive = positive sentiment). Returns 0.0 (neutral) for words not in
    the lexicon, e.g. names, numbers, and most function/content words the lexicon
    doesn't cover. Lookup is case-insensitive."""
    raw_score = _load_lexicon().get(word.lower(), 0.0)
    return raw_score / LEXICON_SCORE_MAGNITUDE
