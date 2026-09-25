"""Turns raw book text into per-neuron input currents for the sensory-input pool."""

import hashlib
import pathlib
import unicodedata

import ebooklib
import ebooklib.epub
import numpy
from bs4 import BeautifulSoup

from eternalfly.sentiment_lexicon import word_valence

# Half-width of the per-neuron random noise added to every token's currents (see
# project_token_to_currents) — purely a texture/diversity signal, carries no sentiment.
NOISE_HALF_WIDTH = 0.15


def _strip_unicode_punctuation(token: str) -> str:
    """Strip leading/trailing characters whose Unicode general category starts with
    "P" (covers every punctuation form, not just ASCII: German „ " quotes, English
    "curly" quotes, em/en dashes, ...), leaving internal punctuation untouched."""
    start = 0
    end = len(token)
    while start < end and unicodedata.category(token[start]).startswith("P"):
        start += 1
    while end > start and unicodedata.category(token[end - 1]).startswith("P"):
        end -= 1
    return token[start:end]


def tokenize_text(raw_text: str) -> list[str]:
    """Split raw_text on whitespace and strip leading/trailing Unicode punctuation
    (see _strip_unicode_punctuation) from each token, dropping tokens that become
    empty after stripping. Case is preserved: the multilingual sentence-embedding
    model (semantic_encoder.py) is cased, so e.g. German "Wein" (wine, a noun) and
    lowercase "wein" (to cry) are meaningfully different inputs to it - unlike the old
    English-only VADER lookup, which was itself always case-insensitive."""
    candidate_tokens = raw_text.split()
    stripped_tokens = [_strip_unicode_punctuation(token) for token in candidate_tokens]
    return [token for token in stripped_tokens if token != ""]


def project_token_to_currents(token: str, pool_size: int, current_scale: float, seed: int) -> numpy.ndarray:
    """Deterministically project a single token onto a pool_size-length array of
    zero-mean noise currents for the sensory-input pool (unique per token, giving each
    word its own texture across the pool), reproducible across separate process runs
    given the same (token, pool_size, current_scale, seed). Carries no sentiment — see
    project_valence_to_currents for that."""
    if not isinstance(pool_size, int) or pool_size <= 0:
        raise ValueError("pool_size must be a positive integer")
    token_digest = hashlib.sha256(f"{token}:{seed}".encode()).hexdigest()
    deterministic_seed = int(token_digest, 16) % (2**32)
    random_generator = numpy.random.default_rng(deterministic_seed)
    return random_generator.uniform(-NOISE_HALF_WIDTH, NOISE_HALF_WIDTH, size=pool_size) * current_scale


def project_valence_to_currents(
    token: str, pool_size: int, current_scale: float, valence_weight: float, channel: str
) -> numpy.ndarray:
    """Return a pool_size-length array of purely excitatory (non-negative) currents for
    one dopaminergic valence channel — "positive" (real reward-coding neurons, the
    mushroom body medial-lobe/PAM-like pool) or "negative" (real punishment-coding
    neurons, the vertical-lobe/PPL1-like pool) — proportional to how strongly token's
    real sentiment (sentiment_lexicon.word_valence) matches that channel.

    A positive word excites only the "positive" channel and a negative word excites
    only the "negative" channel; a word of the opposite sentiment, or a neutral/
    unscored word, contributes zero current here (never a *negative*/suppressive
    current — unlike the old single-channel design, both valence directions are always
    an active, excitatory signal, so neither can be silently overridden by whatever the
    network happened to be doing already). Raises ValueError for any other channel."""
    if channel not in ("positive", "negative"):
        raise ValueError(f"channel must be 'positive' or 'negative', got {channel!r}")
    valence = word_valence(token)
    magnitude = max(0.0, valence if channel == "positive" else -valence)
    current_value = magnitude * valence_weight * current_scale
    return numpy.full(pool_size, current_value, dtype=numpy.float64)


def project_arousal_to_currents(token: str, pool_size: int, current_scale: float, arousal_weight: float) -> numpy.ndarray:
    """Return a pool_size-length array of purely excitatory currents for the
    octopaminergic arousal channel (real arousal/alertness-coding neurons — see
    scripts/build_connectome_cache.py's AROUSAL_OCTOPAMINERGIC_NEUROPILS), proportional
    to how emotionally charged token is: its sentiment magnitude regardless of sign,
    since octopamine drives general arousal in Drosophila rather than a positive/
    negative direction (that's what the valence channels are for). A neutral or
    unscored word contributes zero current here."""
    magnitude = abs(word_valence(token))
    current_value = magnitude * arousal_weight * current_scale
    return numpy.full(pool_size, current_value, dtype=numpy.float64)


def extract_epub_text(epub_path: pathlib.Path) -> str:
    """Read an epub file and return its plain text, stripped of HTML tags, with all
    document items joined by a space in the epub's item order."""
    book = ebooklib.epub.read_epub(str(epub_path))
    document_texts = [
        BeautifulSoup(document_item.get_content(), "html.parser").get_text(separator=" ")
        for document_item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
    ]
    return " ".join(document_texts)


def read_text_file(text_path: pathlib.Path) -> str:
    """Read and return the UTF-8 encoded contents of a plain text file."""
    return text_path.read_text(encoding="utf-8")


def load_and_tokenize_file(file_path: pathlib.Path) -> list[str]:
    """Read file_path (.epub or .txt, case-insensitive) and return its tokenized text.
    Raises ValueError for any other suffix, and FileNotFoundError if the file is missing."""
    suffix = file_path.suffix.lower()
    if suffix == ".epub":
        raw_text = extract_epub_text(file_path)
    elif suffix == ".txt":
        raw_text = read_text_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    return tokenize_text(raw_text)
