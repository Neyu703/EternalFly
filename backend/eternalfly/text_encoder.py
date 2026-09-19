"""Turns raw book text into per-neuron input currents for the sensory-input pool."""

import hashlib
import pathlib
import string

import ebooklib
import ebooklib.epub
import numpy
from bs4 import BeautifulSoup

from eternalfly.sentiment_lexicon import word_valence

# Half-width of the per-neuron random noise added to every token's currents (see
# project_token_to_currents). Calibrated low on purpose: at the original 0.5, noise
# from the many sentiment-neutral words in real prose drowned out the signal from the
# few words that actually carry sentiment (scripts/calibrate_sentiment.py showed the
# rating tracking real text's sentiment only inconsistently); 0.15 keeps just enough
# per-token texture for pool diversity while letting the valence offset dominate.
NOISE_HALF_WIDTH = 0.15


def tokenize_text(raw_text: str) -> list[str]:
    """Lowercase raw_text, split on whitespace, strip surrounding punctuation from each
    token, and drop tokens that become empty after stripping."""
    lowercased_text = raw_text.lower()
    candidate_tokens = lowercased_text.split()
    stripped_tokens = [token.strip(string.punctuation) for token in candidate_tokens]
    return [token for token in stripped_tokens if token != ""]


def project_token_to_currents(
    token: str, pool_size: int, current_scale: float, seed: int, valence_weight: float
) -> numpy.ndarray:
    """Deterministically project a single token onto a pool_size-length array of
    injected currents for the sensory-input-pool neurons, reproducible across separate
    process runs given the same (token, pool_size, current_scale, seed, valence_weight).

    Each neuron gets zero-mean noise (unique per token, for texture/diversity across
    the pool) plus a shared offset from the token's real sentiment valence (see
    sentiment_lexicon.word_valence), so the network receives a signal correlated with
    word meaning instead of pure noise. valence_weight controls how strongly valence
    shifts the mean relative to the noise spread (NOISE_HALF_WIDTH * 2 wide).

    The offset is *negated* relative to word_valence's own sign: calibrating against
    the real FlyWire connectome (see scripts/calibrate_sentiment.py results) showed
    that *more* sensory-pool drive consistently produces *lower* ratings in this
    network's fixed wiring (i.e. stronger stimulation reads as more aversive here, not
    more rewarding) — so positive-valence words need a *lower*-than-baseline current to
    end up rated positively, and vice versa. This is an empirically-measured property
    of this specific connectome, not an assumption."""
    if not isinstance(pool_size, int) or pool_size <= 0:
        raise ValueError("pool_size must be a positive integer")
    token_digest = hashlib.sha256(f"{token}:{seed}".encode()).hexdigest()
    deterministic_seed = int(token_digest, 16) % (2**32)
    random_generator = numpy.random.default_rng(deterministic_seed)
    zero_mean_noise = random_generator.uniform(-NOISE_HALF_WIDTH, NOISE_HALF_WIDTH, size=pool_size)
    valence_offset = -word_valence(token) * valence_weight
    return (zero_mean_noise + valence_offset) * current_scale


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
