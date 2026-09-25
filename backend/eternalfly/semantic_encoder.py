"""Turns a word (and its surrounding context) into real fly sensory drives, via a
multilingual sentence-embedding model's notion of meaning - unlike the old VADER-based
text_encoder, this understands German as well as English, and "nicht gut" as distinct
from "gut" (see context_valence).

The actual model (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) is
loaded in brain_loader.py, not here - everything in this module is pure/injectable
(EmbedFn), so it's fully unit-testable without downloading or running the real ~470MB
model. mean_pool_embeddings is the one piece of real model-output post-processing,
extracted so it's testable with synthetic tensors instead of a real forward pass.
"""

import threading
from dataclasses import dataclass
from typing import Callable

import numpy

# A batch embedding function: a list of words/phrases in, one L2-normalized embedding
# vector per input out, same row order. Implemented for real in brain_loader.py.
EmbedFn = Callable[[list[str]], numpy.ndarray]

# DE+EN anchor words per real sensory channel (see cell_groups.SENSORY_CHANNEL_SELECTORS
# - these names must match exactly, one anchor set per real FlyWire cell group).
SENSORY_CHANNEL_ANCHORS: dict[str, list[str]] = {
    "sweet": ["süß", "zucker", "honig", "sweet", "sugar", "honey"],
    "bitter": ["bitter", "sauer", "ekelhaft", "bitter", "sour", "disgusting"],
    "looming": ["angriff", "monster", "bedrohung", "attack", "monster", "threat", "looming"],
    "heat": ["feuer", "hitze", "brennen", "fire", "heat", "burning"],
    "cold": ["eis", "kälte", "frost", "ice", "cold", "frost"],
    "sound": ["musik", "schrei", "lärm", "music", "scream", "noise"],
    "wind": ["wind", "sturm", "böe", "wind", "storm", "gust"],
    "touch": ["staub", "juckreiz", "kratzen", "dust", "itch", "scratching"],
    "rot": ["fäulnis", "verwesung", "gestank", "rot", "decay", "stench"],
    "fruit": ["obst", "wein", "apfel", "fruit", "wine", "apple"],
    "pheromone": ["liebe", "begierde", "verführung", "love", "desire", "seduction"],
}

# A fixed neutral DE+EN vocabulary shared by every channel's background calibration -
# ordinary words carrying none of the above senses, so each channel's z-score reflects
# "how much more similar to this channel's anchors than everyday vocabulary is", not an
# arbitrary raw cosine cutoff (short-word embeddings in this model sit at a baseline
# cosine similarity around 0.3-0.6 to almost anything - see semantic_encoder tests).
BACKGROUND_WORDS: list[str] = [
    "Tisch", "Stuhl", "Haus", "Straße", "Buch", "Fenster", "Tür", "Auto", "Baum", "Stadt",
    "Lampe", "Teppich", "Wolke", "Berg", "Fluss", "Schuh", "Hemd", "Uhr", "Blatt", "Papier",
    "table", "chair", "house", "street", "book", "window", "door", "car", "tree", "city",
    "lamp", "carpet", "cloud", "mountain", "river", "shoe", "shirt", "clock", "leaf", "paper",
    # Ordinary book prose is mostly function words (articles, conjunctions,
    # prepositions, pronouns, common verbs), not nouns - without these, a short
    # grammatical word sits as a systematic outlier relative to a noun-only baseline
    # and z-scores high on almost every channel at once (verified against the real
    # model: "Der"/"und"/"in" all landed z>=2 on every one of the 11 real channels
    # before this list included any function words).
    "der", "die", "das", "und", "oder", "aber", "ist", "war", "hat", "hatte",
    "mit", "für", "auf", "von", "zu", "er", "sie", "es", "nicht", "auch",
    "the", "and", "or", "but", "is", "was", "has", "had",
    "with", "for", "on", "from", "to", "he", "she", "it", "not", "also",
]

# DE+EN anchors for context_valence's positive/negative poles (see that function).
POSITIVE_CONTEXT_ANCHORS: list[str] = ["gut", "schön", "wunderbar", "good", "wonderful", "great"]
NEGATIVE_CONTEXT_ANCHORS: list[str] = ["schlecht", "schrecklich", "furchtbar", "bad", "terrible", "awful"]

# Verified against the real model + a background vocabulary spanning both nouns and
# common function words (see BACKGROUND_WORDS): a general-purpose (not domain-tuned)
# sentence embedding gives real content-word triggers z-scores mostly in the 1.5-5
# range against a handful of anchor words per channel, while ordinary function words
# ("der", "und", ...) top out around z=1.4 on any single channel - these bounds keep
# function words at 0 while still letting most real triggers fire at least partially.
# Anchor-word/background coverage is intentionally an ongoing tuning target (see
# scripts/audit_brain.py), not something these two constants alone can perfect.
Z_RAMP_START = 1.5
Z_RAMP_END = 3.5


def mean_pool_embeddings(token_embeddings: numpy.ndarray, attention_mask: numpy.ndarray) -> numpy.ndarray:
    """Mean-pool a transformer's per-token output embeddings into one L2-normalized
    embedding per input sequence, masking out padding tokens.

    token_embeddings: (batch, sequence_length, hidden_size). attention_mask: (batch,
    sequence_length), 1 for a real token, 0 for padding. Returns (batch, hidden_size).
    """
    mask = attention_mask[:, :, numpy.newaxis].astype(numpy.float64)
    summed = (token_embeddings * mask).sum(axis=1)
    counts = numpy.clip(mask.sum(axis=1), 1e-9, None)
    mean_pooled = summed / counts
    norms = numpy.clip(numpy.linalg.norm(mean_pooled, axis=1, keepdims=True), 1e-9, None)
    return mean_pooled / norms


def mean_unit_embedding(embeddings: numpy.ndarray) -> numpy.ndarray:
    """Return the L2-normalized mean of a set of L2-normalized embeddings (their
    centroid direction) - used to collapse a channel's or context pole's several
    anchor-word embeddings into one representative direction."""
    centroid = embeddings.mean(axis=0)
    return centroid / numpy.clip(numpy.linalg.norm(centroid), 1e-9, None)


@dataclass(frozen=True)
class ChannelCalibration:
    """One sensory channel's calibrated readout direction: its anchor-word centroid,
    plus the mean/std of how similar ordinary background vocabulary already is to that
    centroid (see channel_drive)."""

    anchor_embedding: numpy.ndarray
    background_mean: float
    background_std: float


def calibrate_channel(anchor_embeddings: numpy.ndarray, background_embeddings: numpy.ndarray) -> ChannelCalibration:
    """Build a ChannelCalibration from a channel's anchor-word embeddings and the
    shared background vocabulary's embeddings (both (n, dim), L2-normalized). Call once
    per channel at startup (see brain_loader.py), not per word."""
    anchor_embedding = mean_unit_embedding(anchor_embeddings)
    background_similarities = background_embeddings @ anchor_embedding
    return ChannelCalibration(
        anchor_embedding=anchor_embedding,
        background_mean=float(background_similarities.mean()),
        background_std=float(background_similarities.std()),
    )


def channel_drive(word_embedding: numpy.ndarray, calibration: ChannelCalibration) -> float:
    """How strongly word_embedding belongs to one calibrated sensory channel, in [0, 1].

    Cosine similarity to the channel's anchor centroid, z-scored against how similar
    ordinary background vocabulary already is (calibration.background_mean/std), then
    ramped linearly from 0 at z<=Z_RAMP_START to 1 at z>=Z_RAMP_END - only words clearly
    more similar to the channel's real anchors than everyday vocabulary drive it at all,
    rather than every word producing some nonzero drive on every channel. Returns 0.0
    if the channel's background similarities had zero variance (degenerate calibration).
    """
    if calibration.background_std == 0.0:
        return 0.0
    similarity = float(word_embedding @ calibration.anchor_embedding)
    z_score = (similarity - calibration.background_mean) / calibration.background_std
    ramped = (z_score - Z_RAMP_START) / (Z_RAMP_END - Z_RAMP_START)
    return max(0.0, min(1.0, ramped))


def channel_drives(word_embedding: numpy.ndarray, calibrations: dict[str, ChannelCalibration]) -> dict[str, float]:
    """channel_drive for every named channel in calibrations, keyed the same way."""
    return {name: channel_drive(word_embedding, calibration) for name, calibration in calibrations.items()}


def build_odor_projection(glomerulus_count: int, embedding_dim: int, seed: int) -> numpy.ndarray:
    """Build a fixed (glomerulus_count, embedding_dim) random unit-row projection
    matrix (a "fly-hash", Dasgupta et al. 2017) for semantic_odor. Deterministic given
    (glomerulus_count, embedding_dim, seed) - call once at startup, reuse for every
    word, so semantically similar words consistently land on overlapping glomeruli."""
    rng = numpy.random.default_rng(seed)
    projection_matrix = rng.normal(size=(glomerulus_count, embedding_dim))
    row_norms = numpy.clip(numpy.linalg.norm(projection_matrix, axis=1, keepdims=True), 1e-9, None)
    return projection_matrix / row_norms


def semantic_odor(word_embedding: numpy.ndarray, projection_matrix: numpy.ndarray, active_fraction: float) -> numpy.ndarray:
    """Project word_embedding onto projection_matrix's fixed random directions and keep
    only the top active_fraction as "active" (1.0, rest 0.0) - the fly-hash sparse code
    for the remaining real ORN glomeruli that have no single fixed real-world meaning
    (see cell_groups.select_remaining_olfactory_receptor_neurons). Semantically similar
    words project similarly and so activate overlapping glomerulus sets, a real
    locality-sensitive hash rather than the old per-token SHA256 noise, which gave
    every word (however similar in meaning) a completely unrelated pattern.

    Raises ValueError if active_fraction is not in (0, 1].
    """
    if not 0 < active_fraction <= 1:
        raise ValueError("active_fraction must be in (0, 1]")
    glomerulus_count = projection_matrix.shape[0]
    projections = projection_matrix @ word_embedding
    active_count = max(1, round(active_fraction * glomerulus_count))
    threshold = numpy.partition(projections, -active_count)[-active_count]
    return (projections >= threshold).astype(numpy.float64)


def context_valence(context_embedding: numpy.ndarray, positive_anchor_embedding: numpy.ndarray, negative_anchor_embedding: numpy.ndarray) -> float:
    """How positive (>0) or negative (<0) a reading-context phrase's embedding is, in
    [-1, 1]: cos(context, positive_anchor) - cos(context, negative_anchor).

    context_embedding should be the embedding of a whole multi-word window (e.g. the
    last ~12 words), embedded as one phrase rather than averaging individually-embedded
    words - a phrase embedding captures negation ("nicht gut" pulls away from "gut"),
    which averaging independent per-word embeddings cannot.
    """
    positive_similarity = float(context_embedding @ positive_anchor_embedding)
    negative_similarity = float(context_embedding @ negative_anchor_embedding)
    return max(-1.0, min(1.0, positive_similarity - negative_similarity))


class WordEmbeddingCache:
    """Caches one embedding per unique word, filled by precompute() (typically called
    from a background lookahead thread that stays ahead of the reading position - see
    reading_session.py) while get() is read concurrently from the simulation thread.
    Thread-safe via a single lock; embedding a word is idempotent (a word already
    cached, or already requested by a concurrent precompute() call, is never
    re-embedded)."""

    def __init__(self, embed_fn: EmbedFn):
        self._embed_fn = embed_fn
        self._cache: dict[str, numpy.ndarray] = {}
        self._lock = threading.Lock()

    def precompute(self, words: list[str]) -> None:
        """Embed every word in words not already cached, in one batch call to embed_fn.
        Does nothing (no call to embed_fn) if every word is already cached."""
        with self._lock:
            uncached_words = [word for word in dict.fromkeys(words) if word not in self._cache]
        if not uncached_words:
            return
        embeddings = self._embed_fn(uncached_words)
        with self._lock:
            for word, embedding in zip(uncached_words, embeddings):
                self._cache[word] = embedding

    def get(self, word: str) -> numpy.ndarray | None:
        """Return word's cached embedding, or None if precompute() hasn't covered it yet."""
        with self._lock:
            return self._cache.get(word)

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)
