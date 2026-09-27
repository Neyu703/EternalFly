"""Turns raw book text into per-neuron input currents for the sensory-input pool."""

import hashlib
import pathlib
import string

import ebooklib
import ebooklib.epub
import numpy
from bs4 import BeautifulSoup, Tag

from eternalfly.sentiment_lexicon import word_valence
from eternalfly.story_start import SpineDocument, TocEntry, plain_text_story, story_start_index

# Half-width of the per-neuron random noise added to every token's currents (see
# project_token_to_currents) — purely a texture/diversity signal, carries no sentiment.
NOISE_HALF_WIDTH = 0.15

# Labels Archive of Our Own puts on a chapter's author notes and summary; the note itself
# follows in a blockquote.
FAN_FICTION_NOTE_LABELS = {"chapter summary", "chapter notes", "chapter end notes"}


def tokenize_text(raw_text: str) -> list[str]:
    """Lowercase raw_text, split on whitespace, strip surrounding punctuation from each
    token, and drop tokens that become empty after stripping."""
    lowercased_text = raw_text.lower()
    candidate_tokens = lowercased_text.split()
    stripped_tokens = [token.strip(string.punctuation) for token in candidate_tokens]
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
    """Read an epub file and return the plain text of its documents in reading (spine)
    order, joined by a space, starting where the story begins: cover, title and copyright
    pages, table of contents, dedication and other front matter are left out (see
    story_start), as are non-linear pages, page titles and SVG image descriptions."""
    book = ebooklib.epub.read_epub(str(epub_path))
    documents = _spine_documents(book)
    start_index = story_start_index(_flatten_toc(book.toc), documents)
    return " ".join(document.text for document in documents[start_index:])


def _spine_documents(book: ebooklib.epub.EpubBook) -> list[SpineDocument]:
    """The book's linear content documents in reading order (navigation documents, which
    are tables of contents, excluded)."""
    documents = []
    for item_id, linear in book.spine:
        item = book.get_item_with_id(item_id)
        if item is not None and linear != "no" and item.get_type() == ebooklib.ITEM_DOCUMENT:
            documents.append(_spine_document(item))
    return documents


def _spine_document(item: ebooklib.epub.EpubItem) -> SpineDocument:
    """One content document's first heading and the visible text of its body."""
    document_soup = BeautifulSoup(item.get_content(), "html.parser")
    body = document_soup.body or document_soup
    for hidden_element in body.find_all(["svg", "script", "style", "noscript"]):
        hidden_element.decompose()
    _remove_fan_fiction_notes(body)
    heading = body.find(["h1", "h2", "h3", "h4", "h5", "h6"])
    return SpineDocument(
        name=item.get_name(),
        heading=heading.get_text(" ", strip=True) if heading else "",
        text=body.get_text(separator=" "),
    )


def _remove_fan_fiction_notes(body: Tag) -> None:
    """Removes Archive of Our Own author notes and summaries from a chapter, in place: each
    labelled note ("Chapter Summary", "Chapter Notes", "Chapter End Notes") together with
    the blockquote holding it, and the "See the end of the chapter for notes" pointers."""
    for notes_pointer in body.find_all("div", class_="endnote-link"):
        notes_pointer.decompose()
    for label in body.find_all(["p", "h3", "h4", "h5", "h6"]):
        if label.decomposed or label.get_text(" ", strip=True).lower() not in FAN_FICTION_NOTE_LABELS:
            continue
        note = label.find_next_sibling()
        if note is not None and note.name == "blockquote":
            note.decompose()
        label.decompose()


def _flatten_toc(toc_items) -> list[TocEntry]:
    """The book's table of contents as one list in reading order; entries that group others
    (ebooklib's (Section, children) pairs) are marked as having children."""
    entries = []
    for toc_item in toc_items:
        if isinstance(toc_item, tuple):
            section, children = toc_item
            entries.append(TocEntry(section.title, getattr(section, "href", "") or "", has_children=True))
            entries.extend(_flatten_toc(children))
        else:
            entries.append(TocEntry(toc_item.title, toc_item.href))
    return entries


def read_text_file(text_path: pathlib.Path) -> str:
    """Read and return the UTF-8 encoded contents of a plain text file."""
    return text_path.read_text(encoding="utf-8")


def load_and_tokenize_file(file_path: pathlib.Path) -> list[str]:
    """Read file_path (.epub or .txt, case-insensitive) and return the tokenized text of its
    story, from the first chapter on. Raises ValueError for any other suffix, and
    FileNotFoundError if the file is missing."""
    suffix = file_path.suffix.lower()
    if suffix == ".epub":
        raw_text = extract_epub_text(file_path)
    elif suffix == ".txt":
        raw_text = plain_text_story(read_text_file(file_path))
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    return tokenize_text(raw_text)
