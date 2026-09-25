"""Turns raw book text into a tokenized word stream for semantic_encoder.py to read."""

import pathlib
import unicodedata

import ebooklib
import ebooklib.epub
from bs4 import BeautifulSoup


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
