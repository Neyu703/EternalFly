import ebooklib.epub
import pytest

from eternalfly.text_encoder import (
    extract_epub_text,
    load_and_tokenize_file,
    read_text_file,
    tokenize_text,
)


def test_tokenize_text_splits_on_whitespace_and_preserves_case():
    # Case is preserved (not lowercased like the old VADER-based tokenizer) - the
    # multilingual embedding model is cased, so e.g. German "Wein" vs "wein" differ.
    assert tokenize_text("Hello World") == ["Hello", "World"]


def test_tokenize_text_strips_leading_and_trailing_punctuation():
    assert tokenize_text('"Hello," she said--well.') == ["Hello", "she", "said--well"]


def test_tokenize_text_strips_german_unicode_quotation_marks():
    # „ (Ps), “ (Pi) - neither is in ASCII string.punctuation, so the old
    # implementation would have left these attached to the token.
    assert tokenize_text("„Hallo“ sagte sie.") == ["Hallo", "sagte", "sie"]


def test_tokenize_text_drops_tokens_that_become_empty_after_stripping():
    assert tokenize_text("hello -- world ...") == ["hello", "world"]


def test_tokenize_text_returns_empty_list_for_empty_string():
    assert tokenize_text("") == []


def _build_tiny_epub(epub_path):
    book = ebooklib.epub.EpubBook()
    book.set_identifier("test-id-123")
    book.set_title("Tiny Test Book")
    book.set_language("en")

    first_chapter = ebooklib.epub.EpubHtml(
        title="Chapter One", file_name="chapter_one.xhtml", lang="en"
    )
    first_chapter.content = "<html><body><p>The dragon flew over the castle.</p></body></html>"

    second_chapter = ebooklib.epub.EpubHtml(
        title="Chapter Two", file_name="chapter_two.xhtml", lang="en"
    )
    second_chapter.content = "<html><body><p>Sunlit meadows stretched onward.</p></body></html>"

    book.add_item(first_chapter)
    book.add_item(second_chapter)
    book.toc = (first_chapter, second_chapter)
    book.add_item(ebooklib.epub.EpubNcx())
    book.add_item(ebooklib.epub.EpubNav())
    book.spine = ["nav", first_chapter, second_chapter]

    ebooklib.epub.write_epub(str(epub_path), book)


def test_extract_epub_text_returns_concatenated_plain_text_from_all_documents(tmp_path):
    epub_path = tmp_path / "tiny.epub"
    _build_tiny_epub(epub_path)

    extracted_text = extract_epub_text(epub_path)

    assert "The dragon flew over the castle." in extracted_text
    assert "Sunlit meadows stretched onward." in extracted_text


def test_read_text_file_reads_utf8_file_contents(tmp_path):
    text_path = tmp_path / "story.txt"
    text_path.write_text("The dragon soared over München.", encoding="utf-8")

    assert read_text_file(text_path) == "The dragon soared over München."


def test_load_and_tokenize_file_tokenizes_a_txt_file(tmp_path):
    text_path = tmp_path / "story.txt"
    text_path.write_text("The dragon flew.", encoding="utf-8")

    assert load_and_tokenize_file(text_path) == ["The", "dragon", "flew"]


def test_load_and_tokenize_file_dispatches_on_uppercase_txt_suffix(tmp_path):
    text_path = tmp_path / "story.TXT"
    text_path.write_text("Sunlit meadows", encoding="utf-8")

    assert load_and_tokenize_file(text_path) == ["Sunlit", "meadows"]


def test_load_and_tokenize_file_tokenizes_an_epub_file(tmp_path):
    epub_path = tmp_path / "tiny.epub"
    _build_tiny_epub(epub_path)

    tokens = load_and_tokenize_file(epub_path)

    assert "dragon" in tokens
    assert "meadows" in tokens


def test_load_and_tokenize_file_dispatches_on_uppercase_epub_suffix(tmp_path):
    epub_path = tmp_path / "tiny.EPUB"
    _build_tiny_epub(epub_path)

    tokens = load_and_tokenize_file(epub_path)

    assert "dragon" in tokens


def test_load_and_tokenize_file_raises_value_error_on_unsupported_suffix(tmp_path):
    unsupported_path = tmp_path / "story.pdf"
    unsupported_path.write_text("irrelevant", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type: .pdf"):
        load_and_tokenize_file(unsupported_path)


def test_load_and_tokenize_file_raises_file_not_found_for_missing_txt_file(tmp_path):
    missing_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        load_and_tokenize_file(missing_path)


def test_load_and_tokenize_file_raises_file_not_found_for_missing_epub_file(tmp_path):
    missing_path = tmp_path / "missing.epub"

    with pytest.raises(FileNotFoundError):
        load_and_tokenize_file(missing_path)
