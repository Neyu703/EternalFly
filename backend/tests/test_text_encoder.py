import ebooklib.epub
import numpy
import pytest

from eternalfly.text_encoder import (
    NOISE_HALF_WIDTH,
    extract_epub_text,
    load_and_tokenize_file,
    project_token_to_currents,
    read_text_file,
    tokenize_text,
)


def test_tokenize_text_splits_and_lowercases_whitespace_separated_words():
    assert tokenize_text("Hello World") == ["hello", "world"]


def test_tokenize_text_strips_leading_and_trailing_punctuation():
    assert tokenize_text('"Hello," she said--well.') == ["hello", "she", "said--well"]


def test_tokenize_text_drops_tokens_that_become_empty_after_stripping():
    assert tokenize_text("hello -- world ...") == ["hello", "world"]


def test_tokenize_text_returns_empty_list_for_empty_string():
    assert tokenize_text("") == []


def test_project_token_to_currents_is_reproducible_across_calls():
    first_call_currents = project_token_to_currents(
        "dragon", pool_size=8, current_scale=1.0, seed=42, valence_weight=1.0
    )
    second_call_currents = project_token_to_currents(
        "dragon", pool_size=8, current_scale=1.0, seed=42, valence_weight=1.0
    )
    numpy.testing.assert_array_equal(first_call_currents, second_call_currents)


def test_project_token_to_currents_differs_between_different_tokens():
    dragon_currents = project_token_to_currents(
        "dragon", pool_size=8, current_scale=1.0, seed=42, valence_weight=1.0
    )
    castle_currents = project_token_to_currents(
        "castle", pool_size=8, current_scale=1.0, seed=42, valence_weight=1.0
    )
    assert not numpy.array_equal(dragon_currents, castle_currents)


def test_project_token_to_currents_returns_pool_size_length_array_scaled_by_current_scale():
    currents = project_token_to_currents(
        "dragon", pool_size=5, current_scale=2.0, seed=1, valence_weight=0.0
    )
    assert currents.shape == (5,)
    # "dragon" is not in the sentiment lexicon (valence 0.0), so with valence_weight=0.0
    # the result is exactly the zero-mean noise term, bounded by current_scale * NOISE_HALF_WIDTH.
    bound = 2.0 * NOISE_HALF_WIDTH
    assert numpy.all(currents >= -bound) and numpy.all(currents <= bound)


def test_project_token_to_currents_shifts_mean_down_for_a_positive_word():
    # Negated relative to word_valence's own sign: see project_token_to_currents's
    # docstring for why (calibrated against the real connectome).
    neutral_currents = project_token_to_currents(
        "dragon", pool_size=2000, current_scale=1.0, seed=1, valence_weight=1.0
    )
    positive_currents = project_token_to_currents(
        "wonderful", pool_size=2000, current_scale=1.0, seed=1, valence_weight=1.0
    )
    assert positive_currents.mean() < neutral_currents.mean()


def test_project_token_to_currents_shifts_mean_up_for_a_negative_word():
    neutral_currents = project_token_to_currents(
        "dragon", pool_size=2000, current_scale=1.0, seed=1, valence_weight=1.0
    )
    negative_currents = project_token_to_currents(
        "kill", pool_size=2000, current_scale=1.0, seed=1, valence_weight=1.0
    )
    assert negative_currents.mean() > neutral_currents.mean()


def test_project_token_to_currents_valence_weight_zero_ignores_sentiment():
    positive_word_currents = project_token_to_currents(
        "wonderful", pool_size=2000, current_scale=1.0, seed=1, valence_weight=0.0
    )
    negative_word_currents = project_token_to_currents(
        "kill", pool_size=2000, current_scale=1.0, seed=1, valence_weight=0.0
    )
    assert positive_word_currents.mean() == pytest.approx(negative_word_currents.mean(), abs=0.05)


def test_project_token_to_currents_raises_on_non_positive_pool_size():
    with pytest.raises(ValueError):
        project_token_to_currents("dragon", pool_size=0, current_scale=1.0, seed=1, valence_weight=1.0)


def test_project_token_to_currents_raises_on_non_integer_pool_size():
    with pytest.raises(ValueError):
        project_token_to_currents("dragon", pool_size=3.5, current_scale=1.0, seed=1, valence_weight=1.0)


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

    assert load_and_tokenize_file(text_path) == ["the", "dragon", "flew"]


def test_load_and_tokenize_file_dispatches_on_uppercase_txt_suffix(tmp_path):
    text_path = tmp_path / "story.TXT"
    text_path.write_text("Sunlit meadows", encoding="utf-8")

    assert load_and_tokenize_file(text_path) == ["sunlit", "meadows"]


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
