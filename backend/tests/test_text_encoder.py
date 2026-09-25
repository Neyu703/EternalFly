import ebooklib.epub
import numpy
import pytest

from eternalfly.text_encoder import (
    NOISE_HALF_WIDTH,
    extract_epub_text,
    load_and_tokenize_file,
    project_arousal_to_currents,
    project_token_to_currents,
    project_valence_to_currents,
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


def test_project_token_to_currents_is_reproducible_across_calls():
    first_call_currents = project_token_to_currents("dragon", pool_size=8, current_scale=1.0, seed=42)
    second_call_currents = project_token_to_currents("dragon", pool_size=8, current_scale=1.0, seed=42)
    numpy.testing.assert_array_equal(first_call_currents, second_call_currents)


def test_project_token_to_currents_differs_between_different_tokens():
    dragon_currents = project_token_to_currents("dragon", pool_size=8, current_scale=1.0, seed=42)
    castle_currents = project_token_to_currents("castle", pool_size=8, current_scale=1.0, seed=42)
    assert not numpy.array_equal(dragon_currents, castle_currents)


def test_project_token_to_currents_returns_pool_size_length_array_scaled_by_current_scale():
    currents = project_token_to_currents("dragon", pool_size=5, current_scale=2.0, seed=1)
    assert currents.shape == (5,)
    bound = 2.0 * NOISE_HALF_WIDTH
    assert numpy.all(currents >= -bound) and numpy.all(currents <= bound)


def test_project_token_to_currents_raises_on_non_positive_pool_size():
    with pytest.raises(ValueError):
        project_token_to_currents("dragon", pool_size=0, current_scale=1.0, seed=1)


def test_project_token_to_currents_raises_on_non_integer_pool_size():
    with pytest.raises(ValueError):
        project_token_to_currents("dragon", pool_size=3.5, current_scale=1.0, seed=1)


def test_project_valence_to_currents_positive_channel_excites_for_a_positive_word():
    currents = project_valence_to_currents(
        "wonderful", pool_size=4, current_scale=10.0, valence_weight=1.0, channel="positive"
    )
    assert numpy.all(currents > 0.0)


def test_project_valence_to_currents_positive_channel_is_zero_for_a_negative_word():
    currents = project_valence_to_currents(
        "kill", pool_size=4, current_scale=10.0, valence_weight=1.0, channel="positive"
    )
    numpy.testing.assert_array_equal(currents, numpy.zeros(4))


def test_project_valence_to_currents_negative_channel_excites_for_a_negative_word():
    currents = project_valence_to_currents(
        "kill", pool_size=4, current_scale=10.0, valence_weight=1.0, channel="negative"
    )
    assert numpy.all(currents > 0.0)


def test_project_valence_to_currents_negative_channel_is_zero_for_a_positive_word():
    currents = project_valence_to_currents(
        "wonderful", pool_size=4, current_scale=10.0, valence_weight=1.0, channel="negative"
    )
    numpy.testing.assert_array_equal(currents, numpy.zeros(4))


def test_project_valence_to_currents_is_zero_for_a_neutral_word_on_both_channels():
    positive_channel = project_valence_to_currents(
        "dragon", pool_size=4, current_scale=10.0, valence_weight=1.0, channel="positive"
    )
    negative_channel = project_valence_to_currents(
        "dragon", pool_size=4, current_scale=10.0, valence_weight=1.0, channel="negative"
    )
    numpy.testing.assert_array_equal(positive_channel, numpy.zeros(4))
    numpy.testing.assert_array_equal(negative_channel, numpy.zeros(4))


def test_project_valence_to_currents_never_negative_regardless_of_channel_or_word():
    for word in ["wonderful", "kill", "dragon"]:
        for channel in ["positive", "negative"]:
            currents = project_valence_to_currents(
                word, pool_size=4, current_scale=10.0, valence_weight=1.0, channel=channel
            )
            assert numpy.all(currents >= 0.0)


def test_project_valence_to_currents_scales_with_valence_weight():
    low_weight_currents = project_valence_to_currents(
        "wonderful", pool_size=4, current_scale=10.0, valence_weight=1.0, channel="positive"
    )
    high_weight_currents = project_valence_to_currents(
        "wonderful", pool_size=4, current_scale=10.0, valence_weight=2.0, channel="positive"
    )
    numpy.testing.assert_allclose(high_weight_currents, low_weight_currents * 2.0)


def test_project_valence_to_currents_raises_on_invalid_channel():
    with pytest.raises(ValueError, match="channel must be"):
        project_valence_to_currents(
            "wonderful", pool_size=4, current_scale=10.0, valence_weight=1.0, channel="sideways"
        )


def test_project_arousal_to_currents_excites_for_a_positive_word():
    currents = project_arousal_to_currents("wonderful", pool_size=4, current_scale=10.0, arousal_weight=1.0)
    assert numpy.all(currents > 0.0)


def test_project_arousal_to_currents_excites_for_a_negative_word():
    currents = project_arousal_to_currents("kill", pool_size=4, current_scale=10.0, arousal_weight=1.0)
    assert numpy.all(currents > 0.0)


def test_project_arousal_to_currents_is_stronger_for_more_extreme_sentiment():
    mild_currents = project_arousal_to_currents("okay", pool_size=4, current_scale=10.0, arousal_weight=1.0)
    extreme_currents = project_arousal_to_currents("kill", pool_size=4, current_scale=10.0, arousal_weight=1.0)
    assert extreme_currents[0] > mild_currents[0]


def test_project_arousal_to_currents_is_zero_for_a_neutral_word():
    currents = project_arousal_to_currents("dragon", pool_size=4, current_scale=10.0, arousal_weight=1.0)
    numpy.testing.assert_array_equal(currents, numpy.zeros(4))


def test_project_arousal_to_currents_never_negative():
    for word in ["wonderful", "kill", "dragon"]:
        currents = project_arousal_to_currents(word, pool_size=4, current_scale=10.0, arousal_weight=1.0)
        assert numpy.all(currents >= 0.0)


def test_project_arousal_to_currents_scales_with_arousal_weight():
    low_weight_currents = project_arousal_to_currents("kill", pool_size=4, current_scale=10.0, arousal_weight=1.0)
    high_weight_currents = project_arousal_to_currents("kill", pool_size=4, current_scale=10.0, arousal_weight=2.0)
    numpy.testing.assert_allclose(high_weight_currents, low_weight_currents * 2.0)


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
