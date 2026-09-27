import zipfile

import ebooklib.epub
import numpy
import pytest
from epub_fixtures import html_document, write_test_epub

from eternalfly.text_encoder import (
    NOISE_HALF_WIDTH,
    SUPPORTED_BOOK_SUFFIXES,
    extract_epub_text,
    extract_plain_text_story,
    load_and_tokenize_file,
    project_arousal_to_currents,
    project_token_to_currents,
    project_valence_to_currents,
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
    first_chapter = ebooklib.epub.EpubHtml(
        title="Chapter One", file_name="chapter_one.xhtml", lang="en"
    )
    first_chapter.content = "<html><body><p>The dragon flew over the castle.</p></body></html>"

    second_chapter = ebooklib.epub.EpubHtml(
        title="Chapter Two", file_name="chapter_two.xhtml", lang="en"
    )
    second_chapter.content = "<html><body><p>Sunlit meadows stretched onward.</p></body></html>"

    chapters = [first_chapter, second_chapter]
    write_test_epub(epub_path, "test-id-123", "Tiny Test Book", chapters, tuple(chapters), ["nav", *chapters])


def _build_epub_with_front_matter(epub_path):
    """Write an EPUB whose spine opens with a title page, a copyright page and a table of
    contents before two chapters and a non-linear footnotes page. The manifest lists
    chapter two before chapter one; the spine has them in reading order."""
    title_page = html_document("title.xhtml", "<h1>The Dragon Book</h1><p>A Novel</p>")
    copyright_page = html_document("copyright.xhtml", "<p>Copyright © 2024. All rights reserved.</p>")
    contents_page = html_document("contents.xhtml", "<h1>Contents</h1><p>Chapter One, Chapter Two</p>")
    chapter_one = html_document(
        "chapter_one.xhtml",
        "<h1>Chapter One</h1><svg><desc>cover-art-url</desc></svg><p>The dragon flew over the castle.</p>",
        head_title="HeadTitle",
    )
    chapter_two = html_document("chapter_two.xhtml", "<h1>Chapter Two</h1><p>Sunlit meadows stretched onward.</p>")
    footnotes = html_document("notes.xhtml", "<p>Footnote about dragons.</p>")
    write_test_epub(
        epub_path,
        "front-matter-test",
        "Front Matter Test",
        [title_page, copyright_page, contents_page, chapter_two, chapter_one, footnotes],
        (
            ebooklib.epub.Link("title.xhtml", "Title Page", "title"),
            ebooklib.epub.Link("copyright.xhtml", "Copyright", "copyright"),
            ebooklib.epub.Link("contents.xhtml", "Contents", "contents"),
            ebooklib.epub.Link("chapter_one.xhtml", "Chapter One", "one"),
            ebooklib.epub.Link("chapter_two.xhtml", "Chapter Two", "two"),
        ),
        [title_page, copyright_page, contents_page, chapter_one, chapter_two, (footnotes, "no")],
    )


def test_extract_epub_text_returns_concatenated_plain_text_from_all_documents(tmp_path):
    epub_path = tmp_path / "tiny.epub"
    _build_tiny_epub(epub_path)

    extracted_text = extract_epub_text(epub_path)

    assert "The dragon flew over the castle." in extracted_text
    assert "Sunlit meadows stretched onward." in extracted_text


def test_extract_epub_text_starts_at_the_first_chapter_skipping_front_matter(tmp_path):
    epub_path = tmp_path / "front_matter.epub"
    _build_epub_with_front_matter(epub_path)

    extracted_text = extract_epub_text(epub_path)

    assert extracted_text.split()[:2] == ["Chapter", "One"]
    assert "A Novel" not in extracted_text
    assert "All rights reserved" not in extracted_text


def test_extract_epub_text_follows_the_spine_order_not_the_manifest_order(tmp_path):
    epub_path = tmp_path / "front_matter.epub"
    _build_epub_with_front_matter(epub_path)

    extracted_text = extract_epub_text(epub_path)

    assert extracted_text.index("The dragon flew") < extracted_text.index("Sunlit meadows")


def test_extract_epub_text_leaves_out_page_titles_svg_descriptions_and_non_linear_pages(tmp_path):
    epub_path = tmp_path / "front_matter.epub"
    _build_epub_with_front_matter(epub_path)

    extracted_text = extract_epub_text(epub_path)

    assert "HeadTitle" not in extracted_text
    assert "cover-art-url" not in extracted_text
    assert "Footnote" not in extracted_text


def _build_fan_fiction_epub(epub_path):
    """Write an EPUB with one Archive of Our Own style chapter: a summary, a notes pointer,
    an epigraph blockquote inside the story, and end notes."""
    chapter = html_document(
        "chapter_one.xhtml",
        '<div id="chapters"><div><h2 class="heading">Chapter 1</h2>'
        '<p>Chapter Summary</p><blockquote class="userstuff"><p>Summary text here.</p></blockquote>'
        '<p>Chapter Notes</p><div class="endnote-link">See the end of the chapter for <a href="#endnotes1">notes</a></div>'
        "</div>"
        '<div class="userstuff2"><blockquote><p>Quoted epigraph.</p></blockquote><p>The dragon flew over the castle.</p></div>'
        '<div id="endnotes1"><p>Chapter End Notes</p><blockquote class="userstuff"><p>Thanks for reading!</p></blockquote></div>'
        "</div>",
    )
    toc = (ebooklib.epub.Link("chapter_one.xhtml", "Chapter 1", "one"),)
    write_test_epub(epub_path, "fan-fiction-test", "Fan Fiction Test", [chapter], toc, [chapter])


def test_extract_epub_text_leaves_out_fan_fiction_chapter_notes_and_summaries(tmp_path):
    epub_path = tmp_path / "fan_fiction.epub"
    _build_fan_fiction_epub(epub_path)

    extracted_text = " ".join(extract_epub_text(epub_path).split())

    assert extracted_text == "Chapter 1 Quoted epigraph. The dragon flew over the castle."


def _build_omnibus_epub(epub_path):
    """Write an EPUB whose nested table of contents groups the first book's chapter under
    a book entry pointing at that book's title page, with a warning page in between."""
    book_title_page = html_document("book_one.xhtml", "<h1>Book One: The Red Pyramid</h1>")
    warning_page = html_document("warning.xhtml", "<p>This is a transcript of an audio recording.</p>")
    chapter_one = html_document("chapter_one.xhtml", "<h1>1. A Death at the Needle</h1><p>We only have a few hours.</p>")
    documents = [book_title_page, warning_page, chapter_one]
    toc = (
        (
            ebooklib.epub.Section("The Kane Chronicles Book 1 The Red Pyramid", href="book_one.xhtml"),
            (ebooklib.epub.Link("chapter_one.xhtml", "1. A Death at the Needle", "one"),),
        ),
    )
    write_test_epub(epub_path, "omnibus-test", "Omnibus Test", documents, toc, documents)


def test_extract_epub_text_steps_into_a_book_entry_that_groups_its_chapters(tmp_path):
    epub_path = tmp_path / "omnibus.epub"
    _build_omnibus_epub(epub_path)

    extracted_text = " ".join(extract_epub_text(epub_path).split())

    assert extracted_text == "1. A Death at the Needle We only have a few hours."


def test_extract_plain_text_story_reads_utf8_file_contents(tmp_path):
    text_path = tmp_path / "story.txt"
    text_path.write_text("The dragon soared over München.", encoding="utf-8")

    assert extract_plain_text_story(text_path) == "The dragon soared over München."


def test_supported_book_suffixes_list_epub_before_txt():
    assert SUPPORTED_BOOK_SUFFIXES == (".epub", ".txt")


def test_extract_epub_text_raises_value_error_for_a_file_that_is_not_a_zip_archive(tmp_path):
    epub_path = tmp_path / "broken.epub"
    epub_path.write_text("not a zip archive", encoding="utf-8")

    with pytest.raises(ValueError, match="Not a readable EPUB: broken.epub"):
        extract_epub_text(epub_path)


def test_extract_epub_text_raises_value_error_when_the_manifest_names_a_missing_file(tmp_path):
    epub_path = tmp_path / "incomplete.epub"
    _build_tiny_epub(epub_path)
    with zipfile.ZipFile(epub_path) as complete_archive:
        entries = {name: complete_archive.read(name) for name in complete_archive.namelist()}
    missing_document = next(name for name in entries if name.endswith("chapter_one.xhtml"))
    with zipfile.ZipFile(epub_path, "w") as incomplete_archive:
        for name, content in entries.items():
            if name != missing_document:
                incomplete_archive.writestr(name, content)

    with pytest.raises(ValueError, match="Not a readable EPUB: incomplete.epub"):
        extract_epub_text(epub_path)


def test_load_and_tokenize_file_tokenizes_a_txt_file(tmp_path):
    text_path = tmp_path / "story.txt"
    text_path.write_text("The dragon flew.", encoding="utf-8")

    assert load_and_tokenize_file(text_path) == ["the", "dragon", "flew"]


def test_load_and_tokenize_file_starts_a_txt_file_at_its_first_chapter(tmp_path):
    text_path = tmp_path / "story.txt"
    chapter_text = " ".join(["dragon"] * 60)
    text_path.write_text(f"Copyright 2024\nAll rights reserved\n\nChapter 1\n{chapter_text}", encoding="utf-8")

    assert load_and_tokenize_file(text_path)[:3] == ["chapter", "1", "dragon"]


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
