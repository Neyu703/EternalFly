import pytest

from eternalfly.story_start import (
    SpineDocument,
    TocEntry,
    document_index_for_href,
    is_chapter_like_title,
    is_front_matter_text,
    is_front_matter_title,
    plain_text_story,
    story_start_index,
)

STORY_TEXT = " ".join(["word"] * 60)


def _document(name, text=STORY_TEXT, heading=""):
    """A spine document with the given name, text and first heading."""
    return SpineDocument(name=name, heading=heading, text=text)


@pytest.mark.parametrize(
    "title",
    [
        "Cover",
        "Front Cover",
        "Title Page",
        "Copyright",
        "Copyright Notice",
        "Copyrights and Credits",
        "Contents",
        "Table of Contents",
        "Table of Conents",
        "Table of Contents Page",
        "Dedication",
        "Epigraph",
        "Acknowledgments",
        "Acknowledgements",
        "About the Author",
        "Also By Rick Riordan",
        "By the Same Author",
        "Other books by Rick Riordan",
        "Map",
        "Color Inserts",
        "Colour Gallery",
        "Information",
        "Synopsis",
        "Preface",
        "Foreword",
        "Introduction",
        "A Note from the Author",
        "Author’s Note",
        "Warning",
        "Disclaimer",
        "Fact",
        "The Star Wars Novels Timeline",
        "Timeline of ‘The 500 Year Climb’",
        "Volume 1: Clown",
        "  Copyright.  ",
        "Impressum",
        "Inhaltsverzeichnis",
        "Widmung",
        "Danksagung",
        "Über die Autorin",
        "Vorwort",
    ],
)
def test_is_front_matter_title_recognizes_front_matter(title):
    assert is_front_matter_title(title)


@pytest.mark.parametrize(
    "title",
    [
        "",
        "Chapter 1",
        "Chapter One | The Dream",
        "Prologue: The War That Never Ended",
        "Prolog",
        "1. Could You Please Stop Killing My Goat",
        "I JASON",
        "Past: The Feast, The Forgotten",
        "THE END",
        "The Cover-Up",
        "Covering Fire",
        "Mapping the Stars",
    ],
)
def test_is_front_matter_title_leaves_story_titles_alone(title):
    assert not is_front_matter_title(title)


@pytest.mark.parametrize(
    "title",
    [
        "Chapter 1",
        "chapter one",
        "Chapter 214: Land of Hope",
        "Kapitel 3",
        "Prologue: Tomb",
        "Prolog",
        "Part One: The Great Disaster",
        "Arc 5: Heaven Official's Blessing",
        "1",
        "1. A Death At The Needle",
        "1 I Accidentally Vaporize My Pre-Algebra Teacher",
        "ONE: Good Morning! You’re Going to Die",
        "I JASON",
        "I: Hazel",
        "IV",
    ],
)
def test_is_chapter_like_title_recognizes_chapter_headings(title):
    assert is_chapter_like_title(title)


@pytest.mark.parametrize(
    "title",
    [
        "",
        "The Kane Chronicles Book 1 The Red Pyramid",
        "Lord of the Mysteries",
        "The Blood of Olympus",
        "Idle Hands",
        "I’m Fine",
    ],
)
def test_is_chapter_like_title_rejects_other_titles(title):
    assert not is_chapter_like_title(title)


def test_is_front_matter_text_recognizes_a_short_copyright_page():
    assert is_front_matter_text("Copyright © 2020 by Someone. All rights reserved. ISBN 978-0-00-000000-0")


def test_is_front_matter_text_recognizes_an_archive_of_our_own_preface():
    assert is_front_matter_text("Posted originally on the Archive of Our Own at http://archiveofourown.org")


def test_is_front_matter_text_ignores_long_texts_that_mention_copyright():
    assert not is_front_matter_text("All rights reserved. " + " ".join(["story"] * 400))


def test_is_front_matter_text_ignores_ordinary_prose():
    assert not is_front_matter_text("The dragon roared and the castle shook with fear.")


def test_document_index_for_href_matches_the_exact_path_and_drops_the_fragment():
    assert document_index_for_href("Text/ch1.xhtml#start", ["Text/cover.xhtml", "Text/ch1.xhtml"]) == 1


def test_document_index_for_href_decodes_url_escapes():
    names = ["titlepage.xhtml", "Breath Mints Battle_split_002.xhtml"]
    assert document_index_for_href("Breath%20Mints%20Battle_split_002.xhtml", names) == 1


def test_document_index_for_href_resolves_paths_relative_to_another_folder():
    assert document_index_for_href("../Text/ch1.xhtml", ["OEBPS/Text/cover.xhtml", "OEBPS/Text/ch1.xhtml"]) == 1


def test_document_index_for_href_returns_none_for_unknown_or_empty_targets():
    assert document_index_for_href("missing.xhtml", ["ch1.xhtml"]) is None
    assert document_index_for_href("#only-a-fragment", ["ch1.xhtml"]) is None


def test_story_start_index_skips_front_matter_entries_and_untitled_pages_before_the_first_chapter():
    documents = [
        _document("preface.xhtml"),
        _document("summary.xhtml"),
        _document("chapter1.xhtml"),
        _document("chapter2.xhtml"),
    ]
    toc_entries = [
        TocEntry("Preface", "preface.xhtml"),
        TocEntry("Past: The Feast", "chapter1.xhtml"),
        TocEntry("Present: The Dream", "chapter2.xhtml"),
    ]
    assert story_start_index(toc_entries, documents) == 2


def test_story_start_index_starts_at_a_prologue():
    documents = [_document("copyright.xhtml"), _document("prologue.xhtml"), _document("chapter1.xhtml")]
    toc_entries = [
        TocEntry("Copyright", "copyright.xhtml"),
        TocEntry("Prologue", "prologue.xhtml"),
        TocEntry("Chapter 1", "chapter1.xhtml"),
    ]
    assert story_start_index(toc_entries, documents) == 1


def test_story_start_index_keeps_a_chapter_whose_own_page_is_empty():
    documents = [_document("title.xhtml", text=""), _document("one.xhtml", text=""), _document("one_text.xhtml")]
    toc_entries = [TocEntry("Title Page", "title.xhtml"), TocEntry("ONE: Good Morning!", "one.xhtml")]
    assert story_start_index(toc_entries, documents) == 1


def test_story_start_index_steps_into_a_book_entry_that_only_groups_chapters():
    documents = [_document("book1.xhtml"), _document("warning.xhtml"), _document("chapter1.xhtml")]
    toc_entries = [
        TocEntry("The Kane Chronicles Book 1 The Red Pyramid", "book1.xhtml", has_children=True),
        TocEntry("1. A Death At The Needle", "chapter1.xhtml"),
    ]
    assert story_start_index(toc_entries, documents) == 2


def test_story_start_index_starts_at_a_part_entry_that_groups_chapters():
    documents = [_document("contents.xhtml"), _document("part1.xhtml", text=""), _document("chapter1.xhtml")]
    toc_entries = [
        TocEntry("Contents", "contents.xhtml"),
        TocEntry("Part One: The Great Disaster", "part1.xhtml", has_children=True),
        TocEntry("Chapter One", "chapter1.xhtml"),
    ]
    assert story_start_index(toc_entries, documents) == 1


def test_story_start_index_skips_entries_pointing_at_copyright_pages_or_unknown_documents():
    documents = [_document("rights.xhtml", text="Text © 2021. All rights reserved."), _document("chapter1.xhtml")]
    toc_entries = [
        TocEntry("Legal Stuff", "rights.xhtml"),
        TocEntry("Missing", "missing.xhtml"),
        TocEntry("Chapter 1", "chapter1.xhtml"),
    ]
    assert story_start_index(toc_entries, documents) == 1


def test_story_start_index_falls_back_to_document_headings_without_a_usable_toc():
    documents = [
        _document("cover.xhtml", text=""),
        _document("copyright.xhtml", text="Copyright © 2020. All rights reserved."),
        _document("dedication.xhtml", text="For my mother", heading="Dedication"),
        _document("thanks.xhtml", text="Thanks to everyone"),
        _document("chapter1.xhtml", text="Chapter 1 It began", heading="Chapter 1"),
    ]
    assert story_start_index([TocEntry("Cover", "cover.xhtml")], documents) == 4


def test_story_start_index_falls_back_to_the_first_long_document_without_headings():
    documents = [_document("title.xhtml", text="A Novel"), _document("text.xhtml")]
    assert story_start_index([], documents) == 1


def test_story_start_index_reads_everything_when_nothing_looks_like_a_story():
    documents = [_document("a.xhtml", text="Short"), _document("b.xhtml", text="Also short")]
    assert story_start_index([], documents) == 0


def test_plain_text_story_starts_at_the_first_chapter_after_a_table_of_contents():
    text = (
        "My Book\nby Someone\n\nContents\nChapter 1. The Start\nChapter 2. The End\n\n"
        f"Chapter 1. The Start\n{STORY_TEXT}\n\nChapter 2. The End\n{STORY_TEXT}\n"
    )
    assert plain_text_story(text).startswith(f"Chapter 1. The Start\n{STORY_TEXT}")


def test_plain_text_story_starts_at_a_prologue_before_chapter_one():
    text = f"Title\n\nPrologue\n{STORY_TEXT}\n\nChapter One\n{STORY_TEXT}"
    assert plain_text_story(text).startswith("Prologue\n")


def test_plain_text_story_understands_german_chapter_headings():
    text = f"Impressum\nAlle Rechte vorbehalten\n\nErstes Kapitel\n{STORY_TEXT}\n\nZweites Kapitel\n{STORY_TEXT}"
    assert plain_text_story(text).startswith("Erstes Kapitel\n")


def test_plain_text_story_strips_the_project_gutenberg_header_and_footer():
    text = (
        "The Project Gutenberg eBook of Something\nLicense text\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\n"
        "The story itself begins here.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\nMore license text"
    )
    assert plain_text_story(text).strip() == "The story itself begins here."


def test_plain_text_story_keeps_text_without_chapter_headings():
    assert plain_text_story("Once upon a time there was a fly.") == "Once upon a time there was a fly."


def test_plain_text_story_keeps_text_when_no_first_chapter_is_followed_by_prose():
    text = "Chapter 1\nChapter 2\nshort"
    assert plain_text_story(text) == text
