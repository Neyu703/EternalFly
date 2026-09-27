"""Finds where a book's story begins, so the fly starts reading at the first chapter (or
prologue) instead of the cover, title and copyright pages, table of contents, dedication,
fan-fiction preface and similar front matter."""

import re
import urllib.parse
from dataclasses import dataclass

# Front-matter section titles, matched against the whole normalized title.
FRONT_MATTER_TITLE_PATTERN = re.compile(
    r"""^(?:
        (?:front\ |full\ |back\ )?cover(?:\ page|\ image)?
      | (?:half[-\ ])?title(?:\ page)?
      | copyrights?\b.* | credits
      | (?:table\ of\ )?contents?(?:\ page)? | table\ of\ \w+(?:\ page)? | toc
      | dedication | epigraph | acknowledge?ments?
      | about\ (?:the\ )?(?:authors?|publishers?|this\ book)
      | also\ by\b.* | (?:by|from)\ the\ same\ author\b.* | other\ (?:books|titles|works)\ by\b.*
      | praise | praise\ for\b.*
      | maps? | map\ of\b.*
      | colou?r\ (?:inserts?|gallery|plates|illustrations?) | gallery | illustrations?
      | (?:book\ )?information | synopsis | blurb
      | preface | foreword | introduction
      | (?:a\ )?note\ (?:from|by)\ the\ author | authors?'?s?\ notes? | (?:a\ )?note\ to\ (?:the\ )?readers?
      | translators?'?s?\ notes?
      | (?:content\ |trigger\ )?warnings? | disclaimer | facts?
      | .*\btimeline | timeline\b.*
      | dramatis\ personae | cast\ of\ characters | (?:list\ of\ )?characters
      | volume\ \d+\b.* | vol\.?\ ?\d+\b.*
      | titel(?:seite|blatt)? | umschlag | impressum | inhalt(?:sverzeichnis)? | widmung | motto
      | danksagung | vorwort | geleitwort | einleitung | einführung
      | über\ (?:den\ autor|die\ autorin|die\ autoren|das\ buch)
      | karten? | personen(?:verzeichnis)? | glossar | zeittafel | zeitleiste | hinweis | triggerwarnung
      | band\ \d+\b.*
    )$""",
    re.VERBOSE | re.IGNORECASE,
)

# Titles that open a chapter, part or prologue: a keyword or a leading number.
CHAPTER_LIKE_TITLE_PATTERN = re.compile(
    r"^(?:chapter|chap|ch|kapitel|prolog(?:ue)?|part|teil|book|buch|act|akt|arc|episode|ep|\d+|one|eins)\b",
    re.IGNORECASE,
)
# Uppercase roman numerals as a whole word ("I JASON", "IV"), but not "I'm" or "Idle".
ROMAN_NUMERAL_TITLE_PATTERN = re.compile(r"^[IVXLCDM]+\b(?!')")

# Short documents with these markers are copyright or metadata pages, whatever their title.
FRONT_MATTER_TEXT_PATTERN = re.compile(r"all rights reserved|\bisbn\b|archive of our own|©", re.IGNORECASE)
MAX_FRONT_MATTER_WORDS = 400
# Without a usable table of contents, a heading-less document this long counts as story.
MIN_STORY_WORDS = 50



def _gutenberg_marker_pattern(marker: str) -> re.Pattern:
    """The line Project Gutenberg opens (marker "START") or closes ("END") its license frame with."""
    return re.compile(rf"^\*{{3}}\s*{marker} OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.IGNORECASE | re.MULTILINE)


# Plain text: Project Gutenberg's license frame, and chapter heading lines.
GUTENBERG_START_PATTERN = _gutenberg_marker_pattern("START")
GUTENBERG_END_PATTERN = _gutenberg_marker_pattern("END")
CHAPTER_HEADING_LINE_PATTERN = re.compile(
    r"^[ \t]*(?:chapter|kapitel|prolog(?:ue)?|epilog(?:ue)?|\w+es[ \t]+kapitel)\b[^\n]{0,80}$",
    re.IGNORECASE | re.MULTILINE,
)
FIRST_CHAPTER_HEADING_PATTERN = re.compile(
    r"^[ \t]*(?:(?:chapter|kapitel)[ \t]+(?:1|one|i|eins)\b|erstes[ \t]+kapitel\b|prolog(?:ue)?\b)",
    re.IGNORECASE,
)
# A table of contents lists headings back to back; a real chapter heading is followed by prose.
MIN_CHAPTER_WORDS = 50


@dataclass(frozen=True)
class TocEntry:
    """One table-of-contents entry: its title, the document (and #fragment) it links to,
    and whether further entries are nested beneath it."""

    title: str
    href: str
    has_children: bool = False


@dataclass(frozen=True)
class SpineDocument:
    """One document of the book in reading order: its path inside the EPUB (relative to the
    package file), its first heading ("" if none) and its visible text."""

    name: str
    heading: str
    text: str


def _word_count(text: str) -> int:
    """The number of whitespace-separated words in text."""
    return len(text.split())


def _normalized_title(title: str) -> str:
    """title with typographic apostrophes and quotes made plain, whitespace collapsed and
    surrounding whitespace and punctuation removed."""
    plain_title = title.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return " ".join(plain_title.split()).strip(" .:;,-–—")


def is_front_matter_title(title: str) -> bool:
    """Whether title names a front-matter section (cover, title page, copyright, contents,
    dedication, epigraph, acknowledgments, about the author, also by, map, preface,
    introduction, author's note, timeline, a volume's title page ...), English or German."""
    return bool(FRONT_MATTER_TITLE_PATTERN.match(_normalized_title(title)))


def is_chapter_like_title(title: str) -> bool:
    """Whether title opens a chapter, part or prologue ("Chapter 1", "Part One", "Prologue",
    "1. A Death", "ONE: ...", "I JASON")."""
    normalized_title = _normalized_title(title)
    return bool(CHAPTER_LIKE_TITLE_PATTERN.match(normalized_title) or ROMAN_NUMERAL_TITLE_PATTERN.match(normalized_title))


def is_front_matter_text(text: str) -> bool:
    """Whether text is a short copyright or metadata page (rights notice, ISBN, © sign, or
    an Archive of Our Own posting note)."""
    return _word_count(text) < MAX_FRONT_MATTER_WORDS and bool(FRONT_MATTER_TEXT_PATTERN.search(text))


def document_index_for_href(href: str, document_names: list[str]) -> int | None:
    """Index of the document an href links to (URL-decoded, #fragment ignored), matched on
    its trailing path so links relative to another folder ("../Text/ch1.xhtml") resolve too.
    None when no document matches or the href has no path."""
    path_parts = [part for part in urllib.parse.unquote(href.partition("#")[0]).split("/") if part not in ("", ".", "..")]
    if not path_parts:
        return None
    for index, name in enumerate(document_names):
        if name.split("/")[-len(path_parts):] == path_parts:
            return index
    return None


def story_start_index(toc_entries: list[TocEntry], documents: list[SpineDocument]) -> int:
    """Index of the document where the story begins. Prefers the table of contents: the
    first entry that is neither front matter by title or content nor a book or volume
    heading that merely groups chapters (the fly steps into those). Without such an entry,
    the first document with a chapter heading, or with enough text and no front-matter
    heading or content. Falls back to 0, reading everything."""
    story_start = _story_start_from_toc(toc_entries, documents)
    if story_start is None:
        story_start = _story_start_from_documents(documents)
    return story_start if story_start is not None else 0


def _is_story_toc_entry(entry: TocEntry, document: SpineDocument) -> bool:
    """Whether a table-of-contents entry, linking to document, opens the story: neither
    front matter by title or content, nor a book or volume heading that merely groups
    chapters."""
    if is_front_matter_title(entry.title) or is_front_matter_text(document.text):
        return False
    return not entry.has_children or is_chapter_like_title(entry.title)


def _story_start_from_toc(toc_entries: list[TocEntry], documents: list[SpineDocument]) -> int | None:
    """Index of the document the first story entry of the table of contents links to, or
    None when no entry qualifies (see _is_story_toc_entry)."""
    document_names = [document.name for document in documents]
    for entry in toc_entries:
        index = document_index_for_href(entry.href, document_names)
        if index is not None and _is_story_toc_entry(entry, documents[index]):
            return index
    return None


def _is_story_document(document: SpineDocument) -> bool:
    """Whether document reads as story: no front-matter heading or content, and a chapter
    heading or at least MIN_STORY_WORDS words."""
    if is_front_matter_title(document.heading) or is_front_matter_text(document.text):
        return False
    return is_chapter_like_title(document.heading) or _word_count(document.text) >= MIN_STORY_WORDS


def _story_start_from_documents(documents: list[SpineDocument]) -> int | None:
    """Index of the first document that reads as story, or None when none does."""
    return next((index for index, document in enumerate(documents) if _is_story_document(document)), None)


def _without_gutenberg_frame(text: str) -> str:
    """text without Project Gutenberg's license header and footer, where present."""
    start_marker = GUTENBERG_START_PATTERN.search(text)
    if start_marker:
        text = text[start_marker.end():]
    end_marker = GUTENBERG_END_PATTERN.search(text)
    return text[: end_marker.start()] if end_marker else text


def plain_text_story(text: str) -> str:
    """The story of a plain-text book: without a Project Gutenberg frame, starting at the
    first "Chapter 1"/"Kapitel 1"/"Erstes Kapitel" or prologue heading that is followed by
    prose (not by the next heading, as in a table of contents). The whole remaining text
    when there's no such heading."""
    text = _without_gutenberg_frame(text)
    headings = list(CHAPTER_HEADING_LINE_PATTERN.finditer(text))
    for heading, next_heading in zip(headings, headings[1:] + [None]):
        if not FIRST_CHAPTER_HEADING_PATTERN.match(heading.group()):
            continue
        section_end = next_heading.start() if next_heading else len(text)
        if _word_count(text[heading.end() : section_end]) >= MIN_CHAPTER_WORDS:
            return text[heading.start() :]
    return text
