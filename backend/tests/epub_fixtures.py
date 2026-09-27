"""Builds small EPUB files for tests."""

import ebooklib.epub


def html_document(file_name: str, body: str, head_title: str = "Page") -> ebooklib.epub.EpubHtml:
    """An EPUB content document with the given body markup and <head> title."""
    document = ebooklib.epub.EpubHtml(title=head_title, file_name=file_name, lang="en")
    document.content = f"<html><head><title>{head_title}</title></head><body>{body}</body></html>"
    return document


def write_test_epub(epub_path, identifier: str, title: str, documents: list, toc: tuple, spine: list) -> None:
    """Write an English EPUB at epub_path holding documents (added to the manifest in the
    given order), with the table of contents toc and the reading order spine (documents,
    "nav" for the navigation page, or (document, "no") for a non-linear page)."""
    book = ebooklib.epub.EpubBook()
    book.set_identifier(identifier)
    book.set_title(title)
    book.set_language("en")
    for document in documents:
        book.add_item(document)
    book.toc = toc
    book.add_item(ebooklib.epub.EpubNcx())
    book.add_item(ebooklib.epub.EpubNav())
    book.spine = spine
    ebooklib.epub.write_epub(str(epub_path), book)
