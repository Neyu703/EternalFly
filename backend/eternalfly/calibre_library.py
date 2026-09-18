"""Read-only access to a Calibre library's metadata.db for listing available books.

Strictly read-only: the sqlite connection is opened via the `file:...?mode=ro` URI
form, and no code path here ever writes, renames, or deletes anything under the
library path or its metadata.db.
"""

import pathlib
import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class CalibreBook:
    """A single book available in a Calibre library, resolved to its best available
    format file per the caller's preferred_formats order."""

    book_id: int
    title: str
    author: str
    file_path: pathlib.Path


def list_books(
    library_path: pathlib.Path,
    preferred_formats: tuple[str, ...] = ("EPUB", "TXT"),
) -> list[CalibreBook]:
    """List books in the Calibre library at library_path, each resolved to the first
    available format found in preferred_formats order. Books with none of the
    preferred formats available are excluded. Raises FileNotFoundError if no
    metadata.db exists at library_path. Never writes to the library."""
    db_path = library_path / "metadata.db"
    if not db_path.exists():
        raise FileNotFoundError(f"No Calibre library found at {library_path}")

    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        books = _fetch_books(connection)
        authors_by_book = _fetch_authors_by_book(connection)
        formats_by_book = _fetch_formats_by_book(connection)
    finally:
        connection.close()

    calibre_books = []
    for book_id, title, book_path in books:
        chosen_format = _pick_preferred_format(formats_by_book.get(book_id, []), preferred_formats)
        if chosen_format is None:
            continue
        format_name, data_name = chosen_format
        file_path = library_path / book_path / f"{data_name}.{format_name.lower()}"
        author = " & ".join(authors_by_book.get(book_id, [])) or "Unbekannt"
        calibre_books.append(CalibreBook(book_id=book_id, title=title, author=author, file_path=file_path))

    calibre_books.sort(key=lambda book: book.title.lower())
    return calibre_books


def _fetch_books(connection: sqlite3.Connection) -> list[tuple[int, str, str]]:
    """Return (id, title, path) for every book in the library."""
    return connection.execute("SELECT id, title, path FROM books").fetchall()


def _fetch_authors_by_book(connection: sqlite3.Connection) -> dict[int, list[str]]:
    """Return a mapping of book id to its author names, in author-id order."""
    rows = connection.execute(
        """
        SELECT books_authors_link.book, authors.name
        FROM books_authors_link
        JOIN authors ON authors.id = books_authors_link.author
        ORDER BY books_authors_link.book, books_authors_link.author
        """
    ).fetchall()
    authors_by_book: dict[int, list[str]] = {}
    for book_id, author_name in rows:
        authors_by_book.setdefault(book_id, []).append(author_name)
    return authors_by_book


def _fetch_formats_by_book(connection: sqlite3.Connection) -> dict[int, list[tuple[str, str]]]:
    """Return a mapping of book id to its available (format, data_name) pairs."""
    rows = connection.execute("SELECT book, format, name FROM data").fetchall()
    formats_by_book: dict[int, list[tuple[str, str]]] = {}
    for book_id, format_name, data_name in rows:
        formats_by_book.setdefault(book_id, []).append((format_name, data_name))
    return formats_by_book


def _pick_preferred_format(
    available_formats: list[tuple[str, str]], preferred_formats: tuple[str, ...]
) -> tuple[str, str] | None:
    """Return the (format, data_name) pair for the first preferred format available,
    or None if none of the preferred formats are present."""
    data_name_by_format = dict(available_formats)
    for preferred_format in preferred_formats:
        if preferred_format in data_name_by_format:
            return preferred_format, data_name_by_format[preferred_format]
    return None
