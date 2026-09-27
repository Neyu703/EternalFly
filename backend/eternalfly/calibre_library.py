"""Read-only access to a Calibre library's metadata.db for listing available books, and
to Calibre's own settings for finding the library it currently uses.

Strictly read-only: the sqlite connection is opened via the `file:...?mode=ro` URI
form, and no code path here ever writes, renames, or deletes anything under the
library path, its metadata.db or Calibre's settings.
"""

import json
import pathlib
import sqlite3
from dataclasses import dataclass

from eternalfly.text_encoder import SUPPORTED_BOOK_SUFFIXES

# Calibre's global settings file inside its config directory (%APPDATA%\calibre on Windows).
CALIBRE_SETTINGS_FILE_NAME = "global.py.json"

# Calibre's format names for the book files EternalFly can read, most preferred first.
PREFERRED_FORMATS = tuple(suffix.removeprefix(".").upper() for suffix in SUPPORTED_BOOK_SUFFIXES)


@dataclass(frozen=True)
class CalibreBook:
    """A single book available in a Calibre library, resolved to its best available
    format file in PREFERRED_FORMATS order."""

    book_id: int
    title: str
    author: str
    file_path: pathlib.Path


def configured_library_path(calibre_config_directory: pathlib.Path) -> pathlib.Path | None:
    """The library Calibre itself currently uses, as named by library_path in its global
    settings under calibre_config_directory, so a moved or switched library is followed
    automatically. None when the settings are missing, unreadable or name no library.
    Only reads the settings file."""
    settings_path = calibre_config_directory / CALIBRE_SETTINGS_FILE_NAME
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    library_path = settings.get("library_path") if isinstance(settings, dict) else None
    return pathlib.Path(library_path) if isinstance(library_path, str) and library_path else None


def list_books(library_path: pathlib.Path) -> list[CalibreBook]:
    """List books in the Calibre library at library_path, each resolved to the first
    available format found in PREFERRED_FORMATS order. Books with none of those formats
    available are excluded. Raises FileNotFoundError if no
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
        chosen_format = _pick_preferred_format(formats_by_book.get(book_id, []))
        if chosen_format is None:
            continue
        format_name, data_name = chosen_format
        file_path = library_path / book_path / f"{data_name}.{format_name.lower()}"
        author = " & ".join(authors_by_book.get(book_id, [])) or "Unknown"
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


def _pick_preferred_format(available_formats: list[tuple[str, str]]) -> tuple[str, str] | None:
    """Return the (format, data_name) pair for the first of PREFERRED_FORMATS available,
    or None if none of them are present."""
    data_name_by_format = dict(available_formats)
    for preferred_format in PREFERRED_FORMATS:
        if preferred_format in data_name_by_format:
            return preferred_format, data_name_by_format[preferred_format]
    return None
