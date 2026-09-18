import pathlib
import sqlite3

import pytest

from eternalfly.calibre_library import CalibreBook, list_books


def _create_library(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a bare Calibre-shaped metadata.db (no rows yet) under a fresh library dir."""
    library_path = tmp_path / "library"
    library_path.mkdir()
    connection = sqlite3.connect(library_path / "metadata.db")
    connection.executescript(
        """
        CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT, path TEXT);
        CREATE TABLE authors (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE books_authors_link (book INTEGER, author INTEGER);
        CREATE TABLE data (id INTEGER PRIMARY KEY, book INTEGER, format TEXT, name TEXT);
        """
    )
    connection.commit()
    connection.close()
    return library_path


def _run_sql(library_path: pathlib.Path, *statements: str) -> None:
    """Execute one or more raw INSERT statements against the library's metadata.db."""
    connection = sqlite3.connect(library_path / "metadata.db")
    for statement in statements:
        connection.execute(statement)
    connection.commit()
    connection.close()


def test_list_books_raises_file_not_found_when_metadata_db_missing(tmp_path):
    missing_library_path = tmp_path / "no_such_library"

    with pytest.raises(FileNotFoundError) as exc_info:
        list_books(missing_library_path)

    assert str(missing_library_path) in str(exc_info.value)


def test_list_books_returns_title_author_and_resolved_path_for_single_book(tmp_path):
    library_path = _create_library(tmp_path)
    _run_sql(
        library_path,
        "INSERT INTO books VALUES (1, 'Dune', 'Frank Herbert/Dune (1)')",
        "INSERT INTO authors VALUES (1, 'Frank Herbert')",
        "INSERT INTO books_authors_link VALUES (1, 1)",
        "INSERT INTO data VALUES (1, 1, 'EPUB', 'dune')",
    )

    books = list_books(library_path)

    assert books == [
        CalibreBook(
            book_id=1,
            title="Dune",
            author="Frank Herbert",
            file_path=library_path / "Frank Herbert/Dune (1)" / "dune.epub",
        )
    ]


def test_list_books_joins_multiple_authors_with_ampersand(tmp_path):
    library_path = _create_library(tmp_path)
    _run_sql(
        library_path,
        "INSERT INTO books VALUES (1, 'Good Omens', 'Pratchett Gaiman/Good Omens (1)')",
        "INSERT INTO authors VALUES (1, 'Terry Pratchett')",
        "INSERT INTO authors VALUES (2, 'Neil Gaiman')",
        "INSERT INTO books_authors_link VALUES (1, 1)",
        "INSERT INTO books_authors_link VALUES (1, 2)",
        "INSERT INTO data VALUES (1, 1, 'EPUB', 'good_omens')",
    )

    books = list_books(library_path)

    assert books[0].author == "Terry Pratchett & Neil Gaiman"


def test_list_books_uses_unbekannt_for_book_with_no_author_link(tmp_path):
    library_path = _create_library(tmp_path)
    _run_sql(
        library_path,
        "INSERT INTO books VALUES (1, 'Anonymous Work', 'Unknown/Anonymous Work (1)')",
        "INSERT INTO data VALUES (1, 1, 'TXT', 'anon')",
    )

    books = list_books(library_path)

    assert books[0].author == "Unbekannt"


def test_list_books_prefers_epub_over_txt_by_default(tmp_path):
    library_path = _create_library(tmp_path)
    _run_sql(
        library_path,
        "INSERT INTO books VALUES (1, 'Dual Format', 'Author/Dual Format (1)')",
        "INSERT INTO data VALUES (1, 1, 'TXT', 'dual')",
        "INSERT INTO data VALUES (2, 1, 'EPUB', 'dual')",
    )

    books = list_books(library_path)

    assert books[0].file_path == library_path / "Author/Dual Format (1)" / "dual.epub"


def test_list_books_honors_custom_preferred_formats_order(tmp_path):
    library_path = _create_library(tmp_path)
    _run_sql(
        library_path,
        "INSERT INTO books VALUES (1, 'Dual Format', 'Author/Dual Format (1)')",
        "INSERT INTO data VALUES (1, 1, 'TXT', 'dual')",
        "INSERT INTO data VALUES (2, 1, 'EPUB', 'dual')",
    )

    books = list_books(library_path, preferred_formats=("TXT", "EPUB"))

    assert books[0].file_path == library_path / "Author/Dual Format (1)" / "dual.txt"


def test_list_books_excludes_book_with_only_non_preferred_format(tmp_path):
    library_path = _create_library(tmp_path)
    _run_sql(
        library_path,
        "INSERT INTO books VALUES (1, 'PDF Only', 'Author/PDF Only (1)')",
        "INSERT INTO data VALUES (1, 1, 'PDF', 'pdfonly')",
    )

    books = list_books(library_path)

    assert books == []


def test_list_books_orders_by_title_case_insensitive_ascending(tmp_path):
    library_path = _create_library(tmp_path)
    _run_sql(
        library_path,
        "INSERT INTO books VALUES (1, 'zebra', 'Author/zebra (1)')",
        "INSERT INTO books VALUES (2, 'Apple', 'Author/Apple (2)')",
        "INSERT INTO books VALUES (3, 'banana', 'Author/banana (3)')",
        "INSERT INTO data VALUES (1, 1, 'EPUB', 'zebra')",
        "INSERT INTO data VALUES (2, 2, 'EPUB', 'apple')",
        "INSERT INTO data VALUES (3, 3, 'EPUB', 'banana')",
    )

    books = list_books(library_path)

    assert [book.title for book in books] == ["Apple", "banana", "zebra"]


def test_list_books_returns_empty_list_when_library_has_no_books(tmp_path):
    library_path = _create_library(tmp_path)

    books = list_books(library_path)

    assert books == []
