"""Builds Calibre-shaped libraries (a metadata.db with Calibre's tables) for tests."""

import pathlib
import sqlite3

CALIBRE_SCHEMA = """
    CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT, path TEXT);
    CREATE TABLE authors (id INTEGER PRIMARY KEY, name TEXT);
    CREATE TABLE books_authors_link (book INTEGER, author INTEGER);
    CREATE TABLE data (id INTEGER PRIMARY KEY, book INTEGER, format TEXT, name TEXT);
"""


def dune_rows(book_format: str) -> tuple[str, ...]:
    """The rows of a library holding Frank Herbert's Dune as a book_format ("EPUB", "TXT")
    file named dune, stored under Frank Herbert/Dune (1)."""
    return (
        "INSERT INTO books VALUES (1, 'Dune', 'Frank Herbert/Dune (1)')",
        "INSERT INTO authors VALUES (1, 'Frank Herbert')",
        "INSERT INTO books_authors_link VALUES (1, 1)",
        f"INSERT INTO data VALUES (1, 1, '{book_format}', 'dune')",
    )


def add_calibre_rows(library_path: pathlib.Path, *insert_statements: str) -> None:
    """Execute INSERT statements against the metadata.db of the library at library_path."""
    connection = sqlite3.connect(library_path / "metadata.db")
    for statement in insert_statements:
        connection.execute(statement)
    connection.commit()
    connection.close()


def create_calibre_library(library_path: pathlib.Path, *insert_statements: str) -> pathlib.Path:
    """Create library_path (with any missing parents) holding a metadata.db with Calibre's
    tables and the rows of insert_statements, and return library_path."""
    library_path.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(library_path / "metadata.db")
    connection.executescript(CALIBRE_SCHEMA)
    connection.commit()
    connection.close()
    add_calibre_rows(library_path, *insert_statements)
    return library_path
