import pathlib

import pytest
from calibre_fixtures import add_calibre_rows, create_calibre_library, dune_rows

from eternalfly.calibre_library import PREFERRED_FORMATS, CalibreBook, configured_library_path, list_books


def test_list_books_raises_file_not_found_when_metadata_db_missing(tmp_path):
    missing_library_path = tmp_path / "no_such_library"

    with pytest.raises(FileNotFoundError) as exc_info:
        list_books(missing_library_path)

    assert str(missing_library_path) in str(exc_info.value)


def test_list_books_returns_title_author_and_resolved_path_for_single_book(tmp_path):
    library_path = create_calibre_library(tmp_path / "library", *dune_rows("EPUB"))

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
    library_path = create_calibre_library(tmp_path / "library")
    add_calibre_rows(
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


def test_list_books_uses_unknown_for_book_with_no_author_link(tmp_path):
    library_path = create_calibre_library(tmp_path / "library")
    add_calibre_rows(
        library_path,
        "INSERT INTO books VALUES (1, 'Anonymous Work', 'Unknown/Anonymous Work (1)')",
        "INSERT INTO data VALUES (1, 1, 'TXT', 'anon')",
    )

    books = list_books(library_path)

    assert books[0].author == "Unknown"


def test_list_books_prefers_epub_over_txt_by_default(tmp_path):
    library_path = create_calibre_library(tmp_path / "library")
    add_calibre_rows(
        library_path,
        "INSERT INTO books VALUES (1, 'Dual Format', 'Author/Dual Format (1)')",
        "INSERT INTO data VALUES (1, 1, 'TXT', 'dual')",
        "INSERT INTO data VALUES (2, 1, 'EPUB', 'dual')",
    )

    books = list_books(library_path)

    assert books[0].file_path == library_path / "Author/Dual Format (1)" / "dual.epub"


def test_list_books_falls_back_to_txt_when_no_epub_is_available(tmp_path):
    library_path = create_calibre_library(tmp_path / "library")
    add_calibre_rows(
        library_path,
        "INSERT INTO books VALUES (1, 'Text Only', 'Author/Text Only (1)')",
        "INSERT INTO data VALUES (1, 1, 'TXT', 'textonly')",
    )

    books = list_books(library_path)

    assert books[0].file_path == library_path / "Author/Text Only (1)" / "textonly.txt"


def test_preferred_formats_are_the_supported_book_suffixes_in_calibre_naming():
    assert PREFERRED_FORMATS == ("EPUB", "TXT")


def test_list_books_excludes_book_with_only_non_preferred_format(tmp_path):
    library_path = create_calibre_library(tmp_path / "library")
    add_calibre_rows(
        library_path,
        "INSERT INTO books VALUES (1, 'PDF Only', 'Author/PDF Only (1)')",
        "INSERT INTO data VALUES (1, 1, 'PDF', 'pdfonly')",
    )

    books = list_books(library_path)

    assert books == []


def test_list_books_orders_by_title_case_insensitive_ascending(tmp_path):
    library_path = create_calibre_library(tmp_path / "library")
    add_calibre_rows(
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
    library_path = create_calibre_library(tmp_path / "library")

    books = list_books(library_path)

    assert books == []


def _write_calibre_settings(config_directory: pathlib.Path, settings_text: str) -> None:
    """Write Calibre's global settings file (global.py.json) with the given raw content."""
    config_directory.mkdir(parents=True, exist_ok=True)
    (config_directory / "global.py.json").write_text(settings_text, encoding="utf-8")


def test_configured_library_path_returns_the_library_calibre_currently_uses(tmp_path):
    config_directory = tmp_path / "calibre"
    _write_calibre_settings(config_directory, r'{"library_path": "E:\\Calibre-Bibliothek", "language": "de"}')

    assert configured_library_path(config_directory) == pathlib.Path("E:/Calibre-Bibliothek")


def test_configured_library_path_returns_none_without_calibre_settings(tmp_path):
    assert configured_library_path(tmp_path / "no_calibre_here") is None


def test_configured_library_path_returns_none_for_unreadable_settings(tmp_path):
    config_directory = tmp_path / "calibre"
    _write_calibre_settings(config_directory, '{"library_path": ')

    assert configured_library_path(config_directory) is None


@pytest.mark.parametrize("settings_text", ['{"language": "de"}', '{"library_path": ""}', '{"library_path": 42}', "[1, 2]"])
def test_configured_library_path_returns_none_when_no_library_is_named(tmp_path, settings_text):
    config_directory = tmp_path / "calibre"
    _write_calibre_settings(config_directory, settings_text)

    assert configured_library_path(config_directory) is None
