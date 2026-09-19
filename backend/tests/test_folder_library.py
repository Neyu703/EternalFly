import pytest

from eternalfly.folder_library import FolderBook, list_books_in_folder


def test_list_books_in_folder_returns_epub_and_txt_files_sorted_by_name(tmp_path):
    (tmp_path / "Zebra.epub").write_bytes(b"")
    (tmp_path / "apple.txt").write_bytes(b"")
    (tmp_path / "Mango.EPUB").write_bytes(b"")

    books = list_books_in_folder(tmp_path)

    assert books == [
        FolderBook(file_name="apple.txt", file_path=tmp_path / "apple.txt"),
        FolderBook(file_name="Mango.EPUB", file_path=tmp_path / "Mango.EPUB"),
        FolderBook(file_name="Zebra.epub", file_path=tmp_path / "Zebra.epub"),
    ]


def test_list_books_in_folder_ignores_unsupported_file_extensions(tmp_path):
    (tmp_path / "book.epub").write_bytes(b"")
    (tmp_path / "cover.jpg").write_bytes(b"")
    (tmp_path / "notes.pdf").write_bytes(b"")

    books = list_books_in_folder(tmp_path)

    assert [book.file_name for book in books] == ["book.epub"]


def test_list_books_in_folder_ignores_subdirectories_not_recursive(tmp_path):
    (tmp_path / "top.epub").write_bytes(b"")
    subfolder = tmp_path / "subfolder"
    subfolder.mkdir()
    (subfolder / "nested.epub").write_bytes(b"")

    books = list_books_in_folder(tmp_path)

    assert [book.file_name for book in books] == ["top.epub"]


def test_list_books_in_folder_returns_empty_list_for_empty_folder(tmp_path):
    books = list_books_in_folder(tmp_path)

    assert books == []


def test_list_books_in_folder_raises_file_not_found_for_missing_folder(tmp_path):
    missing_folder = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError, match="No folder found"):
        list_books_in_folder(missing_folder)


def test_list_books_in_folder_raises_file_not_found_when_path_is_a_file(tmp_path):
    file_path = tmp_path / "not_a_folder.txt"
    file_path.write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="No folder found"):
        list_books_in_folder(file_path)
