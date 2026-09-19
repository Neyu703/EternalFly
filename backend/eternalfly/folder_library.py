"""Read-only listing of loadable book files directly inside an arbitrary folder — for
loading an ad-hoc folder of .epub/.txt files, as opposed to a full Calibre library
with its own metadata.db (see calibre_library.py for that case)."""

import pathlib
from dataclasses import dataclass

SUPPORTED_SUFFIXES = (".epub", ".txt")


@dataclass(frozen=True)
class FolderBook:
    """A single loadable book file found directly inside a folder."""

    file_name: str
    file_path: pathlib.Path


def list_books_in_folder(folder_path: pathlib.Path) -> list[FolderBook]:
    """List .epub/.txt files directly inside folder_path (not recursive), sorted by
    file name case-insensitively. Raises FileNotFoundError if folder_path is not an
    existing directory."""
    if not folder_path.is_dir():
        raise FileNotFoundError(f"No folder found at {folder_path}")

    matching_files = [
        entry
        for entry in folder_path.iterdir()
        if entry.is_file() and entry.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    matching_files.sort(key=lambda entry: entry.name.lower())
    return [FolderBook(file_name=entry.name, file_path=entry) for entry in matching_files]
