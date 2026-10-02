import { useEffect, useState } from "react";
import { BOOK_FILE_EXTENSIONS, listBooksInFolder, type FolderBook } from "../bookApi";
import { BookPickList } from "./BookPickList";
import { LoadingText } from "./LoadingText";

const EMPTY_FOLDER_TEXT = `No ${BOOK_FILE_EXTENSIONS.map((extension) => `.${extension}`).join(" or ")} files in this folder.`;

/** Shown after the user picks a folder via LoadBookButton: lists the book files directly
 * inside it and lets them load one with one click. Give it a key per folder, so picking
 * another folder starts afresh instead of keeping the last folder's list or error. */
export function FolderBookList({ folderPath }: { folderPath: string }) {
  // null until the folder's listing arrives.
  const [books, setBooks] = useState<FolderBook[] | null>(null);
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    listBooksInFolder(folderPath).then((result) => (result.ok ? setBooks(result.books) : setErrorMessage(result.error)));
  }, [folderPath]);

  if (errorMessage) return <p className="error-text">{errorMessage}</p>;
  if (books === null) return <LoadingText>Looking for books…</LoadingText>;
  if (books.length === 0) return <p className="empty-text">{EMPTY_FOLDER_TEXT}</p>;

  return <BookPickList books={books.map((book) => ({ filePath: book.filePath, title: book.fileName }))} />;
}
