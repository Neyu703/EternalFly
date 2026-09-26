import { useEffect, useState } from "react";
import { listBooksInFolder, loadBookByPath, type FolderBook } from "../hooks/useLoadBook";
import "./BookList.css";

/** Shown after the user picks a folder via LoadBookButton: lists the .epub/.txt files
 * directly inside it and lets them load one with one click. */
export function FolderBookList({ folderPath }: { folderPath: string }) {
  const [books, setBooks] = useState<FolderBook[]>([]);
  const [errorMessage, setErrorMessage] = useState("");
  const [loadingPath, setLoadingPath] = useState<string | null>(null);

  useEffect(() => {
    listBooksInFolder(folderPath).then((result) => {
      if (result.ok) {
        setBooks(result.books);
      } else {
        setErrorMessage(result.error);
      }
    });
  }, [folderPath]);

  async function handlePick(filePath: string) {
    setLoadingPath(filePath);
    const result = await loadBookByPath(filePath);
    setLoadingPath(null);
    if (!result.ok) setErrorMessage(result.error);
  }

  if (errorMessage) return <p className="error-text">{errorMessage}</p>;
  if (books.length === 0) return <p className="book-list-empty">Keine .epub- oder .txt-Dateien in diesem Ordner.</p>;

  return (
    <ul className="book-list">
      {books.map((book) => (
        <li key={book.filePath}>
          <button
            className="book-list-item"
            disabled={loadingPath === book.filePath}
            onClick={() => handlePick(book.filePath)}
          >
            <span className="book-list-title">{book.fileName}</span>
            {loadingPath === book.filePath && <span className="book-list-meta">Lädt…</span>}
          </button>
        </li>
      ))}
    </ul>
  );
}
