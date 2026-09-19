import { useEffect, useState } from "react";
import { listBooksInFolder, loadBookByPath, type FolderBook } from "../hooks/useLoadBook";
import "./CalibreBookList.css";

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

  if (errorMessage) return <span className="calibre-list-error">{errorMessage}</span>;
  if (books.length === 0) return <span className="calibre-list-error">Keine .epub/.txt Dateien gefunden.</span>;

  return (
    <ul className="calibre-list">
      {books.map((book) => (
        <li key={book.filePath}>
          <button
            className="calibre-list-item"
            disabled={loadingPath === book.filePath}
            onClick={() => handlePick(book.filePath)}
          >
            {book.fileName}
          </button>
        </li>
      ))}
    </ul>
  );
}
