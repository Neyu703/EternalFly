import { useEffect, useState } from "react";
import { loadBookByPath } from "../hooks/useLoadBook";
import { BACKEND_HTTP_URL } from "../backendUrl";
import "./CalibreBookList.css";

type CalibreBook = { book_id: number; title: string; author: string; file_path: string };

/** Shown when the fly's engagement drops low enough to want a different book: fetches the
 * user's Calibre library (read-only) and lets them pick a replacement with one click. */
export function CalibreBookList() {
  const [books, setBooks] = useState<CalibreBook[]>([]);
  const [errorMessage, setErrorMessage] = useState("");
  const [loadingPath, setLoadingPath] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${BACKEND_HTTP_URL}/calibre-books`)
      .then((response) => response.json())
      .then((body) => setBooks(body.books ?? []))
      .catch(() => setErrorMessage("Calibre-Bibliothek konnte nicht geladen werden."));
  }, []);

  async function handlePick(filePath: string) {
    setLoadingPath(filePath);
    const result = await loadBookByPath(filePath);
    setLoadingPath(null);
    if (!result.ok) setErrorMessage(result.error);
  }

  if (errorMessage) return <span className="calibre-list-error">{errorMessage}</span>;
  if (books.length === 0) return null;

  return (
    <ul className="calibre-list">
      {books.map((book) => (
        <li key={book.book_id}>
          <button
            className="calibre-list-item"
            disabled={loadingPath === book.file_path}
            onClick={() => handlePick(book.file_path)}
          >
            {book.title} — {book.author}
          </button>
        </li>
      ))}
    </ul>
  );
}
