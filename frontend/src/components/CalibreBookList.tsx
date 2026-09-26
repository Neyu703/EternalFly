import { useEffect, useState } from "react";
import { loadBookByPath } from "../hooks/useLoadBook";
import { BACKEND_HTTP_URL } from "../backendUrl";
import "./BookList.css";

type CalibreBook = { book_id: number; title: string; author: string; file_path: string };

/** Shown when the fly's engagement drops low enough to want a different book: fetches the
 * user's Calibre library (read-only) and lets them pick a replacement with one click.
 * Renders nothing when no library is configured or it has no loadable books. */
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

  if (errorMessage) return <p className="error-text">{errorMessage}</p>;
  if (books.length === 0) return null;

  return (
    <>
      <span className="overline">Aus deiner Calibre-Bibliothek</span>
      <ul className="book-list">
        {books.map((book) => (
          <li key={book.book_id}>
            <button
              className="book-list-item"
              disabled={loadingPath === book.file_path}
              onClick={() => handlePick(book.file_path)}
            >
              <span className="book-list-title">{book.title}</span>
              <span className="book-list-meta">{loadingPath === book.file_path ? "Lädt…" : book.author}</span>
            </button>
          </li>
        ))}
      </ul>
    </>
  );
}
