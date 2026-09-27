import { useEffect, useState } from "react";
import { listCalibreBooks, type CalibreBook } from "../bookApi";
import { BookPickList } from "./BookPickList";

/** Shown when the fly's engagement drops low enough to want a different book: fetches the
 * user's Calibre library (read-only) and lets them pick a replacement with one click.
 * Renders nothing when no library is configured or it has no loadable books. */
export function CalibreBookList() {
  const [books, setBooks] = useState<CalibreBook[]>([]);
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    listCalibreBooks().then((result) => (result.ok ? setBooks(result.books) : setErrorMessage(result.error)));
  }, []);

  if (errorMessage) return <p className="error-text">{errorMessage}</p>;
  if (books.length === 0) return null;

  return (
    <>
      <span className="overline">From your Calibre library</span>
      <BookPickList books={books.map((book) => ({ filePath: book.filePath, title: book.title, subtitle: book.author }))} />
    </>
  );
}
