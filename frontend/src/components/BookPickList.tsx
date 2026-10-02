import { useState } from "react";
import { loadBookByPath } from "../bookApi";
import "./BookList.css";

/** One book to offer: the file to load, its title and an optional second line. */
export type BookChoice = { filePath: string; title: string; subtitle?: string };

/** Books to load with one click each. While one loads, its second line says so with a
 * spinner and the others are disabled; a failed load replaces the list with the error. */
export function BookPickList({ books }: { books: BookChoice[] }) {
    const [loadingPath, setLoadingPath] = useState<string | null>(null);
    const [errorMessage, setErrorMessage] = useState("");

    async function handlePick(filePath: string) {
        setLoadingPath(filePath);
        const result = await loadBookByPath(filePath);
        setLoadingPath(null);
        if (!result.ok) setErrorMessage(result.error);
    }

    if (errorMessage) return <p className="error-text">{errorMessage}</p>;

    return (
        <ul className="book-list">
            {books.map((book) => {
                const isLoading = loadingPath === book.filePath;
                return (
                    <li key={book.filePath}>
                        <button
                            className="book-list-item"
                            disabled={loadingPath !== null}
                            aria-busy={isLoading}
                            onClick={() => handlePick(book.filePath)}
                        >
                            <span className="book-list-title truncate">{book.title}</span>
                            {isLoading ? (
                                <span className="book-list-meta">
                                    <span className="spinner" aria-hidden="true" />
                                    Loading…
                                </span>
                            ) : (
                                book.subtitle && <span className="book-list-meta truncate">{book.subtitle}</span>
                            )}
                        </button>
                    </li>
                );
            })}
        </ul>
    );
}
