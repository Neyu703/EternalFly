import { useState } from "react";
import { loadBookByPath } from "../bookApi";
import "./BookList.css";

/** One book to offer: the file to load, its title and an optional second line. */
export type BookChoice = { filePath: string; title: string; subtitle?: string };

/** Books to load with one click each. While one loads, its second line says so; a failed
 * load replaces the list with the error. */
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
                const meta = isLoading ? "Loading…" : book.subtitle;
                return (
                    <li key={book.filePath}>
                        <button className="book-list-item" disabled={isLoading} onClick={() => handlePick(book.filePath)}>
                            <span className="book-list-title truncate">{book.title}</span>
                            {meta && <span className="book-list-meta truncate">{meta}</span>}
                        </button>
                    </li>
                );
            })}
        </ul>
    );
}
