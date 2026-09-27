import { BACKEND_HTTP_URL } from "../backendUrl";

/** The book file types the backend can read, most preferred first. */
export const BOOK_FILE_EXTENSIONS = ["epub", "txt"];

export type LoadBookResult = { ok: true } | { ok: false; error: string };

export type FolderBook = { fileName: string; filePath: string };
export type ListBooksInFolderResult = { ok: true; books: FolderBook[] } | { ok: false; error: string };

export type CalibreBook = { bookId: number; title: string; author: string; filePath: string };
export type ListCalibreBooksResult = { ok: true; books: CalibreBook[] } | { ok: false; error: string };

/** What a backend request returned: its JSON body, or the error to show and, when the
 * backend answered with an error status (rather than being unreachable), that status. */
type BackendJsonResult<Body> = { ok: true; body: Body } | { ok: false; error: string; status?: number };

/** Requests path from the backend and returns its JSON body, or the backend's error
 * detail (for an error status) or the network error. */
async function requestBackendJson<Body>(path: string, init?: RequestInit): Promise<BackendJsonResult<Body>> {
  try {
    const response = await fetch(`${BACKEND_HTTP_URL}${path}`, init);
    if (!response.ok) {
      const body = await response.json().catch(() => null);
      return { ok: false, error: body?.detail ?? `Error ${response.status}`, status: response.status };
    }
    return { ok: true, body: await response.json() };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : "Unknown error" };
  }
}

/** GETs the backend's /books-in-folder for the .epub/.txt files directly inside
 * folderPath, so the user can load an ad-hoc folder of books instead of a full
 * Calibre library. */
export async function listBooksInFolder(folderPath: string): Promise<ListBooksInFolderResult> {
  const result = await requestBackendJson<{ books: { file_name: string; file_path: string }[] }>(
    `/books-in-folder?path=${encodeURIComponent(folderPath)}`,
  );
  if (!result.ok) return result;
  return {
    ok: true,
    books: result.body.books.map((book) => ({ fileName: book.file_name, filePath: book.file_path })),
  };
}

/** GETs the books of the backend's configured Calibre library (read-only). A library the
 * backend can't open reads as an empty one; only an unreachable backend is an error. */
export async function listCalibreBooks(): Promise<ListCalibreBooksResult> {
  const result = await requestBackendJson<{
    books?: { book_id: number; title: string; author: string; file_path: string }[];
  }>("/calibre-books");
  if (!result.ok) {
    return result.status !== undefined ? { ok: true, books: [] } : { ok: false, error: "Couldn't load the Calibre library." };
  }
  return {
    ok: true,
    books: (result.body.books ?? []).map((book) => ({
      bookId: book.book_id,
      title: book.title,
      author: book.author,
      filePath: book.file_path,
    })),
  };
}

/** POSTs an absolute file path to the backend's /load-book endpoint, swapping it in as
 * the session's current book. Shared by the manual file picker and the book lists. */
export async function loadBookByPath(path: string): Promise<LoadBookResult> {
  const result = await requestBackendJson("/load-book", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  return result.ok ? { ok: true } : { ok: false, error: result.error };
}
