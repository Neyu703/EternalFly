import { BACKEND_HTTP_URL } from "../backendUrl";

export type LoadBookResult = { ok: true; totalWords: number } | { ok: false; error: string };

export type FolderBook = { fileName: string; filePath: string };
export type ListBooksInFolderResult = { ok: true; books: FolderBook[] } | { ok: false; error: string };

/** GETs the backend's /books-in-folder for the .epub/.txt files directly inside
 * folderPath, so the user can load an ad-hoc folder of books instead of a full
 * Calibre library. */
export async function listBooksInFolder(folderPath: string): Promise<ListBooksInFolderResult> {
  try {
    const response = await fetch(`${BACKEND_HTTP_URL}/books-in-folder?path=${encodeURIComponent(folderPath)}`);
    if (!response.ok) {
      const body = await response.json().catch(() => null);
      return { ok: false, error: body?.detail ?? `Fehler ${response.status}` };
    }
    const body = await response.json();
    const books: FolderBook[] = body.books.map((book: { file_name: string; file_path: string }) => ({
      fileName: book.file_name,
      filePath: book.file_path,
    }));
    return { ok: true, books };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : "Unbekannter Fehler" };
  }
}

/** POSTs an absolute file path to the backend's /load-book endpoint, swapping it in as
 * the session's current book. Shared by the manual file picker and the Calibre book list. */
export async function loadBookByPath(path: string): Promise<LoadBookResult> {
  try {
    const response = await fetch(`${BACKEND_HTTP_URL}/load-book`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => null);
      return { ok: false, error: body?.detail ?? `Fehler ${response.status}` };
    }
    const body = await response.json();
    return { ok: true, totalWords: body.total_words };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : "Unbekannter Fehler" };
  }
}
