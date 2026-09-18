import { BACKEND_HTTP_URL } from "../backendUrl";

export type LoadBookResult = { ok: true; totalWords: number } | { ok: false; error: string };

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
