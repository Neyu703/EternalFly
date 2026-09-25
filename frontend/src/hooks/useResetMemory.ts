import { BACKEND_HTTP_URL } from "../backendUrl";

export type ResetMemoryResult = { ok: true } | { ok: false; error: string };

/** POSTs to the backend's /reset-memory endpoint, resetting the fly's real KC->MBON
 * synapses back to their un-learned initial weights and deleting any persisted
 * memory.npz (see reading_session.py's ReadingSession.reset_memory). */
export async function resetMemory(): Promise<ResetMemoryResult> {
  try {
    const response = await fetch(`${BACKEND_HTTP_URL}/reset-memory`, { method: "POST" });
    if (!response.ok) {
      const body = await response.json().catch(() => null);
      return { ok: false, error: body?.detail ?? `Fehler ${response.status}` };
    }
    return { ok: true };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : "Unbekannter Fehler" };
  }
}
