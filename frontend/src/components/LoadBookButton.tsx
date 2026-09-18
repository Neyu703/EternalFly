import { useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import "./LoadBookButton.css";

const BACKEND_HTTP_URL = "http://127.0.0.1:8000";

/** Button that opens a native file picker (.txt/.epub) and hands the chosen path to the
 * backend, which restarts the reading position but keeps the simulated brain's ongoing state. */
export function LoadBookButton() {
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
  const [errorMessage, setErrorMessage] = useState("");

  async function handleClick() {
    const selectedPath = await open({
      multiple: false,
      filters: [{ name: "Buch", extensions: ["txt", "epub"] }],
    });
    if (!selectedPath || Array.isArray(selectedPath)) return;

    setStatus("loading");
    setErrorMessage("");
    try {
      const response = await fetch(`${BACKEND_HTTP_URL}/load-book`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: selectedPath }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail ?? `Fehler ${response.status}`);
      }
      setStatus("idle");
    } catch (error) {
      setStatus("error");
      setErrorMessage(error instanceof Error ? error.message : "Unbekannter Fehler");
    }
  }

  return (
    <div className="load-book">
      <button className="load-book-button" onClick={handleClick} disabled={status === "loading"}>
        {status === "loading" ? "Lädt…" : "Buch laden"}
      </button>
      {status === "error" && <span className="load-book-error">{errorMessage}</span>}
    </div>
  );
}
