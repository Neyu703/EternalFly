import { useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { loadBookByPath } from "../hooks/useLoadBook";
import "./LoadBookButton.css";

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
    const result = await loadBookByPath(selectedPath);
    if (result.ok) {
      setStatus("idle");
    } else {
      setStatus("error");
      setErrorMessage(result.error);
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
