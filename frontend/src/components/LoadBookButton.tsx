import { useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { loadBookByPath } from "../hooks/useLoadBook";
import { FolderBookList } from "./FolderBookList";
import "./LoadBookButton.css";

/** Two buttons — one opening a native file picker for a single .txt/.epub, the other a
 * native folder picker whose .epub/.txt files are then listed to choose from — either
 * way handing the chosen path to the backend, which restarts the reading position but
 * keeps the simulated brain's ongoing state. */
export function LoadBookButton() {
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
  const [errorMessage, setErrorMessage] = useState("");
  const [selectedFolderPath, setSelectedFolderPath] = useState<string | null>(null);

  async function loadBookAtPath(path: string) {
    setStatus("loading");
    setErrorMessage("");
    const result = await loadBookByPath(path);
    if (result.ok) {
      setStatus("idle");
    } else {
      setStatus("error");
      setErrorMessage(result.error);
    }
  }

  async function handlePickFile() {
    const selectedPath = await open({
      multiple: false,
      filters: [{ name: "Buch", extensions: ["txt", "epub"] }],
    });
    if (!selectedPath || Array.isArray(selectedPath)) return;
    setSelectedFolderPath(null);
    await loadBookAtPath(selectedPath);
  }

  async function handlePickFolder() {
    const selectedPath = await open({ directory: true, multiple: false });
    if (!selectedPath || Array.isArray(selectedPath)) return;
    setStatus("idle");
    setErrorMessage("");
    setSelectedFolderPath(selectedPath);
  }

  return (
    <div className="load-book">
      <div className="load-book-buttons">
        <button className="load-book-button" onClick={handlePickFile} disabled={status === "loading"}>
          {status === "loading" ? "Lädt…" : "Datei laden"}
        </button>
        <button className="load-book-button" onClick={handlePickFolder} disabled={status === "loading"}>
          Ordner laden
        </button>
      </div>
      {status === "error" && <span className="load-book-error">{errorMessage}</span>}
      {selectedFolderPath && <FolderBookList folderPath={selectedFolderPath} />}
    </div>
  );
}
