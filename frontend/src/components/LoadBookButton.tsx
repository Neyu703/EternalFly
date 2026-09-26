import { useEffect, useState } from "react";
import { isTauri } from "@tauri-apps/api/core";
import { open, type OpenDialogOptions } from "@tauri-apps/plugin-dialog";
import { loadBookByPath } from "../hooks/useLoadBook";
import { FolderBookList } from "./FolderBookList";
import { CloseIcon, FileIcon, FolderIcon } from "./icons";
import "./LoadBookButton.css";

/** Last segment of a file-system path, used as a compact folder heading. */
function lastPathSegment(path: string): string {
  return path.split(/[\\/]/).filter(Boolean).at(-1) ?? path;
}

/** Two header buttons — one opening a native file picker for a single .txt/.epub, the
 * other a native folder picker whose .epub/.txt files are then listed in a popover to
 * choose from — either way handing the chosen path to the backend, which restarts the
 * reading position but keeps the simulated brain's ongoing state. Errors surface in the
 * same popover; it closes via its close button or Escape. */
export function LoadBookButton() {
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
  const [errorMessage, setErrorMessage] = useState("");
  const [selectedFolderPath, setSelectedFolderPath] = useState<string | null>(null);
  const isPopoverOpen = status === "error" || selectedFolderPath !== null;

  useEffect(() => {
    if (!isPopoverOpen) return;
    /** Closes the popover on Escape, wherever focus currently is. */
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") closePopover();
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isPopoverOpen]);

  function closePopover() {
    setStatus("idle");
    setErrorMessage("");
    setSelectedFolderPath(null);
  }

  function showError(message: string) {
    setStatus("error");
    setErrorMessage(message);
  }

  /** Opens a native picker and returns the chosen path, or null if cancelled or not
   * running inside the desktop app (a plain browser has no native dialogs). */
  async function pickPath(options: OpenDialogOptions): Promise<string | null> {
    if (!isTauri()) {
      showError("Die Dateiauswahl gibt es nur in der Desktop-App.");
      return null;
    }
    const selectedPath = await open(options);
    return typeof selectedPath === "string" ? selectedPath : null;
  }

  async function handlePickFile() {
    const selectedPath = await pickPath({ multiple: false, filters: [{ name: "Buch", extensions: ["txt", "epub"] }] });
    if (!selectedPath) return;
    setSelectedFolderPath(null);
    setStatus("loading");
    setErrorMessage("");
    const result = await loadBookByPath(selectedPath);
    if (result.ok) {
      setStatus("idle");
    } else {
      showError(result.error);
    }
  }

  async function handlePickFolder() {
    const selectedPath = await pickPath({ directory: true, multiple: false });
    if (!selectedPath) return;
    setStatus("idle");
    setErrorMessage("");
    setSelectedFolderPath(selectedPath);
  }

  const isLoading = status === "loading";

  return (
    <div className="load-book">
      <button className="button" onClick={handlePickFile} disabled={isLoading}>
        {isLoading ? <span className="spinner spinner--small" aria-hidden="true" /> : <FileIcon />}
        {isLoading ? "Lädt…" : "Datei laden"}
      </button>
      <button className="button" onClick={handlePickFolder} disabled={isLoading}>
        <FolderIcon />
        Ordner laden
      </button>

      {isPopoverOpen && (
        <div className="load-book-popover" role="dialog" aria-label={selectedFolderPath ? "Bücher im Ordner" : "Fehler"}>
          <div className="load-book-popover-header">
            <div className="load-book-popover-heading">
              <span className="overline">{selectedFolderPath ? "Ordner" : "Buch konnte nicht geladen werden"}</span>
              {selectedFolderPath && (
                <span className="load-book-popover-title" title={selectedFolderPath}>
                  {lastPathSegment(selectedFolderPath)}
                </span>
              )}
            </div>
            <button className="icon-button" onClick={closePopover} aria-label="Schließen">
              <CloseIcon />
            </button>
          </div>
          {status === "error" && <p className="error-text">{errorMessage}</p>}
          {selectedFolderPath && <FolderBookList folderPath={selectedFolderPath} />}
        </div>
      )}
    </div>
  );
}
