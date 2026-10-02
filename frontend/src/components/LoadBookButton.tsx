import { useState } from "react";
import { isTauri } from "@tauri-apps/api/core";
import { open, type OpenDialogOptions } from "@tauri-apps/plugin-dialog";
import { BOOK_FILE_EXTENSIONS, loadBookByPath } from "../bookApi";
import { useKeyDown } from "../hooks/useKeyDown";
import { FolderBookList } from "./FolderBookList";
import { CloseIcon, FileIcon, FolderIcon } from "./icons";
import "./LoadBookButton.css";

/** What the buttons are doing: nothing, loading a picked file, showing why it failed, or
 * showing the books of a picked folder. The popover is open in the last two. */
type PickerState =
  | { kind: "closed" }
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "folder"; path: string };

const CLOSED: PickerState = { kind: "closed" };

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
  const [pickerState, setPickerState] = useState<PickerState>(CLOSED);
  const isPopoverOpen = pickerState.kind === "error" || pickerState.kind === "folder";
  const closePopover = () => setPickerState(CLOSED);

  useKeyDown("Escape", isPopoverOpen, closePopover);

  /** Opens a native picker and returns the chosen path, or null if cancelled or not
   * running inside the desktop app (a plain browser has no native dialogs). */
  async function pickPath(options: OpenDialogOptions): Promise<string | null> {
    if (!isTauri()) {
      setPickerState({ kind: "error", message: "File picking is only available in the desktop app." });
      return null;
    }
    const selectedPath = await open(options);
    return typeof selectedPath === "string" ? selectedPath : null;
  }

  async function handlePickFile() {
    const selectedPath = await pickPath({ multiple: false, filters: [{ name: "Book", extensions: BOOK_FILE_EXTENSIONS }] });
    if (!selectedPath) return;
    setPickerState({ kind: "loading" });
    const result = await loadBookByPath(selectedPath);
    setPickerState(result.ok ? CLOSED : { kind: "error", message: result.error });
  }

  async function handlePickFolder() {
    const selectedPath = await pickPath({ directory: true, multiple: false });
    if (selectedPath) setPickerState({ kind: "folder", path: selectedPath });
  }

  const isLoading = pickerState.kind === "loading";

  return (
    <div className="load-book">
      <button className="button" onClick={handlePickFile} disabled={isLoading}>
        {isLoading ? <span className="spinner spinner--small" aria-hidden="true" /> : <FileIcon />}
        {isLoading ? "Loading…" : "Load file"}
      </button>
      <button className="button" onClick={handlePickFolder} disabled={isLoading}>
        <FolderIcon />
        Load folder
      </button>

      {isPopoverOpen && (
        <div
          className="load-book-popover"
          role="dialog"
          aria-label={pickerState.kind === "folder" ? "Books in folder" : "Error"}
        >
          <div className="overlay-header load-book-popover-header">
            <div className="overlay-heading load-book-popover-heading">
              <span className="overline">{pickerState.kind === "folder" ? "Folder" : "Couldn't load the book"}</span>
              {pickerState.kind === "folder" && (
                <span className="load-book-popover-title truncate" title={pickerState.path}>
                  {lastPathSegment(pickerState.path)}
                </span>
              )}
            </div>
            <button className="icon-button" onClick={closePopover} aria-label="Close">
              <CloseIcon />
            </button>
          </div>
          {pickerState.kind === "error" && <p className="error-text">{pickerState.message}</p>}
          {pickerState.kind === "folder" && <FolderBookList key={pickerState.path} folderPath={pickerState.path} />}
        </div>
      )}
    </div>
  );
}
