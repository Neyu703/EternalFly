import { useState } from "react";
import { resetMemory } from "../hooks/useResetMemory";

/** Resets the fly's real KC->MBON synapses (its learned reactions to whatever it's
 * read so far) back to their un-learned initial weights, after a confirmation prompt -
 * this cannot be undone, so it's never fired without the user explicitly confirming. */
export function ResetMemoryButton() {
  const [status, setStatus] = useState<"idle" | "resetting" | "error">("idle");

  async function handleClick() {
    if (!window.confirm("Gedächtnis wirklich löschen? Alles Gelernte geht unwiderruflich verloren.")) return;
    setStatus("resetting");
    const result = await resetMemory();
    setStatus(result.ok ? "idle" : "error");
  }

  return (
    <button className="load-book-button" onClick={handleClick} disabled={status === "resetting"}>
      {status === "resetting" ? "Löscht…" : status === "error" ? "Fehler — erneut versuchen" : "Gedächtnis löschen"}
    </button>
  );
}
