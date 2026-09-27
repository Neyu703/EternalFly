import { useEffect, useEffectEvent } from "react";

/** Calls onEscape whenever Escape is pressed while isActive. The listener is only
 * re-registered when isActive changes, not whenever onEscape does. */
export function useEscapeKey(isActive: boolean, onEscape: () => void): void {
    const handleEscape = useEffectEvent(onEscape);

    useEffect(() => {
        if (!isActive) return;
        /** Forwards Escape presses to the latest onEscape. */
        function handleKeyDown(event: KeyboardEvent) {
            if (event.key === "Escape") handleEscape();
        }
        document.addEventListener("keydown", handleKeyDown);
        return () => document.removeEventListener("keydown", handleKeyDown);
    }, [isActive]);
}
