import { useEffect, useEffectEvent } from "react";

/** Calls onKeyDown whenever key (a KeyboardEvent.key value, e.g. "Escape" or " ") is
 * pressed while isActive. The listener is only re-registered when isActive or key
 * changes, not whenever onKeyDown does. */
export function useKeyDown(key: string, isActive: boolean, onKeyDown: (event: KeyboardEvent) => void): void {
    const handleKeyDown = useEffectEvent(onKeyDown);

    useEffect(() => {
        if (!isActive) return;
        /** Forwards presses of key to the latest onKeyDown. */
        function handleDocumentKeyDown(event: KeyboardEvent) {
            if (event.key === key) handleKeyDown(event);
        }
        document.addEventListener("keydown", handleDocumentKeyDown);
        return () => document.removeEventListener("keydown", handleDocumentKeyDown);
    }, [isActive, key]);
}
