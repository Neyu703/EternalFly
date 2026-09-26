import type { ReactNode } from "react";

/** Shared 16px outline-icon frame: decorative (aria-hidden), inherits the text color. */
function Icon({ children }: { children: ReactNode }) {
    return (
        <svg
            width="16"
            height="16"
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
        >
            {children}
        </svg>
    );
}

/** Solid right-pointing triangle, for "resume". */
export function PlayIcon() {
    return (
        <Icon>
            <path d="M5 3.5v9l7.5-4.5z" fill="currentColor" stroke="none" />
        </Icon>
    );
}

/** Two solid bars, for "pause". */
export function PauseIcon() {
    return (
        <Icon>
            <rect x="4" y="3.5" width="2.75" height="9" rx="0.75" fill="currentColor" stroke="none" />
            <rect x="9.25" y="3.5" width="2.75" height="9" rx="0.75" fill="currentColor" stroke="none" />
        </Icon>
    );
}

/** Document with a folded corner, for "load a file". */
export function FileIcon() {
    return (
        <Icon>
            <path d="M9 2H4.5A1.5 1.5 0 0 0 3 3.5v9A1.5 1.5 0 0 0 4.5 14h7a1.5 1.5 0 0 0 1.5-1.5V6z" />
            <path d="M9 2v4h4" />
        </Icon>
    );
}

/** Folder outline, for "load a folder". */
export function FolderIcon() {
    return (
        <Icon>
            <path d="M2 4.5A1.5 1.5 0 0 1 3.5 3h2.8l1.5 1.5h4.7A1.5 1.5 0 0 1 14 6v5.5a1.5 1.5 0 0 1-1.5 1.5h-9A1.5 1.5 0 0 1 2 11.5z" />
        </Icon>
    );
}

/** Diagonal cross, for "close". */
export function CloseIcon() {
    return (
        <Icon>
            <path d="M4 4l8 8M12 4l-8 8" />
        </Icon>
    );
}

/** EternalFly brand mark: a stylized fly seen from above, two translucent wings behind
 * a solid head and body. */
export function FlyMark({ className }: { className?: string }) {
    return (
        <svg className={className} width="24" height="24" viewBox="0 0 24 24" aria-hidden="true">
            <ellipse cx="7.6" cy="10.4" rx="5.8" ry="3" transform="rotate(-32 7.6 10.4)" fill="currentColor" opacity="0.4" />
            <ellipse cx="16.4" cy="10.4" rx="5.8" ry="3" transform="rotate(32 16.4 10.4)" fill="currentColor" opacity="0.4" />
            <ellipse cx="12" cy="14.6" rx="3" ry="5.4" fill="currentColor" />
            <circle cx="12" cy="7.4" r="2.5" fill="currentColor" />
        </svg>
    );
}
