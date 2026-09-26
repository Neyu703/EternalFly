import { LoadBookButton } from "./LoadBookButton";
import { FlyMark } from "./icons";
import "./AppHeader.css";

/** State of the WebSocket link to the simulation, as shown in the header. */
export type ConnectionStatus = "connecting" | "live" | "paused" | "disconnected";

const STATUS_LABELS: Record<ConnectionStatus, string> = {
    connecting: "Verbinde…",
    live: "Live",
    paused: "Pausiert",
    disconnected: "Getrennt – verbinde neu…",
};

/** Top bar: brand, the simulation's connection status (dot + label, never color alone),
 * and the book-loading actions. */
export function AppHeader({ connectionStatus }: { connectionStatus: ConnectionStatus }) {
    return (
        <header className="app-header">
            <div className="app-brand">
                <FlyMark className="app-brand-mark" />
                <h1 className="app-brand-name">EternalFly</h1>
                <span className="app-brand-tagline">Eine simulierte Fruchtfliege liest – mit ihrem echten Gehirn</span>
            </div>
            <span className={`status-pill status-pill--${connectionStatus}`} role="status">
                <span className="status-pill-dot" aria-hidden="true" />
                {STATUS_LABELS[connectionStatus]}
            </span>
            <LoadBookButton />
        </header>
    );
}
