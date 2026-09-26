import type { TickData } from "../types";
import { formatInteger, formatPercent } from "../utils/format";
import "./ReadingStatus.css";

/** Footer of the reader stage: the word the fly is reading right now and how far
 * through the book it is. */
export function ReadingStatus({ tick }: { tick: TickData }) {
    const progressPercent = Math.round(tick.pageProgress * 100);
    const currentWordText = tick.bookFinished ? "Buch beendet" : (tick.currentWord ?? "—");

    return (
        <div className="reading-status">
            <div className="reading-status-row">
                <div className="reading-status-word-block">
                    <span className="overline">Liest gerade</span>
                    <span className="reading-status-word">{currentWordText}</span>
                </div>
                <span className="reading-status-count">
                    {formatInteger(tick.wordsRead)} / {formatInteger(tick.totalWords)} Wörter · {formatPercent(tick.pageProgress)}
                </span>
            </div>
            <div
                className="progress"
                role="progressbar"
                aria-label="Lesefortschritt"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={progressPercent}
            >
                <div className="progress-fill" style={{ width: `${tick.pageProgress * 100}%` }} />
            </div>
        </div>
    );
}
