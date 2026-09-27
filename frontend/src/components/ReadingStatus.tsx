import type { TickData } from "../types";
import { formatInteger, formatPercent } from "../utils/format";
import "./ReadingStatus.css";

/** Footer of the reader stage: the word the fly is reading right now and how far
 * through the book it is. */
export function ReadingStatus({ tick }: { tick: TickData }) {
    const progressPercent = Math.round(tick.pageProgress * 100);
    const currentWordText = tick.bookFinished ? "Book finished" : (tick.currentWord ?? "—");

    return (
        <div className="reading-status">
            <div className="reading-status-row">
                <div className="reading-status-word-block">
                    <span className="overline">Now reading</span>
                    <span className="reading-status-word truncate">{currentWordText}</span>
                </div>
                <span className="reading-status-count">
                    {formatInteger(tick.wordsRead)} / {formatInteger(tick.totalWords)} words · {formatPercent(tick.pageProgress)}
                </span>
            </div>
            <div
                className="progress"
                role="progressbar"
                aria-label="Reading progress"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={progressPercent}
            >
                <div className="progress-fill reading-status-progress-fill" style={{ width: `${tick.pageProgress * 100}%` }} />
            </div>
        </div>
    );
}
