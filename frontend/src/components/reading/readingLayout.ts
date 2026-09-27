/** Where each word the fly reads sits on the open book, in the book model's own frame:
 * x runs across the spread (spine at 0, left page negative), z runs from a page's top
 * edge (far from the camera, negative) to its bottom edge, y is up. The real book text
 * is mapped onto an endless sequence of two-page spreads so the fly visibly moves along
 * lines, changes pages and turns them as wordsRead grows. */

export const WORDS_PER_LINE = 6;
export const LINES_PER_PAGE = 7;
export const WORDS_PER_PAGE = WORDS_PER_LINE * LINES_PER_PAGE;
export const WORDS_PER_SPREAD = WORDS_PER_PAGE * 2;

export type PageSide = "left" | "right";

/** Text block of one page, measured from the spine outward (|x|) and top to bottom (z). */
export const TEXT_INNER_X = 0.075;
export const TEXT_OUTER_X = 0.33;
export const TEXT_TOP_Z = -0.235;
export const TEXT_BOTTOM_Z = 0.235;
export const LINE_SPACING_Z = (TEXT_BOTTOM_Z - TEXT_TOP_Z) / LINES_PER_PAGE;
export const WORD_DEPTH_Z = LINE_SPACING_Z * 0.34;

const WORD_GAP_X = 0.012;

/** One word's rectangle on the page surface. */
export type WordBox = { centerX: number; centerZ: number; width: number };

/** A word's place in the spread sequence. */
export type ReadingPosition = { spreadIndex: number; side: PageSide; lineIndex: number; wordIndex: number };

/** Maps the count of words read so far onto the spread, page, line and word being read. */
export function readingPositionOf(wordsRead: number): ReadingPosition {
    const wordIndexInBook = Math.max(0, Math.floor(wordsRead));
    const spreadIndex = Math.floor(wordIndexInBook / WORDS_PER_SPREAD);
    const wordInSpread = wordIndexInBook % WORDS_PER_SPREAD;
    const side: PageSide = wordInSpread < WORDS_PER_PAGE ? "left" : "right";
    const wordInPage = wordInSpread % WORDS_PER_PAGE;
    return {
        spreadIndex,
        side,
        lineIndex: Math.floor(wordInPage / WORDS_PER_LINE),
        wordIndex: wordInPage % WORDS_PER_LINE,
    };
}

/** Left edge (lowest x) of a page's text block; text always flows toward +x. */
export function textStartX(side: PageSide): number {
    return side === "left" ? -TEXT_OUTER_X : TEXT_INNER_X;
}

/** Small deterministic PRNG (mulberry32), so a given page always shows the same text. */
function seededRandom(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
        state = (state + 0x6d2b79f5) >>> 0;
        let mixed = Math.imul(state ^ (state >>> 15), 1 | state);
        mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed;
        return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
    };
}

/** The word rectangles of one line: varied but stable widths that fill the line. */
export function lineWordBoxes(spreadIndex: number, side: PageSide, lineIndex: number): WordBox[] {
    const random = seededRandom(spreadIndex * 1009 + (side === "left" ? 0 : 499) + lineIndex * 31 + 7);
    const relativeWidths = Array.from({ length: WORDS_PER_LINE }, () => 0.55 + random() * 0.9);
    const widthSum = relativeWidths.reduce((sum, width) => sum + width, 0);
    const availableWidth = TEXT_OUTER_X - TEXT_INNER_X - WORD_GAP_X * (WORDS_PER_LINE - 1);
    const centerZ = TEXT_TOP_Z + (lineIndex + 0.5) * LINE_SPACING_Z;

    let cursorX = textStartX(side);
    return relativeWidths.map((relativeWidth) => {
        const width = (relativeWidth / widthSum) * availableWidth;
        const box = { centerX: cursorX + width / 2, centerZ, width };
        cursorX += width + WORD_GAP_X;
        return box;
    });
}

/** The word under a fractional word index: between two words of a line it slides from
 * one to the next; at a line's last word it holds until the next line begins. */
export function wordBoxAtIndex(wordIndex: number): WordBox {
    const position = readingPositionOf(wordIndex);
    const lineBoxes = lineWordBoxes(position.spreadIndex, position.side, position.lineIndex);
    const current = lineBoxes[position.wordIndex];
    const next = lineBoxes[position.wordIndex + 1];
    const fraction = wordIndex - Math.floor(wordIndex);
    if (!next || fraction === 0) return current;
    return {
        centerX: current.centerX + (next.centerX - current.centerX) * fraction,
        centerZ: current.centerZ,
        width: current.width + (next.width - current.width) * fraction,
    };
}

/** The whole line holding the word at wordIndex, as one box spanning the text block. */
export function lineBoxAtIndex(wordIndex: number): WordBox {
    const position = readingPositionOf(wordIndex);
    const width = TEXT_OUTER_X - TEXT_INNER_X;
    return {
        centerX: textStartX(position.side) + width / 2,
        centerZ: TEXT_TOP_Z + (position.lineIndex + 0.5) * LINE_SPACING_Z,
        width,
    };
}

/** Word index of the last word on the line holding wordIndex. */
export function lineEndIndex(wordIndex: number): number {
    const wholeIndex = Math.floor(wordIndex);
    return wholeIndex - readingPositionOf(wholeIndex).wordIndex + WORDS_PER_LINE - 1;
}
