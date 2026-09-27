"""Logs what the fly's brain did while it read each word, for inspecting the emotion readout."""

import logging
from dataclasses import dataclass, field

from eternalfly.emotion_decoder import EMOTION_NAMES
from eternalfly.sentiment_lexicon import word_valence
from eternalfly.validation import require_positive_int

logger = logging.getLogger(__name__)

# How many words each INFO-level summary covers.
SUMMARY_WORD_COUNT = 100


def _zero_per_emotion() -> dict[str, float]:
    """A fresh running total of 0.0 per emotion name."""
    return dict.fromkeys(EMOTION_NAMES, 0.0)


@dataclass
class _WordActivity:
    """Raw population rates summed over one word's ticks, plus what was shown at its end."""

    word_index: int
    word: str
    rate_sums: dict[str, float] = field(default_factory=_zero_per_emotion)
    tick_count: int = 0
    emotions: dict[str, float] = field(default_factory=dict)
    rating: float = 0.0

    def mean_rates(self) -> dict[str, float]:
        """Each population's raw firing rate averaged over the word's ticks."""
        return {name: rate_sum / self.tick_count for name, rate_sum in self.rate_sums.items()}


@dataclass
class _SummaryWindow:
    """Running totals over the words since the last summary."""

    word_count: int = 0
    first_word_index: int = 0
    last_word_index: int = 0
    emotion_sums: dict[str, float] = field(default_factory=_zero_per_emotion)
    rating_sum: float = 0.0
    strongest_word_by_emotion: dict[str, tuple[str, float]] = field(default_factory=dict)


class WordActivityLog:
    """Collects the ticks of the word being read and logs it once the next word starts (or
    the book ends): at DEBUG the word, its lexicon sentiment, the raw firing rate of each
    emotion's population averaged over the word's ticks (its immediate response) and the
    displayed, smoothed emotions and dopamine rating; at INFO, every summary_word_count
    words, the average displayed emotions and rating over those words and the word that
    drove each population hardest."""

    def __init__(self, summary_word_count: int = SUMMARY_WORD_COUNT):
        """Create a log that summarizes every summary_word_count words (a positive integer)."""
        require_positive_int("summary_word_count", summary_word_count)
        self._summary_word_count = summary_word_count
        self._current_word: _WordActivity | None = None
        self._summary = _SummaryWindow()

    def record(
        self,
        word_index: int,
        word: str | None,
        raw_rates: dict[str, float],
        emotions: dict[str, float],
        rating: float,
    ) -> None:
        """Add one simulation tick of word (None once the book is finished, which only
        closes the last word) at word_index, with the tick's raw population rates and the
        displayed emotions and rating."""
        if self._current_word is not None and self._current_word.word_index != word_index:
            self._finish_current_word()
        if word is None:
            return
        if self._current_word is None:
            self._current_word = _WordActivity(word_index, word)
        for name in EMOTION_NAMES:
            self._current_word.rate_sums[name] += raw_rates[name]
        self._current_word.tick_count += 1
        self._current_word.emotions = emotions
        self._current_word.rating = rating

    def close_book(self) -> None:
        """Log the word being read and the partial summary of the words since the last
        summary, so neither runs on into the next book (or the restarted one)."""
        if self._current_word is not None:
            self._finish_current_word()
        self._flush_summary()

    def _finish_current_word(self) -> None:
        """Log the word being read, add it to the running summary and forget it."""
        self._finish_word(self._current_word)
        self._current_word = None

    def _flush_summary(self) -> None:
        """Log the summary of the words since the last one, if any, and start a new one."""
        if self._summary.word_count > 0:
            self._log_summary()
        self._summary = _SummaryWindow()

    def _finish_word(self, word_activity: _WordActivity) -> None:
        """Log the finished word and add it to the running summary."""
        mean_rates = word_activity.mean_rates()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "word %d %r lexicon=%+.2f | raw %s | shown %s dopamine=%.2f",
                word_activity.word_index,
                word_activity.word,
                word_valence(word_activity.word),
                _format_values(mean_rates, 4),
                _format_values(word_activity.emotions, 2),
                word_activity.rating,
            )
        self._add_to_summary(word_activity, mean_rates)

    def _add_to_summary(self, word_activity: _WordActivity, mean_rates: dict[str, float]) -> None:
        """Accumulate the word into the summary window, logging it once the window is full."""
        summary = self._summary
        if summary.word_count == 0:
            summary.first_word_index = word_activity.word_index
        summary.word_count += 1
        summary.rating_sum += word_activity.rating
        for name in EMOTION_NAMES:
            summary.emotion_sums[name] += word_activity.emotions[name]
            strongest = summary.strongest_word_by_emotion.get(name)
            if strongest is None or mean_rates[name] > strongest[1]:
                summary.strongest_word_by_emotion[name] = (word_activity.word, mean_rates[name])
        summary.last_word_index = word_activity.word_index
        if summary.word_count == self._summary_word_count:
            self._flush_summary()

    def _log_summary(self) -> None:
        """Log the averages and strongest words of the full summary window."""
        summary = self._summary
        mean_emotions = {name: value / summary.word_count for name, value in summary.emotion_sums.items()}
        strongest_words = ", ".join(
            f"{name} {word!r} ({rate:.4f})" for name, (word, rate) in summary.strongest_word_by_emotion.items()
        )
        logger.info(
            "words %d-%d: mean shown %s dopamine=%.2f | strongest raw response: %s",
            summary.first_word_index,
            summary.last_word_index,
            _format_values(mean_emotions, 2),
            summary.rating_sum / summary.word_count,
            strongest_words,
        )


def _format_values(values: dict[str, float], decimals: int) -> str:
    """name=value pairs separated by spaces, each value with the given decimals."""
    return " ".join(f"{name}={value:.{decimals}f}" for name, value in values.items())
