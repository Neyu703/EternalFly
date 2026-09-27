import logging

import pytest

from eternalfly.word_activity_log import WordActivityLog

LOGGER_NAME = "eternalfly.word_activity_log"
QUIET_EMOTIONS = {"reward": 0.0, "aversion": 0.0, "arousal": 0.0}


def _rates(reward: float, aversion: float, arousal: float) -> dict[str, float]:
    """Raw population rates keyed by emotion name."""
    return {"reward": reward, "aversion": aversion, "arousal": arousal}


def test_word_activity_log_raises_on_non_positive_summary_word_count():
    with pytest.raises(ValueError):
        WordActivityLog(summary_word_count=0)


def test_record_logs_a_word_only_once_the_next_word_starts(caplog):
    word_log = WordActivityLog()

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        word_log.record(0, "good", _rates(0.2, 0.0, 0.1), QUIET_EMOTIONS, 5.0)
        assert caplog.records == []
        word_log.record(1, "table", _rates(0.0, 0.0, 0.0), QUIET_EMOTIONS, 5.0)

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.DEBUG


def test_record_logs_the_words_lexicon_sentiment_mean_raw_rates_and_shown_state(caplog):
    word_log = WordActivityLog()
    shown_emotions = {"reward": 0.5, "aversion": 0.0, "arousal": 0.25}

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        word_log.record(3, "good", _rates(0.2, 0.0, 0.1), QUIET_EMOTIONS, 5.0)
        word_log.record(3, "good", _rates(0.4, 0.2, 0.3), shown_emotions, 7.5)
        word_log.record(4, None, _rates(0.0, 0.0, 0.0), QUIET_EMOTIONS, 5.0)

    assert caplog.records[0].getMessage() == (
        "word 3 'good' lexicon=+0.47 | raw reward=0.3000 aversion=0.1000 arousal=0.2000"
        " | shown reward=0.50 aversion=0.00 arousal=0.25 dopamine=7.50"
    )


def test_record_ignores_ticks_after_the_book_has_finished(caplog):
    word_log = WordActivityLog()

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        word_log.record(0, None, _rates(0.0, 0.0, 0.0), QUIET_EMOTIONS, 5.0)
        word_log.record(0, None, _rates(0.0, 0.0, 0.0), QUIET_EMOTIONS, 5.0)

    assert caplog.records == []


def test_record_logs_an_info_summary_every_summary_word_count_words(caplog):
    word_log = WordActivityLog(summary_word_count=2)

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        word_log.record(0, "good", _rates(0.3, 0.1, 0.2), {"reward": 0.6, "aversion": 0.0, "arousal": 0.4}, 8.0)
        word_log.record(1, "bad", _rates(0.1, 0.4, 0.3), {"reward": 0.2, "aversion": 0.6, "arousal": 0.6}, 3.0)
        word_log.record(2, "table", _rates(0.0, 0.0, 0.0), QUIET_EMOTIONS, 5.0)

    assert [record.getMessage() for record in caplog.records] == [
        "words 0-1: mean shown reward=0.40 aversion=0.30 arousal=0.50 dopamine=5.50"
        " | strongest raw response: reward 'good' (0.3000), aversion 'bad' (0.4000), arousal 'bad' (0.3000)"
    ]


def test_record_starts_a_fresh_summary_window_after_each_summary(caplog):
    word_log = WordActivityLog(summary_word_count=1)

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        word_log.record(0, "good", _rates(0.3, 0.0, 0.0), QUIET_EMOTIONS, 5.0)
        word_log.record(1, "table", _rates(0.1, 0.0, 0.0), QUIET_EMOTIONS, 5.0)
        word_log.record(2, None, _rates(0.0, 0.0, 0.0), QUIET_EMOTIONS, 5.0)

    assert "words 1-1" in caplog.records[1].getMessage()
    assert "reward 'table' (0.1000)" in caplog.records[1].getMessage()


def test_close_book_logs_the_current_word_and_the_partial_summary(caplog):
    word_log = WordActivityLog(summary_word_count=100)

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        word_log.record(40, "good", _rates(0.3, 0.0, 0.2), QUIET_EMOTIONS, 5.0)
        word_log.record(41, "table", _rates(0.1, 0.0, 0.0), QUIET_EMOTIONS, 5.0)
        word_log.close_book()

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 3
    assert "'table'" in messages[1]
    assert messages[2].startswith("words 40-41:")


def test_close_book_starts_the_next_books_summary_from_its_own_first_word(caplog):
    word_log = WordActivityLog(summary_word_count=2)

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        word_log.record(7, "good", _rates(0.3, 0.0, 0.2), QUIET_EMOTIONS, 5.0)
        word_log.close_book()
        word_log.record(0, "bad", _rates(0.0, 0.3, 0.2), QUIET_EMOTIONS, 5.0)
        word_log.record(1, "table", _rates(0.0, 0.0, 0.0), QUIET_EMOTIONS, 5.0)
        word_log.record(2, None, _rates(0.0, 0.0, 0.0), QUIET_EMOTIONS, 5.0)

    assert [record.getMessage()[:9] for record in caplog.records] == ["words 7-7", "words 0-1"]


def test_close_book_logs_nothing_when_no_word_was_read(caplog):
    word_log = WordActivityLog()

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        word_log.close_book()

    assert caplog.records == []
