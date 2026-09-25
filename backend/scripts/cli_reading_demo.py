"""One-off script: run a real ReadingSession (brain_loader.py) over a short DE/EN test
text, printing each word's real sensory drives, emotions/rating, and behavior readouts
to the console - a headless way to verify the fly is genuinely reacting to what it
reads. Not unit-tested itself - composes already-tested eternalfly functions.

Run as `python -m scripts.cli_reading_demo` from backend/.
"""

from eternalfly.brain_loader import build_reading_session
from eternalfly.text_encoder import tokenize_text

TEST_TEXT = """
Der Drache brüllte wütend und die Burg erzitterte vor Angst. Plötzlich zog der
tapfere Ritter sein Schwert und griff an, sein Herz raste vor Aufregung. Es war
die aufregendste Schlacht, die je jemand gesehen hatte. Danach aß er Honig und
lächelte glücklich. Aber ein Monster griff an! Alle rannten in Panik davon.
Afterwards, everyone sat quietly and stared at the wall for a long, dull hour.
"""


def main() -> None:
    """Read TEST_TEXT and print each word's real senses/emotions/behaviors as the
    fly's own connectome-driven readouts, at the session's fixed sim_ms_per_word."""
    tokens = tokenize_text(TEST_TEXT)
    print("token count:", len(tokens))

    session = build_reading_session(tokens)
    session.precompute_upcoming_words(len(tokens))

    for _ in range(len(tokens)):
        result = session.advance(round(session.sim_ms_per_word / session.dt_ms))
        active_senses = {name: round(value, 2) for name, value in result.senses.items() if value > 0.05}
        active_behaviors = {name: round(value, 4) for name, value in result.behaviors.items() if value > 0}
        print(
            f"word={result.current_word!r:15} progress={result.page_progress:.2f} "
            f"rating={result.rating_0_10:.2f} learned={result.learned_valence:+.4f} "
            f"senses={active_senses} behaviors={active_behaviors} bored={result.wants_new_book}"
        )


if __name__ == "__main__":
    main()
