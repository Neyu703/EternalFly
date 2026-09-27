"""One-off script: run a ReadingSession against the real cached connectome with a short
test text, printing the rating and activity at the start of each word (Milestone 2's CLI
verification). Not unit-tested itself - composes already-tested eternalfly functions."""

import sys

from eternalfly.reading_session import ReadingSession
from eternalfly.session_helpers import is_new_word_tick
from eternalfly.text_encoder import tokenize_text
from scripts.session_setup import (
    TEST_TEXT,
    WEIGHT_SCALE,
    build_session_config,
    load_adjacency_as_torch_sparse,
    load_pool_indices,
    select_device,
)


def main() -> None:
    """Run a short reading session and print each word's rating and activity to the console."""
    weight_scale = float(sys.argv[1]) if len(sys.argv) > 1 else WEIGHT_SCALE
    device = select_device()
    print("device:", device, "weight_scale:", weight_scale)

    adjacency_matrix = load_adjacency_as_torch_sparse(device, weight_scale)
    tokens = tokenize_text(TEST_TEXT)
    print("token count:", len(tokens))

    config = build_session_config(device)
    session = ReadingSession(adjacency_matrix.shape[0], adjacency_matrix, load_pool_indices(device), tokens, config)

    total_ticks = len(tokens) * config.ticks_per_word
    for tick_number in range(total_ticks):
        result = session.tick()
        if is_new_word_tick(tick_number, config.ticks_per_word):
            print(
                f"tick {tick_number:4d} word={result.current_word!r:15} "
                f"progress={result.page_progress:.2f} rating={result.rating_0_10:.2f} "
                f"activity={ {name: round(value, 4) for name, value in result.region_activity.items()} } "
                f"bored={result.wants_new_book}"
            )


if __name__ == "__main__":
    main()
