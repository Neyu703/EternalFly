"""One-off script: run a ReadingSession against the real cached connectome with a short
test text, printing emotions/rating per tick (Milestone 2's CLI verification). Not
unit-tested itself - composes already-tested eternalfly functions."""

import sys
from pathlib import Path

import numpy
import scipy.sparse
import torch

from eternalfly.lif import LIFParameters
from eternalfly.reading_session import ReadingSession, ReadingSessionConfig
from eternalfly.text_encoder import tokenize_text

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = DATA_DIR / "cache"

TEST_TEXT = """
The dragon roared and the castle shook with fear. Suddenly, the brave knight
drew his sword and charged forward, heart pounding with excitement. It was
the most thrilling battle anyone had ever seen. Afterwards, everyone sat
quietly and stared at the wall for a long, dull, uneventful hour.
"""


def load_adjacency_as_torch_sparse(device: str) -> torch.Tensor:
    """Load the cached signed adjacency matrix and convert it to a torch sparse CSR tensor."""
    scipy_matrix = scipy.sparse.load_npz(CACHE_DIR / "adjacency.npz").tocsr()
    dense_row_pointers = torch.as_tensor(scipy_matrix.indptr, dtype=torch.int64)
    column_indices = torch.as_tensor(scipy_matrix.indices, dtype=torch.int64)
    values = torch.as_tensor(scipy_matrix.data, dtype=torch.float32)
    return torch.sparse_csr_tensor(
        dense_row_pointers, column_indices, values, size=scipy_matrix.shape, device=device
    )


def load_pool_indices(device: str) -> dict[str, torch.Tensor]:
    """Load the cached per-pool neuron index arrays as torch tensors."""
    raw_pools = numpy.load(CACHE_DIR / "pool_indices.npz")
    return {name: torch.as_tensor(raw_pools[name], dtype=torch.int64, device=device) for name in raw_pools.files}


def main() -> None:
    """Run a short reading session and print per-tick emotions/rating to the console."""
    weight_scale = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "weight_scale:", weight_scale)

    adjacency_matrix = load_adjacency_as_torch_sparse(device)
    scaled_adjacency_matrix = torch.sparse_csr_tensor(
        adjacency_matrix.crow_indices(),
        adjacency_matrix.col_indices(),
        adjacency_matrix.values() * weight_scale,
        size=adjacency_matrix.shape,
        device=device,
    )
    pool_indices = load_pool_indices(device)
    neuron_count = adjacency_matrix.shape[0]

    tokens = tokenize_text(TEST_TEXT)
    print("token count:", len(tokens))

    config = ReadingSessionConfig(
        lif_parameters=LIFParameters(
            membrane_time_constant_ms=20.0,
            spike_threshold=1.0,
            reset_potential=0.0,
            refractory_period_ms=2.0,
            dt_ms=1.0,
        ),
        ticks_per_word=10,
        input_current_scale=30.0,
        valence_weight=1.0,  # calibrated against the real connectome, see scripts/calibrate_sentiment.py
        arousal_weight=1.0,  # calibrated against the real connectome, see scripts/calibrate_sentiment.py
        token_seed=42,
        engagement_window_size=5000,  # ~500 words of context, see scripts/calibrate_sentiment.py
        engagement_threshold=0.15,
        min_ticks_before_boredom_check=5000,
        display_window_size=100,  # ~10 words, see scripts/calibrate_sentiment.py
        positive_valence_ceiling=0.18,  # calibrated against the real connectome, see scripts/calibrate_sentiment.py
        negative_valence_ceiling=0.03,  # calibrated against the real connectome, see scripts/calibrate_sentiment.py
        arousal_ceiling=0.08,  # calibrated against the real connectome, see scripts/calibrate_sentiment.py
        device=device,
    )
    session = ReadingSession(neuron_count, scaled_adjacency_matrix, pool_indices, tokens, config)

    total_ticks = len(tokens) * config.ticks_per_word
    for tick_number in range(total_ticks):
        result = session.tick()
        if tick_number % config.ticks_per_word == 0:
            print(
                f"tick {tick_number:4d} word={result.current_word!r:15} "
                f"progress={result.page_progress:.2f} rating={result.rating_0_10:.2f} "
                f"activity={ {name: round(value, 4) for name, value in result.region_activity.items()} } "
                f"bored={result.wants_new_book}"
            )


if __name__ == "__main__":
    main()
