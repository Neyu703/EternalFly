"""One-off script: measures how strongly the dopaminergic valence_positive/
valence_negative channels (see text_encoder.project_valence_to_currents and
reading_session.ReadingSession._current_external_input) actually shift the final
rating on the real cached connectome, comparing a clearly positive, a clearly
negative, and a neutral passage. Not unit-tested itself - this is exploratory tooling
for re-tuning ReadingSessionConfig.valence_weight (and engagement_window_size) if the
connectome cache or lexicon ever change."""

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
WEIGHT_SCALE = 0.15
INPUT_CURRENT_SCALE = 30.0

POSITIVE_TEXT = (
    "The knight smiled warmly as sunlight filled the garden. Everyone laughed and "
    "celebrated the wonderful, joyful morning together, full of hope and delight. "
) * 6
NEGATIVE_TEXT = (
    "The knight wept bitterly as darkness filled the ruined garden. Everyone screamed "
    "in terror at the horrible, miserable morning, full of dread and despair. "
) * 6
NEUTRAL_TEXT = (
    "The knight walked across the garden. The table had four chairs and a lamp. "
    "The road continued past the bridge toward the old wooden mill building. "
) * 6

TRAILING_TICKS_AVERAGED = 200  # smooths out which single word happened to be last


def load_adjacency_as_torch_sparse(device: str) -> torch.Tensor:
    """Load the cached signed adjacency matrix, scaled by WEIGHT_SCALE, as a torch sparse CSR tensor."""
    scipy_matrix = scipy.sparse.load_npz(CACHE_DIR / "adjacency.npz").tocsr()
    row_pointers = torch.as_tensor(scipy_matrix.indptr, dtype=torch.int64)
    column_indices = torch.as_tensor(scipy_matrix.indices, dtype=torch.int64)
    values = torch.as_tensor(scipy_matrix.data, dtype=torch.float32) * WEIGHT_SCALE
    return torch.sparse_csr_tensor(row_pointers, column_indices, values, size=scipy_matrix.shape, device=device)


def load_pool_indices(cache_path: Path, device: str) -> dict[str, torch.Tensor]:
    """Load a cached per-pool neuron index .npz file as a dict of torch tensors."""
    raw_pools = numpy.load(cache_path)
    return {name: torch.as_tensor(raw_pools[name], dtype=torch.int64, device=device) for name in raw_pools.files}


def average_trailing_rating(
    text: str,
    valence_weight: float,
    engagement_window_size: int,
    device: str,
    adjacency_matrix: torch.Tensor,
    pool_indices: dict[str, torch.Tensor],
) -> float:
    """Run a full ReadingSession over text and return the mean rating over its last
    TRAILING_TICKS_AVERAGED ticks (steadier than a single tick's snapshot)."""
    tokens = tokenize_text(text)
    config = ReadingSessionConfig(
        lif_parameters=LIFParameters(
            membrane_time_constant_ms=20.0,
            spike_threshold=1.0,
            reset_potential=0.0,
            refractory_period_ms=2.0,
            dt_ms=1.0,
        ),
        ticks_per_word=10,
        input_current_scale=INPUT_CURRENT_SCALE,
        valence_weight=valence_weight,
        token_seed=42,
        engagement_window_size=engagement_window_size,
        engagement_threshold=0.15,
        min_ticks_before_boredom_check=40,
        device=device,
    )
    session = ReadingSession(adjacency_matrix.shape[0], adjacency_matrix, pool_indices, tokens, config)
    total_ticks = len(tokens) * config.ticks_per_word
    trailing_window = min(TRAILING_TICKS_AVERAGED, total_ticks)
    trailing_ratings = []
    for tick_number in range(total_ticks):
        result = session.tick()
        if tick_number >= total_ticks - trailing_window:
            trailing_ratings.append(result.rating_0_10)
    return sum(trailing_ratings) / len(trailing_ratings)


def main() -> None:
    """Print positive/negative/neutral ratings for the (valence_weight,
    engagement_window_size) pair given as CLI args (defaults: 1.0, 100)."""
    valence_weight = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    engagement_window_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adjacency_matrix = load_adjacency_as_torch_sparse(device)
    pool_indices = load_pool_indices(CACHE_DIR / "pool_indices.npz", device)

    positive_rating = average_trailing_rating(
        POSITIVE_TEXT, valence_weight, engagement_window_size, device, adjacency_matrix, pool_indices
    )
    negative_rating = average_trailing_rating(
        NEGATIVE_TEXT, valence_weight, engagement_window_size, device, adjacency_matrix, pool_indices
    )
    neutral_rating = average_trailing_rating(
        NEUTRAL_TEXT, valence_weight, engagement_window_size, device, adjacency_matrix, pool_indices
    )
    print(
        f"valence_weight={valence_weight} engagement_window_size={engagement_window_size}  "
        f"positive={positive_rating:.3f} negative={negative_rating:.3f} neutral={neutral_rating:.3f} "
        f"spread={positive_rating - negative_rating:+.3f}"
    )


if __name__ == "__main__":
    main()
