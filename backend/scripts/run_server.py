"""Dev script: start the real WebSocket server against the cached connectome and a
short test text, with uvicorn's auto-reload watching the eternalfly package so `make
dev` picks up backend code changes without a manual restart. Not unit-tested itself -
composes already-tested eternalfly functions, mirrors cli_reading_demo.py's setup.

Run as `python -m scripts.run_server` (from backend/, as the Makefile does) so uvicorn's
reload subprocess can re-import this module by its "scripts.run_server:app" name."""

import sys
from pathlib import Path

import numpy
import scipy.sparse
import torch
import uvicorn

from eternalfly.lif import LIFParameters
from eternalfly.reading_session import ReadingSession, ReadingSessionConfig
from eternalfly.server import create_app
from eternalfly.text_encoder import tokenize_text

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = DATA_DIR / "cache"
CALIBRE_LIBRARY_PATH = Path.home() / "Calibre-Bibliothek"

TEST_TEXT = """
The dragon roared and the castle shook with fear. Suddenly, the brave knight
drew his sword and charged forward, heart pounding with excitement. It was
the most thrilling battle anyone had ever seen. Afterwards, everyone sat
quietly and stared at the wall for a long, dull, uneventful hour.
"""

WEIGHT_SCALE = 0.15
INPUT_CURRENT_SCALE = 30.0
VALENCE_WEIGHT = 1.0  # calibrated against the real connectome, see scripts/calibrate_sentiment.py
TICKS_PER_WORD = 10
CONTEXT_WORD_COUNT = 500  # how many recent words the rating/emotions average over


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


def build_session() -> ReadingSession:
    """Build a ReadingSession over the real connectome and the built-in test text."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adjacency_matrix = load_adjacency_as_torch_sparse(device)
    pool_indices = load_pool_indices(CACHE_DIR / "pool_indices.npz", device)
    neuropil_pool_indices = load_pool_indices(CACHE_DIR / "neuropil_pool_indices.npz", device)
    tokens = tokenize_text(TEST_TEXT)

    config = ReadingSessionConfig(
        lif_parameters=LIFParameters(
            membrane_time_constant_ms=20.0,
            spike_threshold=1.0,
            reset_potential=0.0,
            refractory_period_ms=2.0,
            dt_ms=1.0,
        ),
        ticks_per_word=TICKS_PER_WORD,
        input_current_scale=INPUT_CURRENT_SCALE,
        valence_weight=VALENCE_WEIGHT,
        token_seed=42,
        engagement_window_size=CONTEXT_WORD_COUNT * TICKS_PER_WORD,
        engagement_threshold=0.15,
        min_ticks_before_boredom_check=CONTEXT_WORD_COUNT * TICKS_PER_WORD,
        device=device,
    )
    return ReadingSession(
        adjacency_matrix.shape[0], adjacency_matrix, pool_indices, tokens, config, neuropil_pool_indices
    )


app = create_app(build_session(), tick_interval_seconds=0.05, calibre_library_path=CALIBRE_LIBRARY_PATH)


def main() -> None:
    """Start uvicorn on the given port (default 8000), auto-reloading `app` whenever a
    file under eternalfly/ or scripts/ changes."""
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    uvicorn.run(
        "scripts.run_server:app",
        host="127.0.0.1",
        port=port,
        log_level="info",
        reload=True,
        reload_dirs=["eternalfly", "scripts"],
    )


if __name__ == "__main__":
    main()
