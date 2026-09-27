"""Dev script: start the real WebSocket server against the cached connectome and a
short test text, with uvicorn's auto-reload watching the eternalfly package so `make
dev` picks up backend code changes without a manual restart. Not unit-tested itself -
composes already-tested eternalfly functions, mirrors cli_reading_demo.py's setup.

Run as `python -m scripts.run_server` (from backend/, as the Makefile does) so uvicorn's
reload subprocess can re-import this module by its "scripts.run_server:app" name."""

import logging
import logging.handlers
import os
import sys
from pathlib import Path

import numpy
import scipy.sparse
import torch
import uvicorn

from eternalfly.calibre_library import configured_library_path
from eternalfly.lif import LIFParameters
from eternalfly.neuropils import ALL_NEUROPIL_NAMES
from eternalfly.reading_session import ReadingSession, ReadingSessionConfig, RegionSynapseWeights
from eternalfly.server import create_app
from eternalfly.text_encoder import tokenize_text
from scripts.emotion_calibration import EMOTION_CALIBRATIONS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = DATA_DIR / "cache"
# The library Calibre itself uses (so a moved or switched library is followed), else the
# default location in the home folder.
CALIBRE_CONFIG_DIRECTORY = Path(os.environ.get("APPDATA", "")) / "calibre"
CALIBRE_LIBRARY_PATH = configured_library_path(CALIBRE_CONFIG_DIRECTORY) or Path.home() / "Calibre-Bibliothek"

TEST_TEXT = """
The dragon roared and the castle shook with fear. Suddenly, the brave knight
drew his sword and charged forward, heart pounding with excitement. It was
the most thrilling battle anyone had ever seen. Afterwards, everyone sat
quietly and stared at the wall for a long, dull, uneventful hour.
"""

WEIGHT_SCALE = 0.15
INPUT_CURRENT_SCALE = 30.0
VALENCE_WEIGHT = 1.0  # calibrated against the real connectome, see scripts/calibrate_sentiment.py
AROUSAL_WEIGHT = 1.0  # calibrated against the real connectome, see scripts/calibrate_sentiment.py
TICKS_PER_WORD = 10
TICK_INTERVAL_SECONDS = 0.05
CONTEXT_WORD_COUNT = 500  # how many recent words the boredom/engagement judgment averages over
DISPLAY_WORD_COUNT = 10  # how many recent words the *displayed* rating/emotions/region_activity average over
BASE_WORDS_PER_MINUTE = 60.0 / (TICKS_PER_WORD * TICK_INTERVAL_SECONDS)  # reading pace at speed_multiplier=1.0

# Console log level (DEBUG adds one line per word read, see eternalfly.word_activity_log);
# the log file below always records everything down to DEBUG.
CONSOLE_LOG_LEVEL = os.environ.get("ETERNALFLY_LOG_LEVEL", "INFO").upper()
LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "eternalfly.log"
LOG_FILE_MAX_BYTES = 5_000_000
LOG_FILE_BACKUP_COUNT = 3
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

logger = logging.getLogger("eternalfly.run_server")


def configure_logging() -> None:
    """Send eternalfly's log records to the console at CONSOLE_LOG_LEVEL and, at every
    level, to the rotating LOG_FILE. Does nothing once configured: uvicorn's worker imports
    this module twice (as multiprocessing's __mp_main__, then by name), and a second file
    handler would keep the log file open and block its rotation on Windows."""
    package_logger = logging.getLogger("eternalfly")
    if package_logger.handlers:
        return
    LOG_FILE.parent.mkdir(exist_ok=True)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(CONSOLE_LOG_LEVEL)
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=LOG_FILE_MAX_BYTES, backupCount=LOG_FILE_BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    package_logger.setLevel(logging.DEBUG)
    package_logger.propagate = False
    package_logger.handlers = [console_handler, file_handler]
    for handler in package_logger.handlers:
        handler.setFormatter(logging.Formatter(LOG_FORMAT))


def load_sparse_as_torch(cache_path: Path, device: str, scale: float = 1.0) -> torch.Tensor:
    """Load a cached scipy sparse matrix, multiplied by scale, as a torch sparse CSR tensor."""
    scipy_matrix = scipy.sparse.load_npz(cache_path).tocsr()
    row_pointers = torch.as_tensor(scipy_matrix.indptr, dtype=torch.int64)
    column_indices = torch.as_tensor(scipy_matrix.indices, dtype=torch.int64)
    values = torch.as_tensor(scipy_matrix.data, dtype=torch.float32) * scale
    return torch.sparse_csr_tensor(row_pointers, column_indices, values, size=scipy_matrix.shape, device=device)


def load_adjacency_as_torch_sparse(device: str) -> torch.Tensor:
    """Load the cached signed adjacency matrix, scaled by WEIGHT_SCALE, as a torch sparse CSR tensor."""
    return load_sparse_as_torch(CACHE_DIR / "adjacency.npz", device, WEIGHT_SCALE)


def load_region_synapse_weights(device: str) -> RegionSynapseWeights:
    """Load the cached region x neuron synapse-share matrix, whose rows follow
    ALL_NEUROPIL_NAMES (see scripts/build_connectome_cache.py)."""
    return RegionSynapseWeights(
        names=ALL_NEUROPIL_NAMES, weights=load_sparse_as_torch(CACHE_DIR / "region_synapse_weights.npz", device)
    )


def load_pool_indices(cache_path: Path, device: str) -> dict[str, torch.Tensor]:
    """Load a cached per-pool neuron index .npz file as a dict of torch tensors."""
    raw_pools = numpy.load(cache_path)
    return {name: torch.as_tensor(raw_pools[name], dtype=torch.int64, device=device) for name in raw_pools.files}


def build_session() -> ReadingSession:
    """Build a ReadingSession over the real connectome and the built-in test text."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adjacency_matrix = load_adjacency_as_torch_sparse(device)
    pool_indices = load_pool_indices(CACHE_DIR / "pool_indices.npz", device)
    region_synapse_weights = load_region_synapse_weights(device)
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
        arousal_weight=AROUSAL_WEIGHT,
        token_seed=42,
        engagement_window_size=CONTEXT_WORD_COUNT * TICKS_PER_WORD,
        engagement_threshold=0.15,
        min_ticks_before_boredom_check=CONTEXT_WORD_COUNT * TICKS_PER_WORD,
        display_window_size=DISPLAY_WORD_COUNT * TICKS_PER_WORD,
        emotion_calibrations=EMOTION_CALIBRATIONS,
        device=device,
    )
    logger.info(
        "neuron pools: %s",
        ", ".join(f"{name} {len(indices)} neurons" for name, indices in pool_indices.items()),
    )
    logger.info("emotion calibrations: %s", EMOTION_CALIBRATIONS)
    return ReadingSession(
        adjacency_matrix.shape[0], adjacency_matrix, pool_indices, tokens, config, region_synapse_weights
    )


# Only in uvicorn's worker, which imports this module by name: the reloading parent process
# (run as __main__) must not hold the log file open too, or rotating it fails on Windows.
if __name__ != "__main__":
    configure_logging()
app = create_app(
    build_session(),
    tick_interval_seconds=TICK_INTERVAL_SECONDS,
    calibre_library_path=CALIBRE_LIBRARY_PATH,
    base_words_per_minute=BASE_WORDS_PER_MINUTE,
)


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
