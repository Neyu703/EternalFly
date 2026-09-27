"""Dev script: start the real WebSocket server against the cached connectome and a
short test text, with uvicorn's auto-reload watching the eternalfly package so `make
dev` picks up backend code changes without a manual restart. Not unit-tested itself -
composes already-tested eternalfly functions and the setup shared in
scripts/session_setup.py.

Run as `python -m scripts.run_server` (from backend/, as the Makefile does) so uvicorn's
reload subprocess can re-import this module by its "scripts.run_server:app" name."""

import logging
import logging.handlers
import os
import sys
from pathlib import Path

import uvicorn

from eternalfly.calibre_library import configured_library_path
from eternalfly.reading_session import ReadingSession
from eternalfly.server import create_app
from eternalfly.text_encoder import tokenize_text
from scripts.emotion_calibration import EMOTION_CALIBRATIONS
from scripts.session_setup import (
    TEST_TEXT,
    TICKS_PER_WORD,
    build_session_config,
    load_adjacency_as_torch_sparse,
    load_pool_indices,
    load_region_synapse_weights,
    select_device,
)

# The library Calibre itself uses (so a moved or switched library is followed), else the
# default location in the home folder.
CALIBRE_CONFIG_DIRECTORY = Path(os.environ.get("APPDATA", "")) / "calibre"
CALIBRE_LIBRARY_PATH = configured_library_path(CALIBRE_CONFIG_DIRECTORY) or Path.home() / "Calibre-Bibliothek"

TICK_INTERVAL_SECONDS = 0.05
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


def build_session() -> ReadingSession:
    """Build a ReadingSession over the real connectome and the built-in test text."""
    device = select_device()
    adjacency_matrix = load_adjacency_as_torch_sparse(device)
    pool_indices = load_pool_indices(device)
    logger.info(
        "neuron pools: %s",
        ", ".join(f"{name} {len(indices)} neurons" for name, indices in pool_indices.items()),
    )
    logger.info("emotion calibrations: %s", EMOTION_CALIBRATIONS)
    return ReadingSession(
        adjacency_matrix.shape[0],
        adjacency_matrix,
        pool_indices,
        tokenize_text(TEST_TEXT),
        build_session_config(device),
        load_region_synapse_weights(device),
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
