"""Dev script: start the real WebSocket server against the cached connectome, the real
multilingual embedding model, and a short built-in test text (see brain_loader.py),
with uvicorn's auto-reload watching the eternalfly package so `make dev` picks up
backend code changes without a manual restart. Not unit-tested itself - composes
already-tested eternalfly functions, mirrors cli_reading_demo.py's setup.

Run as `python -m scripts.run_server` (from backend/, as the Makefile does) so uvicorn's
reload subprocess can re-import this module by its "scripts.run_server:app" name."""

import atexit
import sys
from pathlib import Path

import torch
import uvicorn

from eternalfly.brain_loader import build_reading_session, delete_persisted_memory_file, load_memory_into_session, save_memory
from eternalfly.reading_session import ReadingSession
from eternalfly.server import create_app
from eternalfly.text_encoder import tokenize_text

CALIBRE_LIBRARY_PATH = Path.home() / "Calibre-Bibliothek"

TEST_TEXT = """
Der Drache brüllte und die Burg erzitterte vor Angst. Plötzlich zog der
tapfere Ritter sein Schwert und griff an, sein Herz raste vor Aufregung. Danach
aß er Honig und lächelte glücklich. Aber ein Monster griff an! Alle rannten in
Panik davon. Afterwards, everyone sat quietly and stared at the wall for a
long, dull, uneventful hour.
"""


def build_session() -> ReadingSession:
    """Build a real ReadingSession over the real cached connectome, the real
    multilingual embedding model, and the built-in test text."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokens = tokenize_text(TEST_TEXT)
    return build_reading_session(tokens, device=device)


session = build_session()
app = create_app(
    session,
    calibre_library_path=CALIBRE_LIBRARY_PATH,
    load_memory_fn=load_memory_into_session,
    save_memory_fn=save_memory,
    delete_persisted_memory_fn=delete_persisted_memory_file,
)

# Belt-and-suspenders on top of create_app's own FastAPI lifespan shutdown hook: atexit
# runs on normal interpreter exit regardless of platform (unlike a hand-rolled
# signal.signal(SIGTERM/...) handler, which would risk clobbering uvicorn's own
# asyncio-based signal handling, and which SIGTERM barely means anything on Windows
# anyway) - and it also covers uvicorn's own --reload restarts, not just process exit,
# so a code save mid-session no longer silently drops unsaved learning either. Neither
# this nor the lifespan hook can do anything about a hard kill (kill -9, Task Manager
# "End task") - no process can react to those, by design of what a hard kill is.
atexit.register(save_memory, session)


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
