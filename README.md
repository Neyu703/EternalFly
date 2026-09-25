# EternalFly

Live spiking simulation of the [FlyWire](https://flywire.ai/) fruit fly connectome
(~139k real neurons, real synaptic connectivity) *reading a book, through its own real
senses*. Text is embedded with a multilingual sentence-embedding model and translated
into Poisson spikes in the real sensory receptor neurons those words actually match
(taste, smell, sound, temperature, looming threat, ...); a Shiu et al. 2024
(Nature 634:210) -validated, event-driven leaky-integrate-and-fire simulation runs on
the real connectome, and its spiking activity is decoded back into emotions/rating
(from real downstream mushroom-body output neurons) and real descending/motor readouts
(escape jump, feeding, turning, ...), streamed live to a 3D dashboard.

## Architecture

- **`backend/`** — Python/FastAPI. `eternalfly/semantic_encoder.py` turns each word
  into real sensory drives; `eternalfly/reading_session.py` runs the event-driven LIF
  simulation (`eternalfly/lif.py`, `eternalfly/synapses.py`) over the cached FlyWire
  connectome and decodes emotions/behaviors/brain activity from real named cell groups
  (`eternalfly/cell_groups.py`); `eternalfly/server.py` streams per-frame results over
  a WebSocket (`/ws`). `eternalfly/brain_loader.py` is the one place that loads the
  real connectome cache and the real embedding model.
- **`frontend/`** — React + TypeScript + Three.js (`@react-three/fiber`), packaged as a
  desktop app with [Tauri](https://tauri.app/). Renders the fly/book scene, a 3D brain
  with per-region glow, live neural activity charts, and playback controls.

## Setup

Requirements: Python 3.13, Node.js, a CUDA-capable GPU (optional — falls back to CPU).

```bash
make install
```

The simulation needs the real FlyWire connectome data and cell-type annotations (not
checked into git, `backend/data/` is gitignored) — the first run also downloads the
~470MB multilingual embedding model (cached by Hugging Face, not project-specific):

```bash
cd backend
.venv\Scripts\python.exe -m scripts.download_connectome
.venv\Scripts\python.exe -m scripts.build_connectome_cache
```

To verify the real network's reflexes and calibrate its valence/arousal ceilings
against the cached connectome:

```bash
.venv\Scripts\python.exe -m scripts.audit_brain
```

## Running

```bash
make dev
```

Starts the backend (`uvicorn`, auto-reload, port 8000) and the frontend (`tauri dev`)
in parallel. Or run them individually via `make backend` / `make frontend`.

## Testing

```bash
cd backend
.venv\Scripts\pytest.exe
```

## Versioning

Every commit bumps `version` in `backend/pyproject.toml` (kept in lockstep with
`frontend/package.json`) by one patch level; commit subjects are prefixed
`[vX.Y.Z] type: subject`. Every push to `main` is tagged and released automatically
(see `.github/workflows/auto-release.yml`) — GitHub's auto-generated release notes are
the changelog.
