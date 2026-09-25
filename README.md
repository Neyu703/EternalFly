# EternalFly

Live spiking simulation of the [FlyWire](https://flywire.ai/) fruit fly connectome
(~139k real neurons, real synaptic connectivity) *reading a book*. Text is tokenized
and fed into the network as input current; a leaky-integrate-and-fire (LIF) simulation
runs on the real connectome and its spiking activity is decoded back into
valence/arousal and per-neuropil brain activity, streamed live to a 3D dashboard.

## Architecture

- **`backend/`** — Python/FastAPI. Runs the LIF simulation over the cached FlyWire
  connectome and streams per-tick results (spikes, emotion, brain region activity) to
  the frontend over a WebSocket (`/ws`).
- **`frontend/`** — React + TypeScript + Three.js (`@react-three/fiber`), packaged as a
  desktop app with [Tauri](https://tauri.app/). Renders the fly/book scene, a 3D brain
  with per-region glow, live neural activity charts, and playback controls.

## Setup

Requirements: Python 3.13, Node.js, a CUDA-capable GPU (optional — falls back to CPU).

```bash
make install
```

The simulation needs the real FlyWire connectome data (not checked into git,
`backend/data/` is gitignored):

```bash
cd backend
.venv\Scripts\python.exe -m scripts.download_connectome
.venv\Scripts\python.exe -m scripts.build_connectome_cache
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
