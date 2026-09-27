"""Where the downloaded FlyWire connectome files and the simulation cache built from them
live, shared by every script that reads or writes them."""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = DATA_DIR / "cache"

# The public FlyWire (materialization 783) files, see scripts/download_connectome.py.
ROOT_IDS_PATH = DATA_DIR / "proofread_root_ids_783.npy"
CONNECTIONS_PATH = DATA_DIR / "proofread_connections_783.feather"
PRE_NEUROPIL_COUNTS_PATH = DATA_DIR / "per_neuron_neuropil_count_pre_783.feather"
POST_NEUROPIL_COUNTS_PATH = DATA_DIR / "per_neuron_neuropil_count_post_783.feather"
RAW_DATA_PATHS = (CONNECTIONS_PATH, PRE_NEUROPIL_COUNTS_PATH, POST_NEUROPIL_COUNTS_PATH, ROOT_IDS_PATH)

# The simulation-ready cache, see scripts/build_connectome_cache.py.
ADJACENCY_CACHE_PATH = CACHE_DIR / "adjacency.npz"
POOL_INDICES_CACHE_PATH = CACHE_DIR / "pool_indices.npz"
REGION_SYNAPSE_WEIGHTS_CACHE_PATH = CACHE_DIR / "region_synapse_weights.npz"
