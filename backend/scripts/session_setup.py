"""The reading-session setup shared by the server, the calibration script and the CLI demo:
the model's tuned constants, the loaders for the connectome cache and the session config
built from them. Import it rather than scripts.run_server, which builds its whole app (and
a full session) on import."""

import numpy
import scipy.sparse
import torch

from eternalfly.activity_readout import RegionSynapseWeights
from eternalfly.emotion_decoder import PoolCalibration
from eternalfly.lif import LIFParameters, accepting_sparse_csr_beta
from eternalfly.neuropils import ALL_NEUROPIL_NAMES
from eternalfly.reading_session import ReadingSessionConfig
from scripts.data_paths import ADJACENCY_CACHE_PATH, POOL_INDICES_CACHE_PATH, REGION_SYNAPSE_WEIGHTS_CACHE_PATH
from scripts.emotion_calibration import EMOTION_CALIBRATIONS

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
TOKEN_SEED = 42
ENGAGEMENT_THRESHOLD = 0.15
CONTEXT_WORD_COUNT = 500  # how many recent words the boredom/engagement judgment averages over
DISPLAY_WORD_COUNT = 10  # how many recent words the *displayed* rating/emotions/region_activity average over
LIF_PARAMETERS = LIFParameters(
    membrane_time_constant_ms=20.0,
    spike_threshold=1.0,
    reset_potential=0.0,
    refractory_period_ms=2.0,
    dt_ms=1.0,
)


def select_device() -> str:
    """The torch device to simulate on: the GPU when CUDA is available, else the CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_sparse_as_torch(cache_path, device: str, scale: float = 1.0) -> torch.Tensor:
    """Load a cached scipy sparse matrix, multiplied by scale, as a torch sparse CSR tensor.
    The matrix is brought into canonical form (sorted, distinct column indices per row), which
    torch's sparse kernels assume; the invariant check at load time guards that once."""
    scipy_matrix = scipy.sparse.load_npz(cache_path).tocsr()
    scipy_matrix.sum_duplicates()
    row_pointers = torch.as_tensor(scipy_matrix.indptr, dtype=torch.int64)
    column_indices = torch.as_tensor(scipy_matrix.indices, dtype=torch.int64)
    values = torch.as_tensor(scipy_matrix.data, dtype=torch.float32) * scale
    with accepting_sparse_csr_beta():
        return torch.sparse_csr_tensor(
            row_pointers, column_indices, values, size=scipy_matrix.shape, device=device, check_invariants=True
        )


def load_adjacency_as_torch_sparse(device: str, weight_scale: float = WEIGHT_SCALE) -> torch.Tensor:
    """Load the cached signed adjacency matrix, scaled by weight_scale, as a torch sparse CSR tensor."""
    return load_sparse_as_torch(ADJACENCY_CACHE_PATH, device, weight_scale)


def load_pool_indices(device: str) -> dict[str, torch.Tensor]:
    """Load the cached per-pool neuron indices as a dict of torch tensors."""
    raw_pools = numpy.load(POOL_INDICES_CACHE_PATH)
    return {name: torch.as_tensor(raw_pools[name], dtype=torch.int64, device=device) for name in raw_pools.files}


def load_region_synapse_weights(device: str) -> RegionSynapseWeights:
    """Load the cached region x neuron synapse-share matrix, whose rows follow
    ALL_NEUROPIL_NAMES (see scripts/build_connectome_cache.py)."""
    return RegionSynapseWeights(
        names=ALL_NEUROPIL_NAMES, weights=load_sparse_as_torch(REGION_SYNAPSE_WEIGHTS_CACHE_PATH, device)
    )


def build_session_config(
    device: str,
    emotion_calibrations: dict[str, PoolCalibration] = EMOTION_CALIBRATIONS,
    valence_weight: float = VALENCE_WEIGHT,
) -> ReadingSessionConfig:
    """The ReadingSessionConfig every script runs the real connectome with."""
    return ReadingSessionConfig(
        lif_parameters=LIF_PARAMETERS,
        ticks_per_word=TICKS_PER_WORD,
        input_current_scale=INPUT_CURRENT_SCALE,
        valence_weight=valence_weight,
        arousal_weight=AROUSAL_WEIGHT,
        token_seed=TOKEN_SEED,
        engagement_window_size=CONTEXT_WORD_COUNT * TICKS_PER_WORD,
        engagement_threshold=ENGAGEMENT_THRESHOLD,
        min_ticks_before_boredom_check=CONTEXT_WORD_COUNT * TICKS_PER_WORD,
        display_window_size=DISPLAY_WORD_COUNT * TICKS_PER_WORD,
        emotion_calibrations=emotion_calibrations,
        device=device,
    )
