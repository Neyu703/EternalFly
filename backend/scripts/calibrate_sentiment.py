"""One-off script: measures the resting and peak spike rates of the fly's three emotion
populations (reward dopamine neurons, punishment dopamine neurons, octopamine neurons; see
reading_session.ReadingSession) on the real cached connectome, prints them as the
EMOTION_CALIBRATIONS block for scripts/emotion_calibration.py, then checks what the
calibrated emotions show for neutral, positive, negative and mixed passages. Not
unit-tested itself - this is exploratory tooling to re-run whenever the connectome cache,
lexicon or input weights change."""

import statistics
import sys
from pathlib import Path

import numpy
import scipy.sparse
import torch

from eternalfly.emotion_decoder import EMOTION_NAMES, PoolCalibration
from eternalfly.lif import LIFParameters
from eternalfly.reading_session import ReadingSession, ReadingSessionConfig
from eternalfly.text_encoder import tokenize_text

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = DATA_DIR / "cache"
WEIGHT_SCALE = 0.15
INPUT_CURRENT_SCALE = 30.0
TICKS_PER_WORD = 10
DISPLAY_WORD_COUNT = 10

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
MIXED_TEXT = (
    "The knight smiled warmly at the wonderful feast, then screamed in terror as the "
    "horrible fire spread. "
) * 8
# The most strongly charged text there is: every word at the lexicon's extreme.
PEAK_POSITIVE_TEXT = "wonderful " * 80
PEAK_NEGATIVE_TEXT = "horrible " * 80

# Words read before measuring, so the network has settled into its ongoing activity (a
# freshly started network fires at roughly half its settled rate) and the display window
# only holds words of the passage being measured.
SETTLING_WORD_COUNT = 30

# Reads the smoothed raw population rates straight out of TickResult.emotions.
IDENTITY_CALIBRATIONS = {name: PoolCalibration(resting_rate=0.0, peak_rate=1.0) for name in EMOTION_NAMES}


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


def build_session(emotion_calibrations: dict[str, PoolCalibration], valence_weight: float, device: str) -> ReadingSession:
    """A ReadingSession over the real connectome, set up like scripts/run_server.py."""
    adjacency_matrix = load_adjacency_as_torch_sparse(device)
    pool_indices = load_pool_indices(CACHE_DIR / "pool_indices.npz", device)
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
        valence_weight=valence_weight,
        arousal_weight=1.0,
        token_seed=42,
        engagement_window_size=500 * TICKS_PER_WORD,
        engagement_threshold=0.15,
        min_ticks_before_boredom_check=500 * TICKS_PER_WORD,
        display_window_size=DISPLAY_WORD_COUNT * TICKS_PER_WORD,
        emotion_calibrations=emotion_calibrations,
        device=device,
    )
    return ReadingSession(adjacency_matrix.shape[0], adjacency_matrix, pool_indices, tokenize_text(NEUTRAL_TEXT), config)


def mean_emotions_after_settling(session: ReadingSession, text: str) -> dict[str, float]:
    """Read text with session and return its emotions averaged over every tick after the
    first SETTLING_WORD_COUNT words (up to, not including, the book's end)."""
    tokens = tokenize_text(text)
    session.load_new_text(tokens)
    tick_results = [session.tick() for _ in range(len(tokens) * TICKS_PER_WORD)]
    measured = tick_results[SETTLING_WORD_COUNT * TICKS_PER_WORD :]
    return {name: statistics.fmean(result.emotions[name] for result in measured) for name in EMOTION_NAMES}


def measure_calibrations(valence_weight: float, device: str) -> dict[str, PoolCalibration]:
    """Resting rate of each population while reading neutral text, and its peak rate while
    reading the most strongly charged text (arousal: whichever charged text drives it more)."""
    session = build_session(IDENTITY_CALIBRATIONS, valence_weight, device)
    mean_emotions_after_settling(session, NEUTRAL_TEXT)  # settle the freshly started network
    resting_rates = mean_emotions_after_settling(session, NEUTRAL_TEXT)
    peak_positive_rates = mean_emotions_after_settling(session, PEAK_POSITIVE_TEXT)
    mean_emotions_after_settling(session, NEUTRAL_TEXT)  # let the reward response decay again
    peak_negative_rates = mean_emotions_after_settling(session, PEAK_NEGATIVE_TEXT)
    peak_rates = {
        "reward": peak_positive_rates["reward"],
        "aversion": peak_negative_rates["aversion"],
        "arousal": max(peak_positive_rates["arousal"], peak_negative_rates["arousal"]),
    }
    return {name: PoolCalibration(resting_rate=resting_rates[name], peak_rate=peak_rates[name]) for name in EMOTION_NAMES}


def print_calibrations(calibrations: dict[str, PoolCalibration]) -> None:
    """Print calibrations as the EMOTION_CALIBRATIONS block of scripts/emotion_calibration.py."""
    print("EMOTION_CALIBRATIONS = {")
    for name, calibration in calibrations.items():
        print(f'    "{name}": PoolCalibration(resting_rate={calibration.resting_rate:.4f}, peak_rate={calibration.peak_rate:.4f}),')
    print("}")


def print_passage_check(calibrations: dict[str, PoolCalibration], valence_weight: float, device: str) -> None:
    """Print the calibrated emotions shown for each test passage, read one after another."""
    session = build_session(calibrations, valence_weight, device)
    mean_emotions_after_settling(session, NEUTRAL_TEXT)  # settle the freshly started network
    for label, text in (
        ("neutral", NEUTRAL_TEXT),
        ("positive", POSITIVE_TEXT),
        ("neutral", NEUTRAL_TEXT),
        ("negative", NEGATIVE_TEXT),
        ("neutral", NEUTRAL_TEXT),
        ("mixed", MIXED_TEXT),
    ):
        emotions = mean_emotions_after_settling(session, text)
        print(f"{label:9} " + "  ".join(f"{name} {value:.2f}" for name, value in emotions.items()))


def main() -> None:
    """Measure and print the calibrations for the valence_weight given as CLI arg
    (default 1.0), then the passage check using them."""
    valence_weight = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calibrations = measure_calibrations(valence_weight, device)
    print_calibrations(calibrations)
    print_passage_check(calibrations, valence_weight, device)


if __name__ == "__main__":
    main()
