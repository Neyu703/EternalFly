"""One place to load the real connectome cache and the real multilingual embedding
model, and build a fully-wired ReadingSession from them. Composes already-tested
eternalfly functions; not unit-tested itself, same convention as
scripts/build_connectome_cache.py - real model/network loading, no synthetic
substitute worth the complexity (semantic_encoder.py's own logic IS fully tested, via
an injected EmbedFn - see its tests)."""

from pathlib import Path

import numpy
import scipy.sparse
import torch
from transformers import AutoModel, AutoTokenizer

from eternalfly.lif import SHIU_2024_PARAMETERS, SynapticLIFParameters
from eternalfly.neuropils import ALL_NEUROPIL_NAMES
from eternalfly.reading_session import ReadingSession, ReadingSessionConfig, SensoryChannelConfig
from eternalfly.semantic_encoder import (
    BACKGROUND_WORDS,
    NEGATIVE_CONTEXT_ANCHORS,
    POSITIVE_CONTEXT_ANCHORS,
    SENSORY_CHANNEL_ANCHORS,
    EmbedFn,
    build_odor_projection,
    calibrate_channel,
    mean_pool_embeddings,
    mean_unit_embedding,
)
from eternalfly.synapses import EventDrivenSynapses, build_event_driven_synapses, build_signed_edge_weights

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = DATA_DIR / "cache"

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"

ODOR_PROJECTION_SEED = 0
ODOR_ACTIVE_FRACTION = 0.05  # real Drosophila Kenyon-cell sparse coding is ~5% active (Turner et al. 2008)

# Real Drosophila sensory receptor neurons saturate on the order of a few hundred Hz
# (e.g. ORNs ~200-300Hz at high odor concentration, gustatory neurons ~100-200Hz);
# 200Hz is used as one shared ceiling across every named channel and the remaining
# olfactory glomeruli, verified reasonable by scripts/audit_brain.py.
SENSORY_MAX_RATE_HZ = 200.0
# Decay time constants per plan/channel physiology: taste/smell linger across several
# words (Kaissling 2001, olfactory receptor adaptation ~1-2s); looming is a phasic
# visual event with no real "lingering drive" (LC4/LPLC2 fire on stimulus expansion,
# not after it), so its decay is short but never literally zero (LIF discretization
# needs a positive time constant); mechanosensory and thermosensory neurons both show
# fast onset/adaptation (Kamikouchi 2009 for JO; Frank 2015/Enjin 2016 for the VP-
# compartment thermo cells) so both share the same "quick" bucket.
SLOW_DECAY_MS = 1500.0  # sweet, bitter, rot, fruit, pheromone
QUICK_DECAY_MS = 300.0  # sound, wind, touch, heat, cold
LOOMING_DECAY_MS = 50.0
SLOW_CHANNELS = {"sweet", "bitter", "rot", "fruit", "pheromone"}
QUICK_CHANNELS = {"sound", "wind", "touch", "heat", "cold"}

# Real PAM/PPL1 dopaminergic neurons fire up to roughly 20-50Hz during strong
# reinforcement (Cohn et al. 2015; Yamada et al. 2023).
VALENCE_INJECTION_MAX_RATE_HZ = 50.0

SIM_MS_PER_WORD = 150.0
CONTEXT_WINDOW_WORD_COUNT = 12
DISPLAY_TIME_CONSTANT_MS = 300.0
ENGAGEMENT_TIME_CONSTANT_MS = 60_000.0
ENGAGEMENT_THRESHOLD = 0.15
MIN_WORDS_BEFORE_BOREDOM_CHECK = 500

# Real achievable trailing rates measured against the cached connectome by
# scripts/audit_brain.py (30-word sustained stimulation, trailing-10-word mean):
# reward_pam ~0.022 under a strongly positive context, punishment_ppl1 ~0.070 under a
# strongly negative one, arousal_oa ~0.017-0.021 across baseline/sensory stimulation.
# Reward and punishment are asymmetric (PPL1 is a much smaller pool - 16 neurons vs
# PAM's 307 - so its per-neuron rate swings higher for the same relative drive), which
# is exactly why these are measured per-pool rather than assumed equal.
POSITIVE_VALENCE_CEILING = 0.022
NEGATIVE_VALENCE_CEILING = 0.070
AROUSAL_CEILING = 0.020


def channel_config_for(name: str) -> SensoryChannelConfig:
    """The Poisson-injection tuning for one named sensory channel (see
    SLOW_CHANNELS/QUICK_CHANNELS above; anything else, i.e. "looming", gets the short
    LOOMING_DECAY_MS)."""
    if name in SLOW_CHANNELS:
        decay_ms = SLOW_DECAY_MS
    elif name in QUICK_CHANNELS:
        decay_ms = QUICK_DECAY_MS
    else:
        decay_ms = LOOMING_DECAY_MS
    return SensoryChannelConfig(max_rate_hz=SENSORY_MAX_RATE_HZ, decay_ms=decay_ms)


def load_embed_fn(device: str = "cpu") -> EmbedFn:
    """Load the real multilingual sentence-embedding model and return a closure batch-
    embedding a list of words/phrases into L2-normalized vectors."""
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME, revision=EMBEDDING_MODEL_REVISION)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME, revision=EMBEDDING_MODEL_REVISION, use_safetensors=True)
    model.to(device)
    model.eval()

    def embed(texts: list[str]) -> numpy.ndarray:
        with torch.no_grad():
            encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
            output = model(**encoded)
            return mean_pool_embeddings(output.last_hidden_state.cpu().numpy(), encoded["attention_mask"].cpu().numpy())

    return embed


def load_cell_groups(device: str) -> dict[str, torch.Tensor]:
    """Load cell_groups.npz (see scripts/build_connectome_cache.py) as index tensors."""
    raw = numpy.load(CACHE_DIR / "cell_groups.npz")
    return {name: torch.as_tensor(raw[name], dtype=torch.int64, device=device) for name in raw.files}


def load_synapses(lif_parameters: SynapticLIFParameters, device: str) -> EventDrivenSynapses:
    """Load synapses.npz (pre-major CSR, raw signed syn_counts) and scale it into real
    Shiu-calibrated conductance weights (lif_parameters.per_synapse_weight)."""
    raw = scipy.sparse.load_npz(CACHE_DIR / "synapses.npz").tocsr()
    scaled_values = build_signed_edge_weights(numpy.abs(raw.data), numpy.sign(raw.data), lif_parameters.per_synapse_weight)
    scaled = scipy.sparse.csr_matrix((scaled_values, raw.indices, raw.indptr), shape=raw.shape)
    return build_event_driven_synapses(scaled, device=device)


def load_neuropil_readout_matrix(device: str) -> torch.Tensor:
    """Load neuropil_readout.npz as a dense torch tensor (only 78 rows, dense is
    simpler and cheap - see connectome.neuropil_readout_matrix)."""
    sparse_matrix = scipy.sparse.load_npz(CACHE_DIR / "neuropil_readout.npz")
    return torch.as_tensor(sparse_matrix.toarray(), dtype=torch.float32, device=device)


def build_channel_calibrations(embed_fn: EmbedFn) -> dict:
    """Embed every channel's DE+EN anchors plus the shared background vocabulary once,
    and calibrate every real sensory channel (see semantic_encoder.calibrate_channel)."""
    background_embeddings = embed_fn(BACKGROUND_WORDS)
    return {
        name: calibrate_channel(embed_fn(anchors), background_embeddings)
        for name, anchors in SENSORY_CHANNEL_ANCHORS.items()
    }


def build_context_anchors(embed_fn: EmbedFn) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Embed context_valence's positive/negative anchor word lists into their centroids."""
    return (
        mean_unit_embedding(embed_fn(POSITIVE_CONTEXT_ANCHORS)),
        mean_unit_embedding(embed_fn(NEGATIVE_CONTEXT_ANCHORS)),
    )


def build_session_config(embedding_dim: int, device: str = "cpu") -> ReadingSessionConfig:
    return ReadingSessionConfig(
        lif_parameters=SHIU_2024_PARAMETERS,
        sim_ms_per_word=SIM_MS_PER_WORD,
        channel_configs={name: channel_config_for(name) for name in SENSORY_CHANNEL_ANCHORS},
        odor_config=SensoryChannelConfig(max_rate_hz=SENSORY_MAX_RATE_HZ, decay_ms=SLOW_DECAY_MS),
        odor_active_fraction=ODOR_ACTIVE_FRACTION,
        context_window_word_count=CONTEXT_WINDOW_WORD_COUNT,
        valence_injection_max_rate_hz=VALENCE_INJECTION_MAX_RATE_HZ,
        display_time_constant_ms=DISPLAY_TIME_CONSTANT_MS,
        engagement_time_constant_ms=ENGAGEMENT_TIME_CONSTANT_MS,
        engagement_threshold=ENGAGEMENT_THRESHOLD,
        min_words_before_boredom_check=MIN_WORDS_BEFORE_BOREDOM_CHECK,
        positive_valence_ceiling=POSITIVE_VALENCE_CEILING,
        negative_valence_ceiling=NEGATIVE_VALENCE_CEILING,
        arousal_ceiling=AROUSAL_CEILING,
        device=device,
    )


def build_reading_session(tokens: list[str], device: str = "cpu") -> ReadingSession:
    """Load everything and build a real, fully-wired ReadingSession over tokens."""
    root_ids = numpy.load(DATA_DIR / "proofread_root_ids_783.npy")
    neuron_count = len(root_ids)

    config = build_session_config(embedding_dim=384, device=device)
    embed_fn = load_embed_fn(device=device)
    cell_groups = load_cell_groups(device)
    synapses = load_synapses(config.lif_parameters, device)
    neuropil_readout_matrix = load_neuropil_readout_matrix(device)
    odor_projection = torch.as_tensor(
        build_odor_projection(len(cell_groups["olfactory_remaining"]), embedding_dim=384, seed=ODOR_PROJECTION_SEED),
        dtype=torch.float32,
        device=device,
    )
    channel_calibrations = build_channel_calibrations(embed_fn)
    positive_context_anchor, negative_context_anchor = build_context_anchors(embed_fn)

    return ReadingSession(
        neuron_count=neuron_count,
        synapses=synapses,
        cell_groups=cell_groups,
        neuropil_readout_matrix=neuropil_readout_matrix,
        neuropil_names=ALL_NEUROPIL_NAMES,
        odor_projection=odor_projection,
        channel_calibrations=channel_calibrations,
        positive_context_anchor=positive_context_anchor,
        negative_context_anchor=negative_context_anchor,
        embed_fn=embed_fn,
        tokens=tokens,
        config=config,
    )
