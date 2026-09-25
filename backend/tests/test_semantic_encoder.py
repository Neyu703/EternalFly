import numpy
import pytest

from eternalfly.semantic_encoder import (
    SENSORY_CHANNEL_ANCHORS,
    ChannelCalibration,
    WordEmbeddingCache,
    build_odor_projection,
    calibrate_channel,
    channel_drive,
    channel_drives,
    context_valence,
    mean_pool_embeddings,
    mean_unit_embedding,
    semantic_odor,
)


def test_mean_pool_embeddings_averages_only_real_tokens_and_normalizes():
    # sequence 1: two real tokens [1,0] and [0,1], no padding -> mean [.5,.5], normalized.
    # sequence 2: one real token [1,0] and one padding token (masked out) [99,99].
    token_embeddings = numpy.array([
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [99.0, 99.0]],
    ])
    attention_mask = numpy.array([[1, 1], [1, 0]])

    pooled = mean_pool_embeddings(token_embeddings, attention_mask)

    expected_first = numpy.array([0.5, 0.5]) / numpy.linalg.norm([0.5, 0.5])
    assert numpy.allclose(pooled[0], expected_first)
    assert numpy.allclose(pooled[1], [1.0, 0.0])
    assert numpy.allclose(numpy.linalg.norm(pooled, axis=1), [1.0, 1.0])


def test_mean_pool_embeddings_handles_a_sequence_with_no_real_tokens_without_dividing_by_zero():
    token_embeddings = numpy.zeros((1, 3, 2))
    attention_mask = numpy.zeros((1, 3), dtype=int)

    pooled = mean_pool_embeddings(token_embeddings, attention_mask)

    assert not numpy.isnan(pooled).any()


def test_mean_unit_embedding_returns_normalized_centroid():
    embeddings = numpy.array([[1.0, 0.0], [0.0, 1.0]])

    centroid = mean_unit_embedding(embeddings)

    assert numpy.allclose(centroid, [0.7071067811865475, 0.7071067811865475])
    assert numpy.linalg.norm(centroid) == pytest.approx(1.0)


def test_calibrate_channel_computes_centroid_and_background_stats():
    anchor_embeddings = numpy.array([[1.0, 0.0], [0.0, 1.0]])
    background_embeddings = numpy.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    expected_centroid = mean_unit_embedding(anchor_embeddings)
    expected_similarities = background_embeddings @ expected_centroid

    calibration = calibrate_channel(anchor_embeddings, background_embeddings)

    assert numpy.allclose(calibration.anchor_embedding, expected_centroid)
    assert calibration.background_mean == pytest.approx(expected_similarities.mean())
    assert calibration.background_std == pytest.approx(expected_similarities.std())


def test_channel_drive_ramps_linearly_from_zero_at_z_two_to_one_at_z_four():
    calibration = ChannelCalibration(anchor_embedding=numpy.array([1.0, 0.0]), background_mean=0.5, background_std=0.1)

    assert channel_drive(numpy.array([0.7, 0.0]), calibration) == pytest.approx(0.0)  # z = 2.0
    assert channel_drive(numpy.array([0.8, 0.0]), calibration) == pytest.approx(0.5)  # z = 3.0
    assert channel_drive(numpy.array([0.9, 0.0]), calibration) == pytest.approx(1.0)  # z = 4.0


def test_channel_drive_clamps_below_zero_and_above_one():
    calibration = ChannelCalibration(anchor_embedding=numpy.array([1.0, 0.0]), background_mean=0.5, background_std=0.1)

    assert channel_drive(numpy.array([0.5, 0.0]), calibration) == 0.0  # z = 0.0, clamped
    assert channel_drive(numpy.array([1.0, 0.0]), calibration) == 1.0  # z = 5.0, clamped


def test_channel_drive_returns_zero_for_degenerate_zero_variance_background():
    calibration = ChannelCalibration(anchor_embedding=numpy.array([1.0, 0.0]), background_mean=0.5, background_std=0.0)

    assert channel_drive(numpy.array([1.0, 0.0]), calibration) == 0.0


def test_channel_drives_returns_one_value_per_named_channel():
    calibrations = {
        "sweet": ChannelCalibration(numpy.array([1.0, 0.0]), 0.5, 0.1),
        "bitter": ChannelCalibration(numpy.array([0.0, 1.0]), 0.5, 0.1),
    }
    word_embedding = numpy.array([0.9, 0.0])

    drives = channel_drives(word_embedding, calibrations)

    assert set(drives) == {"sweet", "bitter"}
    assert drives["sweet"] == pytest.approx(1.0)
    assert drives["bitter"] == pytest.approx(0.0)


def test_sensory_channel_anchors_cover_every_real_channel_with_de_and_en_words():
    expected_channels = {
        "sweet", "bitter", "looming", "heat", "cold", "sound", "wind", "touch", "rot", "fruit", "pheromone",
    }
    assert set(SENSORY_CHANNEL_ANCHORS) == expected_channels
    for channel_name, anchors in SENSORY_CHANNEL_ANCHORS.items():
        assert len(anchors) >= 2, channel_name


def test_build_odor_projection_returns_unit_rows_of_the_right_shape():
    projection = build_odor_projection(glomerulus_count=5, embedding_dim=3, seed=0)

    assert projection.shape == (5, 3)
    assert numpy.allclose(numpy.linalg.norm(projection, axis=1), 1.0)


def test_build_odor_projection_is_deterministic_given_the_same_seed():
    first = build_odor_projection(10, 4, seed=42)
    second = build_odor_projection(10, 4, seed=42)

    assert numpy.array_equal(first, second)


def test_build_odor_projection_differs_across_seeds():
    first = build_odor_projection(10, 4, seed=1)
    second = build_odor_projection(10, 4, seed=2)

    assert not numpy.array_equal(first, second)


def test_semantic_odor_activates_exactly_the_top_active_fraction_by_projection():
    projection_matrix = numpy.array([[1.0, 0.0], [0.5, 0.8660254], [0.0, 1.0], [-1.0, 0.0]])
    word_embedding = numpy.array([1.0, 0.0])  # projections: [1.0, 0.5, 0.0, -1.0]

    odor = semantic_odor(word_embedding, projection_matrix, active_fraction=0.5)

    assert odor.tolist() == [1.0, 1.0, 0.0, 0.0]


def test_semantic_odor_raises_on_non_positive_active_fraction():
    projection_matrix = numpy.eye(3)
    with pytest.raises(ValueError, match="active_fraction"):
        semantic_odor(numpy.array([1.0, 0.0, 0.0]), projection_matrix, active_fraction=0.0)


def test_semantic_odor_raises_on_active_fraction_above_one():
    projection_matrix = numpy.eye(3)
    with pytest.raises(ValueError, match="active_fraction"):
        semantic_odor(numpy.array([1.0, 0.0, 0.0]), projection_matrix, active_fraction=1.5)


def test_semantic_odor_similar_embeddings_overlap_more_than_dissimilar_ones():
    projection_matrix = build_odor_projection(glomerulus_count=200, embedding_dim=8, seed=7)
    rng = numpy.random.default_rng(1)
    base = rng.normal(size=8)
    base /= numpy.linalg.norm(base)
    nearly_identical = base + rng.normal(size=8) * 0.01
    nearly_identical /= numpy.linalg.norm(nearly_identical)
    unrelated = rng.normal(size=8)
    unrelated /= numpy.linalg.norm(unrelated)

    odor_base = semantic_odor(base, projection_matrix, active_fraction=0.05)
    odor_similar = semantic_odor(nearly_identical, projection_matrix, active_fraction=0.05)
    odor_unrelated = semantic_odor(unrelated, projection_matrix, active_fraction=0.05)

    overlap_similar = numpy.logical_and(odor_base, odor_similar).sum()
    overlap_unrelated = numpy.logical_and(odor_base, odor_unrelated).sum()
    assert overlap_similar > overlap_unrelated


def test_context_valence_positive_context_is_positive():
    positive_anchor = numpy.array([1.0, 0.0])
    negative_anchor = numpy.array([0.0, 1.0])
    context_embedding = numpy.array([1.0, 0.0])

    assert context_valence(context_embedding, positive_anchor, negative_anchor) == pytest.approx(1.0)


def test_context_valence_negative_context_is_negative():
    positive_anchor = numpy.array([1.0, 0.0])
    negative_anchor = numpy.array([0.0, 1.0])
    context_embedding = numpy.array([0.0, 1.0])

    assert context_valence(context_embedding, positive_anchor, negative_anchor) == pytest.approx(-1.0)


def test_context_valence_clamps_to_unit_range():
    # A context embedding correlated with BOTH anchors beyond what a difference of two
    # cosines could naturally produce is engineered here just to exercise the clamp.
    positive_anchor = numpy.array([1.0, 0.0])
    negative_anchor = numpy.array([-1.0, 0.0])
    context_embedding = numpy.array([1.0, 0.0])

    result = context_valence(context_embedding, positive_anchor, negative_anchor)

    assert -1.0 <= result <= 1.0
    assert result == pytest.approx(1.0)


class _CountingEmbedder:
    """Fake EmbedFn recording every word it was actually asked to embed."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, words: list[str]) -> numpy.ndarray:
        self.calls.append(list(words))
        return numpy.array([[hash(word) % 100, 0.0] for word in words])


def test_word_embedding_cache_get_returns_none_before_precompute():
    cache = WordEmbeddingCache(_CountingEmbedder())

    assert cache.get("hallo") is None
    assert len(cache) == 0


def test_word_embedding_cache_precompute_then_get_returns_the_embedding():
    embedder = _CountingEmbedder()
    cache = WordEmbeddingCache(embedder)

    cache.precompute(["hallo", "welt"])

    assert cache.get("hallo") is not None
    assert cache.get("welt") is not None
    assert len(cache) == 2
    assert embedder.calls == [["hallo", "welt"]]


def test_word_embedding_cache_precompute_never_re_embeds_an_already_cached_word():
    embedder = _CountingEmbedder()
    cache = WordEmbeddingCache(embedder)

    cache.precompute(["hallo"])
    cache.precompute(["hallo", "welt"])

    assert embedder.calls == [["hallo"], ["welt"]]


def test_word_embedding_cache_precompute_deduplicates_within_a_single_call():
    embedder = _CountingEmbedder()
    cache = WordEmbeddingCache(embedder)

    cache.precompute(["hallo", "hallo", "welt"])

    assert embedder.calls == [["hallo", "welt"]]


def test_word_embedding_cache_precompute_with_no_new_words_skips_embed_fn_call():
    embedder = _CountingEmbedder()
    cache = WordEmbeddingCache(embedder)
    cache.precompute(["hallo"])

    cache.precompute(["hallo"])

    assert embedder.calls == [["hallo"]]
