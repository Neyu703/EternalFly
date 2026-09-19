import pytest

from eternalfly.sentiment_lexicon import word_valence


def test_word_valence_is_positive_for_a_clearly_positive_word():
    assert word_valence("love") > 0.5


def test_word_valence_is_negative_for_a_clearly_negative_word():
    assert word_valence("kill") < -0.5


def test_word_valence_is_zero_for_a_word_not_in_the_lexicon():
    assert word_valence("dragon") == 0.0


def test_word_valence_is_case_insensitive():
    assert word_valence("LOVE") == word_valence("love")


@pytest.mark.parametrize("word", ["love", "kill", "hate", "wonderful", "terror", "dragon"])
def test_word_valence_is_always_within_unit_range(word):
    assert -1.0 <= word_valence(word) <= 1.0


def test_word_valence_matches_expected_sign_ordering_between_opposite_words():
    assert word_valence("wonderful") > word_valence("dragon") > word_valence("terror")
