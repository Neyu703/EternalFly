import pytest

from eternalfly.neurotransmitters import dominant_neurotransmitter, neurotransmitter_sign


def test_neurotransmitter_sign_ach_is_excitatory():
    assert neurotransmitter_sign("ach") == 1


def test_neurotransmitter_sign_gaba_is_inhibitory():
    assert neurotransmitter_sign("gaba") == -1


def test_neurotransmitter_sign_glut_is_inhibitory():
    assert neurotransmitter_sign("glut") == -1


def test_neurotransmitter_sign_full_word_glutamate_is_inhibitory():
    assert neurotransmitter_sign("glutamate") == -1


def test_neurotransmitter_sign_full_word_acetylcholine_is_excitatory():
    assert neurotransmitter_sign("acetylcholine") == 1


def test_neurotransmitter_sign_full_word_dopamine_is_excitatory():
    assert neurotransmitter_sign("dopamine") == 1


def test_neurotransmitter_sign_full_word_octopamine_is_excitatory():
    assert neurotransmitter_sign("octopamine") == 1


def test_neurotransmitter_sign_full_word_serotonin_is_excitatory():
    assert neurotransmitter_sign("serotonin") == 1


def test_neurotransmitter_sign_da_is_excitatory():
    assert neurotransmitter_sign("da") == 1


def test_neurotransmitter_sign_oct_is_excitatory():
    assert neurotransmitter_sign("oct") == 1


def test_neurotransmitter_sign_ser_is_excitatory():
    assert neurotransmitter_sign("ser") == 1


def test_neurotransmitter_sign_unknown_falls_back_to_excitatory():
    assert neurotransmitter_sign("mystery_nt") == 1


def test_neurotransmitter_sign_is_case_insensitive():
    assert neurotransmitter_sign("GABA") == -1
    assert neurotransmitter_sign("AcH") == 1


def test_dominant_neurotransmitter_returns_argmax_key():
    probabilities = {"ach": 0.1, "gaba": 0.7, "glut": 0.2}
    assert dominant_neurotransmitter(probabilities) == "gaba"


def test_dominant_neurotransmitter_tie_returns_first_in_dict_order():
    probabilities = {"gaba": 0.5, "ach": 0.5}
    assert dominant_neurotransmitter(probabilities) == "gaba"


def test_dominant_neurotransmitter_raises_on_empty_dict():
    with pytest.raises(ValueError):
        dominant_neurotransmitter({})
