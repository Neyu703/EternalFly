from eternalfly.neurotransmitters import neurotransmitter_sign


def test_neurotransmitter_sign_ach_is_excitatory():
    assert neurotransmitter_sign("ach") == 1


def test_neurotransmitter_sign_gaba_is_inhibitory():
    assert neurotransmitter_sign("gaba") == -1


def test_neurotransmitter_sign_glut_is_inhibitory():
    assert neurotransmitter_sign("glut") == -1


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
