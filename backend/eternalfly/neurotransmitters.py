"""Pure lookups mapping FlyWire neurotransmitter labels to synaptic signs."""


_INHIBITORY_NEUROTRANSMITTERS = {"gaba", "glut"}


def neurotransmitter_sign(nt_type: str) -> int:
    """Return +1 (excitatory) or -1 (inhibitory) for a neurotransmitter type code."""
    if nt_type.lower() in _INHIBITORY_NEUROTRANSMITTERS:
        return -1
    return 1


def dominant_neurotransmitter(probabilities: dict[str, float]) -> str:
    """Return the key with the highest probability; ties go to the first key in dict order."""
    if not probabilities:
        raise ValueError("probabilities must not be empty")
    return max(probabilities, key=probabilities.get)
