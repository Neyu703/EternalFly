"""Pure lookups mapping FlyWire neurotransmitter labels to synaptic signs."""


# Short codes (data_prep.py's per-connection NT-probability aggregation) and the full
# words the real FlyWire cell-type annotations use for a neuron's own dominant
# transmitter (Supplemental_file1_neuron_annotations.tsv's top_nt column) both need to
# resolve to the same sign. Glutamate is inhibitory at fly central synapses (GluClalpha),
# the same convention Shiu et al. 2024 and Eichler et al. 2017 use for connectome-scale
# Drosophila models.
_INHIBITORY_NEUROTRANSMITTERS = {"gaba", "glut", "glutamate"}


def neurotransmitter_sign(nt_type: str) -> int:
    """Return +1 (excitatory) or -1 (inhibitory) for a neurotransmitter type code or
    full name (e.g. "gaba"/"glut"/"glutamate", both short codes and the full words the
    real FlyWire annotations use)."""
    if nt_type.lower() in _INHIBITORY_NEUROTRANSMITTERS:
        return -1
    return 1


def dominant_neurotransmitter(probabilities: dict[str, float]) -> str:
    """Return the key with the highest probability; ties go to the first key in dict order."""
    if not probabilities:
        raise ValueError("probabilities must not be empty")
    return max(probabilities, key=probabilities.get)
