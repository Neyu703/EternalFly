"""Declarative FlyWire cell-type selectors: which real annotated neurons make up each
sensory input channel, reward/punishment/arousal readout pool, and descending/motor
behavior readout. One place so the connectome cache build, the reading session and the
frontend all agree on what a "channel" or "readout" means.

Every selector's cell IDENTITY here (cell_class/cell_sub_class/cell_type/side) is a
real annotation curated by the FlyWire team, verified directly against
Supplemental_file1_neuron_annotations.tsv (flywire_annotations v3.1.0) rather than
guessed from a remembered cell-type code - see scripts/download_connectome.py. The
BEHAVIORAL/SENSORY role documented for each in the comments below is established
Drosophila neuroscience layered on top of that real cell identity, cited inline; it is
not itself something the annotation file states.
"""

from typing import Union

import numpy
import pyarrow

# A Selector is an OR of AND-clauses: a neuron matches if it satisfies every condition
# in at least one clause. A condition is a single string (exact match), a set of
# strings (membership), or ("prefix", str) (startswith) - see _condition_matches.
_Condition = Union[str, frozenset[str], tuple]
Selector = list[dict[str, _Condition]]


def _condition_matches(column: pyarrow.ChunkedArray, condition: _Condition) -> numpy.ndarray:
    """Return a boolean mask over column for one clause condition (see Selector)."""
    values = numpy.asarray(column)
    if isinstance(condition, tuple):
        _, prefix = condition
        return numpy.char.startswith(values.astype(str), prefix)
    if isinstance(condition, (set, frozenset)):
        return numpy.isin(values, numpy.array(list(condition)))
    return values == condition


def select_cell_group(annotations: pyarrow.Table, root_ids: numpy.ndarray, selector: Selector) -> numpy.ndarray:
    """Return the root ids of every neuron in annotations matching selector (see
    Selector), restricted to root_ids (the proofread/simulated neuron set), in
    annotations' original row order. An empty selector matches nothing."""
    row_count = annotations.num_rows
    combined_mask = numpy.zeros(row_count, dtype=bool)
    for clause in selector:
        clause_mask = numpy.ones(row_count, dtype=bool)
        for column_name, condition in clause.items():
            clause_mask &= _condition_matches(annotations[column_name], condition)
        combined_mask |= clause_mask

    matching_ids = numpy.asarray(annotations["root_id"])[combined_mask].astype(numpy.int64)
    return matching_ids[numpy.isin(matching_ids, root_ids.astype(numpy.int64))]


def select_remaining_olfactory_receptor_neurons(
    annotations: pyarrow.Table, root_ids: numpy.ndarray, already_named_selectors: list[Selector]
) -> numpy.ndarray:
    """Return every real ORN_* olfactory receptor neuron NOT already claimed by one of
    already_named_selectors (the specific glomeruli with a fixed sensory-channel
    meaning - see cell_groups.SENSORY_CHANNEL_SELECTORS). Feeds semantic_encoder.py's
    fallback "semantic odor" projection (C4): every glomerulus FlyWire annotates that
    isn't already a named channel gets driven by a fixed hash of the word's embedding
    instead, so no real ORN type goes completely unused."""
    all_orns = select_cell_group(annotations, root_ids, [{"cell_class": "olfactory", "cell_type": ("prefix", "ORN_")}])
    named_ids = numpy.concatenate(
        [select_cell_group(annotations, root_ids, selector) for selector in already_named_selectors]
        or [numpy.array([], dtype=numpy.int64)]
    )
    return all_orns[~numpy.isin(all_orns, named_ids)]


# --- sensory input channels ----------------------------------------------------------
# cell_sub_class values are FlyWire's own curated sub-classification (not inferred here)
# - see e.g. Frank et al. 2015 J Neurosci / Enjin et al. 2016 Curr Biol for the VP-
# compartment thermo-/hygrosensory nomenclature, Kamikouchi et al. 2009 Nature for the
# auditory/wind_gravity Johnston's Organ split.
SENSORY_CHANNEL_SELECTORS: dict[str, Selector] = {
    "sweet": [{"cell_class": "gustatory", "cell_sub_class": frozenset({"sugar/water"})}],
    "bitter": [{"cell_class": "gustatory", "cell_sub_class": frozenset({"bitter"})}],
    # LC4/LPLC2: looming-selective visual projection neurons that trigger escape
    # (de Vries & Clandinin 2012 Curr Biol; Klapoetke et al. 2017 Nature).
    "looming": [{"cell_type": frozenset({"LC4", "LPLC2"})}],
    "heat": [{"cell_class": "thermosensory", "cell_sub_class": frozenset({"heating"})}],
    "cold": [{"cell_class": "thermosensory", "cell_sub_class": frozenset({"cold"})}],
    "sound": [{"cell_class": "mechanosensory", "cell_sub_class": frozenset({"auditory"})}],
    "wind": [{"cell_class": "mechanosensory", "cell_sub_class": frozenset({"wind_gravity"})}],
    "touch": [{"cell_class": "mechanosensory", "cell_sub_class": frozenset({"head bristle", "grooming"})}],
    # ORN_DA2 responds to geosmin, the signature volatile of microbial spoilage
    # (Stensmyr et al. 2012 Cell); ORN_V responds to CO2 (Suh et al. 2004 Nature).
    "rot": [{"cell_type": frozenset({"ORN_DA2", "ORN_V"})}],
    # ORN_DM1/DM4 respond most strongly to fruity esters (DoOR 2.0, Münch & Galizia 2016).
    "fruit": [{"cell_type": frozenset({"ORN_DM1", "ORN_DM4"})}],
    # cell_sub_class "pheromone" is FlyWire's own curated label for the 5 real ORN
    # glomeruli known to respond to Drosophila pheromones (cVA and others).
    "pheromone": [{"cell_class": "olfactory", "cell_sub_class": frozenset({"pheromone"})}],
}

# --- reward/punishment/arousal readout pools ------------------------------------------
# PAM = reward-coding dopaminergic cluster, PPL1 = punishment-coding dopaminergic
# cluster (Aso et al. 2014 eLife "Mushroom body output neurons..."); OA- = octopaminergic
# neurons (FlyWire's own cell_type naming convention for them).
DOPAMINE_REWARD_SELECTOR: Selector = [{"cell_class": "DAN", "cell_type": ("prefix", "PAM")}]
DOPAMINE_PUNISHMENT_SELECTOR: Selector = [{"cell_class": "DAN", "cell_type": ("prefix", "PPL1")}]
OCTOPAMINE_AROUSAL_SELECTOR: Selector = [{"cell_type": ("prefix", "OA-")}]
MBON_SELECTOR: Selector = [{"cell_class": "MBON"}]

# --- descending/motor behavior readouts ------------------------------------------------
# Every cell_type below is directly present in the annotation file under this exact
# name; the behavior each drives is established Drosophila literature: DNp01 "Giant
# Fiber" = startle/escape jump (von Reyn et al. 2014 Nat Neurosci); MDN = backward
# walking (Bidaye et al. 2014 Science; Sen et al. 2019 eLife); DNa02 = steering during
# walking (Rayshubskiy et al. 2020 bioRxiv/2024 Nature); brain_motor_neuron/
# proboscis_motor_neuron is FlyWire's own curated label for the real feeding motor pool.
BEHAVIOR_SELECTORS: dict[str, Selector] = {
    "escape": [{"cell_type": "DNp01"}],
    "feeding": [{"cell_class": "brain_motor_neuron", "cell_sub_class": "proboscis_motor_neuron"}],
    "backing": [{"cell_type": "MDN"}],
    "turn_left": [{"cell_type": "DNa02", "side": "left"}],
    "turn_right": [{"cell_type": "DNa02", "side": "right"}],
}
