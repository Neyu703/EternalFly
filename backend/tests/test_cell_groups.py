import numpy
import pyarrow
import pytest

from eternalfly.cell_groups import (
    BEHAVIOR_SELECTORS,
    DOPAMINE_PUNISHMENT_SELECTOR,
    DOPAMINE_REWARD_SELECTOR,
    MBON_SELECTOR,
    OCTOPAMINE_AROUSAL_SELECTOR,
    SENSORY_CHANNEL_SELECTORS,
    select_cell_group,
    select_remaining_olfactory_receptor_neurons,
)


def _annotations(rows: list[dict]) -> pyarrow.Table:
    columns = ["root_id", "super_class", "cell_class", "cell_sub_class", "cell_type", "side", "top_nt"]
    return pyarrow.table({column: [row.get(column, "") if column != "root_id" else row["root_id"] for row in rows] for column in columns})


ROOT_IDS = numpy.array([1, 2, 3, 4, 5])


def test_select_cell_group_exact_match_on_single_column():
    annotations = _annotations([
        {"root_id": 1, "cell_type": "DNp01"},
        {"root_id": 2, "cell_type": "MDN"},
    ])

    matches = select_cell_group(annotations, ROOT_IDS, [{"cell_type": "DNp01"}])

    assert matches.tolist() == [1]


def test_select_cell_group_and_within_a_clause_requires_all_conditions():
    annotations = _annotations([
        {"root_id": 1, "cell_type": "DNa02", "side": "left"},
        {"root_id": 2, "cell_type": "DNa02", "side": "right"},
    ])

    matches = select_cell_group(annotations, ROOT_IDS, [{"cell_type": "DNa02", "side": "left"}])

    assert matches.tolist() == [1]


def test_select_cell_group_or_across_clauses():
    annotations = _annotations([
        {"root_id": 1, "cell_type": "DNa02", "side": "left"},
        {"root_id": 2, "cell_type": "DNa02", "side": "right"},
        {"root_id": 3, "cell_type": "MDN", "side": "left"},
    ])

    matches = select_cell_group(
        annotations, ROOT_IDS, [{"cell_type": "DNa02", "side": "left"}, {"cell_type": "DNa02", "side": "right"}]
    )

    assert sorted(matches.tolist()) == [1, 2]


def test_select_cell_group_set_condition_is_membership():
    annotations = _annotations([
        {"root_id": 1, "cell_class": "gustatory", "cell_sub_class": "bitter"},
        {"root_id": 2, "cell_class": "gustatory", "cell_sub_class": "sugar/water"},
        {"root_id": 3, "cell_class": "gustatory", "cell_sub_class": "low-salt"},
    ])

    matches = select_cell_group(annotations, ROOT_IDS, [{"cell_sub_class": frozenset({"bitter", "sugar/water"})}])

    assert sorted(matches.tolist()) == [1, 2]


def test_select_cell_group_prefix_condition_is_startswith():
    annotations = _annotations([
        {"root_id": 1, "cell_class": "DAN", "cell_type": "PAM01"},
        {"root_id": 2, "cell_class": "DAN", "cell_type": "PPL101"},
        {"root_id": 3, "cell_class": "DAN", "cell_type": "PPL201"},
    ])

    matches = select_cell_group(annotations, ROOT_IDS, [{"cell_class": "DAN", "cell_type": ("prefix", "PPL1")}])

    assert matches.tolist() == [2]


def test_select_cell_group_restricted_to_given_root_ids():
    annotations = _annotations([{"root_id": 1, "cell_type": "DNp01"}, {"root_id": 999, "cell_type": "DNp01"}])

    matches = select_cell_group(annotations, ROOT_IDS, [{"cell_type": "DNp01"}])

    assert matches.tolist() == [1]


def test_select_cell_group_empty_selector_matches_nothing():
    annotations = _annotations([{"root_id": 1, "cell_type": "DNp01"}])

    matches = select_cell_group(annotations, ROOT_IDS, [])

    assert matches.tolist() == []


def test_select_remaining_olfactory_receptor_neurons_excludes_already_named_glomeruli():
    annotations = _annotations([
        {"root_id": 1, "cell_class": "olfactory", "cell_type": "ORN_DA2"},  # named ("rot")
        {"root_id": 2, "cell_class": "olfactory", "cell_type": "ORN_DL1"},  # not named
        {"root_id": 3, "cell_class": "gustatory", "cell_type": "LB3"},  # not an ORN at all
    ])

    remaining = select_remaining_olfactory_receptor_neurons(annotations, ROOT_IDS, [{"rot": SENSORY_CHANNEL_SELECTORS["rot"]}["rot"]])

    assert remaining.tolist() == [2]


@pytest.mark.parametrize("selector_name", list(SENSORY_CHANNEL_SELECTORS))
def test_every_sensory_channel_selector_is_non_empty_list_of_dict_clauses(selector_name):
    selector = SENSORY_CHANNEL_SELECTORS[selector_name]
    assert isinstance(selector, list) and len(selector) > 0
    for clause in selector:
        assert isinstance(clause, dict) and len(clause) > 0


@pytest.mark.parametrize("selector_name", list(BEHAVIOR_SELECTORS))
def test_every_behavior_selector_is_non_empty_list_of_dict_clauses(selector_name):
    selector = BEHAVIOR_SELECTORS[selector_name]
    assert isinstance(selector, list) and len(selector) > 0
    for clause in selector:
        assert isinstance(clause, dict) and len(clause) > 0


def test_dopamine_and_octopamine_and_mbon_selectors_are_non_empty():
    for selector in (DOPAMINE_REWARD_SELECTOR, DOPAMINE_PUNISHMENT_SELECTOR, OCTOPAMINE_AROUSAL_SELECTOR, MBON_SELECTOR):
        assert isinstance(selector, list) and len(selector) > 0
