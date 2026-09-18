from eternalfly.neuropils import ALL_NEUROPIL_NAMES


def test_all_neuropil_names_has_78_unique_entries():
    assert len(ALL_NEUROPIL_NAMES) == 78
    assert len(set(ALL_NEUROPIL_NAMES)) == 78
