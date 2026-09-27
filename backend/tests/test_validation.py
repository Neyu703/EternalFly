import pytest

from eternalfly.validation import require_positive_int


def test_require_positive_int_accepts_a_positive_int():
    require_positive_int("count", 3)


@pytest.mark.parametrize("value", [0, -2, 2.5, "3", None])
def test_require_positive_int_rejects_anything_else_naming_the_argument(value):
    with pytest.raises(ValueError, match=r"^count must be a positive integer, got "):
        require_positive_int("count", value)
