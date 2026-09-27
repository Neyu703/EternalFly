"""Argument checks shared across the simulation modules."""


def require_positive_int(name: str, value: object) -> None:
    """Raise ValueError unless value is an int greater than zero; name labels it in the message."""
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
