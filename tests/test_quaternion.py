from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from spinning_top.quaternion import (
    conjugate,
    from_axis_angle,
    multiply,
    normalize,
    orientation_distance,
    rotate,
    rotation_matrix,
)


def test_multiply_identity_and_conjugate() -> None:
    q = normalize([1.0, 2.0, -3.0, 4.0])
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    assert multiply(identity, q) == pytest.approx(q)
    assert multiply(q, conjugate(q)) == pytest.approx(identity)


def test_rotation_known_axis_angle() -> None:
    q = from_axis_angle([0, 0, 1], np.pi / 2)
    assert rotate(q, [1, 0, 0]) == pytest.approx([0, 1, 0], abs=1e-12)


def test_orientation_distance_is_sign_invariant() -> None:
    q = normalize([0.9, 0.1, 0.2, 0.3])
    assert orientation_distance(q, -q) == pytest.approx(0.0)


@given(st.lists(st.floats(-10, 10, allow_nan=False), min_size=4, max_size=4))
def test_rotation_matrix_is_orthogonal(values: list[float]) -> None:
    if np.linalg.norm(values) < 1e-8:
        return
    matrix = rotation_matrix(values)
    assert matrix.T @ matrix == pytest.approx(np.eye(3), abs=1e-10)
    assert np.linalg.det(matrix) == pytest.approx(1.0, abs=1e-10)


def test_invalid_quaternion_rejected() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        normalize([0, 0, 0, 0])
