"""Scalar-first Hamilton quaternion operations."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def normalize(q: npt.ArrayLike) -> Array:
    """Return a unit quaternion, rejecting zero and non-finite inputs."""
    result = np.asarray(q, dtype=float)
    norm = float(np.linalg.norm(result))
    if result.shape != (4,) or not np.all(np.isfinite(result)) or norm <= 0.0:
        raise ValueError("quaternion must be a finite, nonzero vector of length four")
    return result / norm


def multiply(left: npt.ArrayLike, right: npt.ArrayLike) -> Array:
    """Return the Hamilton product ``left ⊗ right``."""
    w1, x1, y1, z1 = np.asarray(left, dtype=float)
    w2, x2, y2, z2 = np.asarray(right, dtype=float)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def conjugate(q: npt.ArrayLike) -> Array:
    """Return the quaternion conjugate."""
    w, x, y, z = np.asarray(q, dtype=float)
    return np.array([w, -x, -y, -z], dtype=float)


def rotation_matrix(q: npt.ArrayLike) -> Array:
    """Return the body-to-world rotation matrix represented by ``q``."""
    w, x, y, z = normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rotate(q: npt.ArrayLike, vector: npt.ArrayLike) -> Array:
    """Rotate a body-frame vector into world coordinates."""
    return rotation_matrix(q) @ np.asarray(vector, dtype=float)


def from_axis_angle(axis: npt.ArrayLike, angle: float) -> Array:
    """Construct a quaternion from an axis and angle in radians."""
    direction = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(direction))
    if direction.shape != (3,) or not np.all(np.isfinite(direction)) or norm <= 0:
        raise ValueError("axis must be a finite, nonzero vector of length three")
    half = 0.5 * angle
    return np.concatenate(([np.cos(half)], direction / norm * np.sin(half)))


def exponential_map(rotation_vector: npt.ArrayLike) -> Array:
    """Map a three-dimensional rotation vector to a unit quaternion."""
    vector = np.asarray(rotation_vector, dtype=float)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-14:
        return normalize(np.concatenate(([1.0], 0.5 * vector)))
    return from_axis_angle(vector / angle, angle)


def rotation_vector(q: npt.ArrayLike) -> Array:
    """Return the shortest rotation vector represented by a quaternion."""
    unit = normalize(q)
    if unit[0] < 0:
        unit = -unit
    vector_norm = float(np.linalg.norm(unit[1:]))
    if vector_norm < 1e-14:
        return 2.0 * unit[1:]
    angle = 2.0 * np.arctan2(vector_norm, np.clip(unit[0], -1.0, 1.0))
    return unit[1:] * (angle / vector_norm)


def orientation_distance(q1: npt.ArrayLike, q2: npt.ArrayLike) -> float:
    """Return the sign-invariant geodesic distance between orientations."""
    dot = abs(float(np.dot(normalize(q1), normalize(q2))))
    return float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))

