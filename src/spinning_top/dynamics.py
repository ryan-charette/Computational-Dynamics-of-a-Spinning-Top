"""Rigid-body equations and physical invariants."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .parameters import SimulationConfig
from .quaternion import multiply, rotation_matrix

Array = npt.NDArray[np.float64]


def gravitational_torque(q: npt.ArrayLike, config: SimulationConfig) -> Array:
    """Return gravitational torque in the body frame."""
    gravity_world = np.array([0.0, 0.0, -config.gravity])
    gravity_body = rotation_matrix(q).T @ gravity_world
    return np.asarray(
        np.cross(np.asarray(config.center_of_mass), config.mass * gravity_body),
        dtype=float,
    )


def damping_torque(omega: npt.ArrayLike, config: SimulationConfig) -> Array:
    """Return diagonal linear damping torque in the body frame."""
    return np.asarray(-np.asarray(config.damping) * np.asarray(omega, dtype=float), dtype=float)


def rhs(_time: float, state: npt.ArrayLike, config: SimulationConfig) -> Array:
    """Evaluate quaternion and body-angular-velocity derivatives."""
    values = np.asarray(state, dtype=float)
    q = values[:4]
    q_forces = q / np.linalg.norm(q)
    omega = values[4:]
    inertia = np.asarray(config.inertia)
    torque = gravitational_torque(q_forces, config) + damping_torque(omega, config)
    omega_dot = (torque - np.cross(omega, inertia * omega)) / inertia
    q_dot = 0.5 * multiply(q_forces, np.concatenate(([0.0], omega)))
    return np.concatenate((q_dot, omega_dot))


def kinetic_energy(omega: npt.ArrayLike, config: SimulationConfig) -> float:
    """Return rotational kinetic energy."""
    values = np.asarray(omega)
    return float(0.5 * np.dot(values, np.asarray(config.inertia) * values))


def potential_energy(q: npt.ArrayLike, config: SimulationConfig) -> float:
    """Return gravitational potential energy relative to the pivot."""
    position_world = rotation_matrix(q) @ np.asarray(config.center_of_mass)
    return float(config.mass * config.gravity * position_world[2])


def total_energy(state: npt.ArrayLike, config: SimulationConfig) -> float:
    """Return kinetic plus gravitational potential energy."""
    values = np.asarray(state)
    return kinetic_energy(values[4:], config) + potential_energy(values[:4], config)


def angular_momentum_world(state: npt.ArrayLike, config: SimulationConfig) -> Array:
    """Return angular momentum expressed in world coordinates."""
    values = np.asarray(state)
    body = np.asarray(config.inertia) * values[4:]
    return np.asarray(rotation_matrix(values[:4]) @ body, dtype=float)
