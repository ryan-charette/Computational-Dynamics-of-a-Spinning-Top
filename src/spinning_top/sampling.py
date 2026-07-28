"""Reproducible parameter sampling and physical configuration construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from itertools import product

import numpy as np
from scipy.stats import qmc

from .parameters import SimulationConfig
from .quaternion import from_axis_angle, multiply


def generate_samples(
    ranges: Mapping[str, tuple[float, float]],
    count: int,
    seed: int,
    method: str = "sobol",
) -> list[dict[str, float]]:
    """Generate deterministic samples in named ranges."""
    names = sorted(ranges)
    if not names:
        return [{} for _ in range(count)]
    bounds = np.asarray([ranges[name] for name in names], dtype=float)
    if method == "grid":
        points_per_axis = max(2, int(np.ceil(count ** (1 / len(names)))))
        axes = [np.linspace(low, high, points_per_axis) for low, high in bounds]
        values = np.asarray(list(product(*axes)), dtype=float)[:count]
    elif method == "sobol":
        sampler = qmc.Sobol(d=len(names), scramble=True, seed=seed)
        values = qmc.scale(sampler.random(count), bounds[:, 0], bounds[:, 1])
    elif method == "random":
        generator = np.random.default_rng(seed)
        values = generator.uniform(bounds[:, 0], bounds[:, 1], size=(count, len(names)))
    else:
        raise ValueError(f"unsupported sampler: {method}")
    return [
        {name: float(row[index]) for index, name in enumerate(names)} for row in values
    ]


def sample_to_config(base: SimulationConfig, sample: Mapping[str, float]) -> SimulationConfig:
    """Apply a sample record to a base physical configuration."""
    inertia = [
        sample.get("I1", base.inertia[0]),
        sample.get("I2", base.inertia[1]),
        sample.get("I3", base.inertia[2]),
    ]
    damping = [
        sample.get("damping1", base.damping[0]),
        sample.get("damping2", base.damping[1]),
        sample.get("damping3", sample.get("damping", base.damping[2])),
    ]
    center = np.asarray(base.center_of_mass, dtype=float)
    if "center_of_mass_distance" in sample:
        direction_norm = float(np.linalg.norm(center))
        direction = center / direction_norm if direction_norm > 0 else np.array([0.0, 0.0, 1.0])
        center = direction * sample["center_of_mass_distance"]
    for index, key in enumerate(("center_of_mass_x", "center_of_mass_y", "center_of_mass_z")):
        if key in sample:
            center[index] = sample[key]

    tilt = sample.get("initial_tilt", 0.0)
    precession = sample.get("initial_precession", 0.0)
    q_tilt = from_axis_angle((0.0, 1.0, 0.0), tilt)
    q_precession = from_axis_angle((0.0, 0.0, 1.0), precession)
    quaternion = multiply(q_precession, q_tilt)
    transverse = sample.get("initial_transverse_angular_velocity", 0.0)
    transverse_azimuth = sample.get("initial_transverse_azimuth", 0.0)
    omega = (
        transverse * np.cos(transverse_azimuth),
        transverse * np.sin(transverse_azimuth),
        sample.get("initial_spin_rate", base.initial_angular_velocity[2]),
    )
    return replace(
        base,
        inertia=tuple(inertia),
        mass=sample.get("mass", base.mass),
        center_of_mass=tuple(center),
        damping=tuple(damping),
        initial_quaternion=tuple(quaternion),
        initial_angular_velocity=tuple(omega),
        gravity=sample.get("gravity", base.gravity),
        horizon=sample.get("horizon", base.horizon),
    )
