"""Dimensionless feature construction for wobble-time modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_dimensionless_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with physically motivated dimensionless groups."""
    result = frame.copy()
    i1 = result["parameter_inertia_1"].astype(float)
    i2 = result["parameter_inertia_2"].astype(float)
    i3 = result["parameter_inertia_3"].astype(float)
    mass = result["parameter_mass"].astype(float)
    gravity = result["parameter_gravity"].astype(float)
    center = result[
        [
            "parameter_center_of_mass_1",
            "parameter_center_of_mass_2",
            "parameter_center_of_mass_3",
        ]
    ].to_numpy(float)
    radius = np.linalg.norm(center, axis=1)
    i_perp = (i1 + i2) / 2
    valid = radius > 0
    timescale = np.full(len(result), np.nan)
    timescale[valid] = np.sqrt(
        i_perp.to_numpy()[valid]
        / (mass.to_numpy()[valid] * gravity.to_numpy()[valid] * radius[valid])
    )
    result["I_perp"] = i_perp
    result["inertia_asymmetry"] = (i1 - i2) / i_perp
    result["inertia_ratio"] = i3 / i_perp
    result["center_of_mass_distance"] = radius
    result["gravity_timescale"] = timescale
    result["gravity_frequency"] = 1.0 / timescale
    result["dimensionless_spin"] = (
        result["parameter_initial_angular_velocity_3"].astype(float) * timescale
    )
    mean_damping = result[
        ["parameter_damping_1", "parameter_damping_2", "parameter_damping_3"]
    ].mean(axis=1)
    result["dimensionless_damping"] = mean_damping * timescale / i_perp
    result["initial_tilt"] = np.arccos(
        np.clip(
            1
            - 2
            * (
                result["parameter_initial_quaternion_2"].astype(float) ** 2
                + result["parameter_initial_quaternion_3"].astype(float) ** 2
            ),
            -1,
            1,
        )
    )
    result["dimensionless_wobble_time"] = (
        result["wobble_onset_time"].astype(float) / timescale
    )
    return result

