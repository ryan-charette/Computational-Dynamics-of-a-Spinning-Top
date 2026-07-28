from __future__ import annotations

import numpy as np
import pandas as pd

from spinning_top.dimensionless import add_dimensionless_features
from spinning_top.regression import fit_interpretable_models
from spinning_top.survival import fit_weibull_aft


def raw_frame(count: int = 20) -> pd.DataFrame:
    spin = np.linspace(20, 100, count)
    return pd.DataFrame(
        {
            "parameter_inertia_1": 0.2,
            "parameter_inertia_2": 0.21,
            "parameter_inertia_3": 0.3,
            "parameter_mass": 1.0,
            "parameter_gravity": 9.81,
            "parameter_center_of_mass_1": 0.0,
            "parameter_center_of_mass_2": 0.0,
            "parameter_center_of_mass_3": 0.05,
            "parameter_initial_angular_velocity_3": spin,
            "parameter_initial_quaternion_2": 0.0,
            "parameter_initial_quaternion_3": np.linspace(0.01, 0.1, count),
            "parameter_damping_1": 0.002,
            "parameter_damping_2": 0.002,
            "parameter_damping_3": 0.004,
            "wobble_observed": [True] * (count - 4) + [False] * 4,
            "wobble_onset_time": np.concatenate((0.1 * spin[: count - 4], [np.nan] * 4)),
            "observation_horizon": 12.0,
        }
    )


def test_dimensionless_features_and_r_zero() -> None:
    frame = raw_frame(5)
    frame.loc[0, "parameter_center_of_mass_3"] = 0
    result = add_dimensionless_features(frame)
    assert np.isnan(result.loc[0, "gravity_timescale"])
    assert result.loc[1, "dimensionless_spin"] > 0
    assert result.loc[1, "inertia_ratio"] > 0


def test_regression_pipeline_runs() -> None:
    frame = add_dimensionless_features(raw_frame())
    results = fit_interpretable_models(frame.iloc[:14], frame.iloc[14:18])
    assert {item.name for item in results} == {
        "linear",
        "log_linear",
        "polynomial",
        "robust",
        "sparse",
    }
    assert all(np.isfinite(item.metrics["rmse"]) for item in results)


def test_survival_pipeline_uses_censoring() -> None:
    model = fit_weibull_aft(add_dimensionless_features(raw_frame()))
    assert model.median_survival_time_ > 0

