"""Analytical baselines and interpretable regression models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, HuberRegressor, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

FEATURES = [
    "dimensionless_spin",
    "dimensionless_damping",
    "inertia_ratio",
    "inertia_asymmetry",
    "initial_tilt",
]


@dataclass(frozen=True, slots=True)
class ModelResult:
    """Fitted estimator and held-out metrics."""

    name: str
    estimator: Any
    metrics: dict[str, float]


def critical_spin_time(
    initial_spin: np.ndarray,
    critical_spin: np.ndarray,
    inertia_3: np.ndarray,
    damping_3: np.ndarray,
) -> np.ndarray:
    """Evaluate the exponential spin-decay onset formula."""
    with np.errstate(divide="ignore", invalid="ignore"):
        prediction = inertia_3 / damping_3 * np.log(initial_spin / critical_spin)
    prediction[(damping_3 <= 0) | (initial_spin <= critical_spin)] = np.nan
    return prediction


def fit_critical_spin_baseline(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[float, np.ndarray, dict[str, float]]:
    """Estimate a constant critical spin and evaluate exponential decay."""
    observed = train.loc[train["wobble_observed"]].dropna(
        subset=["wobble_onset_time"]
    )
    omega0 = observed["parameter_initial_angular_velocity_3"].to_numpy(float)
    inertia3 = observed["parameter_inertia_3"].to_numpy(float)
    damping3 = observed["parameter_damping_3"].to_numpy(float)
    onset = observed["wobble_onset_time"].to_numpy(float)
    inferred = omega0 * np.exp(-damping3 * onset / inertia3)
    critical = float(np.median(inferred))
    prediction = critical_spin_time(
        test["parameter_initial_angular_velocity_3"].to_numpy(float),
        np.full(len(test), critical),
        test["parameter_inertia_3"].to_numpy(float),
        test["parameter_damping_3"].to_numpy(float),
    )
    observed_test = test["wobble_observed"].to_numpy(bool) & np.isfinite(prediction)
    score = metrics(
        test.loc[observed_test, "wobble_onset_time"].to_numpy(float),
        prediction[observed_test],
    )
    return critical, prediction, score


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Compute standard formula-validation metrics."""
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "median_absolute_error": float(np.median(np.abs(actual - predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def fit_interpretable_models(
    train: pd.DataFrame, test: pd.DataFrame
) -> list[ModelResult]:
    """Fit linear, log-linear, polynomial, robust, and sparse models."""
    train = train.loc[train["wobble_observed"] & train[FEATURES].notna().all(axis=1)]
    test = test.loc[test["wobble_observed"] & test[FEATURES].notna().all(axis=1)]
    if len(train) < 3 or len(test) < 1:
        raise ValueError("insufficient observed wobble cases for regression")
    x_train = train[FEATURES]
    y_train = train["dimensionless_wobble_time"].to_numpy(float)
    x_test = test[FEATURES]
    y_test = test["dimensionless_wobble_time"].to_numpy(float)
    estimators: dict[str, Any] = {
        "linear": Pipeline([("impute", SimpleImputer()), ("model", LinearRegression())]),
        "log_linear": TransformedTargetRegressor(
            regressor=Pipeline(
                [("impute", SimpleImputer()), ("model", LinearRegression())]
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "polynomial": Pipeline(
            [
                ("impute", SimpleImputer()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("scale", StandardScaler()),
                (
                    "model",
                    ElasticNetCV(
                        cv=min(5, len(train)), random_state=0, max_iter=100_000
                    ),
                ),
            ]
        ),
        "robust": Pipeline(
            [
                ("impute", SimpleImputer()),
                ("scale", StandardScaler()),
                ("model", HuberRegressor()),
            ]
        ),
        "sparse": Pipeline(
            [
                ("impute", SimpleImputer()),
                ("scale", StandardScaler()),
                (
                    "model",
                    ElasticNetCV(
                        cv=min(5, len(train)), random_state=0, max_iter=100_000
                    ),
                ),
            ]
        ),
    }
    results = []
    for name, estimator in estimators.items():
        estimator.fit(x_train, y_train)
        prediction = estimator.predict(x_test)
        results.append(ModelResult(name, estimator, metrics(y_test, prediction)))
    return results


def censored_violation_rate(frame: pd.DataFrame, predicted_time: np.ndarray) -> float:
    """Return the fraction of censored predictions before their observation horizon."""
    censored = ~frame["wobble_observed"].to_numpy(bool)
    if not np.any(censored):
        return 0.0
    violations = predicted_time[censored] <= frame.loc[censored, "observation_horizon"].to_numpy()
    return float(np.mean(violations))
