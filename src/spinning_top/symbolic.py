"""Complexity-constrained symbolic regression."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .regression import FEATURES


def fit_symbolic(frame: pd.DataFrame, *, iterations: int = 100) -> Any:
    """Fit PySR to observed, dimensionless wobble times."""
    try:
        from pysr import PySRRegressor
    except ImportError as error:
        raise RuntimeError("install the 'symbolic' extra to run PySR") from error
    data = frame.loc[frame["wobble_observed"], [*FEATURES, "dimensionless_wobble_time"]]
    data = data.dropna()
    if len(data) < 5:
        raise ValueError("at least five observed complete cases are required")
    model = PySRRegressor(
        niterations=iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["log", "exp", "sqrt"],
        constraints={"exp": 5, "log": 5, "sqrt": 5},
        model_selection="best",
        parsimony=0.01,
        random_state=0,
        deterministic=True,
        parallelism="serial",
    )
    model.fit(data[FEATURES].to_numpy(), data["dimensionless_wobble_time"].to_numpy())
    return model
