"""Right-censored survival-analysis baseline."""

from __future__ import annotations

import pandas as pd
from lifelines import WeibullAFTFitter

from .regression import FEATURES


def fit_weibull_aft(frame: pd.DataFrame) -> WeibullAFTFitter:
    """Fit a Weibull accelerated-failure-time model."""
    columns = [*FEATURES, "event_time", "wobble_observed"]
    data = frame.copy()
    data["event_time"] = data["wobble_onset_time"].fillna(data["observation_horizon"])
    data = data[columns].dropna()
    if len(data) < 3:
        raise ValueError("insufficient complete cases for survival analysis")
    model = WeibullAFTFitter()
    model.fit(data, duration_col="event_time", event_col="wobble_observed")
    return model

