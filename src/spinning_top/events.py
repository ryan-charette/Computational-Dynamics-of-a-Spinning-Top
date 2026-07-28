"""Wobble, stopping, and persistence-based event detection."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def causal_rolling_range(time: npt.ArrayLike, values: npt.ArrayLike, window: float) -> Array:
    """Return max-minus-min over each causal interval ``[t-window, t]``."""
    t = np.asarray(time, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.ndim != 1 or y.shape != t.shape or len(t) == 0:
        raise ValueError("time and values must be nonempty one-dimensional arrays")
    if np.any(np.diff(t) <= 0):
        raise ValueError("time must be strictly increasing")
    result = np.empty_like(y)
    for index, current in enumerate(t):
        start = int(np.searchsorted(t, current - window, side="left"))
        segment = y[start : index + 1]
        result[index] = np.max(segment) - np.min(segment)
    return result


def first_persistent_time(
    time: npt.ArrayLike, condition: npt.ArrayLike, hold_time: float
) -> float | None:
    """Return the start of the first continuously true interval of sufficient length."""
    t = np.asarray(time, dtype=float)
    active = np.asarray(condition, dtype=bool)
    if t.shape != active.shape:
        raise ValueError("time and condition must have matching shapes")
    start: int | None = None
    for index, is_active in enumerate(active):
        if is_active and start is None:
            start = index
        elif not is_active:
            start = None
        if start is not None and t[index] - t[start] >= hold_time:
            return float(t[start])
    return None


def wobble_onset(
    time: npt.ArrayLike,
    tilt: npt.ArrayLike,
    window: float,
    threshold: float,
    hold_time: float,
    end_time: float | None = None,
) -> tuple[float | None, Array]:
    """Detect persistent orientation-based wobble and return its rolling amplitude."""
    t = np.asarray(time, dtype=float)
    amplitude = causal_rolling_range(t, tilt, window)
    condition = amplitude > threshold
    if end_time is not None:
        condition &= t < end_time
    return first_persistent_time(t, condition, hold_time), amplitude


def stopping_time(
    time: npt.ArrayLike, axial_spin: npt.ArrayLike, threshold: float, hold_time: float
) -> float | None:
    """Detect persistent low axial spin."""
    return first_persistent_time(
        time, np.abs(np.asarray(axial_spin, dtype=float)) < threshold, hold_time
    )


def classify(
    *,
    success: bool,
    wobble_time: float | None,
    chaos_time: float | None,
    stop_time: float | None,
) -> str:
    """Classify a trajectory using the documented event precedence."""
    if not success:
        return "numerical_failure"
    if chaos_time is not None and (wobble_time is None or chaos_time <= wobble_time):
        return "chaotic_immediately"
    if wobble_time is not None and chaos_time is not None:
        return "chaotic_after_wobble"
    if wobble_time is not None:
        return "regular_wobble"
    if stop_time is not None:
        return "stable_until_stop"
    return "no_event_within_horizon"

