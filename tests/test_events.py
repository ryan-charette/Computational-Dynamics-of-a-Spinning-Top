from __future__ import annotations

import numpy as np
import pytest

from spinning_top.events import (
    causal_rolling_range,
    classify,
    first_persistent_time,
    stopping_time,
    wobble_onset,
)


def test_rolling_range_and_persistent_crossing() -> None:
    time = np.arange(0, 3.1, 0.1)
    tilt = np.zeros_like(time)
    tilt[10:] = np.where(np.arange(len(time) - 10) % 2, 0.2, 0)
    onset, amplitude = wobble_onset(time, tilt, 0.4, 0.1, 0.5)
    assert onset == pytest.approx(1.1)
    assert np.max(amplitude) == pytest.approx(0.2)


def test_noise_does_not_trigger_wobble() -> None:
    time = np.arange(0, 2.1, 0.1)
    tilt = 1e-6 * np.sin(time)
    onset, _ = wobble_onset(time, tilt, 0.5, 1e-3, 0.5)
    assert onset is None


def test_persistence_resets_on_false_sample() -> None:
    time = np.arange(6, dtype=float)
    assert first_persistent_time(time, [1, 1, 0, 1, 1, 1], 2) == pytest.approx(3)


def test_stop_detection() -> None:
    time = np.arange(0, 2.1, 0.1)
    spin = np.where(time < 1, 10, 0.1)
    assert stopping_time(time, spin, 0.5, 0.5) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"success": False, "wobble_time": None, "chaos_time": None, "stop_time": None},
            "numerical_failure",
        ),
        (
            {"success": True, "wobble_time": None, "chaos_time": 1, "stop_time": None},
            "chaotic_immediately",
        ),
        (
            {"success": True, "wobble_time": 1, "chaos_time": 2, "stop_time": None},
            "chaotic_after_wobble",
        ),
        (
            {"success": True, "wobble_time": 1, "chaos_time": None, "stop_time": None},
            "regular_wobble",
        ),
        (
            {"success": True, "wobble_time": None, "chaos_time": None, "stop_time": 1},
            "stable_until_stop",
        ),
    ],
)
def test_classification_precedence(kwargs: dict[str, object], expected: str) -> None:
    assert classify(**kwargs) == expected


def test_rolling_range_rejects_unsorted_time() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        causal_rolling_range([0, 2, 1], [0, 0, 0], 1)
