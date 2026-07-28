from __future__ import annotations

import pytest

from spinning_top.parameters import DetectionConfig, SimulationConfig, SweepConfig


def test_config_normalizes_orientation_and_axis() -> None:
    config = SimulationConfig(initial_quaternion=(2, 0, 0, 0), reference_axis=(0, 0, 5))
    assert config.initial_quaternion == pytest.approx((1, 0, 0, 0))
    assert config.reference_axis == pytest.approx((0, 0, 1))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mass", 0),
        ("inertia", (1, 1, 3)),
        ("damping", (0, -1, 0)),
        ("reference_axis", (0, 0, 0)),
    ],
)
def test_nonphysical_config_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        SimulationConfig(**{field: value})


def test_detector_rejects_nonpositive_values() -> None:
    with pytest.raises(ValueError):
        DetectionConfig(hold_time=0)


def test_sweep_rejects_decreasing_range() -> None:
    with pytest.raises(ValueError):
        SweepConfig(ranges={"mass": (2, 1)})
