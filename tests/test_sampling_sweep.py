from __future__ import annotations

import pandas as pd
import pytest

from spinning_top.parameters import SimulationConfig, SweepConfig
from spinning_top.sampling import generate_samples, sample_to_config
from spinning_top.sweep import run_sweep, stable_run_id


def test_sampling_is_reproducible() -> None:
    ranges = {"mass": (1.0, 2.0), "initial_spin_rate": (10.0, 20.0)}
    first = generate_samples(ranges, 8, 42)
    second = generate_samples(ranges, 8, 42)
    assert first == second
    assert all(1 <= row["mass"] <= 2 for row in first)


def test_sample_builds_tilted_configuration() -> None:
    config = sample_to_config(
        SimulationConfig(), {"initial_tilt": 0.2, "initial_spin_rate": 50}
    )
    assert config.initial_angular_velocity[2] == pytest.approx(50)
    assert config.initial_quaternion != (1, 0, 0, 0)


def test_stable_run_id_is_order_independent() -> None:
    config = SimulationConfig()
    assert stable_run_id(config, 1, {"a": 1, "b": 2}) == stable_run_id(
        config, 1, {"b": 2, "a": 1}
    )


@pytest.mark.integration
def test_sweep_resumes_and_records_censoring(tmp_path) -> None:
    settings = SweepConfig(
        simulation=SimulationConfig(horizon=0.03, output_interval=0.01, max_step=0.01),
        ranges={"initial_spin_rate": (20.0, 30.0)},
        sample_count=3,
        output_path=str(tmp_path / "sweep"),
        save_diagnostics=True,
    )
    first = run_sweep(settings)
    first_frame = pd.read_parquet(first)
    second = run_sweep(settings)
    second_frame = pd.read_parquet(second)
    pd.testing.assert_frame_equal(first_frame, second_frame)
    assert len(first_frame) == 3
    assert not first_frame["wobble_observed"].any()
    assert first_frame["wobble_onset_time"].isna().all()
    assert set(first_frame["censoring_reason"]) == {"not_observed_within_horizon"}
    assert len(list((tmp_path / "sweep" / "diagnostics").glob("*.npz"))) == 3


def test_invalid_sample_is_isolated(tmp_path) -> None:
    settings = SweepConfig(
        simulation=SimulationConfig(horizon=0.01, output_interval=0.01),
        ranges={"I3": (1.0, 1.0)},
        sample_count=1,
        output_path=str(tmp_path / "invalid"),
    )
    frame = pd.read_parquet(run_sweep(settings))
    assert frame.loc[0, "classification"] == "numerical_failure"
    assert frame.loc[0, "exception_type"] == "ValueError"

