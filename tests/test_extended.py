from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("work/matplotlib").resolve()))

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from spinning_top.chaos import estimate_ftle
from spinning_top.cli import main
from spinning_top.parameters import (
    ChaosConfig,
    DetectionConfig,
    SimulationConfig,
    load_simulation_bundle,
    load_sweep_config,
)
from spinning_top.plotting import (
    load_diagnostics,
    plot_convergence,
    plot_ftle,
    plot_parameter_map,
    plot_phase_space,
    plot_prediction_validation,
    plot_representatives,
    plot_residuals,
    plot_tilt,
)
from spinning_top.quaternion import (
    exponential_map,
    from_axis_angle,
    multiply,
    rotation_vector,
)
from spinning_top.regression import (
    censored_violation_rate,
    critical_spin_time,
    fit_critical_spin_baseline,
)
from spinning_top.report import generate_report
from spinning_top.sampling import generate_samples, sample_to_config
from spinning_top.simulation import make_outcome, simulate
from spinning_top.symbolic import fit_symbolic


def model_frame(count: int = 24) -> pd.DataFrame:
    spin = np.linspace(20.0, 100.0, count)
    return pd.DataFrame(
        {
            "dimensionless_spin": spin,
            "dimensionless_damping": np.linspace(0.01, 0.03, count),
            "inertia_ratio": np.linspace(1.1, 1.4, count),
            "inertia_asymmetry": np.linspace(-0.05, 0.05, count),
            "initial_tilt": np.linspace(0.02, 0.2, count),
            "dimensionless_wobble_time": 0.5 * spin + 2,
            "wobble_observed": True,
            "wobble_onset_time": 0.1 * spin + 1,
            "observation_horizon": 20.0,
            "parameter_initial_angular_velocity_3": spin,
            "parameter_inertia_3": 0.3,
            "parameter_damping_3": 0.004,
            "success": True,
            "classification": "regular_wobble",
            "run_id": [f"run-{index}" for index in range(count)],
        }
    )


def test_quaternion_maps_round_trip_and_invalid_axis() -> None:
    vector = np.array([0.2, -0.1, 0.3])
    assert rotation_vector(exponential_map(vector)) == pytest.approx(vector)
    combined = multiply(from_axis_angle([1, 0, 0], 0.1), exponential_map([0, 0.2, 0]))
    assert np.linalg.norm(combined) == pytest.approx(1)
    with pytest.raises(ValueError):
        from_axis_angle([0, 0, 0], 1)


def test_ftle_smoke() -> None:
    config = SimulationConfig(
        gravity=0,
        damping=(0, 0, 0),
        center_of_mass=(0, 0, 0),
        initial_angular_velocity=(0.1, 0.2, 10),
        horizon=0.12,
        output_interval=0.02,
        max_step=0.01,
    )
    settings = ChaosConfig(
        enabled=True,
        renormalization_interval=0.03,
        ftle_window=0.06,
        hold_time=0.03,
    )
    result = estimate_ftle(config, settings)
    assert len(result.time) == 4
    assert np.isfinite(result.ftle).all()
    assert result.max_ftle is not None


def test_configuration_loaders(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "simulation:\n  horizon: 1\n"
        "detection:\n  hold_time: 0.2\n"
        "chaos:\n  enabled: true\n"
        "ranges:\n  mass: [1, 2]\n"
        "sample_count: 2\n",
        encoding="utf-8",
    )
    simulation, detection, chaos = load_simulation_bundle(path)
    assert simulation.horizon == 1
    assert detection.hold_time == 0.2
    assert chaos.enabled
    assert load_sweep_config(path).sample_count == 2


def test_all_sampling_modes_and_overrides() -> None:
    ranges = {"mass": (1.0, 2.0), "initial_tilt": (0.1, 0.2)}
    assert len(generate_samples(ranges, 3, 1, "grid")) == 3
    assert len(generate_samples(ranges, 3, 1, "random")) == 3
    assert generate_samples({}, 2, 1) == [{}, {}]
    with pytest.raises(ValueError, match="unsupported"):
        generate_samples(ranges, 2, 1, "bad")
    config = sample_to_config(
        SimulationConfig(center_of_mass=(0, 0, 0)),
        {
            "center_of_mass_distance": 0.2,
            "center_of_mass_x": 0.01,
            "damping1": 0.1,
            "damping2": 0.2,
            "damping3": 0.3,
            "initial_transverse_angular_velocity": 2,
            "initial_transverse_azimuth": np.pi / 2,
        },
    )
    assert config.center_of_mass == pytest.approx((0.01, 0, 0.2))
    assert config.damping == pytest.approx((0.1, 0.2, 0.3))
    assert config.initial_angular_velocity[1] == pytest.approx(2)


def test_baseline_and_censoring_metrics() -> None:
    frame = model_frame()
    critical, prediction, scores = fit_critical_spin_baseline(frame.iloc[:18], frame.iloc[18:])
    assert critical > 0
    assert np.isfinite(prediction).all()
    assert np.isfinite(scores["rmse"])
    direct = critical_spin_time(
        np.array([10.0, 1.0]),
        np.array([2.0, 2.0]),
        np.ones(2),
        np.array([0.1, 0.1]),
    )
    assert np.isfinite(direct[0])
    assert np.isnan(direct[1])
    censored = frame.iloc[:2].copy()
    censored["wobble_observed"] = [False, True]
    assert censored_violation_rate(censored, np.array([1.0, 1.0])) == 1.0


def test_plotting_and_report_from_saved_data(tmp_path: Path) -> None:
    result = simulate(
        SimulationConfig(horizon=0.03, output_interval=0.01, max_step=0.01),
        DetectionConfig(),
    )
    diagnostics = {
        "time": result.time,
        "state": result.state,
        "tilt": result.tilt,
        "wobble_onset_time": np.nan,
        "ftle_time": np.array([0.1, 0.2]),
        "ftle": np.array([0.0, 0.1]),
        "chaos_onset_time": np.nan,
    }
    assert plot_tilt(diagnostics, tmp_path / "tilt.png").exists()
    assert plot_ftle(diagnostics, tmp_path / "ftle.png").exists()
    assert plot_phase_space(diagnostics, tmp_path / "phase.png").exists()
    frame = model_frame(6)
    frame["category"] = ["a", "b", "a", "b", "a", "b"]
    assert plot_parameter_map(
        frame, "dimensionless_spin", "initial_tilt", "wobble_onset_time", tmp_path / "map.png"
    ).exists()
    assert plot_parameter_map(
        frame, "dimensionless_spin", "initial_tilt", "category", tmp_path / "classes.png"
    ).exists()
    actual = np.arange(1.0, 5.0)
    predicted = actual + 0.1
    assert plot_prediction_validation(actual, predicted, tmp_path / "prediction.png").exists()
    assert plot_residuals(actual, actual, predicted, tmp_path / "residual.png").exists()
    assert plot_convergence(
        result.time, result.state, result.state + 1e-6, tmp_path / "convergence.png"
    ).exists()
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"selected_formula": "T*=0.5 Pi_spin+2", "metrics": {"rmse": 0.1}}),
        encoding="utf-8",
    )
    report = generate_report(frame, tmp_path / "report.md", summary)
    assert "T*=0.5 Pi_spin+2" in report.read_text(encoding="utf-8")


def test_representative_plot_and_diagnostic_loader(tmp_path: Path) -> None:
    result = simulate(SimulationConfig(horizon=0.02, output_interval=0.01, max_step=0.01))
    diagnostic_dir = tmp_path / "diagnostics"
    diagnostic_dir.mkdir()
    from spinning_top.io import save_diagnostics

    archive_path = save_diagnostics(diagnostic_dir / "example.npz", result)
    assert load_diagnostics(archive_path)["state"].shape == result.state.shape
    frame = pd.DataFrame({"run_id": ["example"], "classification": ["stable_until_stop"]})
    created = plot_representatives(frame, diagnostic_dir, tmp_path / "figures")
    assert len(created) == 2


def test_outcome_censoring_branches() -> None:
    config = SimulationConfig(horizon=0.02, output_interval=0.01, max_step=0.01)
    result = simulate(config)
    assert make_outcome("a", config, result).censoring_reason == "not_observed_within_horizon"
    stopped = replace(result, stopping_time=0.01, classification="stable_until_stop")
    assert make_outcome("b", config, stopped).censoring_reason == "stopped_before_wobble"
    failed = replace(result, success=False, classification="numerical_failure")
    assert make_outcome("c", config, failed).censoring_reason == "numerical_failure"


def test_cli_analysis_discovery_and_report(tmp_path: Path) -> None:
    frame = model_frame()
    features = tmp_path / "features.parquet"
    frame.to_parquet(features)
    models = tmp_path / "models"
    main(["discover-formula", "--input", str(features), "--output", str(models)])
    assert (models / "summary.json").exists()
    report = tmp_path / "final.md"
    main(
        [
            "report",
            "--input",
            str(features),
            "--model-summary",
            str(models / "summary.json"),
            "--output",
            str(report),
        ]
    )
    assert report.exists()


def test_symbolic_dependency_message() -> None:
    with pytest.raises(RuntimeError, match="symbolic"):
        fit_symbolic(model_frame())
