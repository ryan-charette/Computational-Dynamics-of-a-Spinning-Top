"""Reproducible plotting from saved tabular and trajectory data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_tilt(diagnostics: dict[str, Any], path: str | Path) -> Path:
    """Plot tilt and detected wobble onset."""
    fig, axis = plt.subplots()
    axis.plot(diagnostics["time"], np.rad2deg(diagnostics["tilt"]), label="tilt")
    onset = float(diagnostics["wobble_onset_time"])
    if np.isfinite(onset):
        axis.axvline(onset, color="tab:red", linestyle="--", label="wobble onset")
    axis.set(xlabel="Time [s]", ylabel="Tilt [deg]")
    axis.legend()
    return _save(fig, Path(path))


def plot_ftle(diagnostics: dict[str, Any], path: str | Path) -> Path:
    """Plot rolling FTLE and detected chaos onset."""
    fig, axis = plt.subplots()
    axis.plot(diagnostics["ftle_time"], diagnostics["ftle"])
    onset = float(diagnostics["chaos_onset_time"])
    if np.isfinite(onset):
        axis.axvline(onset, color="tab:red", linestyle="--")
    axis.set(xlabel="Time [s]", ylabel="Rolling FTLE [1/s]")
    return _save(fig, Path(path))


def plot_phase_space(diagnostics: dict[str, Any], path: str | Path) -> Path:
    """Plot the body angular velocity trajectory."""
    state = diagnostics["state"]
    fig = plt.figure()
    axis = fig.add_subplot(projection="3d")
    axis.plot(state[:, 4], state[:, 5], state[:, 6], linewidth=0.8)
    axis.set(xlabel=r"$\omega_1$", ylabel=r"$\omega_2$", zlabel=r"$\omega_3$")
    return _save(fig, Path(path))


def plot_parameter_map(
    frame: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    path: str | Path,
) -> Path:
    """Plot a continuous or categorical parameter-space map."""
    fig, axis = plt.subplots()
    if pd.api.types.is_numeric_dtype(frame[color]):
        artist = axis.scatter(frame[x], frame[y], c=frame[color], cmap="viridis")
        fig.colorbar(artist, ax=axis, label=color)
    else:
        codes, labels = pd.factorize(frame[color])
        artist = axis.scatter(frame[x], frame[y], c=codes, cmap="tab10")
        handles, _ = artist.legend_elements()
        axis.legend(handles, labels.astype(str), title=color)
    axis.set(xlabel=x, ylabel=y)
    return _save(fig, Path(path))


def plot_prediction_validation(actual: np.ndarray, predicted: np.ndarray, path: str | Path) -> Path:
    """Plot predicted versus simulated onset time."""
    finite = np.isfinite(actual) & np.isfinite(predicted)
    fig, axis = plt.subplots()
    axis.scatter(actual[finite], predicted[finite], alpha=0.7)
    bounds = [
        min(np.min(actual[finite]), np.min(predicted[finite])),
        max(np.max(actual[finite]), np.max(predicted[finite])),
    ]
    axis.plot(bounds, bounds, color="black", linestyle="--")
    axis.set(xlabel="Simulated onset [s]", ylabel="Predicted onset [s]")
    return _save(fig, Path(path))


def plot_residuals(
    feature: np.ndarray, actual: np.ndarray, predicted: np.ndarray, path: str | Path
) -> Path:
    """Plot residuals against one feature."""
    fig, axis = plt.subplots()
    axis.scatter(feature, predicted - actual, alpha=0.7)
    axis.axhline(0.0, color="black", linestyle="--")
    axis.set(xlabel="Feature value", ylabel="Prediction residual [s]")
    return _save(fig, Path(path))


def plot_convergence(
    time: np.ndarray, adaptive: np.ndarray, rk4: np.ndarray, path: str | Path
) -> Path:
    """Plot an adaptive-versus-RK4 convergence comparison."""
    fig, axis = plt.subplots()
    axis.plot(time, np.linalg.norm(adaptive - rk4, axis=1))
    axis.set(xlabel="Time [s]", ylabel="State-vector difference")
    axis.set_yscale("log")
    return _save(fig, Path(path))


def load_diagnostics(path: str | Path) -> dict[str, Any]:
    """Load a diagnostics archive into ordinary arrays."""
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def plot_representatives(
    frame: pd.DataFrame, diagnostics_dir: str | Path, output_dir: str | Path
) -> list[Path]:
    """Plot one saved trajectory for each available dynamic class."""
    diagnostic_root = Path(diagnostics_dir)
    output_root = Path(output_dir)
    created: list[Path] = []
    for category in ("stable_until_stop", "regular_wobble", "chaotic_after_wobble"):
        candidates = frame.loc[frame["classification"] == category, "run_id"]
        if candidates.empty:
            continue
        run_id = str(candidates.iloc[0])
        archive = load_diagnostics(diagnostic_root / f"{run_id}.npz")
        created.append(plot_tilt(archive, output_root / f"{category}_tilt.png"))
        created.append(plot_phase_space(archive, output_root / f"{category}_phase.png"))
        if len(archive["ftle"]):
            created.append(plot_ftle(archive, output_root / f"{category}_ftle.png"))
    return created
