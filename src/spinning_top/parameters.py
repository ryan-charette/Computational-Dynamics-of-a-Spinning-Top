"""Validated configuration models and YAML loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .quaternion import normalize

Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


def _finite_tuple(value: tuple[float, ...], length: int, name: str) -> None:
    if len(value) != length or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain {length} finite values")


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Physical parameters, initial state, and integration controls."""

    inertia: Vector3 = (0.2, 0.2, 0.3)
    mass: float = 1.0
    center_of_mass: Vector3 = (0.0, 0.0, 0.05)
    reference_axis: Vector3 = (0.0, 0.0, 1.0)
    gravity: float = 9.81
    damping: Vector3 = (0.002, 0.002, 0.004)
    initial_quaternion: Quaternion = (1.0, 0.0, 0.0, 0.0)
    initial_angular_velocity: Vector3 = (0.0, 0.0, 100.0)
    horizon: float = 20.0
    output_interval: float = 0.01
    method: str = "DOP853"
    rtol: float = 1e-9
    atol: float = 1e-11
    max_step: float = 0.02

    def __post_init__(self) -> None:
        for name in (
            "inertia",
            "center_of_mass",
            "reference_axis",
            "damping",
            "initial_angular_velocity",
        ):
            _finite_tuple(tuple(getattr(self, name)), 3, name)
        _finite_tuple(self.initial_quaternion, 4, "initial_quaternion")
        inertia = np.asarray(self.inertia)
        if np.any(inertia <= 0):
            raise ValueError("principal moments of inertia must be positive")
        if np.any(inertia > np.sum(inertia) - inertia + 1e-12):
            raise ValueError("principal moments violate rigid-body triangle inequalities")
        if not np.isfinite(self.mass) or self.mass <= 0:
            raise ValueError("mass must be positive and finite")
        if not np.isfinite(self.gravity) or self.gravity < 0:
            raise ValueError("gravity must be finite and nonnegative")
        if np.any(np.asarray(self.damping) < 0):
            raise ValueError("diagonal damping coefficients must be nonnegative")
        for name in ("horizon", "output_interval", "rtol", "atol", "max_step"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        axis = np.asarray(self.reference_axis, dtype=float)
        norm_axis = float(np.linalg.norm(axis))
        if norm_axis <= 0:
            raise ValueError("reference_axis must be nonzero")
        object.__setattr__(self, "reference_axis", tuple(axis / norm_axis))
        object.__setattr__(self, "initial_quaternion", tuple(normalize(self.initial_quaternion)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-compatible representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    """Wobble and stopping detector settings."""

    rolling_window: float = 0.5
    wobble_threshold: float = np.deg2rad(2.0)
    hold_time: float = 0.5
    stop_spin_rate: float = 0.5
    stop_hold_time: float = 0.5
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        for name in (
            "rolling_window",
            "wobble_threshold",
            "hold_time",
            "stop_spin_rate",
            "stop_hold_time",
            "epsilon",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True, slots=True)
class ChaosConfig:
    """Nearby-trajectory finite-time Lyapunov settings."""

    enabled: bool = False
    perturbation: float = 1e-8
    renormalization_interval: float = 0.1
    ftle_window: float = 1.0
    threshold: float = 0.1
    hold_time: float = 0.5
    angular_velocity_scale: float | None = None

    def __post_init__(self) -> None:
        for name in ("perturbation", "renormalization_interval", "ftle_window", "hold_time"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not np.isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        if self.angular_velocity_scale is not None and self.angular_velocity_scale <= 0:
            raise ValueError("angular_velocity_scale must be positive")


@dataclass(frozen=True, slots=True)
class SweepConfig:
    """Parameter-sweep controls."""

    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    chaos: ChaosConfig = field(default_factory=ChaosConfig)
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    sample_count: int = 16
    seed: int = 0
    workers: int = 1
    sampler: str = "sobol"
    output_path: str = "data/raw/sweep"
    schema_version: int = 1
    save_diagnostics: bool = True

    def __post_init__(self) -> None:
        if self.sample_count <= 0 or self.workers <= 0 or self.schema_version <= 0:
            raise ValueError("sample_count, workers, and schema_version must be positive")
        if self.sampler not in {"sobol", "random", "grid"}:
            raise ValueError("sampler must be 'sobol', 'random', or 'grid'")
        for name, bounds in self.ranges.items():
            _finite_tuple(bounds, 2, f"range {name}")
            if bounds[0] > bounds[1]:
                raise ValueError(f"range {name} has decreasing bounds")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    return value


def load_simulation_bundle(
    path: str | Path,
) -> tuple[SimulationConfig, DetectionConfig, ChaosConfig]:
    """Load simulation, detection, and chaos sections from YAML."""
    raw = load_yaml(path)
    simulation = SimulationConfig(**raw.get("simulation", raw))
    detection = DetectionConfig(**raw.get("detection", {}))
    chaos = ChaosConfig(**raw.get("chaos", {}))
    return simulation, detection, chaos


def load_sweep_config(path: str | Path) -> SweepConfig:
    """Load a complete sweep configuration from YAML."""
    raw = load_yaml(path)
    return SweepConfig(
        simulation=SimulationConfig(**raw.get("simulation", {})),
        detection=DetectionConfig(**raw.get("detection", {})),
        chaos=ChaosConfig(**raw.get("chaos", {})),
        ranges={key: tuple(value) for key, value in raw.get("ranges", {}).items()},
        sample_count=raw.get("sample_count", 16),
        seed=raw.get("seed", 0),
        workers=raw.get("workers", 1),
        sampler=raw.get("sampler", "sobol"),
        output_path=raw.get("output_path", "data/raw/sweep"),
        schema_version=raw.get("schema_version", 1),
        save_diagnostics=raw.get("save_diagnostics", True),
    )

