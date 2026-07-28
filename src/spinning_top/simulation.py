"""High-level simulation, diagnostics, event detection, and outcome records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from .chaos import ChaosResult, estimate_ftle
from .dynamics import angular_momentum_world, total_energy
from .events import classify, stopping_time, wobble_onset
from .integrators import integrate_adaptive, integrate_rk4
from .parameters import ChaosConfig, DetectionConfig, SimulationConfig
from .quaternion import rotate

Array = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Trajectory, derived diagnostics, and event times."""

    time: Array
    state: Array
    tilt: Array
    wobble_amplitude: Array
    transverse_ratio: Array
    energy: Array
    angular_momentum_world: Array
    wobble_onset_time: float | None
    chaos_onset_time: float | None
    stopping_time: float | None
    max_ftle: float | None
    ftle_time: Array
    ftle: Array
    classification: str
    success: bool
    message: str
    raw_quaternion_norm_error: float


@dataclass(frozen=True, slots=True)
class OutcomeRecord:
    """Serializable one-row summary of a simulation."""

    run_id: str
    wobble_observed: bool
    wobble_onset_time: float | None
    observation_horizon: float
    chaos_observed: bool
    chaos_onset_time: float | None
    stopping_time: float | None
    max_ftle: float | None
    classification: str
    success: bool
    censoring_reason: str | None
    raw_quaternion_norm_error: float
    message: str
    parameters: dict[str, Any]
    exception_type: str | None = None

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten parameters for tabular storage."""
        record = asdict(self)
        parameters = record.pop("parameters")
        for key, value in parameters.items():
            if isinstance(value, (list, tuple)):
                for index, item in enumerate(value, start=1):
                    record[f"parameter_{key}_{index}"] = item
            else:
                record[f"parameter_{key}"] = value
        return record


def _diagnostics(
    config: SimulationConfig, time: Array, state: Array, detection: DetectionConfig
) -> tuple[Array, Array, Array, Array, float | None, float | None]:
    axis = np.asarray(config.reference_axis)
    world_axis = np.array([rotate(q, axis) for q in state[:, :4]])
    tilt = np.arccos(np.clip(world_axis[:, 2], -1.0, 1.0))
    axial_spin = state[:, 4:] @ axis
    transverse = state[:, 4:] - axial_spin[:, None] * axis
    ratio = np.linalg.norm(transverse, axis=1) / np.maximum(np.abs(axial_spin), detection.epsilon)
    stop = stopping_time(time, axial_spin, detection.stop_spin_rate, detection.stop_hold_time)
    wobble, amplitude = wobble_onset(
        time,
        tilt,
        detection.rolling_window,
        detection.wobble_threshold,
        detection.hold_time,
        stop,
    )
    return tilt, amplitude, ratio, axial_spin, wobble, stop


def simulate(
    config: SimulationConfig,
    detection: DetectionConfig | None = None,
    chaos: ChaosConfig | None = None,
    *,
    integrator: str = "adaptive",
) -> SimulationResult:
    """Run one simulation and compute all configured diagnostics."""
    detection = detection or DetectionConfig()
    chaos = chaos or ChaosConfig()
    integrated = integrate_adaptive(config) if integrator == "adaptive" else integrate_rk4(config)
    time = integrated.time
    state = integrated.state
    tilt, amplitude, ratio, _axial, wobble, stop = _diagnostics(config, time, state, detection)
    energy = np.array([total_energy(row, config) for row in state])
    momentum = np.array([angular_momentum_world(row, config) for row in state])
    chaos_result = (
        estimate_ftle(config, chaos)
        if integrated.success
        else ChaosResult(np.array([]), np.array([]), None, None)
    )
    category = classify(
        success=integrated.success,
        wobble_time=wobble,
        chaos_time=chaos_result.chaos_onset_time,
        stop_time=stop,
    )
    return SimulationResult(
        time=time,
        state=state,
        tilt=tilt,
        wobble_amplitude=amplitude,
        transverse_ratio=ratio,
        energy=energy,
        angular_momentum_world=momentum,
        wobble_onset_time=wobble,
        chaos_onset_time=chaos_result.chaos_onset_time,
        stopping_time=stop,
        max_ftle=chaos_result.max_ftle,
        ftle_time=chaos_result.time,
        ftle=chaos_result.ftle,
        classification=category,
        success=integrated.success,
        message=integrated.message,
        raw_quaternion_norm_error=integrated.raw_quaternion_norm_error,
    )


def make_outcome(run_id: str, config: SimulationConfig, result: SimulationResult) -> OutcomeRecord:
    """Create a right-censoring-aware outcome record."""
    if not result.success:
        censoring = "numerical_failure"
    elif result.wobble_onset_time is not None:
        censoring = None
    elif result.stopping_time is not None:
        censoring = "stopped_before_wobble"
    else:
        censoring = "not_observed_within_horizon"
    return OutcomeRecord(
        run_id=run_id,
        wobble_observed=result.wobble_onset_time is not None,
        wobble_onset_time=result.wobble_onset_time,
        observation_horizon=config.horizon,
        chaos_observed=result.chaos_onset_time is not None,
        chaos_onset_time=result.chaos_onset_time,
        stopping_time=result.stopping_time,
        max_ftle=result.max_ftle,
        classification=result.classification,
        success=result.success,
        censoring_reason=censoring,
        raw_quaternion_norm_error=result.raw_quaternion_norm_error,
        message=result.message,
        parameters=config.to_dict(),
    )
