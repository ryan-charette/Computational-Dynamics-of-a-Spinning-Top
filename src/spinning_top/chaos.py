"""Finite-time Lyapunov analysis using periodically renormalized trajectories."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from .events import first_persistent_time
from .integrators import integrate_adaptive
from .parameters import ChaosConfig, SimulationConfig
from .quaternion import conjugate, exponential_map, multiply, normalize, rotation_vector

Array = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ChaosResult:
    """Finite-time Lyapunov diagnostics."""

    time: Array
    ftle: Array
    chaos_onset_time: float | None
    max_ftle: float | None


def _segment(config: SimulationConfig, state: Array, duration: float) -> Array:
    segment_config = replace(
        config,
        initial_quaternion=tuple(state[:4]),
        initial_angular_velocity=tuple(state[4:]),
        horizon=duration,
        output_interval=duration,
        max_step=min(config.max_step, duration),
    )
    result = integrate_adaptive(segment_config)
    if not result.success:
        raise RuntimeError(result.message)
    return np.asarray(result.state[-1], dtype=float)


def _difference(reference: Array, nearby: Array, velocity_scale: float) -> tuple[Array, float]:
    relative = multiply(conjugate(reference[:4]), nearby[:4])
    orientation = rotation_vector(relative)
    velocity = (nearby[4:] - reference[4:]) / velocity_scale
    vector = np.concatenate((orientation, velocity))
    return vector, float(np.linalg.norm(vector))


def estimate_ftle(config: SimulationConfig, settings: ChaosConfig) -> ChaosResult:
    """Estimate a rolling largest finite-time Lyapunov exponent."""
    if not settings.enabled:
        return ChaosResult(np.array([], dtype=float), np.array([], dtype=float), None, None)
    scale = settings.angular_velocity_scale
    if scale is None:
        scale = max(float(np.linalg.norm(config.initial_angular_velocity)), 1.0)
    epsilon = settings.perturbation
    reference = np.asarray(
        (*config.initial_quaternion, *config.initial_angular_velocity), dtype=float
    )
    nearby = reference.copy()
    nearby[4] += epsilon * scale
    times: list[float] = []
    local_rates: list[float] = []
    elapsed = 0.0
    while elapsed < config.horizon - 1e-15:
        duration = min(settings.renormalization_interval, config.horizon - elapsed)
        reference = _segment(config, reference, duration)
        nearby = _segment(config, nearby, duration)
        difference, distance = _difference(reference, nearby, scale)
        distance = max(distance, np.finfo(float).tiny)
        local_rates.append(float(np.log(distance / epsilon) / duration))
        elapsed += duration
        times.append(elapsed)
        scaled = difference * (epsilon / distance)
        delta_q = exponential_map(scaled[:3])
        nearby[:4] = normalize(multiply(reference[:4], delta_q))
        nearby[4:] = reference[4:] + scale * scaled[3:]

    t = np.asarray(times)
    local = np.asarray(local_rates)
    rolling = np.empty_like(local)
    for index, current in enumerate(t):
        start = int(np.searchsorted(t, current - settings.ftle_window, side="left"))
        rolling[index] = float(np.mean(local[start : index + 1]))
    onset = first_persistent_time(t, rolling > settings.threshold, settings.hold_time)
    return ChaosResult(t, rolling, onset, float(np.max(rolling)) if len(rolling) else None)
