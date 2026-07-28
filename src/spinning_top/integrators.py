"""Adaptive and fixed-step integration methods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp

from .dynamics import rhs
from .parameters import SimulationConfig

Array = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class IntegrationResult:
    """Raw integration output with normalized sampled quaternions."""

    time: Array
    state: Array
    raw_quaternion_norm_error: float
    success: bool
    message: str


def initial_state(config: SimulationConfig) -> Array:
    """Build the seven-component initial state."""
    return np.asarray((*config.initial_quaternion, *config.initial_angular_velocity), dtype=float)


def _sample_times(config: SimulationConfig) -> Array:
    count = int(np.floor(config.horizon / config.output_interval))
    times = np.arange(count + 1, dtype=float) * config.output_interval
    if times[-1] < config.horizon:
        times = np.append(times, config.horizon)
    return times


def integrate_adaptive(config: SimulationConfig) -> IntegrationResult:
    """Integrate with SciPy's adaptive ``solve_ivp``."""
    solution = solve_ivp(
        lambda time, state: rhs(time, state, config),
        (0.0, config.horizon),
        initial_state(config),
        t_eval=_sample_times(config),
        method=config.method,
        rtol=config.rtol,
        atol=config.atol,
        max_step=config.max_step,
    )
    state = solution.y.T.copy()
    raw_norms = np.linalg.norm(state[:, :4], axis=1)
    norm_error = float(np.max(np.abs(raw_norms - 1.0))) if len(state) else np.inf
    state[:, :4] /= raw_norms[:, None]
    return IntegrationResult(
        solution.t.astype(float),
        state,
        norm_error,
        bool(solution.success),
        solution.message,
    )


def _rk4_step(time: float, state: Array, step: float, config: SimulationConfig) -> Array:
    k1 = rhs(time, state, config)
    k2 = rhs(time + step / 2, state + step * k1 / 2, config)
    k3 = rhs(time + step / 2, state + step * k2 / 2, config)
    k4 = rhs(time + step, state + step * k3, config)
    result = state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    result[:4] /= np.linalg.norm(result[:4])
    return result


def integrate_rk4(config: SimulationConfig, step: float | None = None) -> IntegrationResult:
    """Integrate with fixed-step RK4, primarily for convergence comparisons."""
    dt = step or config.output_interval
    times = _sample_times(config)
    states = np.empty((len(times), 7), dtype=float)
    states[0] = initial_state(config)
    current_time = 0.0
    for index in range(1, len(times)):
        target = times[index]
        current = states[index - 1].copy()
        while current_time < target - 1e-15:
            local_step = min(dt, target - current_time)
            current = _rk4_step(current_time, current, local_step, config)
            current_time += local_step
        states[index] = current
    return IntegrationResult(times, states, 0.0, True, "fixed-step RK4 completed")

