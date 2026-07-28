"""Numerical convergence studies for saved validation evidence."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .dynamics import angular_momentum_world, total_energy
from .integrators import integrate_adaptive, integrate_rk4
from .parameters import SimulationConfig


def convergence_study(
    config: SimulationConfig, steps: tuple[float, ...] | None = None
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Compare RK4 step sizes with the adaptive reference solution."""
    adaptive = integrate_adaptive(config)
    if not adaptive.success:
        raise RuntimeError(adaptive.message)
    step_sizes = steps or (
        config.output_interval,
        config.output_interval / 2,
        config.output_interval / 4,
    )
    rows: list[dict[str, float]] = []
    finest_state = np.empty_like(adaptive.state)
    for step in step_sizes:
        fixed = integrate_rk4(config, step)
        aligned = fixed.state.copy()
        signs = np.sign(np.sum(aligned[:, :4] * adaptive.state[:, :4], axis=1))
        signs[signs == 0] = 1
        aligned[:, :4] *= signs[:, None]
        error = np.linalg.norm(aligned - adaptive.state, axis=1)
        energy = np.asarray([total_energy(state, config) for state in aligned])
        momentum = np.asarray([angular_momentum_world(state, config) for state in aligned])
        rows.append(
            {
                "rk4_step": step,
                "final_state_error": float(error[-1]),
                "maximum_state_error": float(np.max(error)),
                "relative_energy_drift": float(
                    np.max(np.abs(energy - energy[0])) / max(abs(energy[0]), 1e-15)
                ),
                "maximum_world_momentum_drift": float(
                    np.max(np.linalg.norm(momentum - momentum[0], axis=1))
                ),
                "adaptive_quaternion_norm_error": adaptive.raw_quaternion_norm_error,
            }
        )
        finest_state = aligned
    return pd.DataFrame(rows), adaptive.time, adaptive.state, finest_state
