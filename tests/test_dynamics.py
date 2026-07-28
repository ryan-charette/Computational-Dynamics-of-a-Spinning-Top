from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from spinning_top.dynamics import gravitational_torque
from spinning_top.parameters import DetectionConfig, SimulationConfig
from spinning_top.simulation import simulate


def torque_free_config() -> SimulationConfig:
    return SimulationConfig(
        inertia=(0.2, 0.25, 0.3),
        gravity=0,
        damping=(0, 0, 0),
        center_of_mass=(0, 0, 0),
        initial_angular_velocity=(1, 2, 20),
        horizon=0.5,
        output_interval=0.005,
        max_step=0.002,
        rtol=1e-11,
        atol=1e-13,
    )


def test_gravity_torque_depends_on_orientation() -> None:
    config = SimulationConfig(center_of_mass=(0, 0, 0.1))
    upright = gravitational_torque((1, 0, 0, 0), config)
    tilted = gravitational_torque((np.sqrt(0.5), np.sqrt(0.5), 0, 0), config)
    assert upright == pytest.approx([0, 0, 0])
    assert np.linalg.norm(tilted) > 0


def test_torque_free_world_angular_momentum_and_energy_conserved() -> None:
    result = simulate(torque_free_config(), DetectionConfig(stop_spin_rate=0.01))
    momentum_error = np.max(
        np.linalg.norm(result.angular_momentum_world - result.angular_momentum_world[0], axis=1)
    )
    relative_energy_error = np.max(np.abs(result.energy - result.energy[0])) / result.energy[0]
    assert momentum_error < 2e-8
    assert relative_energy_error < 1e-9
    assert result.raw_quaternion_norm_error < 1e-8


def test_gravity_only_conserves_energy_and_vertical_momentum() -> None:
    config = replace(
        torque_free_config(),
        gravity=9.81,
        center_of_mass=(0.01, 0, 0.05),
        initial_quaternion=(0.996194698, 0, 0.087155743, 0),
        horizon=0.2,
    )
    result = simulate(config, DetectionConfig(stop_spin_rate=0.01))
    energy_error = np.max(np.abs(result.energy - result.energy[0])) / abs(result.energy[0])
    vertical_error = np.max(
        np.abs(result.angular_momentum_world[:, 2] - result.angular_momentum_world[0, 2])
    )
    assert energy_error < 1e-8
    assert vertical_error < 1e-8


def test_damping_reduces_energy() -> None:
    config = replace(torque_free_config(), damping=(0.01, 0.01, 0.01))
    result = simulate(config, DetectionConfig(stop_spin_rate=0.01))
    assert result.energy[-1] < result.energy[0]
    assert np.max(np.diff(result.energy)) < 1e-8


def test_rk4_converges_toward_adaptive_solution() -> None:
    config = replace(torque_free_config(), horizon=0.1, output_interval=0.01)
    adaptive = simulate(config, integrator="adaptive")
    coarse = simulate(config, integrator="rk4")
    fine = simulate(replace(config, output_interval=0.005), integrator="rk4")
    coarse_error = np.linalg.norm(coarse.state[-1] - adaptive.state[-1])
    fine_error = np.linalg.norm(fine.state[-1] - adaptive.state[-1])
    assert fine_error < coarse_error
