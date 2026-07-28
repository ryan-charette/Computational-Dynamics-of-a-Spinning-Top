"""Serialization helpers for trajectories and outcomes."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np

from .simulation import SimulationResult


def save_diagnostics(path: str | Path, result: SimulationResult) -> Path:
    """Atomically save diagnostics needed to reproduce trajectory plots."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp.npz")
    np.savez_compressed(
        temporary,
        time=result.time,
        state=result.state,
        tilt=result.tilt,
        wobble_amplitude=result.wobble_amplitude,
        transverse_ratio=result.transverse_ratio,
        energy=result.energy,
        angular_momentum_world=result.angular_momentum_world,
        ftle_time=result.ftle_time,
        ftle=result.ftle,
        wobble_onset_time=np.nan if result.wobble_onset_time is None else result.wobble_onset_time,
        chaos_onset_time=np.nan if result.chaos_onset_time is None else result.chaos_onset_time,
        stopping_time=np.nan if result.stopping_time is None else result.stopping_time,
    )
    temporary.replace(target)
    return target
