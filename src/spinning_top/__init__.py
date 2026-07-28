"""Validated dynamics and wobble analysis for a spinning top."""

from .parameters import ChaosConfig, DetectionConfig, SimulationConfig, SweepConfig
from .simulation import OutcomeRecord, SimulationResult, simulate

__all__ = [
    "ChaosConfig",
    "DetectionConfig",
    "OutcomeRecord",
    "SimulationConfig",
    "SimulationResult",
    "SweepConfig",
    "simulate",
]

__version__ = "0.1.0"
