"""Parallel, resumable parameter sweeps with atomic output fragments."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from .io import save_diagnostics
from .parameters import ChaosConfig, DetectionConfig, SimulationConfig, SweepConfig
from .sampling import generate_samples, sample_to_config
from .simulation import OutcomeRecord, SimulationResult, make_outcome, simulate


def stable_run_id(config: SimulationConfig, schema_version: int, sample: dict[str, float]) -> str:
    """Return a stable content-derived run identifier."""
    payload = json.dumps(
        {"schema_version": schema_version, "config": config.to_dict(), "sample": sample},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.blake2b(payload.encode(), digest_size=12).hexdigest()


def _run_one(
    task: tuple[str, SimulationConfig, DetectionConfig, ChaosConfig],
) -> tuple[OutcomeRecord, SimulationResult | None]:
    run_id, config, detection, chaos = task
    try:
        result = simulate(config, detection, chaos)
        return make_outcome(run_id, config, result), result
    except Exception as error:
        return (
            OutcomeRecord(
                run_id=run_id,
                wobble_observed=False,
                wobble_onset_time=None,
                observation_horizon=config.horizon,
                chaos_observed=False,
                chaos_onset_time=None,
                stopping_time=None,
                max_ftle=None,
                classification="numerical_failure",
                success=False,
                censoring_reason="numerical_failure",
                raw_quaternion_norm_error=float("nan"),
                message=str(error),
                parameters=config.to_dict(),
                exception_type=type(error).__name__,
            ),
            None,
        )


def _atomic_parquet(path: Path, row: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    pd.DataFrame([row]).to_parquet(temporary, index=False)
    temporary.replace(path)


def consolidate(fragment_directory: str | Path, output: str | Path) -> Path:
    """Consolidate outcome fragments without deleting them."""
    fragments = sorted(Path(fragment_directory).glob("*.parquet"))
    if not fragments:
        raise ValueError("no outcome fragments found")
    frame = pd.concat((pd.read_parquet(path) for path in fragments), ignore_index=True)
    frame = frame.sort_values("run_id").drop_duplicates("run_id", keep="last")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    return target


def run_sweep(settings: SweepConfig) -> Path:
    """Run or resume a sweep and return the consolidated Parquet path."""
    root = Path(settings.output_path)
    fragments = root / "fragments"
    diagnostics = root / "diagnostics"
    fragments.mkdir(parents=True, exist_ok=True)
    diagnostics.mkdir(parents=True, exist_ok=True)
    samples = generate_samples(
        settings.ranges, settings.sample_count, settings.seed, settings.sampler
    )
    tasks: list[tuple[str, SimulationConfig, DetectionConfig, ChaosConfig]] = []
    for sample in samples:
        try:
            config = sample_to_config(settings.simulation, sample)
            run_id = stable_run_id(config, settings.schema_version, sample)
        except Exception as error:  # invalid physical samples are isolated records
            run_id = hashlib.blake2b(
                json.dumps(sample, sort_keys=True).encode(), digest_size=12
            ).hexdigest()
            invalid = OutcomeRecord(
                run_id=run_id,
                wobble_observed=False,
                wobble_onset_time=None,
                observation_horizon=settings.simulation.horizon,
                chaos_observed=False,
                chaos_onset_time=None,
                stopping_time=None,
                max_ftle=None,
                classification="numerical_failure",
                success=False,
                censoring_reason="numerical_failure",
                raw_quaternion_norm_error=float("nan"),
                message=str(error),
                parameters={**asdict(settings.simulation), **sample},
                exception_type=type(error).__name__,
            )
            target = fragments / f"{run_id}.parquet"
            if not target.exists():
                _atomic_parquet(target, invalid.to_flat_dict())
            continue
        if not (fragments / f"{run_id}.parquet").exists():
            tasks.append((run_id, config, settings.detection, settings.chaos))

    def consume(results: Any) -> None:
        for outcome, result in results:
            if settings.save_diagnostics and result is not None:
                save_diagnostics(diagnostics / f"{outcome.run_id}.npz", result)
            _atomic_parquet(fragments / f"{outcome.run_id}.parquet", outcome.to_flat_dict())

    if settings.workers == 1:
        consume(map(_run_one, tasks))
    else:
        with ProcessPoolExecutor(max_workers=settings.workers) as executor:
            consume(executor.map(_run_one, tasks))
    return consolidate(fragments, root / "outcomes.parquet")
