"""Command-line interface for simulation, sweeps, analysis, and reports."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .io import save_diagnostics
from .parameters import load_simulation_bundle, load_sweep_config
from .simulation import simulate
from .sweep import run_sweep


def _simulate(args: argparse.Namespace) -> None:
    config, detection, chaos = load_simulation_bundle(args.config)
    result = simulate(config, detection, chaos, integrator=args.integrator)
    output = save_diagnostics(args.output, result)
    if args.figures:
        from .plotting import plot_phase_space, plot_tilt

        archive = {
            "time": result.time,
            "state": result.state,
            "tilt": result.tilt,
            "wobble_onset_time": np.nan
            if result.wobble_onset_time is None
            else result.wobble_onset_time,
        }
        plot_tilt(archive, Path(args.figures) / "tilt.png")
        plot_phase_space(archive, Path(args.figures) / "phase_space.png")
    print(json.dumps({"output": str(output), "classification": result.classification}))


def _sweep(args: argparse.Namespace) -> None:
    print(run_sweep(load_sweep_config(args.config)))


def _analyze(args: argparse.Namespace) -> None:
    import pandas as pd

    from .dimensionless import add_dimensionless_features

    result = add_dimensionless_features(pd.read_parquet(args.input))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    print(output)


def _discover(args: argparse.Namespace) -> None:
    import joblib
    import pandas as pd

    from .regression import (
        FEATURES,
        fit_critical_spin_baseline,
        fit_interpretable_models,
    )
    from .survival import fit_weibull_aft
    from .symbolic import fit_symbolic

    frame = pd.read_parquet(args.input).sort_values("dimensionless_spin")
    split = max(1, int(0.8 * len(frame)))
    train, test = frame.iloc[:split], frame.iloc[split:]
    if test.empty:
        raise ValueError("model discovery requires at least two rows")
    models = fit_interpretable_models(train, test)
    best = min(models, key=lambda item: item.metrics["rmse"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.estimator, output / f"{best.name}.joblib")
    critical, _baseline_prediction, baseline_metrics = fit_critical_spin_baseline(train, test)
    survival = fit_weibull_aft(train)
    joblib.dump(survival, output / "weibull_aft.joblib")
    selected_formula = f"{best.name} model over features: " + ", ".join(FEATURES)
    if args.symbolic:
        symbolic = fit_symbolic(train, iterations=args.symbolic_iterations)
        joblib.dump(symbolic, output / "symbolic.joblib")
        selected_formula = str(symbolic.sympy())
    summary = {
        "selected_model": best.name,
        "selected_formula": selected_formula,
        "metrics": best.metrics,
        "all_metrics": {item.name: item.metrics for item in models},
        "analytical_baseline": {
            "critical_spin": critical,
            "formula": "t_w = (I3 / b3) * log(omega3_0 / omega_crit)",
            "metrics": baseline_metrics,
        },
        "holdout": "highest 20% of dimensionless_spin",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(output / "summary.json")


def _report(args: argparse.Namespace) -> None:
    import pandas as pd

    from .report import generate_report

    frame = pd.read_parquet(args.input)
    print(generate_report(frame, args.output, args.model_summary))


def build_parser() -> argparse.ArgumentParser:
    """Build the public command-line parser."""
    parser = argparse.ArgumentParser(prog="spinning-top")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("simulate", help="run one simulation")
    single.add_argument("--config", required=True)
    single.add_argument("--output", default="outputs/single_run.npz")
    single.add_argument("--figures")
    single.add_argument("--integrator", choices=["adaptive", "rk4"], default="adaptive")
    single.set_defaults(function=_simulate)

    sweep = subparsers.add_parser("sweep", help="run or resume a parameter sweep")
    sweep.add_argument("--config", required=True)
    sweep.set_defaults(function=_sweep)

    analyze = subparsers.add_parser("analyze", help="build dimensionless features")
    analyze.add_argument("--input", required=True)
    analyze.add_argument("--output", default="data/processed/features.parquet")
    analyze.set_defaults(function=_analyze)

    discover = subparsers.add_parser(
        "discover-formula", help="fit interpretable wobble-time models"
    )
    discover.add_argument("--input", required=True)
    discover.add_argument("--output", default="outputs/models")
    discover.add_argument("--symbolic", action="store_true")
    discover.add_argument("--symbolic-iterations", type=int, default=100)
    discover.set_defaults(function=_discover)

    report = subparsers.add_parser("report", help="generate the Markdown report")
    report.add_argument("--input", required=True)
    report.add_argument("--model-summary")
    report.add_argument("--output", default="outputs/reports/final_report.md")
    report.set_defaults(function=_report)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command-line interface."""
    args = build_parser().parse_args(argv)
    args.function(args)
