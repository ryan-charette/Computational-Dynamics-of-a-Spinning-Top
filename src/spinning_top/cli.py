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
    critical, baseline_prediction, baseline_metrics = fit_critical_spin_baseline(train, test)
    survival = fit_weibull_aft(train)
    joblib.dump(survival, output / "weibull_aft.joblib")
    selected_formula = f"{best.name} model over features: " + ", ".join(FEATURES)
    if args.symbolic:
        symbolic = fit_symbolic(train, iterations=args.symbolic_iterations)
        joblib.dump(symbolic, output / "symbolic.joblib")
        selected_formula = str(symbolic.sympy())
    from .plotting import (
        plot_baseline_comparison,
        plot_prediction_validation,
        plot_residuals,
    )

    complete_test = test[FEATURES].notna().all(axis=1) & test["wobble_observed"]
    validation = test.loc[complete_test]
    formula_star = best.estimator.predict(validation[FEATURES])
    actual_star = validation["dimensionless_wobble_time"].to_numpy(float)
    figure_root = output / "figures"
    plot_prediction_validation(
        actual_star, formula_star, figure_root / "predicted_vs_simulated.png"
    )
    for feature in FEATURES:
        plot_residuals(
            validation[feature].to_numpy(float),
            actual_star,
            formula_star,
            figure_root / f"residual_{feature}.png",
        )
    gravity_scale = (
        validation["gravity_timescale"].to_numpy(float)
        if "gravity_timescale" in validation
        else np.ones(len(validation))
    )
    plot_baseline_comparison(
        validation["wobble_onset_time"].to_numpy(float),
        baseline_prediction[complete_test.to_numpy()],
        formula_star * gravity_scale,
        figure_root / "baseline_vs_formula.png",
    )
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
    if args.figures:
        from .plotting import plot_parameter_map, plot_representatives

        figure_root = Path(args.figures)
        if {"dimensionless_spin", "dimensionless_damping"}.issubset(frame.columns):
            plot_parameter_map(
                frame,
                "dimensionless_spin",
                "dimensionless_damping",
                "wobble_onset_time",
                figure_root / "parameter_wobble_time.png",
            )
            plot_parameter_map(
                frame,
                "dimensionless_spin",
                "dimensionless_damping",
                "classification",
                figure_root / "parameter_classification.png",
            )
        if args.diagnostics:
            plot_representatives(frame, args.diagnostics, figure_root)
    print(generate_report(frame, args.output, args.model_summary))


def _validate(args: argparse.Namespace) -> None:
    from .plotting import plot_convergence
    from .validation import convergence_study

    config, _detection, _chaos = load_simulation_bundle(args.config)
    frame, time, adaptive, fixed = convergence_study(config)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    if args.figure:
        plot_convergence(time, adaptive, fixed, args.figure)
    print(target)


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
    report.add_argument("--figures")
    report.add_argument("--diagnostics")
    report.set_defaults(function=_report)

    validate = subparsers.add_parser("validate", help="run a numerical convergence study")
    validate.add_argument("--config", required=True)
    validate.add_argument("--output", default="outputs/reports/convergence.parquet")
    validate.add_argument("--figure", default="outputs/figures/convergence.png")
    validate.set_defaults(function=_validate)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command-line interface."""
    args = build_parser().parse_args(argv)
    args.function(args)
