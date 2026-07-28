"""Markdown report generation from saved analysis products."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def generate_report(
    frame: pd.DataFrame,
    output: str | Path,
    model_summary: str | Path | None = None,
) -> Path:
    """Generate a concise reproducibility and validation report."""
    summary: dict[str, object] = {}
    if model_summary is not None and Path(model_summary).exists():
        summary = json.loads(Path(model_summary).read_text(encoding="utf-8"))
    successful = frame.loc[frame["success"]]
    censoring_rate = (
        float((~successful["wobble_observed"]).mean()) if len(successful) else float("nan")
    )
    classifications = frame["classification"].value_counts().to_dict()
    formula = summary.get("selected_formula", "No formula has been selected yet.")
    ranges = {
        column: [float(frame[column].min()), float(frame[column].max())]
        for column in (
            "dimensionless_spin",
            "dimensionless_damping",
            "inertia_ratio",
            "inertia_asymmetry",
            "initial_tilt",
        )
        if column in frame
    }
    text = f"""# Spinning-Top Wobble Analysis Report

## Corrected model

The state contains a body-to-world unit quaternion and body-frame angular
velocity. Gravity is rotated into the body frame before torque is computed;
linear damping is applied in the body frame. Wobble is a persistent rolling
tilt-amplitude event and chaos is a separate nearby-trajectory FTLE event.

## Dataset

- Simulations: {len(frame)}
- Successful simulations: {len(successful)}
- Right-censoring rate among successful runs: {censoring_rate:.3f}
- Dynamic classifications: `{json.dumps(classifications, sort_keys=True)}`

## Predictive expression

```text
{formula}
```

This expression is empirical or semi-analytical, not a rigorous derivation.

Tested dimensionless ranges:

```json
{json.dumps(ranges, indent=2, sort_keys=True)}
```

## Validation

```json
{json.dumps(summary.get("metrics", {}), indent=2, sort_keys=True)}
```

Numerical convergence, censored-case violations, parameter-region holdouts,
and residual plots must be inspected alongside aggregate errors.

## Limitations

The pivot is fixed. Contact, translation, slipping, impacts, aerodynamic lift,
and nonlinear pivot friction are outside the model. Predictions should not be
extrapolated beyond the listed parameter ranges without new validation.
"""
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target

