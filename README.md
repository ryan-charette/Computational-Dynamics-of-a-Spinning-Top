# Computational Dynamics of a Spinning Top

[![CI](https://github.com/ryan-charette/Computational-Dynamics-of-a-Spinning-Top/actions/workflows/ci.yml/badge.svg)](https://github.com/ryan-charette/Computational-Dynamics-of-a-Spinning-Top/actions/workflows/ci.yml)

This project simulates a damped rigid body about a fixed pivot, detects the
onset of observable wobble, separately estimates chaotic divergence, and fits
interpretable models for wobble-onset time.

The original notebook integrated angular velocity alone and incorrectly used
angular velocity to determine gravitational torque. The package now integrates
orientation and angular velocity together:

```text
state = [q_w, q_x, q_y, q_z, omega_1, omega_2, omega_3]
```

The scalar-first Hamilton quaternion maps body coordinates into world
coordinates. In the principal-axis body frame,

```text
g_body = R(q).T @ [0, 0, -g]
tau_gravity = r_cm x (m * g_body)
tau_damping = -B @ omega
I * domega/dt = tau_gravity + tau_damping - omega x (I * omega)
dq/dt = 0.5 * q ⊗ [0, omega]
```

## Installation

Python 3.11 or newer is required.

```bash
python -m pip install -e .
python -m pip install -e ".[analysis]"
```

Install `.[symbolic]` only when running PySR. Development dependencies are
declared as PEP 735 groups and can be installed with a compatible package
manager, or through the Nox sessions.

## Quick start

Run one simulation:

```bash
python -m spinning_top simulate \
  --config configs/default_simulation.yaml \
  --output outputs/single_run.npz \
  --figures outputs/figures
```

Run or resume a parameter sweep:

```bash
python -m spinning_top sweep --config configs/default_sweep.yaml
```

Build dimensionless features and discover a formula:

```bash
python -m spinning_top analyze \
  --input data/raw/sweep/outcomes.parquet \
  --output data/processed/features.parquet
python -m spinning_top discover-formula \
  --input data/processed/features.parquet \
  --output outputs/models
python -m spinning_top report \
  --input data/processed/features.parquet \
  --model-summary outputs/models/summary.json
```

## Event definitions

The configured body reference axis (body `e3` by default) is rotated into the
world frame. Its angle from vertical is `theta`. The primary wobble amplitude
is the max-minus-min range of `theta` over a causal rolling window. Wobble onset
is the first threshold crossing that persists for the configured hold time.

Stopping is a separate persistent low-axial-spin event. Chaos is also separate:
a nearby trajectory is periodically renormalized and used to estimate a
rolling finite-time largest Lyapunov exponent. Periodic nutation can therefore
be classified as regular wobble rather than chaos.

If wobble is not observed, `wobble_onset_time` remains null. A run that reaches
its horizon is right-censored; a run that stops first and a numerical failure
have distinct censoring reasons.

## Formula discovery

The pipeline stores dimensionless spin, damping, inertia ratio/asymmetry,
initial tilt, and dimensionless wobble time. It evaluates the analytical
spin-decay baseline

```text
t_w = (I3 / b3) * log(omega3_0 / omega_crit)
```

before linear, robust, sparse, survival, and optional symbolic models.
Parameter-region holdouts are used instead of relying solely on random row
splits. Any discovered expression is empirical or semi-analytical unless it is
separately derived.

## Validation and development

```bash
python -m pytest
ruff check .
ruff format --check .
mypy -p spinning_top
nox -s build
pipx run "sp-repo-review[cli]" .
```

Tests cover quaternion geometry, torque-free and gravity-only conservation,
damping energy loss, convergence, detector persistence, right-censoring,
sampling reproducibility, resumability, and modeling smoke cases.

## Limitations

The pivot is fixed. The model does not include translation, contact, slipping,
impacts, aerodynamic lift, or nonlinear pivot friction. A configurable linear
body-frame rotational damping model is used. Do not extrapolate a fitted
formula beyond its reported parameter range without new simulations and
validation.

See the [documentation](docs/index.md), [contribution guide](CONTRIBUTING.md),
and [citation metadata](CITATION.cff) for more.

