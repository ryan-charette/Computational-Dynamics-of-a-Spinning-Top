# Reproducible workflow

1. Validate or edit a YAML configuration.
2. Run a single simulation and inspect its saved diagnostics.
3. Run a seeded Sobol sweep. Each run receives a stable content hash.
4. Resume safely after interruption; completed fragments are skipped.
5. Build dimensionless features from the consolidated Parquet dataset.
6. Fit the analytical, regression, survival, and optional symbolic models.
7. Regenerate plots and the report from saved results.

Run `python -m spinning_top validate --config configs/default_simulation.yaml`
to save an adaptive-versus-RK4 convergence table and figure.

Do not record the horizon as an onset time. If wobble is unobserved,
`wobble_observed` is false and `wobble_onset_time` is null.

Large generated datasets are intentionally ignored by Git. Record the
configuration, seed, package commit, parameter ranges, and report alongside
any externally archived dataset.
