# Contributing

Contributions are welcome through focused issues and pull requests.

1. Create a virtual environment using a supported Python version.
2. Install the package, analysis extra, and development dependency groups.
3. Add tests for every behavioral change.
4. Run `nox -s tests lint typing docs build repo_review`.
5. Keep generated sweeps, figures, models, and reports out of commits.

Scientific changes should state the physical assumptions, include a
convergence or conservation check where applicable, and preserve deterministic
seeds. Bug reports should include a minimal configuration, package versions,
platform, expected behavior, and observed behavior.

Use NumPy-style docstrings for public APIs and add a changelog entry for
user-visible changes. By participating, contributors agree to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

