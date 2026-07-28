#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["nox>=2025.5.1"]
# ///
"""Reproducible development tasks."""

from __future__ import annotations

from pathlib import Path

import nox

nox.needs_version = ">=2025.5.1"
nox.options.default_venv_backend = "uv|virtualenv"
ROOT = Path(__file__).parent


@nox.session(python=["3.11", "3.12", "3.13", "3.14"])
def tests(session: nox.Session) -> None:
    """Run tests against an installed package."""
    session.install("-e.", "pytest>=8.3", "hypothesis>=6.100")
    session.run("python", "-m", "pytest", *session.posargs)


@nox.session
def lint(session: nox.Session) -> None:
    """Run formatting and lint checks."""
    session.install("ruff>=0.12", "codespell>=2.4")
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")
    session.run("codespell", ".")


@nox.session
def typing(session: nox.Session) -> None:
    """Run strict static type checking."""
    session.install("-e.[analysis]", "mypy>=1.15", "types-PyYAML", "pandas-stubs")
    session.run("mypy", "-p", "spinning_top")


@nox.session
def coverage(session: nox.Session) -> None:
    """Run branch coverage with the configured threshold."""
    session.install("-e.[analysis]", "coverage[toml]", "pytest", "hypothesis")
    session.run("coverage", "run", "-m", "pytest")
    session.run("coverage", "report")


@nox.session
def docs(session: nox.Session) -> None:
    """Build documentation with warnings treated as errors."""
    session.install("-e.[analysis]", "sphinx", "myst-parser", "pydata-sphinx-theme")
    session.run("sphinx-build", "-W", "--keep-going", "-b", "html", "docs", "docs/_build/html")


@nox.session
def build(session: nox.Session) -> None:
    """Build and validate the wheel and source distribution."""
    session.install("build", "twine")
    session.run("python", "-m", "build")
    session.run("twine", "check", "dist/*")


@nox.session
def repo_review(session: nox.Session) -> None:
    """Run the Scientific Python repository review."""
    session.install("sp-repo-review[cli]")
    session.run("sp-repo-review", str(ROOT))


if __name__ == "__main__":
    nox.main()
