from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spinning_top.cli import main
from spinning_top.io import save_diagnostics
from spinning_top.parameters import SimulationConfig
from spinning_top.simulation import simulate


def test_diagnostics_round_trip(tmp_path: Path) -> None:
    result = simulate(SimulationConfig(horizon=0.02, output_interval=0.01, max_step=0.01))
    target = save_diagnostics(tmp_path / "diagnostics.npz", result)
    with np.load(target) as archive:
        assert archive["state"] == pytest.approx(result.state)
        assert "wobble_onset_time" in archive


def test_cli_simulate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "simulation:\n  horizon: 0.02\n  output_interval: 0.01\n  max_step: 0.01\n",
        encoding="utf-8",
    )
    output = tmp_path / "run.npz"
    main(["simulate", "--config", str(config), "--output", str(output)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert output.exists()
    assert payload["classification"] == "no_event_within_horizon"
