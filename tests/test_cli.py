"""CLI aracı için basit entegrasyon (subprocess) testleri."""

import json
import os
import subprocess
import sys
import tempfile

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
CLI_PATH = os.path.join(ROOT, "cli.py")

SAMPLE_PROBLEM = {
    "criteria": ["Fiyat", "Kalite"],
    "weights": [0.5, 0.5],
    "alternatives": ["A", "B"],
    "criteria_types": ["cost", "benefit"],
    "decision_matrix": [[100, 5], [200, 9]],
}


@pytest.fixture
def problem_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(SAMPLE_PROBLEM, f)
        path = f.name
    yield path
    os.unlink(path)


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, CLI_PATH, *args],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )


def test_cli_rank_runs_successfully(problem_file):
    result = _run_cli(problem_file)
    assert result.returncode == 0
    assert "TOPSIS" in result.stdout


def test_cli_compare_runs_successfully(problem_file):
    result = _run_cli(problem_file, "--compare")
    assert result.returncode == 0
    assert "WSM" in result.stdout and "WPM" in result.stdout and "TOPSIS" in result.stdout


def test_cli_sensitivity_runs_successfully(problem_file):
    result = _run_cli(problem_file, "--sensitivity")
    assert result.returncode == 0
    assert "Duyarlılık Özeti" in result.stdout


def test_cli_rejects_missing_file():
    result = _run_cli("nonexistent_file.json")
    assert result.returncode != 0
