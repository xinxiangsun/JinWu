"""Regression checks for the local and tagged-release validation entry points."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    "jinwu",
    "jinwu-ep",
    "jinwu-swift",
    "jinwu-fermi",
    "jinwu-gw",
    "jinwu-gecam",
    "jinwurs",
)
RECIPES = (
    "recipe/meta.yaml",
    "packages/jinwurs/conda-recipe/meta.yaml",
)


def _make_dry_run(target: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", "-n", target],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_make_check_includes_swift_contract_tests():
    result = _make_dry_run("check")
    assert "packages/jinwu-swift/tests" in result.stdout


def test_make_release_stages_jinwurs_conda_recipe():
    lines = _make_dry_run("release").stdout.splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("git add "))
    command: list[str] = []
    for line in lines[start:]:
        command.append(line)
        if not line.endswith("\\"):
            break
    assert "packages/jinwurs/conda-recipe/meta.yaml" in " ".join(command)


def _workflow_step_script(name: str) -> str:
    lines = (ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8").splitlines()
    marker = f"      - name: {name}"
    start = lines.index(marker) + 2
    script: list[str] = []
    for line in lines[start:]:
        if line.startswith("          "):
            script.append(line[10:])
        elif not line.strip():
            script.append("")
        else:
            break
    return "\n".join(script)


def test_publish_wheel_gate_includes_swift_contract_tests():
    script = _workflow_step_script("Run offline gate tests against installed wheels")
    assert "packages/jinwu-swift/tests" in script


def _release_fixture(root: Path, version: str) -> None:
    for package in PACKAGES:
        path = root / "packages" / package / "pyproject.toml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'[project]\nversion = "{version}"\n', encoding="utf-8")
    cargo = root / "packages/jinwurs/Cargo.toml"
    cargo.write_text(f'[package]\nversion = "{version}"\n', encoding="utf-8")
    for recipe in RECIPES:
        path = root / recipe
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '{% set version = "' + version + '" %}\n'
            "package:\n"
            f'  version: "{version}"\n',
            encoding="utf-8",
        )


def _run_release_check(root: Path, version: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["GITHUB_REF"] = f"refs/tags/v{version}"
    return subprocess.run(
        ["bash", "-c", _workflow_step_script("Require a release tag matching every package version")],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
    )


def test_publish_release_check_accepts_matching_conda_recipes(tmp_path):
    version = "9.8.7"
    _release_fixture(tmp_path, version)
    result = _run_release_check(tmp_path, version)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("recipe", RECIPES)
def test_publish_release_check_rejects_stale_conda_recipe(tmp_path, recipe):
    version = "9.8.7"
    _release_fixture(tmp_path, version)
    path = tmp_path / recipe
    path.write_text(path.read_text(encoding="utf-8").replace(
        f'version: "{version}"', 'version: "9.8.6"', 1
    ), encoding="utf-8")

    result = _run_release_check(tmp_path, version)

    assert result.returncode != 0
    assert recipe in result.stdout + result.stderr
