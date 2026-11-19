"""Execution and path boundaries; real mission comparisons are run separately."""
import os
from pathlib import Path
import subprocess

import pytest

from jinwu.core import DockerMount, DockerRunner
from jinwu.ep import FXTDockerRunner, WXTDockerRunner
from jinwu.ep.docker import _INITIALIZE


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "work space"
    root.mkdir()
    (root / "input").mkdir()
    caldb = tmp_path / "caldb"
    caldb.mkdir()
    return root, caldb


def test_ep_paths_parameters_and_environment(workspace):
    root, caldb = workspace
    runner = FXTDockerRunner.from_env(
        root, environ={"FXTCALDB": str(caldb), "FXTDASIMG": "fxt:tag"}, context="test",
    )
    command = runner.chain_command("input", "output", srcffregion=Path("source ;$.reg"),
                                   expr="ELV>-10 && SAA==0", module="A")
    assert "srcffregion=/work/source ;$.reg" in command
    assert "expr=ELV>-10 && SAA==0" in command
    assert not any(arg.startswith("clobber=") for arg in command)
    docker = runner.build_command(command)
    assert docker[:3] == ["docker", "--context", "test"]
    assert docker[docker.index("--workdir") + 1] == "/tmp"
    assert docker[docker.index("jinwu-ep") + 1] == "/work/output"
    assert f"type=bind,src={caldb},dst=/caldb,readonly" in docker
    assert docker[-len(command):] == command
    assert runner.container_path("input/../output") == "/work/output"


def test_paths_reject_escape_and_overlap(workspace, tmp_path):
    root, caldb = workspace
    runner = WXTDockerRunner("wxt:tag", caldb, root)
    (root / "escape").symlink_to(tmp_path, target_is_directory=True)
    for output in ("../outside", "escape/output", "input", "input/output", "."):
        with pytest.raises(ValueError):
            runner.pipeline_command("input", output)
    with pytest.raises(ValueError):
        runner.pipeline_command("input", "output", bad=None)
    with pytest.raises(ValueError):
        runner.build_command("wxtpipeline")
    assert "clobber=no" in runner.pipeline_command("input", "output")


def test_fxt_output_protection(workspace):
    root, caldb = workspace
    runner = FXTDockerRunner("fxt:tag", caldb, root)
    (root / "output").mkdir()
    (root / "output" / "existing-product").touch()
    with pytest.raises(ValueError, match="empty or absent"):
        runner.chain_command("input", "output")
    assert runner.chain_command("input", "output", allow_existing=True)[0] == "fxtchain"
    with pytest.raises(ValueError, match="no clobber"):
        runner.chain_command("input", "another-output", clobber=True)


@pytest.mark.parametrize("runner_class,task,command_method,run_method", [
    (WXTDockerRunner, "wxtpipeline", "pipeline_command", "run_pipeline"),
    (FXTDockerRunner, "fxtchain", "chain_command", "run_chain"),
])
def test_default_sibling_output(workspace, monkeypatch, runner_class, task, command_method, run_method):
    root, caldb = workspace
    runner = runner_class("image:tag", caldb, root)
    build = getattr(runner, command_method)
    expected = build(root / "input", root / f"input_{task}")
    assert build("input/") == expected
    assert build(root / "input") == expected
    assert f"outdir=/work/input_{task}" in expected
    assert "outdir=/work/custom" in build("input", "custom")
    (root / "input-alias").symlink_to(root / "input", target_is_directory=True)
    assert build("input-alias") == expected
    monkeypatch.setattr(runner, "run", lambda command, **options: command)
    assert getattr(runner, run_method)("input") == expected
    assert getattr(runner, run_method)("input", "custom") == build("input", "custom")
    with pytest.raises(ValueError, match="inside workspace"):
        build(root)  # Its default sibling is outside the mounted workspace.


def test_default_output_protection(workspace, tmp_path):
    root, caldb = workspace
    runner = FXTDockerRunner("fxt:tag", caldb, root)
    default = root / "input_fxtchain"
    default.mkdir()
    (default / "existing-product").touch()
    with pytest.raises(ValueError, match="empty or absent"):
        runner.chain_command("input")
    with pytest.raises(ValueError, match="empty or absent"):
        runner.run_chain("input")
    assert "outdir=/work/input_fxtchain" in runner.chain_command("input", allow_existing=True)
    assert "outdir=/work/custom" in runner.chain_command("input", "custom")
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "input_wxtpipeline").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="inside workspace"):
        WXTDockerRunner("wxt:tag", caldb, root).pipeline_command("input")


def test_image_archive_requires_explicit_load(workspace, tmp_path):
    root, caldb = workspace
    archive = tmp_path / "image.tar"
    archive.write_bytes(b"fake")
    with pytest.raises(ValueError, match="load_image"):
        WXTDockerRunner.from_env(root, environ={"WXTDASIMG": str(archive), "WXTCALDB": str(caldb)})
    runner = WXTDockerRunner.from_env(root, image="wxt:loaded", caldb=caldb, environ={})
    assert runner.image == "wxt:loaded"


def test_shell_preserves_arguments_and_isolates_pfiles(tmp_path):
    headas = tmp_path / "headas"
    headas.mkdir()
    (headas / "headas-init.sh").write_text('export PFILES="/old/user;/mission/syspfiles"\n')
    command = ['printf', '%s\n', 'value=$(touch injected); space', 'second value']
    script = _INITIALIZE.replace('exec "$@"', 'printf "%s\\n" "$PFILES"; exec "$@"')
    env = os.environ | {"HEADAS": str(headas), "JINWU_CALDB_CONFIG": "/caldb/test"}
    result = subprocess.run(["bash", "--noprofile", "--norc", "-c", script, "test", "", *command],
                            cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    pfiles, *args = result.stdout.splitlines()
    user_pfiles, system_pfiles = pfiles.split(";")
    assert system_pfiles == "/mission/syspfiles"
    assert Path(user_pfiles).is_dir()
    Path(user_pfiles).rmdir()
    assert args == command[2:]
    assert not (tmp_path / "injected").exists()


def test_task_cwd_without_mutable_runner_state(workspace):
    root, caldb = workspace
    runner = WXTDockerRunner("wxt:tag", caldb, root)
    first = runner.build_command(runner.pipeline_command("input", "first output"))
    second = runner.build_command(runner.pipeline_command("input", "second output"))
    generic = runner.build_command(["printf", "hello"])
    assert first[first.index("jinwu-ep") + 1] == "/work/first output"
    assert second[second.index("jinwu-ep") + 1] == "/work/second output"
    assert generic[generic.index("jinwu-ep") + 1] == ""
    assert runner.workdir == "/tmp"
    help_command = runner.build_command(["wxtpipeline", "-h"])
    assert help_command[help_command.index("jinwu-ep") + 1] == ""
    for command in (["wxtpipeline", "outdir=relative"],
                    ["wxtpipeline", "outdir=/one", "outdir=/two"]):
        with pytest.raises(ValueError, match="exactly one absolute outdir"):
            runner.build_command(command)


def test_initializer_creates_output_and_changes_directory(tmp_path):
    headas = tmp_path / "headas"
    headas.mkdir()
    (headas / "headas-init.sh").write_text('export PFILES="/old;/mission/syspfiles"\n')
    destination = tmp_path / "output ; $literal"
    env = os.environ | {"HEADAS": str(headas), "JINWU_CALDB_CONFIG": "/caldb/test"}
    result = subprocess.run(["bash", "-c", _INITIALIZE, "test", str(destination),
                             "bash", "-c", 'printf "%s\\n" "$PWD" "$PFILES"'],
                            cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    cwd, pfiles = result.stdout.splitlines()
    assert cwd == str(destination)
    Path(pfiles.split(";")[0]).rmdir()
    assert destination.is_dir()


def test_mount_csv_and_existing_exec(tmp_path):
    import csv
    path = tmp_path / 'space,comma"quote'
    path.mkdir()
    mount = DockerMount(path, "/data,with comma", read_only=True)
    fields = next(csv.reader([mount.argument()]))
    assert fields == ["type=bind", f"src={path}", "dst=/data,with comma", "readonly"]
    runner = DockerRunner(container="running", context="test", mounts={"missing": "/data"})
    assert runner.build_command(["true"], existing=True)[-2:] == ["running", "true"]
    with pytest.raises(ValueError):
        runner.build_command(["true"])


def test_subprocess_errors_logs_and_start(tmp_path, monkeypatch):
    executable = tmp_path / "docker"
    executable.write_text('#!/bin/sh\nprintf "out\\n"\nprintf "err\\n" >&2\nexit 7\n')
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    runner = DockerRunner("image")
    with pytest.raises(subprocess.CalledProcessError) as error:
        runner.run(["false"])
    assert error.value.stdout == "out\n"
    assert error.value.stderr == "err\n"
    log = tmp_path / "output.log"
    result = runner.run(["false"], check=False, log_path=log)
    assert result.returncode == 7
    assert log.read_text() == "out\nerr\n"
    process = runner.start(["false"], log_path=log)
    assert process.wait(timeout=10) == 7
    assert log.read_text() == "out\nerr\nout\nerr\n"


def test_load_image_multiple_references(tmp_path, monkeypatch):
    archive = tmp_path / "image.tar"
    archive.touch()
    def load(args, **kwargs):
        assert args == ["docker", "--context", "test", "image", "load", "--input", str(archive)]
        return subprocess.CompletedProcess(args, 0, "Loaded image: one:tag\nLoaded image ID: sha256:two\n", "")
    monkeypatch.setattr(subprocess, "run", load)
    assert DockerRunner.load_image(archive, context="test") == ["one:tag", "sha256:two"]
