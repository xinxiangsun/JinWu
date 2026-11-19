"""Initialization regressions without requiring a HEASoft installation."""

import os
import sys

import pytest

from jinwu.core.heasoft import HeasoftEnvManager


def test_init_respects_headas_script_symlink_and_restores_process(tmp_path, monkeypatch):
    headas = tmp_path / "hea's install"
    build_dir = headas / "BUILD_DIR"
    build_dir.mkdir(parents=True)
    (headas / "lib" / "python").mkdir(parents=True)
    (build_dir / "headas-init.sh").write_text(
        'test -d "$HEADAS/lib/python" || return 7\n'
        'export PATH="$HEADAS/bin:$PATH"\n'
        'export PFILES="/user/par;$HEADAS/syspfiles"\n'
        'export JINWU_HEA_TEST="line one\nline two"\n'
        'echo "initializing"\n'
    )
    (headas / "headas-init.sh").symlink_to("BUILD_DIR/headas-init.sh")
    monkeypatch.setenv("HEADAS", str(headas))
    monkeypatch.delenv("PFILES", raising=False)
    before = dict(os.environ), sys.path[:]
    manager = HeasoftEnvManager()
    assert not manager.is_heasoft_initialized()

    try:
        manager.init_heasoft()
        assert manager.is_heasoft_initialized()
        assert os.environ["HEADAS"] == str(headas)
        assert os.environ["JINWU_HEA_TEST"] == "line one\nline two"
        assert sys.path[0] == str(headas / "lib" / "python")

        (build_dir / "headas-init.sh").write_text("return 7\n")
        initialized_env = dict(os.environ)
        assert HeasoftEnvManager.init_heasoft_in_notebook().is_heasoft_initialized()
        assert dict(os.environ) == initialized_env
        with pytest.raises(RuntimeError, match="初始化失败"):
            HeasoftEnvManager(headas).init_heasoft()
    finally:
        manager.reset_environment()
    assert (dict(os.environ), sys.path) == before


def test_init_reports_false_success_without_mutating_process(tmp_path, monkeypatch):
    init = tmp_path / "custom-init.sh"
    init.write_text('echo "setup failed" >&2\nreturn 0\n')
    monkeypatch.delenv("HEADAS", raising=False)
    monkeypatch.delenv("PFILES", raising=False)
    before = dict(os.environ), sys.path[:]
    with pytest.raises(RuntimeError, match="setup failed"):
        HeasoftEnvManager(init).init_heasoft()
    assert (dict(os.environ), sys.path) == before
