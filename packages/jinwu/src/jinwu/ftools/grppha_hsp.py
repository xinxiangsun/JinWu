"""HEASoft ``grppha`` wrapper used by prepared spectrum workflows."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import os
import shutil
import subprocess
from typing import Any

try:
    import heasoftpy as _heasoftpy
except ImportError:  # pragma: no cover - exercised when HEASoft is unavailable
    _heasoftpy = None

__all__ = ["grppha_hsp"]


@contextmanager
def _workdir(path: Path):
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def _keyword_command(keyword: str, value: str | Path | None) -> str | None:
    if value is None:
        return None
    return f"chkey {keyword} {Path(value).name}"


def _commands(
    *,
    min_counts: int,
    rmf: str | Path | None,
    arf: str | Path | None,
    bkg: str | Path | None,
    comm: str | None,
) -> str:
    commands = [
        command
        for command in (
            _keyword_command("respfile", rmf),
            _keyword_command("ancrfile", arf),
            _keyword_command("backfile", bkg),
        )
        if command is not None
    ]
    commands.append(comm if comm is not None else f"group min {int(min_counts)}")
    commands.append("exit")
    return " & ".join(commands)


def _result(
    *,
    success: bool,
    outfile: Path,
    message: str,
    error: str | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "success": success,
        "outfile": str(outfile),
        "message": message,
    }
    if error is not None:
        result["error"] = error
    if backend is not None:
        result["backend"] = backend
    return result


def _subprocess_grppha(
    *,
    infile: Path,
    outfile: Path,
    command: str,
    clobber: bool,
    timeout: int,
) -> dict[str, Any]:
    executable = shutil.which("grppha")
    if executable is None:
        return _result(
            success=False,
            outfile=outfile,
            message="Failed",
            error="grppha executable not found",
            backend="subprocess",
        )

    try:
        with _workdir(infile.parent):
            proc = subprocess.run(
                [executable, infile.name, str(outfile), command],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
    except subprocess.TimeoutExpired:
        return _result(
            success=False,
            outfile=outfile,
            message="Failed",
            error="grppha command timed out",
            backend="subprocess",
        )
    except OSError as exc:
        return _result(
            success=False,
            outfile=outfile,
            message="Failed",
            error=str(exc),
            backend="subprocess",
        )

    if outfile.exists() and outfile.stat().st_size > 0:
        return _result(
            success=True,
            outfile=outfile,
            message="Success: grppha completed",
            backend="subprocess",
        )
    stderr = (proc.stderr or proc.stdout or "").strip()
    return _result(
        success=False,
        outfile=outfile,
        message="Failed",
        error=stderr or f"grppha exited with status {proc.returncode}",
        backend="subprocess",
    )


def grppha_hsp(
    infile: str | Path,
    outfile: str | Path,
    min_counts: int,
    *,
    rmf: str | Path | None = None,
    arf: str | Path | None = None,
    bkg: str | Path | None = None,
    comm: str | None = None,
    clobber: bool = True,
    verbose: int = 0,
    timeout: int = 60,
) -> dict[str, Any]:
    """Group one PHA and write its response/background header links.

    The output path is preserved even though HEASoft executes from the input
    PHA directory. Response/background header values are written as bare file
    names, matching XSPEC workflows that change into the spectrum directory
    before loading the grouped PHA.
    """
    source = Path(infile).expanduser().resolve()
    target = Path(outfile).expanduser().resolve()
    if not source.exists():
        return _result(
            success=False,
            outfile=target,
            message="Failed",
            error=f"Input PHA does not exist: {source}",
        )
    if min_counts < 1:
        raise ValueError("min_counts must be at least 1")
    if target.exists():
        if not clobber:
            return _result(
                success=False,
                outfile=target,
                message="Failed",
                error=f"Output PHA already exists: {target}",
            )
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)

    command = _commands(min_counts=min_counts, rmf=rmf, arf=arf, bkg=bkg, comm=comm)
    if _heasoftpy is not None:
        try:
            with _workdir(source.parent):
                result = _heasoftpy.grppha(
                    infile=source.name,
                    outfile=str(target),
                    comm=command,
                    tempc="",
                    clobber=clobber,
                    verbose=verbose,
                    allow_failure=True,
                )
            if target.exists() and target.stat().st_size > 0:
                return _result(
                    success=True,
                    outfile=target,
                    message="Success: heasoftpy grppha completed",
                    backend="heasoftpy",
                )
            detail = getattr(result, "stderr", None) or getattr(result, "messages", None)
            return _result(
                success=False,
                outfile=target,
                message="Failed",
                error=str(detail or "heasoftpy grppha did not create output PHA"),
                backend="heasoftpy",
            )
        except Exception as exc:
            fallback = _subprocess_grppha(
                infile=source,
                outfile=target,
                command=command,
                clobber=clobber,
                timeout=timeout,
            )
            fallback.setdefault("heasoftpy_error", str(exc))
            return fallback

    return _subprocess_grppha(
        infile=source,
        outfile=target,
        command=command,
        clobber=clobber,
        timeout=timeout,
    )
