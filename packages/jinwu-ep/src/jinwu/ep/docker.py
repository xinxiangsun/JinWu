"""Docker entry points for the official WXT and FXT reduction commands."""
from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from jinwu.core.docker import DockerMount, DockerRunner

__all__ = ["WXTDockerRunner", "FXTDockerRunner"]

# Arguments are passed separately via "$@", never interpolated into this script.
# Initialise first: some official images set HEADAS but have an incomplete PATH.
_INITIALIZE = '''set -e
source "${HEADAS:?Image must define HEADAS}/headas-init.sh"
export CALDB=/caldb
export CALDBCONFIG="$JINWU_CALDB_CONFIG"
export CALDBALIAS=/caldb/software/tools/alias_config.fits
export PFILES="$(mktemp -d /tmp/jinwu-pfiles.XXXXXX);${PFILES##*;}"
export HEADASNOQUERY=1 HEADASPROMPT=/dev/null
if [ -n "$1" ]; then
    mkdir -p -- "$1"
    cd -- "$1"
fi
shift
exec "$@"
'''


class _EPDockerRunner(DockerRunner):
    """Shared deployment details for the two official HEASoft-based images."""

    instrument: str
    caldb_config: str

    def __init__(self, image: str, caldb: str | os.PathLike,
                 workspace: str | os.PathLike, *, context=None, user=None, env=None):
        self.workspace = Path(workspace).expanduser().resolve(strict=True)
        self.caldb = Path(caldb).expanduser().resolve(strict=True)
        if not self.workspace.is_dir() or not self.caldb.is_dir():
            raise ValueError("workspace and caldb must be existing directories")
        if Path(image).expanduser().is_file():
            raise ValueError("image must be a loaded Docker tag/ID; call DockerRunner.load_image(archive) first")
        super().__init__(
            image, mounts=[DockerMount(self.workspace, "/work"),
                           DockerMount(self.caldb, "/caldb", read_only=True)],
            workdir="/tmp", context=context, user=user, entrypoint="/bin/bash",
            env={"JINWU_CALDB_CONFIG": self.caldb_config} | dict(env or {}),
        )

    @classmethod
    def from_env(cls, workspace, *, image=None, caldb=None,
                 environ: Mapping[str, str] | None = None, **kwargs):
        """Read INSTRUMENTCALDB / INSTRUMENTDASIMG at call time.

        ``environ`` can be an explicit mapping; otherwise use ``os.environ``.
        No shell startup file is executed. Explicit image/caldb take precedence.
        DASIMG may name an archive: load it explicitly first and pass its tag as
        ``image``. No automatic image loading or ambiguous tag guessing occurs.
        """
        values = os.environ if environ is None else environ
        image = image if image is not None else values.get(f"{cls.instrument}DASIMG")
        caldb = caldb if caldb is not None else values.get(f"{cls.instrument}CALDB")
        if not image or not caldb:
            raise ValueError(f"Specify image/caldb or set {cls.instrument}DASIMG and {cls.instrument}CALDB")
        return cls(image, caldb, workspace, **kwargs)

    def container_path(self, path: str | os.PathLike) -> str:
        """Map a host path inside workspace to /work; relative paths use workspace.

        Symlinks are resolved, including existing parents of future output paths.
        Paths outside the mounted workspace are rejected.
        """
        path = Path(path).expanduser()
        path = (self.workspace / path).resolve() if not path.is_absolute() else path.resolve()
        try:
            return str(PurePosixPath("/work") / path.relative_to(self.workspace).as_posix())
        except ValueError:
            raise ValueError(f"Path must lie inside workspace {self.workspace}: {path}") from None

    def build_command(self, command, *, existing=False, env=None):
        """Preview initialization; official tasks run in their output directory."""
        if existing:
            raise ValueError("EP runners use fresh containers; use DockerRunner for docker exec")
        if isinstance(command, (str, bytes)) or not command:
            raise ValueError("command must be a nonempty argument sequence")
        directory = ""
        if command[0] in {"wxtpipeline", "fxtchain"}:
            outputs = [os.fspath(arg)[len("outdir="):] for arg in command[1:]
                       if os.fspath(arg).startswith("outdir=")]
            if outputs and (len(outputs) != 1 or not PurePosixPath(outputs[0]).is_absolute()):
                raise ValueError("Official task commands require exactly one absolute outdir")
            directory = outputs[0] if outputs else ""
        return super().build_command(
            ["--noprofile", "--norc", "-c", _INITIALIZE, "jinwu-ep", directory, *command], env=env,
        )

    def _task_command(self, task, indir, outdir, params) -> tuple[list[str], Path]:
        input_path = Path(indir).expanduser()
        input_path = (self.workspace / input_path).resolve() if not input_path.is_absolute() else input_path.resolve()
        if not input_path.is_dir():
            raise ValueError(f"indir must be an existing directory: {input_path}")
        output_path = (
            input_path.with_name(f"{input_path.name}_{task}") if outdir is None
            else (self.workspace / Path(outdir).expanduser()).resolve()
        )
        source = self.container_path(input_path)
        destination = self.container_path(output_path)
        src, dst = PurePosixPath(source), PurePosixPath(destination)
        if src == dst or src in dst.parents or dst in src.parents:
            raise ValueError("indir and outdir must be separate, non-nested directories")
        args = [task, f"indir={source}", f"outdir={destination}"]
        for key, value in params.items():
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key):
                raise ValueError(f"Invalid task parameter: {key!r}")
            if isinstance(value, os.PathLike):
                value = self.container_path(value)
            elif isinstance(value, bool):
                value = "yes" if value else "no"
            elif value is None:
                raise ValueError(f"Omit {key} instead of passing None")
            args.append(f"{key}={value}")
        return args, output_path


class WXTDockerRunner(_EPDockerRunner):
    """Official WXTDAS runner; workspace is writable, CALDB is read-only.

    Task parameters retain official meanings. Host file parameters must be Path
    objects for automatic mapping; strings pass through as container values.
    No observing-mode, GTI, coordinate or scientific screening choices are made.
    """

    instrument = "WXT"
    caldb_config = "/caldb/software/tools/caldb.config"

    def pipeline_command(self, indir, outdir=None, *, clobber=False, **params) -> list[str]:
        """Build wxtpipeline argv; outdir defaults to an indir_wxtpipeline sibling."""
        command, _ = self._task_command("wxtpipeline", indir, outdir, {"clobber": clobber} | params)
        return command

    def run_pipeline(self, indir, outdir=None, *, params=None, log_path=None, timeout=None,
                     capture_output=True, check=True):
        """Run wxtpipeline; ``params`` maps official parameter names to values.

        clobber defaults to no, overriding the official WXT default of yes.
        Omitted outdir is the resolved input's sibling named indir_wxtpipeline.
        The container creates outdir before starting the task in that directory.
        """
        return self.run(self.pipeline_command(indir, outdir, **dict(params or {})),
                        log_path=log_path, timeout=timeout, capture_output=capture_output, check=check)


class FXTDockerRunner(_EPDockerRunner):
    """Official FXTSoft runner with the same path/parameter contract as WXT."""

    instrument = "FXT"
    caldb_config = "/caldb/caldb.config"

    def chain_command(self, indir, outdir=None, *, allow_existing=False, **params) -> list[str]:
        """Build fxtchain argv, refusing nonempty output directories by default.

        FXTSoft has no clobber parameter. ``allow_existing=True`` explicitly
        permits reusing products under the official task's overwrite policy.
        Omitted outdir is the resolved input's sibling named indir_fxtchain.
        """
        if "clobber" in params:
            raise ValueError("fxtchain has no clobber parameter; choose a fresh outdir or allow_existing=True")
        command, output = self._task_command("fxtchain", indir, outdir, params)
        if output.exists() and (not output.is_dir() or (not allow_existing and any(output.iterdir()))):
            raise ValueError(f"outdir must be empty or absent; use allow_existing=True to reuse it: {output}")
        return command

    def run_chain(self, indir, outdir=None, *, params=None, allow_existing=False, log_path=None, timeout=None,
                  capture_output=True, check=True):
        """Run fxtchain, defaulting to an indir_fxtchain sibling output directory.

        Explicitly opt in before reusing a nonempty outdir.
        """
        return self.run(self.chain_command(indir, outdir, allow_existing=allow_existing, **dict(params or {})),
                        log_path=log_path, timeout=timeout, capture_output=capture_output, check=check)
