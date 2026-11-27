"""Small, argv-based Docker CLI runner; no Docker Python dependency required."""
from __future__ import annotations

import csv
import io
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from collections.abc import Mapping, Sequence

__all__ = ["DockerMount", "DockerRunner"]


@dataclass(frozen=True)
class DockerMount:
    """An existing host file/directory bound to an absolute container path."""

    host: str | os.PathLike
    target: str
    read_only: bool = False

    def argument(self) -> str:
        host = Path(self.host).expanduser().resolve(strict=True)
        if not PurePosixPath(self.target).is_absolute():
            raise ValueError("Mount target must be an absolute container path")
        fields = ["type=bind", f"src={host}", f"dst={self.target}"]
        if self.read_only:
            fields.append("readonly")
        # Docker's --mount parser uses CSV, independently of shell quoting.
        buffer = io.StringIO()
        csv.writer(buffer, lineterminator="").writerow(fields)
        return buffer.getvalue()


class DockerRunner:
    """Run commands through Docker with explicit mounts and container settings.

    ``mounts`` accepts DockerMount objects or a host-to-container mapping. ``env``
    contains container variables; the Docker CLI inherits the caller's normal
    environment. ``context=None`` uses Docker's configured context. Construction
    and command previews do not contact Docker or load images.
    """

    def __init__(
        self, image: str | None = None, *,
        mounts: Sequence[DockerMount] | Mapping = (),
        workdir: str | None = None, container: str | None = None,
        user: str | None = None, context: str | None = None,
        env: Mapping[str, str] | None = None, entrypoint: str | None = None,
    ):
        self.image = image
        self.mounts = tuple(DockerMount(k, v) for k, v in mounts.items()) if isinstance(mounts, Mapping) else tuple(mounts)
        self.workdir = workdir
        self.container = container
        self.user = user
        self.context = context
        self.env = dict(env or {})
        self.entrypoint = entrypoint

    def build_command(self, command: Sequence[str], *, existing=False, env=None) -> list[str]:
        """Return the exact argv, validating mounts without starting a container.

        ``existing=True`` executes in ``container`` using its existing mounts and
        entrypoint; image and configured mounts are only used for ``docker run``.
        Shell syntax is not interpreted. Use an explicit shell for shell programs.
        """
        if isinstance(command, (str, bytes)) or not command:
            raise ValueError("command must be a nonempty argument sequence")
        target = self.container if existing else self.image
        if not target or str(target).startswith("-"):
            raise ValueError("Specify container for exec, or image for run")
        args = ["docker"]
        if self.context:
            args += ["--context", self.context]
        args.append("exec" if existing else "run")
        if not existing:
            args.append("--rm")
            for mount in self.mounts:
                args += ["--mount", mount.argument()]
            if self.entrypoint:
                args += ["--entrypoint", self.entrypoint]
        if self.workdir:
            if not PurePosixPath(self.workdir).is_absolute():
                raise ValueError("workdir must be an absolute container path")
            args += ["--workdir", self.workdir]
        if self.user is not None:
            args += ["--user", str(self.user)]
        for key, value in (self.env | dict(env or {})).items():
            if not key or "=" in key or "\0" in key:
                raise ValueError(f"Invalid environment variable name: {key!r}")
            args += ["--env", f"{key}={value}"]
        return args + [str(target)] + [os.fspath(arg) for arg in command]


    def run(self, command, *, existing=False, env=None, timeout=None,
            capture_output=True, check=True, log_path=None):
        """Return CompletedProcess; failures raise CalledProcessError by default.

        With ``log_path``, write combined output to that file (overwriting it),
        instead of capturing it in memory. Parent directories must exist. With
        no log and ``capture_output=False``, output streams to the terminal.
        A timeout stops the Docker CLI; the container may need manual stopping.
        """
        args = self.build_command(command, existing=existing, env=env)
        if log_path is not None:
            with Path(log_path).open("w", encoding="utf-8") as log:
                return subprocess.run(args, stdin=subprocess.DEVNULL, stdout=log,
                                      stderr=subprocess.STDOUT, text=True,
                                      timeout=timeout, check=check)
        return subprocess.run(args, stdin=subprocess.DEVNULL, capture_output=capture_output,
                              text=True, timeout=timeout, check=check)

    def start(self, command, *, existing=False, env=None, log_path=None):
        """Return Popen immediately; call ``wait`` and check its return code.

        Combined output appends to ``log_path``, or inherits the terminal. No
        implicit wait or container cleanup on Python process termination occurs.
        Terminating this CLI process does not reliably stop the container.
        """
        args = self.build_command(command, existing=existing, env=env)
        log = Path(log_path).open("a", encoding="utf-8") if log_path is not None else None
        try:
            return subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT if log else None, text=True)
        finally:
            if log:
                log.close()

    @staticmethod
    def load_image(archive: str | os.PathLike, *, context=None) -> list[str]:
        """Explicitly load a Docker save archive and return its image tags/IDs.

        Loading mutates the selected Docker daemon. It is never implicit in
        construction or from_env. Multiple returned images require a selection.
        """
        path = Path(archive).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"Not an image archive file: {path}")
        args = ["docker"] + (["--context", context] if context else [])
        result = subprocess.run(args + ["image", "load", "--input", str(path)],
                                capture_output=True, text=True, check=True)
        images = []
        for line in result.stdout.splitlines():
            for prefix in ("Loaded image: ", "Loaded image ID: "):
                if line.startswith(prefix):
                    images.append(line[len(prefix):].strip())
        if not images:
            raise RuntimeError(f"Docker did not report a loaded image: {result.stdout} {result.stderr}")
        return images
