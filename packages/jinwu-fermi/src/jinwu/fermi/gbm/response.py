"""Safe helpers for generating Fermi/GBM continuous-data responses.

The official ``SA_GBM_RSP_Gen.pl`` utility is an external dependency.  This
module deliberately constructs an argument vector instead of a shell command,
so source names and paths never become shell syntax.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Iterable, Sequence

__all__ = [
    "GBMResponseCommand",
    "GBMResponseRun",
    "build_gbm_response_command",
    "generate_gbm_response",
    "contgbmrsp",
]

_DETECTOR_NUMBERS = {
    **{f"n{index}": index for index in range(10)},
    "na": 10,
    "nb": 11,
    "b0": 12,
    "b1": 13,
}


def _normalise_detector(detector: str | int) -> tuple[str, int]:
    if isinstance(detector, int):
        if detector not in range(14):
            raise ValueError("GBM detector number must be in [0, 13]")
        name = next(name for name, number in _DETECTOR_NUMBERS.items() if number == detector)
        return name, detector
    name = str(detector).strip().lower()
    if name.startswith("nai_"):
        name = f"n{int(name.split('_', 1)[1]) - 1}"
    elif name.startswith("bgo_"):
        name = f"b{int(name.split('_', 1)[1]) - 1}"
    if name not in _DETECTOR_NUMBERS:
        choices = ", ".join(_DETECTOR_NUMBERS)
        raise ValueError(f"Unknown GBM detector {detector!r}; expected one of {choices}")
    return name, _DETECTOR_NUMBERS[name]


@dataclass(frozen=True, slots=True)
class GBMResponseCommand:
    """Validated command for the official GBM response generator."""

    arguments: tuple[str, ...]
    workdir: Path
    detector_names: tuple[str, ...]
    start_met: float
    stop_met: float


@dataclass(frozen=True, slots=True)
class GBMResponseRun:
    """Result of one response-generator invocation."""

    command: GBMResponseCommand
    returncode: int
    stdout: str
    stderr: str
    response_paths: tuple[Path, ...]


def build_gbm_response_command(
    *,
    ra_deg: float,
    dec_deg: float,
    start_met: float,
    stop_met: float,
    detectors: Iterable[str | int],
    workdir: str | Path,
    executable: str = "SA_GBM_RSP_Gen.pl",
    data_type: str = "cspec",
) -> GBMResponseCommand:
    """Build a non-shell command for an arbitrary-source GBM response.

    ``data_type`` is intentionally limited to CSPEC/CTIME, the two modes
    supported by the official response generator.  ``start_met`` and
    ``stop_met`` are Fermi MET seconds and generate an RSP2 when appropriate.
    """
    # 方法：按官方 GBM 响应生成器（Trigger Mode 2，任意源位置/任意时间）构造
    #       参数向量：-C<cspec|ctime> 选数据类型（决定能道边界）；-d<N> 逐探测器，
    #       编号 0-13（n0-n9=0..9, na=10, nb=11, b0=12, b1=13，即 12/13 为两台 BGO）；
    #       -R/-D 为源 J2000 RA/Dec（度）；-S/-E 为 MET 起止（秒），两者同给时
    #       输出带时间序列多矩阵的 RSP2（指向不变时为单矩阵 .rsp）；工作目录
    #       必须作为最后一个位置参数放在所有选项之后。
    # 参考：Fermi GBM 官方文档 "Documentation for the GBM Response Generator"
    #       https://fermi.gsfc.nasa.gov/ssc/data/analysis/gbm/DOCUMENTATION.html
    #       （-C/-d/-R/-D/-S/-E 语义、探测器编号、RSP2 规则、目录参数位置）；
    #       Meegan et al., 2009, ApJ 702, 791 (doi:10.1088/0004-637X/702/1/791)
    #       （GBM 探测器布局与 poshist/cspec 数据产品）。
    if data_type.lower() not in {"cspec", "ctime"}:
        raise ValueError("data_type must be 'cspec' or 'ctime'")
    ra = float(ra_deg)
    dec = float(dec_deg)
    start = float(start_met)
    stop = float(stop_met)
    if not 0.0 <= ra < 360.0:
        raise ValueError("ra_deg must be in [0, 360)")
    if not -90.0 <= dec <= 90.0:
        raise ValueError("dec_deg must be in [-90, 90]")
    if not start < stop:
        raise ValueError("stop_met must be larger than start_met")
    normalised = tuple(_normalise_detector(detector) for detector in detectors)
    if not normalised:
        raise ValueError("at least one GBM detector is required")
    if len({number for _, number in normalised}) != len(normalised):
        raise ValueError("GBM detectors must be unique")
    directory = Path(workdir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"GBM response working directory does not exist: {directory}")
    arguments = [
        str(executable),
        f"-C{data_type.lower()}",
        *[f"-d{number}" for _, number in normalised],
        f"-R{ra:.10g}",
        f"-D{dec:.10g}",
        f"-S{start:.9f}",
        f"-E{stop:.9f}",
        str(directory),
    ]
    return GBMResponseCommand(
        arguments=tuple(arguments),
        workdir=directory,
        detector_names=tuple(name for name, _ in normalised),
        start_met=start,
        stop_met=stop,
    )


def generate_gbm_response(
    *,
    ra_deg: float,
    dec_deg: float,
    start_met: float,
    stop_met: float,
    detectors: Iterable[str | int],
    workdir: str | Path,
    executable: str = "SA_GBM_RSP_Gen.pl",
    data_type: str = "cspec",
    timeout_s: float | None = 600.0,
) -> GBMResponseRun:
    """Run the official GBM response generator without invoking a shell."""
    command = build_gbm_response_command(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        start_met=start_met,
        stop_met=stop_met,
        detectors=detectors,
        workdir=workdir,
        executable=executable,
        data_type=data_type,
    )
    executable_path = shutil.which(command.arguments[0])
    if executable_path is None:
        raise FileNotFoundError(
            f"GBM response generator not found: {command.arguments[0]!r}. "
            "Install gbmrsp and put SA_GBM_RSP_Gen.pl on PATH."
        )
    before = {path.resolve() for path in command.workdir.glob("*.rsp*")}
    completed = subprocess.run(
        (executable_path, *command.arguments[1:]),
        cwd=command.workdir,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    after = {path.resolve() for path in command.workdir.glob("*.rsp*")}
    result = GBMResponseRun(
        command=command,
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
        response_paths=tuple(sorted(after - before)),
    )
    if result.returncode != 0:
        raise RuntimeError(
            "GBM response generator failed "
            f"(exit {result.returncode}): {result.stderr.strip() or result.stdout.strip()}"
        )
    if not result.response_paths:
        raise RuntimeError("GBM response generator succeeded but created no new .rsp/.rsp2 file")
    return result


class contgbmrsp:
    """Backward-compatible wrapper around :func:`generate_gbm_response`.

    New code should use the functions above.  The historical constructor and
    method names are retained because :class:`GBMObservation` exposes them.
    """

    def __init__(self, ra, dec, start_time, end_time, detector):
        self.ra = float(ra)
        self.dec = float(dec)
        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.detector = tuple(detector)

    @property
    def _det_num(self) -> list[int]:
        return [_normalise_detector(item)[1] for item in self.detector]

    def commandpl(self) -> str:
        command = build_gbm_response_command(
            ra_deg=self.ra,
            dec_deg=self.dec,
            start_met=self.start_time,
            stop_met=self.end_time,
            detectors=self.detector,
            workdir=Path.cwd(),
        )
        return " ".join(command.arguments)

    def gbmrsppl(self) -> GBMResponseRun:
        return generate_gbm_response(
            ra_deg=self.ra,
            dec_deg=self.dec,
            start_met=self.start_time,
            stop_met=self.end_time,
            detectors=self.detector,
            workdir=Path.cwd(),
        )
