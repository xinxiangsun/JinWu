"""Resumable, instrument-independent analysis pipeline primitives."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import re
import tempfile
from typing import Any, ClassVar, Generic, Iterator, Mapping, TypeVar

from .config import InstrumentConfig
from .products import jsonable as _jsonable

__all__ = [
    "InstrumentPipeline",
    "PipelineInput",
    "PipelineStage",
    "PipelineStatus",
    "PipelineRunState",
    "StageResult",
    "pipeline",
    "register_pipeline",
]


InputT = TypeVar("InputT", bound="PipelineInput")
ResultT = TypeVar("ResultT")


class PipelineStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    NEEDS_REVIEW = "needs_review"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class PipelineInput:
    target_id: str
    root: Path | str
    output_root: Path | str | None = None

    def resolved_root(self) -> Path:
        return Path(self.root).expanduser().resolve()

    def resolved_output_root(self) -> Path:
        if self.output_root is not None:
            return Path(self.output_root).expanduser().resolve()
        root = self.resolved_root()
        target = re.sub(r"[^A-Za-z0-9._-]+", "_", self.target_id.strip())
        target = re.sub(r"_+", "_", target).strip("._-")
        if not target:
            raise ValueError("target_id must contain at least one filename-safe character")
        return root.parent / f"{target}_jinwu"


@dataclass(frozen=True, slots=True)
class PipelineStage:
    name: str
    dependencies: tuple[str, ...] = ()


@dataclass(slots=True)
class StageResult:
    status: PipelineStatus = PipelineStatus.COMPLETED
    outputs: dict[str, str] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    message: str | None = None


@dataclass(frozen=True, slots=True)
class PipelineRunState:
    status: PipelineStatus
    current_stage: str | None
    completed_stages: tuple[str, ...]
    message: str | None = None


_PIPELINES: dict[str, type["InstrumentPipeline[Any, Any]"]] = {}


def register_pipeline(key: str):
    """Register one concrete pipeline without importing it from config."""

    normalized = key.strip().lower()

    def decorator(cls):
        if normalized in _PIPELINES and _PIPELINES[normalized] is not cls:
            raise ValueError(f"Pipeline key already registered: {key}")
        _PIPELINES[normalized] = cls
        return cls

    return decorator


def pipeline(config: InstrumentConfig, input_data: PipelineInput, **kwargs):
    """Construct the concrete pipeline selected by an instrument config."""
    if not config.pipeline:
        raise ValueError(f"Instrument {config.name} has no pipeline configured")
    key = config.pipeline.strip().lower()
    if key not in _PIPELINES:
        if key == "ep.wxt.pointing":
            try:
                __import__("jinwu.ep.wxt.pipeline")
            except ImportError as exc:  # pragma: no cover - depends on optional runtime env
                raise ImportError(
                    "The 'ep.wxt.pointing' pipeline requires the EP instrument "
                    "package; install it with `pip install jinwu-ep`"
                ) from exc
    try:
        cls = _PIPELINES[key]
    except KeyError as exc:
        choices = ", ".join(sorted(_PIPELINES)) or "none"
        raise ValueError(f"Unknown pipeline {config.pipeline!r}; registered: {choices}") from exc
    return cls(input_data, config=config, **kwargs)


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_fingerprint(path: Path) -> str:
    from .products import sha256_file

    return sha256_file(path)


def _output_fingerprints(outputs: Mapping[str, str]) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for key, value in outputs.items():
        path = Path(value)
        if path.is_file():
            fingerprints[key] = _file_fingerprint(path)
    return fingerprints


class InstrumentPipeline(ABC, Generic[InputT, ResultT]):
    """Base class for deterministic, manifest-backed instrument workflows."""

    stages: ClassVar[tuple[PipelineStage, ...]] = ()

    def __init__(self, input_data: InputT, *, config: InstrumentConfig):
        self.input = input_data
        self.config = config
        configured = config.execution.workspace
        self.workspace = (
            Path(configured).expanduser().resolve()
            if configured is not None
            else input_data.resolved_output_root()
        )
        self._manifest_dir = self.workspace / ".pipeline"
        self._state = PipelineRunState(PipelineStatus.PENDING, None, ())

    @abstractmethod
    def validate_input(self) -> None:
        """Validate immutable top-level inputs before any stage runs."""

    @abstractmethod
    def execute_stage(
        self,
        stage: PipelineStage,
        context: Mapping[str, StageResult],
    ) -> StageResult:
        """Execute one concrete stage."""

    @abstractmethod
    def build_result(self, context: Mapping[str, StageResult]) -> ResultT:
        """Build the public result, including partial review results."""

    def status(self) -> PipelineRunState:
        return self._state

    def _stage_path(self, name: str) -> Path:
        return self._manifest_dir / f"{name}.json"

    def _input_fingerprint(self) -> str:
        implementation = inspect.getsourcefile(type(self))
        implementation_hash = None
        if implementation is not None and Path(implementation).is_file():
            implementation_hash = _file_fingerprint(Path(implementation))
        return _fingerprint(
            {
                "input": self.input,
                "config": self.config,
                "pipeline_class": f"{type(self).__module__}.{type(self).__qualname__}",
                "implementation_sha256": implementation_hash,
            }
        )

    def _dependency_fingerprint(
        self,
        stage: PipelineStage,
        context: Mapping[str, StageResult],
    ) -> str:
        return _fingerprint(
            {
                name: {
                    "result": context[name],
                    "outputs": _output_fingerprints(context[name].outputs),
                }
                for name in stage.dependencies
            }
        )

    def stage_code_dependencies(self, stage: PipelineStage) -> tuple[Path, ...]:
        """Additional source files whose changes invalidate a cached stage."""
        return ()

    def _stage_code_fingerprint(self, stage: PipelineStage) -> str:
        files = []
        for path in self.stage_code_dependencies(stage):
            resolved = Path(path).expanduser().resolve()
            files.append(
                {
                    "path": str(resolved),
                    "sha256": _file_fingerprint(resolved) if resolved.is_file() else None,
                }
            )
        return _fingerprint(files)

    def _load_cached_stage(
        self,
        stage: PipelineStage,
        context: Mapping[str, StageResult],
    ) -> StageResult | None:
        path = self._stage_path(stage.name)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("input_fingerprint") != self._input_fingerprint():
            return None
        if payload.get("dependency_fingerprint") != self._dependency_fingerprint(stage, context):
            return None
        if payload.get("stage_code_fingerprint") != self._stage_code_fingerprint(stage):
            return None
        result_payload = payload.get("result", {})
        status = PipelineStatus(result_payload.get("status", PipelineStatus.FAILED.value))
        # Review stages are intentionally re-executed so an approval marker can
        # advance the pipeline without changing the immutable input model.
        if status != PipelineStatus.COMPLETED:
            return None
        outputs = {str(key): str(value) for key, value in result_payload.get("outputs", {}).items()}
        if any(not Path(path_value).exists() for path_value in outputs.values()):
            return None
        if payload.get("output_fingerprints", {}) != _output_fingerprints(outputs):
            return None
        return StageResult(
            status=status,
            outputs=outputs,
            data=dict(result_payload.get("data", {})),
            message=result_payload.get("message"),
        )

    def _write_stage_manifest(
        self,
        stage: PipelineStage,
        result: StageResult,
        context: Mapping[str, StageResult],
    ) -> None:
        self._manifest_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "stage": stage.name,
            "input_fingerprint": self._input_fingerprint(),
            "dependency_fingerprint": self._dependency_fingerprint(stage, context),
            "stage_code_fingerprint": self._stage_code_fingerprint(stage),
            "result": _jsonable(result),
            "output_fingerprints": _output_fingerprints(result.outputs),
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
        }
        target = self._stage_path(stage.name)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self._manifest_dir, delete=False
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(target)

    @contextmanager
    def stage_environment(self, stage_name: str) -> Iterator[dict[str, str]]:
        """Yield an external-tool environment with process-local PFILES."""
        env = os.environ.copy()
        pfiles = self.workspace / ".pfiles" / stage_name
        pfiles.mkdir(parents=True, exist_ok=True)
        headas = env.get("HEADAS")
        env["PFILES"] = f"{pfiles};{headas}/syspfiles" if headas else str(pfiles)
        env["HEADASNOQUERY"] = ""
        env["HEADASPROMPT"] = "/dev/null"
        yield env

    def run(
        self,
        *,
        until: str | None = None,
        resume: bool | None = None,
    ) -> ResultT:
        self.validate_input()
        self.workspace.mkdir(parents=True, exist_ok=True)
        use_resume = self.config.execution.resume if resume is None else bool(resume)
        context: dict[str, StageResult] = {}
        completed: list[str] = []

        stage_names = {stage.name for stage in self.stages}
        if until is not None and until not in stage_names:
            raise ValueError(f"Unknown pipeline stage {until!r}")

        for stage in self.stages:
            missing = [name for name in stage.dependencies if name not in context]
            if missing:
                raise RuntimeError(f"Stage {stage.name} has missing dependencies: {missing}")
            self._state = PipelineRunState(
                PipelineStatus.RUNNING, stage.name, tuple(completed)
            )
            result = self._load_cached_stage(stage, context) if use_resume else None
            if result is None:
                try:
                    result = self.execute_stage(stage, context)
                except Exception as exc:
                    failed = StageResult(PipelineStatus.FAILED, message=str(exc))
                    self._write_stage_manifest(stage, failed, context)
                    self._state = PipelineRunState(
                        PipelineStatus.FAILED, stage.name, tuple(completed), str(exc)
                    )
                    raise
                self._write_stage_manifest(stage, result, context)
            context[stage.name] = result

            if result.status == PipelineStatus.NEEDS_REVIEW:
                self._state = PipelineRunState(
                    PipelineStatus.NEEDS_REVIEW,
                    stage.name,
                    tuple(completed),
                    result.message,
                )
                return self.build_result(context)
            if result.status != PipelineStatus.COMPLETED:
                raise RuntimeError(f"Stage {stage.name} returned invalid status {result.status}")
            completed.append(stage.name)
            if stage.name == until:
                break

        final_status = (
            PipelineStatus.COMPLETED
            if not self.stages or completed[-1] == self.stages[-1].name
            else PipelineStatus.PENDING
        )
        self._state = PipelineRunState(final_status, None, tuple(completed))
        return self.build_result(context)

    def run_stage(self, stage: str, *, resume: bool = True) -> ResultT:
        return self.run(until=stage, resume=resume)
