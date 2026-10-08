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
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Iterator, Mapping, Protocol, TypeVar

from .products import jsonable as _jsonable

if TYPE_CHECKING:
    from .config import ExecutionConfig

__all__ = [
    "InstrumentPipeline",
    "PipelineConfigProtocol",
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


class PipelineConfigProtocol(Protocol):
    """Structural config contract required by the pipeline machinery.

    Only ``name`` (error messages), ``pipeline`` (the registered key) and
    ``execution`` (workspace/resume) are consumed by the base class.  The
    full :class:`~jinwu.core.config.InstrumentConfig` satisfies this
    protocol, but non-instrument pipelines (e.g. ``jinwu.gw``) may ship
    their own lightweight frozen dataclass instead of constructing energy
    bands and detector fields they never use.
    """

    name: str
    pipeline: str | None
    execution: "ExecutionConfig"


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
        """解析输入根目录 / Resolve the input root to an absolute expanded path.

        不创建或校验目录存在性；调用者仍需 validate_input。
        Do not create the directory or validate existence; validate_input is separate."""
        return Path(self.root).expanduser().resolve()

    def resolved_output_root(self) -> Path:
        """解析独立输出目录 / Resolve an explicit or target-derived output root.

        显式 output_root 直接展开解析；否则清理 target_id 中不适合文件名的
        字符，在输入 root 的同级创建路径 <target>_jinwu（本函数不 mkdir）。
        清理后无有效字符抛 ValueError；不同原始标识可能归一化到同一名称。
        Expand/resolve an explicit output_root, otherwise sanitize target_id and return
        sibling <target>_jinwu. No directory is created. Empty sanitized identifiers
        raise ValueError; distinct identifiers can normalize to the same filename."""
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
    """注册流水线类装饰器 / Return a decorator registering a concrete pipeline.

    key 去首尾空白并小写，用于进程内 _PIPELINES。相同类重复注册可复用；
    不同类若 inspect.getsourcefile 返回相同值也复用已有类，适配 -m 的
    二次加载。来源不同却占用同 key 时抛 ValueError。导入后即注册，不运行。
    Strip/lowercase key for the process-local registry. Reuse an identical class,
    or an existing class whose inspect.getsourcefile value matches the new class
    (for -m reloads). Different-source key collisions raise ValueError. Registration
    occurs on decoration; no pipeline stages execute."""

    normalized = key.strip().lower()

    def decorator(cls):
        """登记或复用类 / Register the class, or return its same-source predecessor.

        修改进程注册表；返回实际注册类，键冲突抛 ValueError。
        Mutate the process registry; return the registered class or raise on collision."""
        existing = _PIPELINES.get(normalized)
        if existing is not None and existing is not cls:
            if inspect.getsourcefile(existing) == inspect.getsourcefile(cls):
                return existing
            raise ValueError(f"Pipeline key already registered: {key}")
        _PIPELINES[normalized] = cls
        return cls

    return decorator


def _discover_pipelines(key: str) -> None:
    """延迟加载仪器流水线 / Discover instrument pipelines through entry points.

    优先加载 jinwu.instruments 中与 key 首段同名的插件，未匹配则尝试全部。
    导入触发注册装饰器，必要时导入余下 key 对应子模块。不返回流水线对象；
    依赖导入失败会报告 ImportError，不能视为插件已成功安装或运行。
    Prefer entry points matching key's first segment, otherwise try all. Loading
    triggers registration and may import the dotted submodule. Return None;
    ImportError reports dependency failures. Discovery does not execute stages."""
    try:
        from importlib import import_module
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - stdlib on all supported versions
        return
    eps = list(entry_points(group="jinwu.instruments"))
    prefix = key.split(".", 1)[0]
    matched = [ep for ep in eps if ep.name == prefix]
    for ep in (matched or eps):
        try:
            ep.load()
            # A plugin entry point commonly exposes its package rather than
            # importing every optional sub-pipeline eagerly.  Once the
            # package is loaded, ask for the module represented by the
            # remaining dotted pipeline key (for example
            # ``swift.bat.survey`` -> ``jinwu.swift.bat.survey``).  This keeps
            # ``python -m`` entry points free from runpy double-import
            # warnings while still allowing ``pipeline(config, input)`` to
            # discover lazily registered sub-pipelines.
            if key not in _PIPELINES:
                module_name = getattr(ep, "module", "")
                suffix = ".".join(key.split(".")[1:])
                if module_name and suffix:
                    try:
                        import_module(f"{module_name}.{suffix}")
                    except ModuleNotFoundError as exc:
                        # Only suppress a missing derived submodule.  A
                        # dependency imported by that module must still be
                        # reported to the caller.
                        expected = f"{module_name}.{suffix}"
                        if exc.name != expected:
                            raise
        except ImportError as exc:
            raise ImportError(
                f"The {key!r} pipeline requires the {ep.name!r} instrument "
                f"package; install it with `pip install jinwu-{ep.name}`"
            ) from exc


def pipeline(config: PipelineConfigProtocol, input_data: PipelineInput, **kwargs):
    """按配置构造流水线 / Instantiate the pipeline selected by configuration.

    config 须提供 name、pipeline、execution；input_data 为 PipelineInput。
    kwargs 传给注册类构造器。未配置或未注册时抛 ValueError，按需发现
    仪器插件。返回具体流水线实例；不会调用 run，也不创建科学产物。
    Require config.name/pipeline/execution and PipelineInput. Forward kwargs to
    the registered constructor, discovering plugins lazily. Missing/unknown keys
    raise ValueError. Return the concrete instance without running stages."""
    if not config.pipeline:
        raise ValueError(f"Instrument {config.name} has no pipeline configured")
    key = config.pipeline.strip().lower()
    if key not in _PIPELINES:
        _discover_pipelines(key)
    try:
        cls = _PIPELINES[key]
    except KeyError as exc:
        choices = ", ".join(sorted(_PIPELINES)) or "none"
        raise ValueError(
            f"Unknown pipeline {config.pipeline!r}; registered: {choices}. "
            "Instrument pipelines are discovered via the 'jinwu.instruments' "
            "entry points; install the matching instrument package "
            "(e.g. `pip install jinwu-ep`)."
        ) from exc
    return cls(input_data, config=config, **kwargs)


def _fingerprint(value: Any) -> str:
    """计算规范 JSON 指纹 / Hash the jsonable representation as canonical JSON.

    通过 products.jsonable 处理对象，按键排序并去分隔空格，返回 SHA-256
    十六进制字符串；此处不读取路径所指文件内容。
    Serialize via products.jsonable with sorted keys/compact separators and return
    SHA-256 hex. A path value alone does not cause its file contents to be read."""
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_fingerprint(path: Path) -> str:
    """计算文件内容指纹 / Return a file's content SHA-256 via products.sha256_file.

    读取文件但不修改；I/O 错误向上传递。
    Read without modifying the file; propagate I/O errors."""
    from .products import sha256_file

    return sha256_file(path)


def _output_fingerprints(outputs: Mapping[str, str]) -> dict[str, str]:
    """为实际输出文件建立指纹映射 / Hash outputs that currently identify files.

    outputs 为名称到路径字符串的映射。跳过目录及不存在路径；不递归校验
    目录内容。返回 {name: SHA256}，文件读取错误上抛。
    Return {name: SHA256} for existing files, skipping directories/missing paths.
    Directory contents are not recursively validated; file-read errors propagate."""
    fingerprints: dict[str, str] = {}
    for key, value in outputs.items():
        path = Path(value)
        if path.is_file():
            fingerprints[key] = _file_fingerprint(path)
    return fingerprints


class InstrumentPipeline(ABC, Generic[InputT, ResultT]):
    """Base class for deterministic, manifest-backed instrument workflows."""

    stages: ClassVar[tuple[PipelineStage, ...]] = ()

    def __init__(self, input_data: InputT, *, config: PipelineConfigProtocol):
        """保存输入配置并推导工作区 / Initialize pipeline identity and workspace.

        优先使用 config.execution.workspace，再使用 input_data 输出目录；
        设置 .pipeline 清单路径和 pending 状态，不创建目录或执行阶段。
        Prefer configured workspace over the input-derived output root. Initialize
        manifest location and pending state, without creating directories/running stages."""
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
        """子类输入校验接口 / Validate top-level inputs before stages run.

        子类负责实现，成功返回 None，失败抛异常；此接口不定义通用科学验收。
        Subclass hook: return None on success and raise on invalid input. The base
        contract does not provide instrument-specific scientific validation."""

    @abstractmethod
    def execute_stage(
        self,
        stage: PipelineStage,
        context: Mapping[str, StageResult],
    ) -> StageResult:
        """子类单阶段执行接口 / Execute one declared stage in a subclass.

        stage 给出名称及依赖，context 包含已执行或缓存的阶段结果；须返回
        StageResult。由子类负责具体 I/O 与科学计算，框架处理状态及清单。
        stage names dependencies; context contains previous executed/cached results.
        Return StageResult. The subclass owns scientific work/I/O; the framework owns
        manifest/state handling. Raising an exception signals execution failure."""

    @abstractmethod
    def build_result(self, context: Mapping[str, StageResult]) -> ResultT:
        """组装公共结果的子类接口 / Build the public result from stage context.

        context 可只包含部分阶段，例如 until 提前结束或 needs_review；
        子类须保留这种部分状态，不假定所有科学产物都已存在。
        Context may be partial after until or needs_review; subclasses must preserve
        that status rather than assume all scientific products exist."""

    def status(self) -> PipelineRunState:
        """读取当前内存状态 / Return the current in-memory PipelineRunState.

        不读取磁盘清单、不启动阶段，也不证明输出科学有效。
        No manifest reload or execution; status alone does not establish scientific validity."""
        return self._state

    def _stage_path(self, name: str) -> Path:
        """推导阶段清单路径 / Return workspace/.pipeline/<name>.json without creating it."""
        return self._manifest_dir / f"{name}.json"

    def _input_fingerprint(self) -> str:
        """建立整次运行身份 / Hash run input, implementation identity and optional config.

        包含输入对象、具体类全名和可读取的类源码文件 SHA；配置是否纳入由
        include_config_in_input_fingerprint 决定。源码字节改变（包括注释）可
        影响该指纹；无法定位源码时实现哈希为 None。
        Include input, concrete class identity and its source-file hash when readable;
        include config according to the hook. Source-byte changes, including comments,
        can invalidate identity. Unlocatable implementation source is recorded as None."""
        implementation = inspect.getsourcefile(type(self))
        implementation_hash = None
        if implementation is not None and Path(implementation).is_file():
            implementation_hash = _file_fingerprint(Path(implementation))
        payload: dict[str, Any] = {
            "input": self.input,
            "pipeline_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "implementation_sha256": implementation_hash,
        }
        # Older/general pipelines use the complete configuration as part of
        # their immutable run identity.  Instrument adapters that can prove a
        # stage-level configuration dependency override
        # ``include_config_in_input_fingerprint`` and store the narrower
        # fingerprint in each stage manifest below.
        if self.include_config_in_input_fingerprint():
            payload["config"] = self.config
        return _fingerprint(payload)

    def include_config_in_input_fingerprint(self) -> bool:
        """决定运行指纹是否含全部配置 / Decide whether run identity includes full config.

        默认 True，保持整体配置变更使缓存失效。独立阶段适配器可覆盖为 False，
        并用 stage_config_dependencies 声明每阶段真正依赖的配置。
        Default True conservatively invalidates on any config change. Subclasses may
        return False with precise per-stage stage_config_dependencies declarations."""
        return True

    def stage_config_dependencies(self, stage: PipelineStage) -> Any:
        """返回阶段消费的配置 / Return configuration values consumed by a stage.

        默认返回完整 self.config，忽略 stage。子类可返回较小的可 JSON 化
        配置子集；遗漏真实依赖会导致陈旧缓存，因此需与阶段实现一起维护。
        Default returns full self.config, ignoring stage. Subclasses may return a
        smaller jsonable subset; declarations must reflect actual stage dependencies."""
        del stage
        return self.config

    def _stage_config_fingerprint(self, stage: PipelineStage) -> str:
        """计算阶段配置指纹 / Hash the values returned by stage_config_dependencies."""
        return _fingerprint(self.stage_config_dependencies(stage))

    def _dependency_fingerprint(
        self,
        stage: PipelineStage,
        context: Mapping[str, StageResult],
    ) -> str:
        """对上游结果及输出文件建立指纹 / Hash declared upstream results and file outputs.

        按 stage.dependencies 从 context 取 StageResult，并加入已存在输出文件
        内容哈希；缺少依赖会抛 KeyError，目录内容不递归哈希。
        Read declared dependencies from context and hash results plus existing output
        files. Missing dependencies raise KeyError; directories are not recursively hashed."""
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
        """声明额外代码依赖 / Declare extra source files that invalidate a cached stage.

        默认空元组。子类返回当前安装中的实际文件路径，指纹按内容而非 AST。
        Default empty tuple; subclasses return actual installed paths. Fingerprints
        use file contents, so documentation changes also invalidate declared dependencies."""
        return ()

    def stage_input_dependencies(self, stage: PipelineStage) -> tuple[Path, ...]:
        """声明外部数据文件依赖 / Declare external scientific inputs of a stage.

        默认空元组；目录型顶层输入不自动递归展开，子类应声明会影响结果的
        具体数据、响应和配置文件。
        Default empty tuple. Top-level directory inputs are not recursively expanded;
        subclasses declare specific data, response and config files affecting results."""
        return ()

    def _stage_input_fingerprint(self, stage: PipelineStage) -> str:
        """指纹化外部输入路径及内容 / Fingerprint declared external input dependencies.

        常规文件使用内容 SHA；目录只记录路径与 mtime_ns，不递归读取；缺失
        路径记录 missing。返回规范 JSON 的 SHA，不修改输入文件。
        Hash files by content, directories by path/mtime_ns without recursion, and
        record missing paths. Return canonical payload SHA; do not modify inputs."""
        files = []
        for path in self.stage_input_dependencies(stage):
            resolved = Path(path).expanduser().resolve()
            record: dict[str, Any] = {
                "path": str(resolved),
                "sha256": _file_fingerprint(resolved) if resolved.is_file() else None,
            }
            # A small number of optional adapters expose a calibration
            # directory rather than one concrete file.  Preserve existence
            # and directory metadata in that case without recursively hashing
            # a potentially large CALDB tree; concrete config/response files
            # should still be declared separately when their contents matter.
            if resolved.is_dir():
                try:
                    stat = resolved.stat()
                    record.update(
                        {
                            "kind": "directory",
                            "mtime_ns": int(stat.st_mtime_ns),
                        }
                    )
                except OSError:
                    record["kind"] = "directory_unreadable"
            elif resolved.exists():
                record["kind"] = "file"
            else:
                record["kind"] = "missing"
            files.append(record)
        return _fingerprint(files)

    def _stage_code_fingerprint(self, stage: PipelineStage) -> str:
        """指纹化额外源文件 / Hash declared code dependency paths and contents.

        路径须为实际文件，否则 RuntimeError；空依赖集合也有确定的指纹。
        使用完整字节哈希，含注释，不按算法语义判断是否改变。
        Require actual files or raise RuntimeError. Empty dependencies still yield a
        deterministic hash. Full bytes, including comments, determine invalidation."""
        files = []
        for path in self.stage_code_dependencies(stage):
            resolved = Path(path).expanduser().resolve()
            if not resolved.is_file():
                # 缺失时静默写 sha256=None 会让指纹退化为常量、代码改动
                # 永不触发缓存失效（AUD-01 的潜伏机制），必须显式报错。
                raise RuntimeError(
                    f"stage {stage.name!r}: 声明的代码依赖不存在: {resolved}；"
                    "请检查 stage_code_dependencies 的模块定位或包安装完整性"
                )
            files.append(
                {
                    "path": str(resolved),
                    "sha256": _file_fingerprint(resolved),
                }
            )
        return _fingerprint(files)

    def _load_cached_stage(
        self,
        stage: PipelineStage,
        context: Mapping[str, StageResult],
    ) -> StageResult | None:
        """读取可复用的完成阶段 / Load a valid completed-stage cache, otherwise None.

        要求输入、配置、上游、代码、外部输入与输出文件指纹一致，且所有声明
        输出路径存在。needs_review 和非 completed 状态始终重跑。文件缺失、
        JSON 读取失败或指纹不匹配返回 None；部分非法 payload 仍可能抛异常。
        Require matching input/config/upstream/code/external/output fingerprints and
        existing output paths. Only completed stages are reusable; review stages rerun.
        Missing/unreadable/stale manifests return None; malformed payload types/statuses
        can still raise. Existing directory outputs are not content-hashed."""
        path = self._stage_path(stage.name)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("input_fingerprint") != self._input_fingerprint():
            return None
        # Manifests written before stage-level configuration fingerprints were
        # introduced are intentionally treated as stale.  Rebuilding once is
        # safer than allowing a fit to reuse products made with another model
        # or confidence convention.
        if payload.get("stage_config_fingerprint") != self._stage_config_fingerprint(stage):
            return None
        if payload.get("dependency_fingerprint") != self._dependency_fingerprint(stage, context):
            return None
        if payload.get("stage_code_fingerprint") != self._stage_code_fingerprint(stage):
            return None
        if payload.get("stage_input_fingerprint") != self._stage_input_fingerprint(stage):
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
        """原子替换阶段清单 / Write a stage manifest through a temporary sibling file.

        创建 .pipeline 目录，记录 schema=2、结果、运行环境和各类指纹；
        临时文件写完后 replace 目标 JSON。返回 None；指纹/I/O 异常向上传递。
        Create manifest directory and store schema 2, result, runtime and fingerprints.
        Write a temporary sibling then replace target. Return None; hashing/I/O errors
        propagate. This records status and does not independently validate science."""
        self._manifest_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 2,
            "stage": stage.name,
            "input_fingerprint": self._input_fingerprint(),
            "stage_config_fingerprint": self._stage_config_fingerprint(stage),
            "dependency_fingerprint": self._dependency_fingerprint(stage, context),
            "stage_code_fingerprint": self._stage_code_fingerprint(stage),
            "stage_input_fingerprint": self._stage_input_fingerprint(stage),
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
        """提供阶段工具环境副本 / Yield an external-tool environment with local PFILES.

        context manager 创建 workspace/.pfiles/<stage_name>，复制 os.environ，
        设置 PFILES、HEADASNOQUERY 和 HEADASPROMPT。不修改父进程环境，
        退出时不删除参数目录。调用者须将 yield 的字典传给外部进程。
        Create a stage-local PFILES directory and yield a copy of os.environ with
        HEASoft prompt settings. Parent environment is unchanged; directories persist
        on exit. Callers pass the returned mapping to external tool processes."""
        env = os.environ.copy()
        pfiles = self.workspace / ".pfiles" / stage_name
        pfiles.mkdir(parents=True, exist_ok=True)
        headas = env.get("HEADAS")
        env["PFILES"] = f"{pfiles};{headas}/syspfiles" if headas else str(pfiles)
        # Keep HEASoft tasks non-interactive when the caller prepared the
        # environment that way (the Swift BAT pipeline sets this explicitly
        # for its stage-local context).  An unset value also defaults to the
        # safe batch behavior; callers that need prompts can override the
        # returned mapping before launching a task.
        env["HEADASNOQUERY"] = env.get("HEADASNOQUERY") or "1"
        env["HEADASPROMPT"] = "/dev/null"
        yield env

    def run(
        self,
        *,
        until: str | None = None,
        resume: bool | None = None,
    ) -> ResultT:
        """顺序执行或恢复流水线 / Run or resume stages in their declared order.

        Parameters
        ----------
        until : str or None
            执行到该阶段并包含该阶段；None 执行全部。未知名称抛 ValueError。
            Stop after this stage, inclusive; None runs all. Unknown names raise ValueError.
        resume : bool or None
            True 尝试复用校验通过的 completed 清单；None 使用 execution.resume。
            Try valid completed caches if True; None uses execution.resume.

        Returns
        -------
        ResultT
            经 build_result 组装的完整或部分结果。needs_review 立即返回部分结果；
            until 在非末阶段停止时总体状态为 pending，而非 completed。
            Public full/partial result from build_result. needs_review returns early;
            stopping before the final stage leaves overall state pending.

        Notes
        -----
        先 validate_input 再创建工作区。缺少前置依赖抛 RuntimeError。执行阶段
        抛异常时尝试记录 failed 清单与状态后重新抛出；返回非 completed/
        needs_review 的状态会报 RuntimeError。清单写入或 build_result 也可报错。
        Validate input before mkdir. Missing predecessors raise RuntimeError. Stage
        exceptions are recorded as failed when manifest writing succeeds, then reraised.
        Returned statuses other than completed/needs_review raise RuntimeError. Manifest
        writing and result-building errors also propagate. Completion describes workflow
        execution; scientific acceptance belongs to the concrete stages."""
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
        """从首阶段执行至指定阶段 / Run from the first stage through the named stage.

        转交 run(until=stage, resume=resume)，包含此前依赖；不是只执行单一阶段。
        默认 resume=True，返回完整或部分公共结果。
        Delegate to run(until=stage, resume=resume), including predecessors. This is
        not a standalone one-stage call. Default resume=True; return the public result."""
        return self.run(until=stage, resume=resume)
