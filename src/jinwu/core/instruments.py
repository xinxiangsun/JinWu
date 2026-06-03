from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
import re
from typing import Any, Iterable

from astropy.io import fits

__all__ = [
    "DataFile",
    "Manifest",
    "Catalog",
    "SpectrumBundle",
    "JointSpectrumBundle",
    "LayoutScanner",
    "FXTScanner",
    "WXTScanner",
    "scan",
]

_HEADER_KEYS = (
    "TELESCOP",
    "INSTRUME",
    "DETNAM",
    "OBS_ID",
    "DATAMODE",
    "FILTER",
    "BACKFILE",
    "RESPFILE",
    "ANCRFILE",
)
_FITS_SUFFIXES = {".arf", ".fits", ".fit", ".pha", ".rmf", ".evt"}
_SCANNERS: dict[str, type["LayoutScanner"]] = {}
_KEY_FILE_ROLES = {
    "arf",
    "rmf",
    "source_pha",
    "background_pha",
    "arm_region",
    "source_region",
    "background_region",
}


def _clean_header_value(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


def _header(path: Path) -> dict[str, str]:
    """Read the small cross-instrument header subset used by scanners."""
    if path.suffix.lower() not in _FITS_SUFFIXES:
        return {}
    try:
        with fits.open(path, memmap=False) as hdus:
            values: dict[str, str] = {}
            for hdu in hdus:
                for key in _HEADER_KEYS:
                    if key in values:
                        continue
                    value = _clean_header_value(hdu.header.get(key))
                    if value is not None:
                        values[key] = value
            return values
    except OSError:
        return {}


def _identity_warnings(
    header: dict[str, str],
    *,
    instrument: str,
    obsid: str | None = None,
    detector: str | None = None,
) -> list[str]:
    warnings: list[str] = []
    telescope = header.get("TELESCOP")
    if telescope and telescope.upper() != "EP":
        warnings.append(f"TELESCOP={telescope} is not EP")
    header_instrument = header.get("INSTRUME")
    if header_instrument and header_instrument.upper() != instrument.upper():
        warnings.append(f"INSTRUME={header_instrument} is not {instrument}")
    header_obsid = header.get("OBS_ID")
    if obsid and header_obsid and header_obsid != obsid:
        warnings.append(f"OBS_ID={header_obsid} does not match {obsid}")
    header_detector = header.get("DETNAM")
    if detector and header_detector and header_detector.upper() != detector.upper():
        warnings.append(f"DETNAM={header_detector} does not match {detector}")
    return warnings


def _identity_ok(
    header: dict[str, str],
    *,
    instrument: str,
    obsid: str | None = None,
    detector: str | None = None,
) -> bool:
    return bool(header) and not _identity_warnings(
        header,
        instrument=instrument,
        obsid=obsid,
        detector=detector,
    )


def _module_name(short_name: str) -> str:
    return f"FXT{short_name.upper()}"


def _html(value: Any) -> str:
    return escape(str(value)) if value not in (None, "") else "&mdash;"


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _table(headers: tuple[str, ...], rows: Iterable[Iterable[Any]]) -> str:
    head = "".join(f"<th>{_html(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_html(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return (
        '<table style="border-collapse:collapse;text-align:left">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    )


def _definition_list(items: Iterable[tuple[str, Any]]) -> str:
    entries = "".join(
        f"<dt style='font-weight:600'>{_html(label)}</dt><dd>{_html(value)}</dd>"
        for label, value in items
    )
    return (
        "<dl style='display:grid;grid-template-columns:max-content auto;"
        f"gap:0.2rem 0.8rem;margin:0.4rem 0'>{entries}</dl>"
    )


def _messages_html(label: str, messages: list[str]) -> str:
    if not messages:
        return ""
    rows = ((message,) for message in messages)
    return f"<h4 style='margin-bottom:0.2rem'>{_html(label)}</h4>{_table(('message',), rows)}"


def _file_rows(files: Iterable["DataFile"], root: Path) -> Iterable[tuple[Any, ...]]:
    for data_file in files:
        yield (
            data_file.role,
            data_file.provenance,
            data_file.source_id,
            data_file.module,
            data_file.detector,
            _relative_path(data_file.path, root),
        )


def _key_file_rows(
    manifest: "Manifest",
    *,
    source_id: str | None = None,
    include_shared: bool = True,
) -> Iterable[tuple[Any, ...]]:
    key_files = [data_file for data_file in manifest.files if data_file.role in _KEY_FILE_ROLES]
    if source_id is not None:
        key_files = [
            data_file
            for data_file in key_files
            if data_file.source_id == source_id or (include_shared and data_file.source_id is None)
        ]
    yield from _file_rows(key_files, manifest.root)

    if not any(data_file.role == "source_region" for data_file in key_files):
        yield (
            "source_region",
            None,
            source_id,
            manifest.module,
            manifest.detector,
            "None",
        )


def _key_source_ids(manifest: "Manifest") -> list[str]:
    return sorted(
        {
            data_file.source_id
            for data_file in manifest.files
            if data_file.role in _KEY_FILE_ROLES and data_file.source_id is not None
        }
    )


def _key_files_html(manifest: "Manifest", *, title: str = "Key files") -> str:
    headers = ("role", "provenance", "source_id", "module", "detector", "path")
    source_ids = _key_source_ids(manifest)
    if len(source_ids) <= 1:
        return (
            f"<h4 style='margin-bottom:0.2rem'>{_html(title)}</h4>"
            + _table(headers, _key_file_rows(manifest))
        )

    source_tables = "".join(
        "<details open><summary>Source "
        + _html(source_id)
        + "</summary>"
        + _table(headers, _key_file_rows(manifest, source_id=source_id))
        + "</details>"
        for source_id in source_ids
    )
    return f"<h4 style='margin-bottom:0.2rem'>{_html(title)}</h4>{source_tables}"


@dataclass(slots=True)
class DataFile:
    path: Path
    role: str
    provenance: str
    mission: str
    instrument: str
    obsid: str | None = None
    module: str | None = None
    detector: str | None = None
    source_id: str | None = None
    header: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        scope = self.module or self.detector
        fields = [f"role={self.role!r}"]
        if scope:
            fields.append(f"scope={scope!r}")
        if self.source_id:
            fields.append(f"source_id={self.source_id!r}")
        fields.append(f"path={str(self.path)!r}")
        return f"DataFile({', '.join(fields)})"

    def _repr_html_(self) -> str:
        header_keys = ", ".join(f"{key}={value}" for key, value in self.header.items())
        return (
            "<div class='jinwu-data-file'><strong>DataFile</strong>"
            + _definition_list(
                (
                    ("role", self.role),
                    ("provenance", self.provenance),
                    ("path", self.path),
                    ("instrument", f"{self.mission}/{self.instrument}"),
                    ("obsid", self.obsid),
                    ("module", self.module),
                    ("detector", self.detector),
                    ("source_id", self.source_id),
                    ("header", header_keys or None),
                )
            )
            + "</div>"
        )


@dataclass(slots=True)
class SpectrumBundle:
    source_pha: DataFile | None
    background_pha: DataFile | None
    arf: DataFile | None
    rmf: DataFile | None
    module: str | None
    obsid: str | None
    diagnostics: list[str] = field(default_factory=list)
    status: str = "ready"
    detector: str | None = None
    source_id: str | None = None

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def __repr__(self) -> str:
        scope = self.module or self.detector
        source_text = f", source_id={self.source_id!r}" if self.source_id else ""
        return f"SpectrumBundle(scope={scope!r}, obsid={self.obsid!r}{source_text}, status={self.status!r})"


@dataclass(slots=True)
class JointSpectrumBundle:
    bundles: tuple[SpectrumBundle, ...]
    obsid: str
    modules: tuple[str, ...]
    diagnostics: list[str] = field(default_factory=list)
    status: str = "ready"

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def __repr__(self) -> str:
        return (
            f"JointSpectrumBundle(obsid={self.obsid!r}, modules={self.modules!r}, "
            f"status={self.status!r})"
        )


@dataclass(slots=True)
class Manifest:
    root: Path
    mission: str
    instrument: str
    layout: str
    obsid: str | None = None
    module: str | None = None
    detector: str | None = None
    files: list[DataFile] = field(default_factory=list)
    bundles: list[SpectrumBundle] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    status: str = "ready"

    def by_role(self, role: str) -> list[DataFile]:
        return [data_file for data_file in self.files if data_file.role == role]

    def by_source(self, source_id: str) -> list[DataFile]:
        return [data_file for data_file in self.files if data_file.source_id == source_id]

    def __repr__(self) -> str:
        scope = self.module or self.detector
        scope_text = f" scope={scope}" if scope else ""
        return (
            f"Manifest({self.instrument} {self.layout} obsid={self.obsid}"
            f"{scope_text} files={len(self.files)} bundles={len(self.bundles)} "
            f"status={self.status})"
        )

    def _repr_html_(self) -> str:
        roles = Counter(data_file.role for data_file in self.files)
        role_table = _table(("role", "files"), sorted(roles.items()))
        bundle_table = ""
        if self.bundles:
            bundle_table = (
                "<h4 style='margin-bottom:0.2rem'>Spectrum bundles</h4>"
                + _table(
                    ("module", "obsid", "status", "source_pha", "background_pha", "arf", "rmf"),
                    (
                        (
                            bundle.module,
                            bundle.obsid,
                            bundle.status,
                            bundle.source_pha.path.name if bundle.source_pha else None,
                            bundle.background_pha.path.name if bundle.background_pha else None,
                            bundle.arf.path.name if bundle.arf else None,
                            bundle.rmf.path.name if bundle.rmf else None,
                        )
                        for bundle in self.bundles
                    ),
                )
            )
        files_table = (
            "<details><summary>Files "
            f"({len(self.files)})</summary>"
            + _table(
                ("role", "provenance", "source_id", "module", "detector", "path"),
                _file_rows(self.files, self.root),
            )
            + "</details>"
        )
        summary = (
            "<details><summary>Manifest summary</summary>"
            + _definition_list(
                (
                    ("root", self.root),
                    ("mission/instrument", f"{self.mission}/{self.instrument}"),
                    ("layout", self.layout),
                    ("obsid", self.obsid),
                    ("module", self.module),
                    ("detector", self.detector),
                    ("status", self.status),
                )
            )
            + "<h4 style='margin-bottom:0.2rem'>File roles</h4>"
            + role_table
            + "</details>"
        )
        return (
            "<div class='jinwu-manifest'><strong>Manifest</strong>"
            + summary
            + _key_files_html(self)
            + bundle_table
            + _messages_html("Warnings", self.warnings)
            + _messages_html("Diagnostics", self.diagnostics)
            + files_table
            + "</div>"
        )


@dataclass(slots=True)
class Catalog:
    root: Path
    manifests: list[Manifest] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def bundles(self) -> list[SpectrumBundle]:
        return [bundle for manifest in self.manifests for bundle in manifest.bundles]

    @property
    def joint_bundles(self) -> list[JointSpectrumBundle]:
        by_obsid: dict[str, dict[str, SpectrumBundle]] = {}
        for bundle in self.bundles:
            if not bundle.ready or not bundle.obsid or bundle.module not in {"FXTA", "FXTB"}:
                continue
            by_obsid.setdefault(bundle.obsid, {})[bundle.module] = bundle

        joints: list[JointSpectrumBundle] = []
        for obsid, modules in by_obsid.items():
            if {"FXTA", "FXTB"} <= modules.keys():
                joints.append(
                    JointSpectrumBundle(
                        bundles=(modules["FXTA"], modules["FXTB"]),
                        obsid=obsid,
                        modules=("FXTA", "FXTB"),
                    )
                )
        return joints

    def select_source(self, source_id: str) -> "Catalog":
        """Return one source view of a catalog with shared files preserved."""
        selected_manifests: list[Manifest] = []
        for manifest in self.manifests:
            bundles = [bundle for bundle in manifest.bundles if bundle.source_id == source_id]
            if not bundles:
                continue
            files = [
                data_file
                for data_file in manifest.files
                if data_file.source_id in {None, source_id}
            ]
            selected_manifests.append(
                Manifest(
                    root=manifest.root,
                    mission=manifest.mission,
                    instrument=manifest.instrument,
                    layout=manifest.layout,
                    obsid=manifest.obsid,
                    module=manifest.module,
                    detector=manifest.detector,
                    files=files,
                    bundles=bundles,
                    warnings=list(manifest.warnings),
                    diagnostics=list(manifest.diagnostics),
                    status=manifest.status,
                )
            )
        if not selected_manifests:
            raise ValueError(f"Catalog has no spectrum source {source_id!r}")
        return Catalog(
            root=self.root,
            manifests=selected_manifests,
            warnings=list(self.warnings),
        )

    def __repr__(self) -> str:
        return (
            f"Catalog(root={str(self.root)!r}, manifests={len(self.manifests)}, "
            f"bundles={len(self.bundles)}, joint_bundles={len(self.joint_bundles)}, "
            f"warnings={len(self.warnings)})"
        )

    def _repr_html_(self) -> str:
        manifest_rows = (
            (
                manifest.instrument,
                manifest.layout,
                manifest.obsid,
                manifest.module,
                manifest.detector,
                manifest.status,
                len(manifest.files),
                len(manifest.bundles),
                len(manifest.warnings),
                len(manifest.diagnostics),
            )
            for manifest in self.manifests
        )
        manifest_table = _table(
            (
                "instrument",
                "layout",
                "obsid",
                "module",
                "detector",
                "status",
                "files",
                "bundles",
                "warnings",
                "diagnostics",
            ),
            manifest_rows,
        )
        key_file_sections = "".join(
            _key_files_html(
                manifest,
                title=(
                    f"Key files - {manifest.instrument} {manifest.obsid}"
                    if len(self.manifests) > 1
                    else "Key files"
                ),
            )
            for manifest in self.manifests
        )
        manifest_sections = "".join(
            "<details><summary>Manifest details"
            + (
                f" - {_html(manifest.instrument)} {_html(manifest.obsid)}"
                if len(self.manifests) > 1
                else ""
            )
            + "</summary>"
            + manifest._repr_html_()
            + "</details>"
            for manifest in self.manifests
        )
        return (
            "<div class='jinwu-catalog'><strong>Catalog</strong>"
            + _definition_list(
                (
                    ("root", self.root),
                    ("manifests", len(self.manifests)),
                    ("bundles", len(self.bundles)),
                    ("joint bundles", len(self.joint_bundles)),
                    ("warnings", len(self.warnings)),
                )
            )
            + "<h4 style='margin-bottom:0.2rem'>Manifests</h4>"
            + manifest_table
            + key_file_sections
            + manifest_sections
            + _messages_html("Catalog warnings", self.warnings)
            + "</div>"
        )


class LayoutScanner(ABC):
    """Base class for one instrument layout family."""

    key = ""
    instrument = ""

    @classmethod
    @abstractmethod
    def matches(cls, root: Path) -> bool:
        raise NotImplementedError

    @abstractmethod
    def scan(self, root: Path) -> Manifest:
        raise NotImplementedError


def _register_scanner(cls: type[LayoutScanner]) -> type[LayoutScanner]:
    _SCANNERS[cls.key] = cls
    return cls


def _data_file(
    path: Path,
    *,
    role: str,
    provenance: str,
    instrument: str,
    obsid: str | None = None,
    module: str | None = None,
    detector: str | None = None,
    source_id: str | None = None,
    meta: dict[str, Any] | None = None,
) -> DataFile:
    return DataFile(
        path=path,
        role=role,
        provenance=provenance,
        mission="EP",
        instrument=instrument,
        obsid=obsid,
        module=module,
        detector=detector,
        source_id=source_id,
        header=_header(path),
        meta=meta or {},
    )


def _append_identity_warnings(
    manifest: Manifest,
    data_file: DataFile,
    *,
    require_header: bool = False,
) -> None:
    if require_header and not data_file.header:
        manifest.warnings.append(f"{data_file.path.name}: FITS header could not be read")
        return
    detector = data_file.module or data_file.detector
    for warning in _identity_warnings(
        data_file.header,
        instrument=data_file.instrument,
        obsid=data_file.obsid,
        detector=detector,
    ):
        manifest.warnings.append(f"{data_file.path.name}: {warning}")


def _candidate_by_basename(candidates: Iterable[DataFile], name: str | None) -> list[DataFile]:
    if not name or name.strip().upper() in {"NONE", "NULL"}:
        return []
    basename = Path(name.strip()).name
    return [candidate for candidate in candidates if candidate.path.name == basename]


def _choose_related(
    source: DataFile,
    candidates: list[DataFile],
    *,
    relation: str,
    label: str,
    diagnostics: list[str],
) -> DataFile | None:
    relation_value = source.header.get(relation)
    has_relation = bool(relation_value and relation_value.strip().upper() not in {"NONE", "NULL"})
    related = _candidate_by_basename(candidates, relation_value)
    if len(related) == 1:
        return related[0]
    if len(related) > 1:
        diagnostics.append(f"{label} relation {relation} resolves to multiple files")
        return None
    if has_relation and candidates:
        diagnostics.append(
            f"{source.path.name} references missing {label} file {relation_value}"
        )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        diagnostics.append(f"missing {label} for {source.path.name}")
    else:
        names = ", ".join(candidate.path.name for candidate in candidates)
        diagnostics.append(f"ambiguous {label} for {source.path.name}: {names}")
    return None


def _bundle_from_fxt_files(files: list[DataFile], *, obsid: str | None, module: str | None) -> SpectrumBundle:
    diagnostics: list[str] = []
    sources = [data_file for data_file in files if data_file.role == "source_pha"]
    valid_sources = [
        source
        for source in sources
        if _identity_ok(source.header, instrument="FXT", obsid=obsid, detector=module)
    ]
    source = valid_sources[0] if len(valid_sources) == 1 else None
    if not source:
        if not valid_sources:
            diagnostics.append(f"missing validated source PHA for {module or 'FXT'}")
        else:
            names = ", ".join(candidate.path.name for candidate in valid_sources)
            diagnostics.append(f"ambiguous source PHA for {module or 'FXT'}: {names}")
        return SpectrumBundle(None, None, None, None, module, obsid, diagnostics, "partial")

    background = _choose_related(
        source,
        [data_file for data_file in files if data_file.role == "background_pha"],
        relation="BACKFILE",
        label="background PHA",
        diagnostics=diagnostics,
    )
    arf = _choose_related(
        source,
        [data_file for data_file in files if data_file.role == "arf"],
        relation="ANCRFILE",
        label="ARF",
        diagnostics=diagnostics,
    )
    rmf = _choose_related(
        source,
        [data_file for data_file in files if data_file.role == "rmf"],
        relation="RESPFILE",
        label="RMF",
        diagnostics=diagnostics,
    )
    status = "ready" if background and arf and rmf else "partial"
    return SpectrumBundle(source, background, arf, rmf, module, obsid, diagnostics, status)


def _wxt_bundle_candidates(
    files: list[DataFile],
    *,
    role: str,
    source_id: str | None = None,
    shared: bool = False,
) -> list[DataFile]:
    return [
        data_file
        for data_file in files
        if data_file.role == role
        and (
            source_id is None
            or data_file.source_id == source_id
            or (shared and data_file.source_id is None)
        )
    ]


def _bundles_from_wxt_files(
    files: list[DataFile],
    *,
    obsid: str | None,
    detector: str | None,
) -> list[SpectrumBundle]:
    source_ids = sorted(
        {
            data_file.source_id
            for data_file in files
            if data_file.role == "source_pha" and data_file.source_id is not None
        }
    )
    bundles: list[SpectrumBundle] = []

    for source_id in source_ids:
        diagnostics: list[str] = []
        sources = [
            data_file
            for data_file in _wxt_bundle_candidates(files, role="source_pha", source_id=source_id)
            if _identity_ok(data_file.header, instrument="WXT", obsid=obsid, detector=detector)
        ]
        source = sources[0] if len(sources) == 1 else None
        if source is None:
            if not sources:
                diagnostics.append(f"missing validated source PHA for {detector or 'WXT'} {source_id}")
            else:
                names = ", ".join(data_file.path.name for data_file in sources)
                diagnostics.append(f"ambiguous source PHA for {detector or 'WXT'} {source_id}: {names}")
            bundles.append(
                SpectrumBundle(
                    None,
                    None,
                    None,
                    None,
                    None,
                    obsid,
                    diagnostics,
                    "partial",
                    detector=detector,
                    source_id=source_id,
                )
            )
            continue

        background = _choose_related(
            source,
            _wxt_bundle_candidates(files, role="background_pha", source_id=source_id),
            relation="BACKFILE",
            label="background PHA",
            diagnostics=diagnostics,
        )
        arf = _choose_related(
            source,
            _wxt_bundle_candidates(files, role="arf", source_id=source_id),
            relation="ANCRFILE",
            label="ARF",
            diagnostics=diagnostics,
        )
        rmf = _choose_related(
            source,
            _wxt_bundle_candidates(files, role="rmf", source_id=source_id, shared=True),
            relation="RESPFILE",
            label="RMF",
            diagnostics=diagnostics,
        )
        status = "ready" if background and arf and rmf else "partial"
        bundles.append(
            SpectrumBundle(
                source,
                background,
                arf,
                rmf,
                None,
                obsid,
                diagnostics,
                status,
                detector=detector,
                source_id=source_id,
            )
        )

    return bundles


@_register_scanner
class FXTScanner(LayoutScanner):
    key = "fxt"
    instrument = "FXT"

    _flat_root = re.compile(
        r"^ep(?P<obsid>\d{11})_FXT(?P<module>[AB])(?:_[A-Za-z0-9]+)+$",
        re.IGNORECASE,
    )
    _raw_event = re.compile(
        r"^fxt_(?P<module>[ab])_(?P<obsid>\d{11})_(?P<mode>ff|pw|tm)_"
        r"(?P<filter>0[123])_(?P<pointing>po|sl)_uf_(?P<event_type>evt|fsaevt)_"
        r"(?P<version>[A-Za-z0-9]+)\.fits$",
        re.IGNORECASE,
    )
    _clean_event = re.compile(
        r"^fxt_(?P<module>[ab])_(?P<obsid>\d{11})_(?P<mode>ff|pw|tm)_"
        r"(?P<filter>0[123])_(?P<pointing>po|sl)_cl_(?P<version>[A-Za-z0-9]+)\.fits$",
        re.IGNORECASE,
    )
    _auxiliary = re.compile(
        r"^ep_(?P<obsid>\d{11})_(?P<role>att|orb)_(?P<version>[A-Za-z0-9]+)\.fits$",
        re.IGNORECASE,
    )
    _housekeeping = re.compile(
        r"^fxt_(?P<obsid>\d{11})_(?P<role>mkf|hk)_(?P<version>[A-Za-z0-9]+)\.fits$",
        re.IGNORECASE,
    )
    _flat_pha_response = re.compile(
        r"^fxt_(?P<module>[ab])_(?P<obsid>\d{11})_(?P<mode>ff|pw|tm)_"
        r"(?P<filter>0[123])_(?P<pointing>po|sl)_(?P<region>src|bkg)_"
        r"(?P<source_id>\d+)(?P<grouped>_grp)?_(?P<version>[A-Za-z0-9]+)"
        r"(?:_(?P<release>[A-Za-z0-9]+))?\.(?P<suffix>pha|arf|rmf)$",
        re.IGNORECASE,
    )
    _flat_extracted_event = re.compile(
        r"^fxt_(?P<module>[ab])_(?P<obsid>\d{11})_(?P<mode>ff|pw|tm)_"
        r"(?P<filter>0[123])_(?P<pointing>po|sl)_(?P<region>src|bkg)_"
        r"(?P<source_id>\d+)_cl_(?P<version>[A-Za-z0-9]+)"
        r"(?:_(?P<release>[A-Za-z0-9]+))?\.fits$",
        re.IGNORECASE,
    )
    _flat_lightcurve = re.compile(
        r"^fxt_(?P<module>[ab])_(?P<obsid>\d{11})_(?P<mode>ff|pw|tm)_"
        r"(?P<filter>0[123])_(?P<pointing>po|sl)_(?P<region>src|bkg)_"
        r"(?P<source_id>\d+)_w_(?P<binsize>[A-Za-z0-9]+)_"
        r"(?P<version>[A-Za-z0-9]+)(?:_(?P<release>[A-Za-z0-9]+))?\.lc$",
        re.IGNORECASE,
    )

    @classmethod
    def matches(cls, root: Path) -> bool:
        if (root / "auxil").is_dir() and (root / "fxt" / "event").is_dir():
            return True
        if cls._flat_root.match(root.name):
            return True
        return any(cls._flat_pha_response.match(path.name) for path in root.glob("fxt_*"))

    def scan(self, root: Path) -> Manifest:
        if (root / "auxil").is_dir() and (root / "fxt" / "event").is_dir():
            return self._scan_level1(root)
        return self._scan_flat(root)

    def _scan_level1(self, root: Path) -> Manifest:
        obsid = root.name if root.name.isdigit() else None
        manifest = Manifest(root=root, mission="EP", instrument="FXT", layout="fxt_level1", obsid=obsid)

        for path in sorted((root / "auxil").glob("*.fits")):
            match = self._auxiliary.match(path.name)
            if not match:
                continue
            role = "attitude" if match.group("role").lower() == "att" else "orbit"
            data_file = _data_file(
                path,
                role=role,
                provenance="level1",
                instrument="FXT",
                obsid=match.group("obsid"),
                meta={"version": match.group("version")},
            )
            manifest.files.append(data_file)
            _append_identity_warnings(manifest, data_file)

        hk_dir = root / "fxt" / "hk"
        for path in sorted(hk_dir.glob("*.fits")) if hk_dir.is_dir() else []:
            match = self._housekeeping.match(path.name)
            if not match:
                continue
            role = "mkf" if match.group("role").lower() == "mkf" else "housekeeping"
            data_file = _data_file(
                path,
                role=role,
                provenance="level1",
                instrument="FXT",
                obsid=match.group("obsid"),
                meta={"version": match.group("version")},
            )
            manifest.files.append(data_file)
            _append_identity_warnings(manifest, data_file)

        for path in sorted((root / "fxt" / "event").glob("*.fits")):
            match = self._raw_event.match(path.name)
            if not match:
                continue
            module = _module_name(match.group("module"))
            event_type = match.group("event_type").lower()
            role = "unfiltered_event" if event_type == "evt" else "frame_store_event"
            data_file = _data_file(
                path,
                role=role,
                provenance="level1",
                instrument="FXT",
                obsid=match.group("obsid"),
                module=module,
                meta={
                    "datamode": match.group("mode").upper(),
                    "filter": match.group("filter"),
                    "pointing": match.group("pointing").lower(),
                    "event_type": event_type,
                    "version": match.group("version"),
                },
            )
            manifest.files.append(data_file)
            _append_identity_warnings(manifest, data_file, require_header=True)

        products = root / "fxt" / "products"
        for path in sorted(products.glob("*.fits")) if products.is_dir() else []:
            match = self._clean_event.match(path.name)
            if not match:
                continue
            data_file = _data_file(
                path,
                role="cleaned_event",
                provenance="pipeline",
                instrument="FXT",
                obsid=match.group("obsid"),
                module=_module_name(match.group("module")),
                meta={
                    "datamode": match.group("mode").upper(),
                    "filter": match.group("filter"),
                    "pointing": match.group("pointing").lower(),
                    "version": match.group("version"),
                },
            )
            manifest.files.append(data_file)
            _append_identity_warnings(manifest, data_file, require_header=True)

        for path in sorted(root.glob("*.reg")):
            lower_name = path.name.lower()
            if "bkg" in lower_name:
                role = "background_region"
            elif "src" in lower_name:
                role = "source_region"
            else:
                role = "region"
            manifest.files.append(
                _data_file(path, role=role, provenance="local", instrument="FXT", obsid=obsid)
            )

        if not manifest.files:
            manifest.status = "partial"
            manifest.diagnostics.append("FXT Level 1 directory has no recognized files")
        return manifest

    def _scan_flat(self, root: Path) -> Manifest:
        root_match = self._flat_root.match(root.name)
        obsid = root_match.group("obsid") if root_match else None
        module = _module_name(root_match.group("module")) if root_match else None
        manifest = Manifest(
            root=root,
            mission="EP",
            instrument="FXT",
            layout="fxt_flat",
            obsid=obsid,
            module=module,
        )

        for path in sorted(root.iterdir()):
            if not path.is_file():
                continue
            match = self._flat_pha_response.match(path.name)
            if match:
                data_file = self._flat_pha_or_response(path, match)
            else:
                match = self._flat_extracted_event.match(path.name)
                if match:
                    data_file = self._flat_extracted_data(path, match, role="cleaned_event")
                else:
                    match = self._flat_lightcurve.match(path.name)
                    if match:
                        region = match.group("region").lower()
                        role = "source_lightcurve" if region == "src" else "background_lightcurve"
                        data_file = self._flat_extracted_data(path, match, role=role)
                    elif path.suffix.lower() == ".pha" and re.search(
                        r"(?:group|min|grp)", path.name, re.IGNORECASE
                    ):
                        data_file = _data_file(
                            path,
                            role="local_grouped_pha",
                            provenance="local_grouped",
                            instrument="FXT",
                            obsid=obsid,
                            module=module,
                        )
                    else:
                        data_file = self._flat_ancillary(path, obsid=obsid, module=module)
            if data_file is None:
                continue
            manifest.files.append(data_file)
            _append_identity_warnings(
                manifest,
                data_file,
                require_header=data_file.path.suffix.lower() in _FITS_SUFFIXES,
            )
            obsid = obsid or data_file.obsid or data_file.header.get("OBS_ID")
            module = module or data_file.module or data_file.header.get("DETNAM")

        manifest.obsid = obsid
        manifest.module = module
        bundle = _bundle_from_fxt_files(manifest.files, obsid=obsid, module=module)
        if any(data_file.role == "source_pha" for data_file in manifest.files):
            manifest.bundles.append(bundle)
            manifest.diagnostics.extend(bundle.diagnostics)
        if not manifest.files or manifest.warnings or (manifest.bundles and not bundle.ready):
            manifest.status = "partial"
        return manifest

    def _flat_pha_or_response(self, path: Path, match: re.Match[str]) -> DataFile:
        region = match.group("region").lower()
        suffix = match.group("suffix").lower()
        grouped = bool(match.group("grouped"))
        if suffix == "pha" and region == "src" and grouped:
            role = "official_grouped_pha"
            provenance = "official_grouped"
        elif suffix == "pha" and region == "src":
            role = "source_pha"
            provenance = "official"
        elif suffix == "pha" and region == "bkg":
            role = "background_pha"
            provenance = "official"
        elif suffix == "arf" and region == "src":
            role = "arf"
            provenance = "official"
        elif suffix == "rmf" and region == "src":
            role = "rmf"
            provenance = "official"
        else:
            role = f"{region}_{suffix}"
            provenance = "official"
        return _data_file(
            path,
            role=role,
            provenance=provenance,
            instrument="FXT",
            obsid=match.group("obsid"),
            module=_module_name(match.group("module")),
            source_id=match.group("source_id"),
            meta={
                "datamode": match.group("mode").upper(),
                "filter": match.group("filter"),
                "pointing": match.group("pointing").lower(),
                "version": match.group("version"),
                "release": match.group("release"),
            },
        )

    def _flat_extracted_data(self, path: Path, match: re.Match[str], *, role: str) -> DataFile:
        meta = {
            "datamode": match.group("mode").upper(),
            "filter": match.group("filter"),
            "pointing": match.group("pointing").lower(),
            "region": match.group("region").lower(),
            "version": match.group("version"),
            "release": match.group("release"),
        }
        if "binsize" in match.groupdict():
            meta["binsize"] = match.group("binsize")
        return _data_file(
            path,
            role=role,
            provenance="official",
            instrument="FXT",
            obsid=match.group("obsid"),
            module=_module_name(match.group("module")),
            source_id=match.group("source_id"),
            meta=meta,
        )

    def _flat_ancillary(self, path: Path, *, obsid: str | None, module: str | None) -> DataFile | None:
        suffix = path.suffix.lower()
        if suffix == ".reg":
            return _data_file(
                path,
                role="region",
                provenance="official",
                instrument="FXT",
                obsid=obsid,
                module=module,
            )
        if suffix in {".gif", ".png", ".pdf", ".log", ".pds"}:
            return _data_file(
                path,
                role="ancillary",
                provenance="official",
                instrument="FXT",
                obsid=obsid,
                module=module,
            )
        return None


@_register_scanner
class WXTScanner(LayoutScanner):
    key = "wxt"
    instrument = "WXT"

    _flat_root = re.compile(
        r"^ep(?P<obsid>\d{11})wxt(?P<detector>CMOS\d+)(?:[A-Za-z0-9_]+)?$",
        re.IGNORECASE,
    )
    _base = re.compile(r"^ep(?P<obsid>\d{11})wxt(?P<cmos>\d+)(?P<rest>.*)$", re.IGNORECASE)
    _source = re.compile(r"^s(?P<source_id>\d+)", re.IGNORECASE)

    @classmethod
    def matches(cls, root: Path) -> bool:
        if cls._flat_root.match(root.name):
            return True
        return any(cls._base.match(path.name) for path in root.iterdir() if path.is_file())

    def scan(self, root: Path) -> Manifest:
        root_match = self._flat_root.match(root.name)
        obsid = root_match.group("obsid") if root_match else None
        detector = root_match.group("detector").upper() if root_match else None
        manifest = Manifest(
            root=root,
            mission="EP",
            instrument="WXT",
            layout="wxt_l23_product",
            obsid=obsid,
            detector=detector,
        )

        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            match = self._base.match(path.name)
            if not match and path.parent.name != "ProductReport":
                continue
            role = self._role(path.name)
            if not role:
                continue
            path_obsid = match.group("obsid") if match else obsid
            path_detector = f"CMOS{match.group('cmos')}" if match else detector
            source_id = self._source_id(match.group("rest")) if match else None
            data_file = _data_file(
                path,
                role=role,
                provenance="official",
                instrument="WXT",
                obsid=path_obsid,
                detector=path_detector,
                source_id=source_id,
            )
            manifest.files.append(data_file)
            _append_identity_warnings(
                manifest,
                data_file,
                require_header=path.suffix.lower() in _FITS_SUFFIXES,
            )
            obsid = obsid or data_file.header.get("OBS_ID") or data_file.obsid
            detector = detector or data_file.header.get("DETNAM") or data_file.detector

        manifest.obsid = obsid
        manifest.detector = detector
        if any(data_file.role == "source_pha" for data_file in manifest.files):
            manifest.bundles.extend(
                _bundles_from_wxt_files(manifest.files, obsid=obsid, detector=detector)
            )
            for bundle in manifest.bundles:
                manifest.diagnostics.extend(bundle.diagnostics)
        if (
            not manifest.files
            or manifest.warnings
            or any(not bundle.ready for bundle in manifest.bundles)
        ):
            manifest.status = "partial"
        return manifest

    def _role(self, name: str) -> str | None:
        lower_name = name.lower()
        if lower_name.endswith("po_cl.evt"):
            return "cleaned_event"
        if lower_name.endswith(("po_uf.evt", "sl_uf.evt")):
            return "unfiltered_event"
        if lower_name.endswith("clgti.fits"):
            return "gti"
        if re.search(r"s\d+bk\.pha$", lower_name):
            return "background_pha"
        if re.search(r"s\d+\.pha$", lower_name):
            return "source_pha"
        if lower_name.endswith(".arf"):
            return "arf"
        if lower_name.endswith(".rmf"):
            return "rmf"
        if re.search(r"s\d+bk\.lc$", lower_name):
            return "background_lightcurve"
        if re.search(r"s\d+\.lc$", lower_name):
            return "source_lightcurve"
        if re.search(r"s\d+bk\.reg$", lower_name):
            return "background_region"
        if lower_name.endswith("arm.reg"):
            return "arm_region"
        if lower_name.endswith(".reg"):
            return "region"
        if lower_name.endswith(".mkf"):
            return "mkf"
        if lower_name.endswith(".expcorr"):
            return "exposure_correction"
        if lower_name.endswith((".exp", ".exp.gz")):
            return "exposure"
        if lower_name.endswith(
            (".gif", ".pdf", ".cat", ".conf", ".img", ".prefilter", "_ufbp.fits", "_ufhp.fits")
        ):
            return "ancillary"
        return None

    def _source_id(self, rest: str) -> str | None:
        match = self._source.match(rest)
        return f"s{match.group('source_id')}" if match else None


def _scanner_for(instrument_name: str) -> LayoutScanner:
    key = instrument_name.strip().lower().replace("ep/", "").replace("ep_", "")
    if key not in _SCANNERS:
        choices = ", ".join(sorted(_SCANNERS))
        raise ValueError(f"Unsupported instrument scanner: {instrument_name}. Available: {choices}")
    return _SCANNERS[key]()


def _scan_one(path: str | Path, instrument: str | None = None) -> Manifest:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {root}")
    if instrument:
        scanner = _scanner_for(instrument)
        if not scanner.matches(root):
            raise ValueError(f"{root} does not match the {instrument} scanner")
        return scanner.scan(root)

    matches = [scanner_cls() for scanner_cls in _SCANNERS.values() if scanner_cls.matches(root)]
    if len(matches) != 1:
        keys = ", ".join(scanner.key for scanner in matches) or "none"
        raise ValueError(f"Could not choose one scanner for {root}. Matches: {keys}")
    return matches[0].scan(root)


def scan(path: str | Path, instrument: str | None = None) -> Catalog:
    """Scan an EP data directory or directory collection into a catalog.

    English
    -------
    ``scan`` is the public entry point. It first tries ``path`` itself as one
    supported data directory. If that does not match, it scans direct child
    directories and collects each successful manifest in a :class:`Catalog`.
    A single data directory therefore returns a catalog with one manifest.

    ``scan`` intentionally scans only the root directory or its direct children;
    instrument scanners own the deeper directory traversal inside a recognized
    layout. Pass ``instrument`` to restrict matching to one scanner family.

    中文
    ----
    ``scan`` 是统一扫描入口。它会先尝试把 ``path`` 本身当作一个受支持的
    数据目录处理；若根目录不能直接匹配，则扫描其直接子目录，并将每个成功的
    manifest 汇总为 :class:`Catalog`。输入单个数据目录时，返回的 catalog 只含
    一个 manifest。

    ``scan`` 只负责根目录和直接子目录这一层；某个目录一旦被识别，目录内部的
    深层遍历由对应仪器扫描器负责。传入 ``instrument`` 可将扫描限制到指定
    仪器扫描器。
    """
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {root}")

    try:
        return Catalog(root=root, manifests=[_scan_one(root, instrument=instrument)])
    except ValueError as exc:
        catalog = Catalog(root=root, warnings=[str(exc)])

    for child in sorted(child for child in root.iterdir() if child.is_dir()):
        try:
            catalog.manifests.append(_scan_one(child, instrument=instrument))
        except ValueError:
            continue
    if not catalog.manifests:
        raise ValueError(f"No supported EP data directories found under {root}")
    return catalog
