"""Prepared spectrum inputs built from scanned EP instrument products."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
from typing import Iterable

from .config import instrument
from .instruments import Catalog, Manifest, SpectrumBundle
from ..ftools.grppha_hsp import grppha_hsp

__all__ = [
    "PreparedSpectrum",
    "PreparedJointSpectrum",
    "PreparedCatalog",
    "prepare_spectra",
]


@dataclass(slots=True)
class PreparedSpectrum:
    instrument: str
    obsid: str | None
    module: str | None
    detector: str | None
    source_id: str | None
    source_pha: Path | None
    grouped_pha: Path | None
    background_pha: Path | None
    arf: Path | None
    rmf: Path | None
    group_min: int
    energy_range_keV: tuple[float, float]
    diagnostics: list[str] = field(default_factory=list)
    status: str = "ready"

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def __repr__(self) -> str:
        scope = self.module or self.detector
        source_text = f", source_id={self.source_id!r}" if self.source_id else ""
        return (
            f"PreparedSpectrum(instrument={self.instrument!r}, obsid={self.obsid!r}, "
            f"scope={scope!r}{source_text}, grouped_pha={self.grouped_pha!s}, "
            f"status={self.status!r})"
        )


@dataclass(slots=True)
class PreparedJointSpectrum:
    spectra: tuple[PreparedSpectrum, ...]
    obsid: str
    modules: tuple[str, ...]
    diagnostics: list[str] = field(default_factory=list)
    status: str = "ready"

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def __repr__(self) -> str:
        return (
            f"PreparedJointSpectrum(obsid={self.obsid!r}, modules={self.modules!r}, "
            f"status={self.status!r})"
        )


@dataclass(slots=True)
class PreparedCatalog:
    root: Path
    spectra: list[PreparedSpectrum] = field(default_factory=list)
    joint_spectra: list[PreparedJointSpectrum] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    status: str = "ready"

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def __repr__(self) -> str:
        return (
            f"PreparedCatalog(root={str(self.root)!r}, spectra={len(self.spectra)}, "
            f"joint_spectra={len(self.joint_spectra)}, status={self.status!r})"
        )


def _manifests(data: Catalog | Manifest) -> list[Manifest]:
    return list(data.manifests) if isinstance(data, Catalog) else [data]


def _scan_root(data: Catalog | Manifest) -> Path:
    return data.root


def _default_outdir(data: Catalog | Manifest) -> Path:
    root = _scan_root(data).expanduser().resolve()
    return root.parent / f"{root.name}_jinwu_fit"


def _paths(bundle: SpectrumBundle) -> tuple[Path | None, Path | None, Path | None, Path | None]:
    return (
        bundle.source_pha.path if bundle.source_pha else None,
        bundle.background_pha.path if bundle.background_pha else None,
        bundle.arf.path if bundle.arf else None,
        bundle.rmf.path if bundle.rmf else None,
    )


def _group_directory(root: Path, bundle: SpectrumBundle, *, instrument_name: str) -> Path:
    obsid = bundle.obsid or "unknown_obsid"
    if instrument_name.upper() == "FXT":
        return root / obsid / (bundle.module or "FXT")
    return root / obsid / (bundle.detector or "WXT") / (bundle.source_id or "source")


def _stage_ancillary_file(
    directory: Path,
    source: Path,
    *,
    name: str,
    overwrite: bool,
) -> Path:
    staged = directory / name
    if staged.exists() or staged.is_symlink():
        if not overwrite:
            return staged
        staged.unlink()
    try:
        staged.symlink_to(source)
    except OSError:
        shutil.copy2(source, staged)
    return staged


def _stage_ancillary_files(
    directory: Path,
    *,
    background: Path,
    arf: Path,
    rmf: Path,
    overwrite: bool,
) -> dict[str, Path]:
    """Expose prepared ancillary files next to grouped PHA by filename."""
    return {
        "BACKFILE": _stage_ancillary_file(
            directory,
            background,
            name=background.name,
            overwrite=overwrite,
        ),
        "ANCRFILE": _stage_ancillary_file(
            directory,
            arf,
            name=arf.name,
            overwrite=overwrite,
        ),
        "RESPFILE": _stage_ancillary_file(
            directory,
            rmf,
            name=rmf.name,
            overwrite=overwrite,
        ),
    }


def _rewrite_grouped_header_links(grouped_pha: Path, staged_files: dict[str, Path]) -> None:
    """Keep grouped PHA response links portable inside its prepared directory."""
    from astropy.io import fits

    with fits.open(grouped_pha, mode="update") as hdus:
        header = hdus[1].header
        for keyword, staged in staged_files.items():
            header[keyword] = staged.name
        hdus.flush()


def _prepared_from_bundle(
    bundle: SpectrumBundle,
    *,
    outdir: Path,
    group_min: int | None,
    overwrite: bool,
) -> PreparedSpectrum:
    source, background, arf, rmf = _paths(bundle)
    instrument_name = bundle.source_pha.instrument if bundle.source_pha else (
        "FXT" if bundle.module else "WXT"
    )
    cfg = instrument(instrument_name)
    resolved_group_min = int(group_min if group_min is not None else cfg.group_min_counts or 1)
    diagnostics = list(bundle.diagnostics)

    if not bundle.ready or any(path is None for path in (source, background, arf, rmf)):
        diagnostics.append("spectrum bundle is not ready for grouping")
        return PreparedSpectrum(
            instrument=instrument_name,
            obsid=bundle.obsid,
            module=bundle.module,
            detector=bundle.detector,
            source_id=bundle.source_id,
            source_pha=source,
            grouped_pha=None,
            background_pha=background,
            arf=arf,
            rmf=rmf,
            group_min=resolved_group_min,
            energy_range_keV=cfg.energy_range_keV,
            diagnostics=diagnostics,
            status="partial",
        )

    grouped_dir = _group_directory(outdir, bundle, instrument_name=instrument_name)
    grouped_pha = grouped_dir / f"grouped_g{resolved_group_min}.pha"
    if grouped_pha.exists() and not overwrite:
        raise FileExistsError(f"Prepared grouped PHA already exists: {grouped_pha}")
    grouped_dir.mkdir(parents=True, exist_ok=True)
    staged_files = _stage_ancillary_files(
        grouped_dir,
        background=background,
        arf=arf,
        rmf=rmf,
        overwrite=overwrite,
    )
    result = grppha_hsp(
        infile=source,
        outfile=grouped_pha,
        min_counts=resolved_group_min,
        rmf=staged_files["RESPFILE"],
        arf=staged_files["ANCRFILE"],
        bkg=staged_files["BACKFILE"],
        clobber=overwrite,
    )
    if not bool(result.get("success")):
        diagnostics.append(f"grppha failed for {source.name}: {result.get('error') or result.get('message')}")
        status = "failed"
    else:
        grouped_pha = Path(result.get("outfile", grouped_pha)).expanduser().resolve()
        try:
            _rewrite_grouped_header_links(grouped_pha, staged_files)
        except Exception as exc:
            diagnostics.append(f"failed to rewrite grouped PHA ancillary links: {exc}")
        status = "ready"

    return PreparedSpectrum(
        instrument=instrument_name,
        obsid=bundle.obsid,
        module=bundle.module,
        detector=bundle.detector,
        source_id=bundle.source_id,
        source_pha=source,
        grouped_pha=grouped_pha,
        background_pha=background,
        arf=arf,
        rmf=rmf,
        group_min=resolved_group_min,
        energy_range_keV=cfg.energy_range_keV,
        diagnostics=diagnostics,
        status=status,
    )


def _joint_spectra(prepared: Iterable[PreparedSpectrum]) -> list[PreparedJointSpectrum]:
    by_obsid: dict[str, dict[str, PreparedSpectrum]] = {}
    for spectrum in prepared:
        if not spectrum.ready or spectrum.instrument.upper() != "FXT":
            continue
        if not spectrum.obsid or spectrum.module not in {"FXTA", "FXTB"}:
            continue
        by_obsid.setdefault(spectrum.obsid, {})[spectrum.module] = spectrum

    return [
        PreparedJointSpectrum(
            spectra=(modules["FXTA"], modules["FXTB"]),
            obsid=obsid,
            modules=("FXTA", "FXTB"),
        )
        for obsid, modules in by_obsid.items()
        if {"FXTA", "FXTB"} <= modules.keys()
    ]


def prepare_spectra(
    data: Catalog | Manifest,
    *,
    outdir: str | Path | None = None,
    group_min: int | None = None,
    overwrite: bool = False,
) -> PreparedCatalog:
    """Group scanned existing spectra into fit-ready prepared inputs."""
    target_root = Path(outdir).expanduser().resolve() if outdir is not None else _default_outdir(data)
    manifests = _manifests(data)
    spectra: list[PreparedSpectrum] = []
    warnings: list[str] = list(data.warnings) if isinstance(data, Catalog) else []
    diagnostics: list[str] = []

    for manifest in manifests:
        warnings.extend(manifest.warnings)
        diagnostics.extend(manifest.diagnostics)
        for bundle in manifest.bundles:
            spectra.append(
                _prepared_from_bundle(
                    bundle,
                    outdir=target_root,
                    group_min=group_min,
                    overwrite=overwrite,
                )
            )

    if not spectra:
        diagnostics.append("no spectrum bundles found in scanned data")
    diagnostics.extend(
        diagnostic
        for spectrum in spectra
        for diagnostic in spectrum.diagnostics
        if diagnostic not in diagnostics
    )
    status = "ready" if spectra and all(spectrum.ready for spectrum in spectra) else "partial"
    if any(spectrum.status == "failed" for spectrum in spectra):
        status = "failed"
    return PreparedCatalog(
        root=target_root,
        spectra=spectra,
        joint_spectra=_joint_spectra(spectra),
        warnings=warnings,
        diagnostics=diagnostics,
        status=status,
    )
