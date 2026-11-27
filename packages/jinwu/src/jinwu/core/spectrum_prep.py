"""Prepared spectrum inputs built from scanned EP instrument products."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Iterable

from .config import instrument
from .instruments import Catalog, Manifest, SpectrumBundle
from ..ftools.grppha_hsp import grppha_hsp  # TODO: migrate to ftgrouppha (deprecated)

__all__ = [
    "PreparedSpectrum",
    "PreparedJointSpectrum",
    "PreparedCatalog",
    "prepare_spectra",
    "prepare_ogip_spectrum",
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
    spectrum_key: str | None = None

    @property
    def ready(self) -> bool:
        """读取准备状态标记 / Test whether the stored status is ready.

        不重新校验文件；True 仅表示 status == 'ready'。
        Does not revalidate files; True only means status == 'ready'.
        """
        return self.status == "ready"

    def __repr__(self) -> str:
        """显示单谱身份与状态 / Summarize spectrum identity, path and status."""
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
        """读取联合谱状态 / Test the stored joint-spectrum readiness flag.

        不递归检查成员状态或磁盘文件。
        Does not recursively check member statuses or files on disk.
        """
        return self.status == "ready"

    def __repr__(self) -> str:
        """显示联合谱观测与模块 / Summarize joint observation, modules and status."""
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
        """读取目录总体状态 / Test the stored catalog readiness flag.

        仅比较 status 字段，不重跑分组或校验。
        Compare status only; do not rerun grouping or validation.
        """
        return self.status == "ready"

    def __repr__(self) -> str:
        """显示目录、谱数量与状态 / Summarize root, spectrum counts and status."""
        return (
            f"PreparedCatalog(root={str(self.root)!r}, spectra={len(self.spectra)}, "
            f"joint_spectra={len(self.joint_spectra)}, status={self.status!r})"
        )


def _manifests(data: Catalog | Manifest) -> list[Manifest]:
    """统一扫描结果为清单列表 / Normalize scanned input to a manifest list.

    Catalog 的 manifests 浅拷贝为列表；单个 Manifest 包装为一项列表。
    Shallow-copy Catalog manifests or wrap a single Manifest in a one-item list.
    """
    return list(data.manifests) if isinstance(data, Catalog) else [data]


def _scan_root(data: Catalog | Manifest) -> Path:
    """返回扫描根路径 / Return the scanned root path without resolving it."""
    return data.root


def _default_outdir(data: Catalog | Manifest) -> Path:
    """推导独立准备目录 / Derive a sibling directory for prepared spectra.

    解析扫描根目录，返回同级的 <root-name>_jinwu_fit，不创建目录。
    Resolve the scan root and return sibling <root-name>_jinwu_fit; no mkdir.
    """
    root = _scan_root(data).expanduser().resolve()
    return root.parent / f"{root.name}_jinwu_fit"


def _paths(bundle: SpectrumBundle) -> tuple[Path | None, Path | None, Path | None, Path | None]:
    """提取源、背景、ARF、RMF 路径 / Extract source, background, ARF, RMF paths.

    返回固定顺序四元组，缺失产品为 None，不访问文件系统。
    Return that ordered four-tuple, using None for absent products; no file access.
    """
    return (
        bundle.source_pha.path if bundle.source_pha else None,
        bundle.background_pha.path if bundle.background_pha else None,
        bundle.arf.path if bundle.arf else None,
        bundle.rmf.path if bundle.rmf else None,
    )


def _group_directory(root: Path, bundle: SpectrumBundle, *, instrument_name: str) -> Path:
    """按观测和仪器作用域组织目录 / Derive an observation-scoped output path.

    FXT 用 obsid/module，其他仪器用 obsid/detector/source_id；缺失身份
    使用显式占位名。本函数不创建目录。
    FXT uses obsid/module; others use obsid/detector/source_id. Missing identity
    uses placeholder names. Return a path without creating directories.
    """
    obsid = bundle.obsid or "unknown_obsid"
    if instrument_name.upper() == "FXT":
        return root / obsid / (bundle.module or "FXT")
    return root / obsid / (bundle.detector or "WXT") / (bundle.source_id or "source")


def _stage_ancillary_file(
    directory: Path,
    source: Path,
    *,
    name: str,
) -> Path:
    """在准备目录链接或复制辅助文件 / Stage one ancillary link or copy.

    同名目标已指向 source 时复用；否则移除旧目标再建立符号链接，链接
    失败则 copy2。directory 必须已存在。返回 staged 路径并修改磁盘。
    Reuse a same-source target, otherwise replace it with a symlink or copy2
    fallback. The directory must already exist. Return the staged path and
    modify disk.
    """
    staged = directory / name
    if staged.exists() or staged.is_symlink():
        try:
            if staged.resolve() == source.resolve() or staged.samefile(source):
                return staged
        except (OSError, RuntimeError):
            # A broken or looping old symlink is stale and must be replaced.
            pass
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
) -> dict[str, Path]:
    """将辅助谱文件放到分组谱旁 / Stage ancillary files beside grouped PHA.

    返回 BACKFILE、ANCRFILE、RESPFILE 到本地路径的映射；分别链接/复制
    背景、ARF、RMF。目标使用原始 basename，调用者须避免同名文件冲突。
    Return those three keyword-to-path mappings, staging background/ARF/RMF
    with original basenames. Callers must avoid ancillary basename collisions.
    """
    return {
        "BACKFILE": _stage_ancillary_file(
            directory,
            background,
            name=background.name,
        ),
        "ANCRFILE": _stage_ancillary_file(
            directory,
            arf,
            name=arf.name,
        ),
        "RESPFILE": _stage_ancillary_file(
            directory,
            rmf,
            name=rmf.name,
        ),
    }


def _validate_grouped_header_links(grouped_pha: Path, staged_files: dict[str, Path]) -> None:
    """检查分组谱辅助文件链接 / Verify grouped PHA ancillary header links.

    打开 SPECTRUM 扩展，检查各目标是文件且头值等于 staged.name。
    成功返回 None；扩展缺失/头值不符抛 ValueError，目标缺失抛
    FileNotFoundError。不校验响应内容或科学适用性。
    Read SPECTRUM and require existing targets plus matching basenames.
    Return None on success; missing extension/wrong values raise ValueError,
    absent files raise FileNotFoundError. No calibration/content validation.
    """
    from astropy.io import fits

    with fits.open(grouped_pha) as hdus:
        try:
            header = hdus["SPECTRUM"].header
        except KeyError as exc:
            raise ValueError("grouped PHA has no SPECTRUM extension") from exc
        for keyword, staged in staged_files.items():
            if not staged.is_file():
                raise FileNotFoundError(f"Staged {keyword} target does not exist: {staged}")
            actual = header.get(keyword)
            if str(actual).strip() != staged.name:
                raise ValueError(
                    f"SPECTRUM {keyword}={actual!r}; expected staged filename {staged.name!r}"
                )


def _prepared_from_bundle(
    bundle: SpectrumBundle,
    *,
    outdir: Path,
    group_min: int | None,
    overwrite: bool,
) -> PreparedSpectrum:
    """准备单组已扫描谱 / Prepare one scanned spectrum bundle for fitting.

    解析仪器配置的分组阈值和 keV 拟合范围；输入不完整返回 partial。
    完整输入在独立目录链接辅助文件、调用 grppha，并检查头链接与通道
    兼容性，将结果及诊断写入 PreparedSpectrum。不会执行谱拟合。
    Resolve grouping threshold and fit energy range (keV) from configuration.
    Incomplete input returns partial. Otherwise stage ancillary files, invoke
    grppha, check links/channel compatibility, and return status/diagnostics.
    No spectral fit is performed. Existing grouped output raises FileExistsError
    unless overwrite=True; other staging/tool exceptions may propagate.
    """
    source, background, arf, rmf = _paths(bundle)
    instrument_name = bundle.source_pha.instrument if bundle.source_pha else (
        "FXT" if bundle.module else "WXT"
    )
    cfg = instrument(instrument_name)
    resolved_group_min = int(group_min if group_min is not None else cfg.spectrum.group_min_counts or cfg.group_min_counts or 1)
    fit_energy_range = cfg.spectrum.fit_energy_range_keV or cfg.energy_range_keV
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
            energy_range_keV=fit_energy_range,
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
    )
    # 方法：采用 HEASoft grppha 标准 "group min N" 最小计数分组（N 由仪器配置给出，WXT=1/FXT=3），原 PHA 的 EXPOSURE/BACKSCAL/AREASCAL 关键词保持原值、由 XSPEC 按其标准语义解释面积/背景标度；轻分组适配 Poisson 似然（cstat/wstat）拟合
    # 参考：OGIP CAL/GEN/92-002 "The Calibration Requirements for Spectral Analysis"（EXPOSURE/BACKSCAL/AREASCAL 语义，George, Arnaud, Pence et al., HEASARC）；分组工具语义见 HEASoft ftools/grppha
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
        status = "ready"
        try:
            _validate_grouped_header_links(grouped_pha, staged_files)
        except Exception as exc:
            diagnostics.append(f"failed to validate grouped PHA ancillary links: {exc}")
            status = "partial"

        try:
            from .io import read_pha, read_rmf, read_arf
            from .ogip import check_response_compatibility

            compatibility = check_response_compatibility(
                read_pha(grouped_pha), read_rmf(staged_files["RESPFILE"]),
                arf=read_arf(staged_files["ANCRFILE"]),
            )
            diagnostics.extend(
                f"response compatibility {message.level.lower()} [{message.code}]: {message.message}"
                for message in compatibility.messages
            )
            if not compatibility.ok:
                status = "failed"
            elif any(
                message.level == "WARN" or message.code == "COMPAT_NOT_CHECKED"
                for message in compatibility.messages
            ) and status != "failed":
                status = "partial"
        except Exception as exc:
            diagnostics.append(f"failed to check grouped PHA/RMF/ARF compatibility: {exc}")
            if status != "failed":
                status = "partial"

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
        energy_range_keV=fit_energy_range,
        diagnostics=diagnostics,
        status=status,
    )


def _joint_spectra(prepared: Iterable[PreparedSpectrum]) -> list[PreparedJointSpectrum]:
    """配对同观测 ready 的 FXTA/B / Pair ready FXTA/FXTB spectra per observation.

    只纳入身份完整的 FXT 谱；每个 obsid/module 重复时保留最后一个。
    返回有 A、B 两模块的联合谱列表，成员顺序固定为 FXTA、FXTB。
    Include ready FXT spectra with obsid and known module; last duplicate wins.
    Return complete pairs ordered FXTA then FXTB. No fitting or file validation.
    """
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
    """准备已扫描的能谱供拟合 / Group scanned spectra into prepared inputs.

    Parameters
    ----------
    data : Catalog or Manifest
        scan 得到的产品清单；须包含源谱、背景谱、ARF、RMF 配对。
        Scanned product catalog/manifest with source, background, ARF, RMF bundles.
    outdir : str or Path or None
        独立输出目录；默认扫描根目录的同级 <name>_jinwu_fit。
        Output root; defaults to sibling <scan-root-name>_jinwu_fit.
    group_min : int or None
        最小分组计数；None 使用仪器配置，最终回退到 1。
        Minimum grouping counts; None uses instrument configuration, then 1.
    overwrite : bool
        默认 False，已有分组谱会报 FileExistsError；True 允许工具覆盖。
        Default False rejects existing grouped PHA; True permits tool overwrite.

    Returns
    -------
    PreparedCatalog
        包含单谱、同观测 FXTA/B 联合谱及诊断。全部单谱 ready 且非空才
        总体 ready；存在 failed 则 failed；其余 partial。检查逐谱状态。
        Single/joint spectra and diagnostics. Overall ready requires nonempty,
        all-ready spectra; any failed gives failed, otherwise partial. Inspect
        individual statuses. This function writes products but does not fit them.
    """
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


def _ogip_type1(path: Path, row: int | None):
    """Copy one spectrum, retaining double precision and native grouping flags."""
    import numpy as np
    from astropy.io import fits

    with fits.open(path) as hdus:
        original = hdus["SPECTRUM"]
        data = original.data
        channels = data["CHANNEL"]
        is_type2 = (
            original.header.get("HDUCLAS4", "").upper() == "TYPE:II"
            or "SPEC_NUM" in original.columns.names
            or (len(channels) and np.ndim(channels[0]) != 0)
        )
        if is_type2:
            if not isinstance(row, Integral) or isinstance(row, bool) or not 1 <= row <= len(data):
                raise ValueError(f"Explicit 1-based spectrum row required for {path}")
            selected = data[int(row) - 1]
            selected_channels = np.atleast_1d(selected["CHANNEL"])
            columns = []
            for name in ("CHANNEL", "COUNTS", "RATE", "STAT_ERR", "SYS_ERR", "QUALITY", "GROUPING", "AREASCAL", "BACKSCAL"):
                if name not in original.columns.names:
                    continue
                values = np.asarray(selected[name])
                if values.ndim == 0 and name in {"SYS_ERR", "QUALITY", "GROUPING", "AREASCAL", "BACKSCAL"}:
                    continue
                values = np.atleast_1d(values)
                if values.shape != selected_channels.shape:
                    raise ValueError(f"Invalid {name} vector in {path}")
                fmt = "J" if name in {"CHANNEL", "QUALITY", "GROUPING"} else "D"
                columns.append(fits.Column(name=name, format=fmt, unit=original.columns[name].unit, array=values))
            spectrum = fits.BinTableHDU.from_columns(columns, name="SPECTRUM")
            # Non-structural metadata belongs to the selected spectrum.
            for key, value in original.header.items():
                if key and key not in spectrum.header and not key.startswith(
                    ("TFORM", "TTYPE", "TUNIT", "TDIM", "TLMIN", "TLMAX", "TSCAL", "TZERO", "TNULL", "TDISP",
                     "CHECKSUM", "DATASUM", "THEAP")
                ):
                    spectrum.header[key] = value
            for key in ("EXPOSURE", "BACKSCAL", "AREASCAL", "RESPFILE", "ANCRFILE", "BACKFILE", "POISSERR",
                        "TIME", "TIMEDEL", "QUALITY", "GROUPING", "SYS_ERR", "CORRFILE", "CORRSCAL"):
                if key in original.columns.names and np.ndim(selected[key]) == 0:
                    spectrum.header[key] = selected[key].item() if isinstance(selected[key], np.generic) else selected[key]
        else:
            if row is not None:
                raise ValueError(f"A row selector is only valid for type-II spectra: {path}")
            spectrum = original.copy()
        header = spectrum.header
        header["HDUCLASS"], header["HDUCLAS1"] = "OGIP", "SPECTRUM"
        header["HDUCLAS4"], header["HDUVERS"] = "TYPE:I", "1.2.1"
        if not len(spectrum.data) or ("COUNTS" in spectrum.columns.names) == ("RATE" in spectrum.columns.names):
            raise ValueError(f"Expected nonempty COUNTS or RATE spectrum in {path}")
        header["HDUCLAS3"] = "COUNT" if "COUNTS" in spectrum.columns.names else "RATE"
        header["DETCHANS"] = int(original.header.get("DETCHANS", len(spectrum.data)))
        original_channel_index = original.columns.names.index("CHANNEL") + 1
        channel_index = spectrum.columns.names.index("CHANNEL") + 1
        for limit, channel in (("TLMIN", spectrum.data["CHANNEL"].min()), ("TLMAX", spectrum.data["CHANNEL"].max())):
            header[f"{limit}{channel_index}"] = int(original.header.get(f"{limit}{original_channel_index}", channel))
        for key in ("EXPOSURE", "BACKSCAL", "AREASCAL"):
            default = 1.0 if key != "EXPOSURE" else np.nan
            if key in spectrum.columns.names:
                values = spectrum.data[key]
            else:
                values = header.get(key, hdus[0].header.get(key, default))
                header[key] = values
            if np.any(~np.isfinite(values)) or np.any(np.asarray(values) <= 0):
                raise ValueError(f"Invalid {key} in {path}")
        copies = [hdus[0].copy(), spectrum]
        copies.extend(h.copy() for h in hdus[1:] if h.name != "SPECTRUM")
        return fits.HDUList(copies)


def prepare_ogip_spectrum(
    source: str | Path, *, outdir: str | Path, instrument: str,
    response: str | Path, energy_range, background: str | Path | None = None,
    arf: str | Path | None = None, source_row: int | None = None,
    background_row: int | None = None, source_poisson: bool | None = None,
    background_poisson: bool | None = None, obsid: str | None = None,
    detector: str | None = None, source_id: str | None = None,
    spectrum_key: str | None = None,
) -> PreparedSpectrum:
    """Prepare existing OGIP products, including one explicitly selected type-II row.

    ``energy_range`` is a two-element Quantity convertible to keV. ``source_row``
    and ``background_row`` are 1-based for type-II files and None for type-I.
    Explicit response/background/ARF paths override header associations. An
    existing background association may not silently be discarded: supply its
    file explicitly. No regrouping, fitting, or count correction is performed.
    Original counts, errors, exposure, channel numbers and grouping are retained;
    type-II vectors are written as type-I doubles (the generic PHA writer uses
    single precision). ``outdir`` must be new or empty. Source and background
    Poisson flags may be explicitly supplied when an upstream producer omits
    them, but count data are checked before accepting ``True``.
    Set ``spectrum_key`` when observation and detector metadata would identify
    more than one spectrum with the same key.
    """
    import astropy.units as u
    import numpy as np
    from .io import read_pha, read_rmf, read_arf
    from .ogip import check_response_compatibility

    if not isinstance(energy_range, u.Quantity):
        raise TypeError("energy_range must be a Quantity")
    band = np.asarray(energy_range.to_value(u.keV))
    if band.shape != (2,) or not np.all(np.isfinite(band)) or not 0 < band[0] < band[1]:
        raise ValueError("energy_range must contain positive increasing limits")
    paths = [Path(p).expanduser().resolve(strict=True) if p is not None else None
             for p in (source, background, response, arf)]
    source, background, response, arf = paths
    if background is None and (background_row is not None or background_poisson is not None):
        raise ValueError("Background metadata requires a background file")
    output = Path(outdir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new or empty prepared directory: {output}")
    spectra = [_ogip_type1(source, source_row)]
    if background is not None:
        spectra.append(_ogip_type1(background, background_row))
    association = str(spectra[0]["SPECTRUM"].header.get("BACKFILE", "")).strip().lower()
    if background is None and association not in {"", "none"}:
        raise ValueError("Supply the associated background file explicitly")
    for hdus, poisson in zip(spectra, (source_poisson, background_poisson)):
        hdu = hdus["SPECTRUM"]
        correction = str(hdu.header.get("CORRFILE", "NONE")).strip().lower()
        if correction not in {"", "none"}:
            raise ValueError("Preparing a spectrum with a correction file is unsupported")
        hdu.header["CORRFILE"] = "NONE"
        hdu.header.setdefault("CORRSCAL", 0.0)
        if poisson is not None:
            hdu.header["POISSERR"] = bool(poisson)
        if "POISSERR" not in hdu.header:
            raise ValueError("POISSERR is missing; explicitly specify the source/background count convention")
        column = "COUNTS" if "COUNTS" in hdu.columns.names else "RATE"
        counts = np.asarray(hdu.data[column], dtype=float)
        if not np.all(np.isfinite(counts)):
            raise ValueError("Nonfinite spectrum data")
        if hdu.header["POISSERR"]:
            raw = counts if column == "COUNTS" else counts * hdu.header["EXPOSURE"]
            if np.any(raw < 0) or not np.allclose(raw, np.rint(raw), rtol=0, atol=1e-5):
                raise ValueError("Poisson spectrum must contain nonnegative integer counts")
        else:
            if "STAT_ERR" not in hdu.columns.names:
                raise ValueError("Non-Poisson spectrum requires STAT_ERR")
            error = np.asarray(hdu.data["STAT_ERR"])
            if np.any(~np.isfinite(error)) or np.any(error < 0):
                raise ValueError("Invalid STAT_ERR")
    if len(spectra) == 2 and not np.array_equal(spectra[0]["SPECTRUM"].data["CHANNEL"], spectra[1]["SPECTRUM"].data["CHANNEL"]):
        raise ValueError("Source/background channels do not match")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".ogip-", dir=output.parent) as temporary:
        staging = Path(temporary)
        for name, original in (("response.rsp", response), ("effective_area.arf", arf)):
            if original is not None:
                shutil.copy2(original, staging / name)
        for index, hdus in enumerate(spectra):
            header = hdus["SPECTRUM"].header
            header["BACKFILE"] = "background.pha" if index == 0 and background is not None else "NONE"
            header["RESPFILE"] = "response.rsp"
            header["ANCRFILE"] = "effective_area.arf" if arf is not None else "NONE"
            hdus.writeto(staging / ("source.pha" if index == 0 else "background.pha"), checksum=True)
            hdus.close()
        rmf = read_rmf(staging / "response.rsp")
        pha = read_pha(staging / "source.pha")
        compatibility = check_response_compatibility(
            pha, rmf,
            arf=read_arf(staging / "effective_area.arf") if arf is not None else None,
        )
        if not compatibility.ok:
            raise ValueError(f"Incompatible response: {compatibility.errors()}")
        if (rmf.channel is None or pha.det_chans != rmf.det_chans
                or len(rmf.channel) != rmf.det_chans or np.any(np.diff(rmf.channel) != 1)
                or np.any(np.diff(pha.channels) <= 0) or not np.all(np.isin(pha.channels, rmf.channel))):
            raise ValueError("Response EBOUNDS and spectrum channel definitions do not match")
        if (np.any(~np.isfinite(rmf.e_min)) or np.any(~np.isfinite(rmf.e_max))
                or np.any(rmf.e_min < 0) or np.any(rmf.e_max < rmf.e_min)):
            raise ValueError("Invalid response EBOUNDS energy limits")
        ignored = pha.channels[pha.quality > 0] if pha.quality is not None else ()
        inputs = (pha, read_pha(staging / "background.pha")) if background is not None else (pha,)
        for spectrum in inputs:
            if spectrum.det_chans != rmf.det_chans:
                raise ValueError("Response EBOUNDS and spectrum DETCHANS do not match")
            if spectrum.ebounds is None:
                continue
            channels, lower, upper = spectrum.ebounds
            indices = channels - rmf.channel[0]
            # GECAM pads excluded overflow channels with zero-width PHA bins.
            compare = ~((lower == upper) & np.isin(channels, ignored))
            if (np.any(indices < 0) or np.any(indices >= len(rmf.channel)) or np.any(np.diff(channels) <= 0)
                    or not np.all(np.isin(spectrum.channels, channels))
                    or not np.allclose(lower[compare], rmf.e_min[indices[compare]], rtol=1e-6, atol=1e-8)
                    or not np.allclose(upper[compare], rmf.e_max[indices[compare]], rtol=1e-6, atol=1e-8)):
                raise ValueError(f"Response EBOUNDS do not match {spectrum.path.name}")
        staging.rename(output)
    return PreparedSpectrum(
        instrument=instrument, obsid=obsid, module=None, detector=detector, source_id=source_id,
        source_pha=source, grouped_pha=output / "source.pha",
        background_pha=output / "background.pha" if background is not None else None,
        arf=output / "effective_area.arf" if arf is not None else None,
        rmf=output / "response.rsp", group_min=0, energy_range_keV=tuple(map(float, band)),
        diagnostics=["Existing grouping preserved; no regrouping", f"Selected source row: {source_row}; background row: {background_row}"],
        spectrum_key=spectrum_key,
    )
