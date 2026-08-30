"""XSELECT product extraction and lightweight pure-Python helpers.

The module exposes two deliberately separate paths:

* :func:`extract_products_with_xselect` drives the HEASoft ``xselect``
  executable non-interactively and writes standard spectra, light curves,
  filtered events, and images.
* :class:`XSelectSession` and the module-level ``extract_*`` functions provide
  lightweight pure-Python event filtering and product construction.

The external runner requires an initialized HEASoft environment with
``xselect`` on ``PATH``. It does not initialize or mutate HEASoft itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple, cast
import warnings

import numpy as np
from astropy.io import fits

from .base import OgipMeta
from .data import EventData, PhaData, LightcurveData
from .io import read_evt
from ..ftools import region as regionmod
from ..ftools import ftselect as exprmod
from . import gti as gtimod
from ..ftools import xselect_mdb
# ftools: local pure-Python replacements for common HEASOFT utilities
from .. import ftools

__all__ = [
    'XSelectExecutionError',
    'XSelectOutputPaths',
    'XSelectProductKind',
    'XSelectRole',
    'XSelectRunResult',
    'build_xselect_output_paths',
    'build_effective_ds9_region',
    'extract_products_with_xselect',
    'extract_spectrum_with_xselect',
    'select_events',
    'accumulate_spectrum_from_events',
    'write_pha',
    'XSelectSession',
]


XSelectProductKind = Literal['spectrum', 'lightcurve', 'events', 'image']
XSelectRole = Literal['src', 'bkg', 'all']

_DS9_SHAPE_LINE = re.compile(
    r'^(?P<prefix>\s*(?:[a-z][a-z0-9_]*\s*;\s*)?)'
    r'(?P<sign>[+-]?)\s*'
    r'(?P<body>(?:circle|annulus|polygon|ellipse|box|sector)\s*\([^)]*\))'
    r'(?P<tail>.*)$',
    re.IGNORECASE,
)

_XSELECT_PRODUCT_ORDER: tuple[XSelectProductKind, ...] = (
    'spectrum',
    'lightcurve',
    'image',
    'events',  # Keep events last: EXTRACT EVENTS changes XSELECT's workspace.
)
_XSELECT_PRODUCT_LAYOUT: dict[XSelectProductKind, tuple[str, str]] = {
    'spectrum': ('spec', '.pha'),
    'lightcurve': ('lc', '.lc'),
    'events': ('evt', '.evt'),
    'image': ('img', '.img'),
}
_XSELECT_PRODUCT_ALIASES: dict[str, XSelectProductKind | Literal['all']] = {
    'spectrum': 'spectrum',
    'spec': 'spectrum',
    'pha': 'spectrum',
    'lightcurve': 'lightcurve',
    'curve': 'lightcurve',
    'lc': 'lightcurve',
    'events': 'events',
    'event': 'events',
    'evt': 'events',
    'image': 'image',
    'img': 'image',
    'all': 'all',
}
_XSELECT_ROLE_ALIASES: dict[str, XSelectRole] = {
    'src': 'src',
    'source': 'src',
    'bkg': 'bkg',
    'background': 'bkg',
    'back': 'bkg',
    'all': 'all',
    'full': 'all',
}


@dataclass(frozen=True)
class XSelectOutputPaths:
    """Deterministic output paths for one XSELECT extraction.

    Product names follow this contract::

        <prefix>[_<label>]_<role>_<product-tag>.<extension>

    The standard product tags and extensions are ``spec.pha``, ``lc.lc``,
    ``evt.evt``, and ``img.img``. Control files use ``xselect.xco`` and
    ``xselect.log``. For example::

        grb050904_wt_seg003_src_spec.pha
        grb050904_wt_seg003_src_lc.lc
        grb050904_wt_seg003_src_evt.evt
        grb050904_wt_seg003_src_img.img
    """

    spectrum: Path | None
    lightcurve: Path | None
    events: Path | None
    image: Path | None
    command_file: Path
    log_file: Path

    def selected(self) -> dict[XSelectProductKind, Path]:
        """Return only requested science products in execution order."""
        out: dict[XSelectProductKind, Path] = {}
        for kind in _XSELECT_PRODUCT_ORDER:
            path = getattr(self, kind)
            if path is not None:
                out[kind] = path
        return out


@dataclass(frozen=True)
class XSelectRunResult:
    """Result and provenance for a completed external XSELECT run."""

    event_path: Path
    outputs: XSelectOutputPaths
    session_name: str
    returncode: int
    commands: tuple[str, ...]
    requested_lc_binsize_s: float | None
    effective_lc_binsize_s: float | None
    event_timedel_s: float | None
    image_binsize: int | None = None

    @property
    def spectrum(self) -> Path | None:
        return self.outputs.spectrum

    @property
    def lightcurve(self) -> Path | None:
        return self.outputs.lightcurve

    @property
    def events(self) -> Path | None:
        return self.outputs.events

    @property
    def image(self) -> Path | None:
        return self.outputs.image


class XSelectExecutionError(RuntimeError):
    """External XSELECT failed or did not create valid requested products."""

    def __init__(
        self,
        message: str,
        *,
        log_path: Path,
        returncode: int | None = None,
    ) -> None:
        super().__init__(f"{message}; see {log_path}")
        self.log_path = log_path
        self.returncode = returncode


def _safe_filename_token(value: str, *, field: str) -> str:
    token = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value).strip())
    token = re.sub(r'_+', '_', token).strip('._-')
    if not token:
        raise ValueError(f'{field} must contain at least one filename-safe character')
    return token


def _default_xselect_prefix(event: Path) -> str:
    """Strip FITS/event and compression suffixes while preserving mission tags."""
    name = event.name
    for suffix in ('.gz', '.bz2', '.xz'):
        if name.lower().endswith(suffix):
            name = name[:-len(suffix)]
            break
    for suffix in ('.evt', '.fits', '.fit'):
        if name.lower().endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name


def _normalize_xselect_role(role: str) -> XSelectRole:
    try:
        return _XSELECT_ROLE_ALIASES[str(role).strip().lower()]
    except KeyError as exc:
        allowed = ', '.join(sorted(_XSELECT_ROLE_ALIASES))
        raise ValueError(f'Unknown XSELECT role {role!r}; expected one of: {allowed}') from exc


def _normalize_xselect_products(
    products: str | Sequence[str],
) -> tuple[XSelectProductKind, ...]:
    values = [products] if isinstance(products, str) else list(products)
    if not values:
        raise ValueError('products must contain at least one product kind')

    requested: set[XSelectProductKind] = set()
    for value in values:
        key = str(value).strip().lower()
        try:
            normalized = _XSELECT_PRODUCT_ALIASES[key]
        except KeyError as exc:
            allowed = ', '.join(sorted(_XSELECT_PRODUCT_ALIASES))
            raise ValueError(
                f'Unknown XSELECT product {value!r}; expected one of: {allowed}'
            ) from exc
        if normalized == 'all':
            requested.update(_XSELECT_PRODUCT_ORDER)
        else:
            requested.add(normalized)
    return tuple(kind for kind in _XSELECT_PRODUCT_ORDER if kind in requested)


def build_xselect_output_paths(
    event_path: str | Path,
    output_dir: str | Path,
    *,
    products: str | Sequence[str] = ('spectrum',),
    prefix: str | None = None,
    label: str | None = None,
    role: str = 'all',
) -> XSelectOutputPaths:
    """Build product paths using the public XSELECT naming convention.

    ``prefix`` defaults to the complete event-file stem. It is intentionally
    not stripped of mission suffixes such as ``_cl`` because doing so can make
    products from distinct event files collide. ``label`` should describe a
    caller-defined selection such as ``wt_seg003``; floating time bounds are
    not encoded automatically.
    """
    event = Path(event_path)
    outdir = Path(output_dir)
    normalized_products = _normalize_xselect_products(products)
    normalized_role = _normalize_xselect_role(role)

    name_parts = [
        _safe_filename_token(
            prefix if prefix is not None else _default_xselect_prefix(event),
            field='prefix',
        )
    ]
    if label is not None:
        name_parts.append(_safe_filename_token(label, field='label'))
    name_parts.append(normalized_role)
    base = '_'.join(name_parts)

    selected = set(normalized_products)

    def product_path(kind: XSelectProductKind) -> Path | None:
        if kind not in selected:
            return None
        tag, extension = _XSELECT_PRODUCT_LAYOUT[kind]
        return outdir / f'{base}_{tag}{extension}'

    return XSelectOutputPaths(
        spectrum=product_path('spectrum'),
        lightcurve=product_path('lightcurve'),
        events=product_path('events'),
        image=product_path('image'),
        command_file=outdir / f'{base}_xselect.xco',
        log_file=outdir / f'{base}_xselect.log',
    )


def _xselect_session_name(command_path: Path) -> str:
    stem = re.sub(r'[^A-Za-z0-9]', '', command_path.stem)
    return f'jw{stem[-18:] or "session"}'


def _xselect_safe_lc_binsize(
    event_path: Path,
    requested_binsize_s: float | None,
) -> tuple[float | None, float | None]:
    """Avoid XSELECT's interactive prompt when binsize is below TIMEDEL."""
    timedel_values: list[float] = []
    try:
        with fits.open(event_path, memmap=False) as hdul:
            for hdu in hdul:
                value = hdu.header.get('TIMEDEL')
                try:
                    value_float = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value_float) and value_float > 0:
                    timedel_values.append(value_float)
    except OSError:
        pass

    event_timedel = max(timedel_values) if timedel_values else None
    if requested_binsize_s is None:
        return None, event_timedel

    requested = float(requested_binsize_s)
    if not np.isfinite(requested) or requested <= 0:
        raise ValueError('lc_binsize must be a finite positive number of seconds')
    if event_timedel is not None and requested < event_timedel:
        effective = event_timedel * 1.01
        warnings.warn(
            f'lc_binsize={requested:g} s is below event TIMEDEL={event_timedel:g} s; '
            f'using {effective:g} s to keep XSELECT non-interactive',
            RuntimeWarning,
            stacklevel=3,
        )
        return effective, event_timedel
    return requested, event_timedel


def _normalize_region_paths(
    region: str | Path | Sequence[str | Path] | None,
) -> tuple[Path, ...]:
    if region is None:
        return ()
    values: Sequence[str | Path]
    if isinstance(region, (str, Path)):
        values = (region,)
    else:
        values = region
    paths: list[Path] = []
    for value in values:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'XSELECT region file not found: {path}')
        if any(char.isspace() for char in str(path)):
            raise ValueError(f'XSELECT region paths cannot contain whitespace: {path}')
        paths.append(path)
    return tuple(paths)


def build_effective_ds9_region(
    include_region: str | Path,
    exclusion_regions: Sequence[str | Path],
    output_path: str | Path,
    *,
    default_frame: str | None = None,
) -> Path:
    """Write one DS9 region representing include minus all exclusions.

    Geometry lines in exclusion files are normalized to negative DS9 shapes,
    regardless of whether the input mask used ``polygon`` or ``-polygon``.
    Coordinate-system directives are retained.  ``default_frame`` is inserted
    before a frameless inclusion file, which is useful for mission products
    whose DS9 regions are implicitly in physical detector coordinates.  The
    source files are never modified.
    """

    include = Path(include_region).expanduser().resolve()
    exclusions = tuple(Path(path).expanduser().resolve() for path in exclusion_regions)
    output = Path(output_path).expanduser().resolve()
    if not include.is_file():
        raise FileNotFoundError(f'DS9 inclusion region not found: {include}')
    for path in exclusions:
        if not path.is_file():
            raise FileNotFoundError(f'DS9 exclusion region not found: {path}')
    if output == include or output in exclusions:
        raise ValueError('Effective DS9 output must not overwrite an input region')

    include_text = include.read_text(encoding='utf-8', errors='replace').rstrip()
    if default_frame is not None:
        normalized_frame = str(default_frame).strip().lower()
        valid_frames = {'physical', 'image', 'fk5', 'icrs', 'galactic', 'ecliptic', 'fk4'}
        if normalized_frame not in valid_frames:
            raise ValueError(f'Unsupported DS9 default frame: {default_frame}')
        coordinate_lines = {
            line.strip().lower().split(';', 1)[0].strip()
            for line in include_text.splitlines()
            if line.strip() and not line.lstrip().startswith('#')
        }
        if not coordinate_lines & valid_frames:
            include_text = f'{normalized_frame}\n{include_text}'
    include_shapes = [
        line for line in include_text.splitlines() if _DS9_SHAPE_LINE.match(line)
    ]
    if not include_shapes:
        raise ValueError(f'DS9 inclusion region contains no supported shapes: {include}')

    blocks = [
        '# Jinwu effective region: inclusion minus exclusion masks',
        f'# inclusion: {include}',
        include_text,
    ]
    exclusion_shape_count = 0
    for path in exclusions:
        normalized_lines = [f'# exclusion: {path}']
        for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
            match = _DS9_SHAPE_LINE.match(line)
            if match is None:
                normalized_lines.append(line)
                continue
            exclusion_shape_count += 1
            normalized_lines.append(
                f'{match.group("prefix")}-{match.group("body")}{match.group("tail")}'
            )
        blocks.append('\n'.join(normalized_lines).rstrip())
    if exclusions and exclusion_shape_count == 0:
        raise ValueError('DS9 exclusion region files contain no supported shapes')

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(blocks) + '\n', encoding='utf-8')
    return output


def _format_xselect_time_bound(value: float | str, time_format: str) -> str:
    if time_format == 'ut':
        text = str(value).strip()
        if not text:
            raise ValueError('UT time bounds cannot be empty')
        return text
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f'{time_format.upper()} time bounds must be finite')
    return f'{numeric:.9f}'


def _build_external_xselect_commands(
    *,
    event_path: Path,
    outputs: XSelectOutputPaths,
    session_name: str,
    products: tuple[XSelectProductKind, ...],
    time_range: tuple[float | str, float | str] | None,
    time_format: str,
    pha_range: tuple[int, int] | None,
    region_paths: tuple[Path, ...],
    lc_binsize_s: float | None,
    image_binsize: int | None,
) -> tuple[str, ...]:
    commands = [
        session_name,
        'read events',
        str(event_path.parent),
        event_path.name,
        'yes',
        # EP mission initialization may consume one additional response after
        # "Reset the mission?".  A blank is a no-op at the normal command
        # prompt and prevents the first real filter command from being lost.
        '',
    ]

    if time_range is not None:
        if len(time_range) != 2:
            raise ValueError('time_range must contain exactly (start, stop)')
        start = _format_xselect_time_bound(time_range[0], time_format)
        stop = _format_xselect_time_bound(time_range[1], time_format)
        if time_format != 'ut' and float(stop) <= float(start):
            raise ValueError('time_range stop must be greater than start')
        commands.extend([f'filter time {time_format}', f'{start}, {stop}', 'x'])

    if pha_range is not None:
        if len(pha_range) != 2:
            raise ValueError('pha_range must contain exactly (lower, upper)')
        lower, upper = int(pha_range[0]), int(pha_range[1])
        if lower < 0 or upper <= lower:
            raise ValueError('pha_range must satisfy 0 <= lower < upper')
        commands.append(f'filter pha_cutoff {lower} {upper}')

    if region_paths:
        joined = ' '.join(str(path) for path in region_paths)
        commands.append(f'filter region "{joined}"')

    selected_paths = outputs.selected()
    for kind in products:
        output = selected_paths[kind]
        if kind == 'lightcurve' and lc_binsize_s is not None:
            commands.append(f'set binsize {lc_binsize_s:.9g}')
        if kind == 'image' and image_binsize is not None:
            commands.append(f'set xybinsize {image_binsize}')
        extract_kind = 'curve' if kind == 'lightcurve' else kind
        if kind == 'spectrum':
            extract_kind = 'spectrum'
        commands.extend([f'extract {extract_kind}', f'save {extract_kind} {output.name}'])
        if kind == 'events':
            # SAVE EVENTS asks whether to read the filtered event list back in.
            commands.append('no')

    commands.extend(['exit', 'no', ''])
    return tuple(commands)


def _resolve_xselect_executable(
    executable: str | Path | None,
    environment: Mapping[str, str],
) -> str:
    requested = str(executable) if executable is not None else 'xselect'
    if os.sep in requested:
        path = Path(requested).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
        raise RuntimeError(f'XSELECT executable is not executable: {path}')
    found = shutil.which(requested, path=environment.get('PATH'))
    if found is None:
        raise RuntimeError(
            'xselect executable not found. Initialize HEASoft/HEADAS in the '
            'active environment or pass xselect_executable explicitly.'
        )
    return found


def _write_external_xselect_log(
    path: Path,
    *,
    executable: str,
    commands: tuple[str, ...],
    returncode: int | None,
    stdout: str,
    stderr: str,
    error: str | None = None,
) -> None:
    sections = [
        f'COMMAND: {executable} < {path.with_suffix(".xco").name}',
        f'RETURN_CODE: {returncode if returncode is not None else "NOT_STARTED"}',
    ]
    if error is not None:
        sections.extend(['ERROR:', error])
    sections.extend(
        [
            'XCO:',
            '\n'.join(commands),
            'STDOUT:',
            stdout,
            'STDERR:',
            stderr,
        ]
    )
    path.write_text('\n'.join(sections), encoding='utf-8')


def _validate_external_xselect_product(path: Path, kind: XSelectProductKind) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f'{kind} output is missing or empty: {path.name}')
    try:
        with fits.open(path, memmap=False) as hdul:
            if len(hdul) == 0:
                raise ValueError('FITS file contains no HDUs')
    except Exception as exc:
        raise ValueError(f'{kind} output is not a readable FITS file: {path.name}') from exc


def extract_products_with_xselect(
    event_path: str | Path,
    output_dir: str | Path,
    *,
    products: str | Sequence[str] = ('spectrum',),
    prefix: str | None = None,
    label: str | None = None,
    role: str = 'all',
    time_range: tuple[float | str, float | str] | None = None,
    time_format: Literal['scc', 'mjd', 'ut'] = 'scc',
    pha_range: tuple[int, int] | None = None,
    region: str | Path | Sequence[str | Path] | None = None,
    lc_binsize: float | None = None,
    image_binsize: int | None = None,
    overwrite: bool = False,
    xselect_executable: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> XSelectRunResult:
    """Extract one or more products by driving HEASoft XSELECT.

    Parameters
    ----------
    event_path:
        Input FITS event file. Its parent directory and basename are supplied
        separately to XSELECT, matching the interactive ``read events`` flow.
    output_dir:
        Directory for science products, the replayable ``.xco`` file, and log.
    products:
        Any of ``spectrum``, ``lightcurve``, ``events``, ``image`` or their
        aliases. ``all`` requests all four products.
    prefix, label, role:
        Naming components. The resulting contract is
        ``<prefix>[_<label>]_<role>_<tag>.<extension>``.
    time_range, time_format:
        Optional XSELECT time filter. Numeric SCC/MJD values are passed without
        conversion; UT values are passed as strings.
    pha_range:
        Inclusive XSELECT PHA/PI cutoff pair, emitted as
        ``filter pha_cutoff lower upper``.
    region:
        One or more DS9 region files in physical or WCS coordinates. XSELECT
        concatenates multiple files in the supplied order. Consequently an
        inclusion file followed by a file containing ``-shape(...)`` entries
        represents the inclusion region with those shapes excluded.
    lc_binsize:
        Requested light-curve bin size in seconds. If it is below an event-file
        ``TIMEDEL``, the effective value is raised to ``1.01 * TIMEDEL`` to
        avoid XSELECT's interactive continue prompt.
    image_binsize:
        Positive integer spatial rebinning factor passed through
        ``set xybinsize`` before image extraction.
    overwrite:
        Replace requested science products. Command and log files are always
        refreshed because they describe the current run.
    env:
        Environment overrides merged onto the current process environment.

    Notes
    -----
    XSELECT runs in an isolated temporary working directory so stale session
    files cannot contaminate another extraction. Products are moved into
    ``output_dir`` only after every requested output exists and opens as FITS.
    """
    event = Path(event_path).expanduser().resolve()
    if not event.is_file():
        raise FileNotFoundError(f'XSELECT event file not found: {event}')

    normalized_products = _normalize_xselect_products(products)
    if time_format not in {'scc', 'mjd', 'ut'}:
        raise ValueError("time_format must be one of 'scc', 'mjd', or 'ut'")
    if lc_binsize is not None and 'lightcurve' not in normalized_products:
        raise ValueError('lc_binsize requires the lightcurve product')
    if image_binsize is not None and 'image' not in normalized_products:
        raise ValueError('image_binsize requires the image product')
    if image_binsize is not None:
        if isinstance(image_binsize, bool) or int(image_binsize) != image_binsize or int(image_binsize) < 1:
            raise ValueError('image_binsize must be a positive integer')
        image_binsize = int(image_binsize)

    outdir = Path(output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = build_xselect_output_paths(
        event,
        outdir,
        products=normalized_products,
        prefix=prefix,
        label=label,
        role=role,
    )

    selected_paths = outputs.selected()
    existing = [path for path in selected_paths.values() if path.exists()]
    if existing and not overwrite:
        names = ', '.join(path.name for path in existing)
        raise FileExistsError(f'XSELECT output already exists: {names}')

    region_paths = _normalize_region_paths(region)
    requested_binsize = float(lc_binsize) if lc_binsize is not None else None
    effective_binsize, event_timedel = _xselect_safe_lc_binsize(
        event,
        requested_binsize if 'lightcurve' in normalized_products else None,
    )

    run_env = os.environ.copy()
    if env is not None:
        run_env.update({str(key): str(value) for key, value in env.items()})
    executable = _resolve_xselect_executable(xselect_executable, run_env)
    session_name = _xselect_session_name(outputs.command_file)
    commands = _build_external_xselect_commands(
        event_path=event,
        outputs=outputs,
        session_name=session_name,
        products=normalized_products,
        time_range=time_range,
        time_format=time_format,
        pha_range=pha_range,
        region_paths=region_paths,
        lc_binsize_s=effective_binsize,
        image_binsize=image_binsize,
    )
    xco_text = '\n'.join(commands)
    outputs.command_file.write_text(xco_text, encoding='utf-8')

    stdout = ''
    stderr = ''
    returncode: int | None = None
    with tempfile.TemporaryDirectory(prefix='.jinwu_xselect_', dir=outdir) as workdir_text:
        workdir = Path(workdir_text)
        try:
            proc = subprocess.run(
                [executable],
                input=xco_text,
                cwd=workdir,
                env=run_env,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
            stdout = proc.stdout or ''
            stderr = proc.stderr or ''
            returncode = int(proc.returncode)
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ''
            stderr = exc.stderr if isinstance(exc.stderr, str) else ''
            _write_external_xselect_log(
                outputs.log_file,
                executable=executable,
                commands=commands,
                returncode=None,
                stdout=stdout,
                stderr=stderr,
                error=f'XSELECT timed out after {timeout} seconds',
            )
            raise XSelectExecutionError(
                f'XSELECT timed out for {event.name}',
                log_path=outputs.log_file,
            ) from exc

        product_errors: list[str] = []
        transcript_markers = (
            'Command not found; type ? for a command listing',
            'You should not see this!',
        )
        for marker in transcript_markers:
            if marker in stdout:
                product_errors.append(f'XSELECT interaction failed: {marker}')
        for kind, final_path in selected_paths.items():
            temporary_path = workdir / final_path.name
            try:
                _validate_external_xselect_product(temporary_path, kind)
            except ValueError as exc:
                product_errors.append(str(exc))

        # If xselect produced valid output files, a non-zero exit code from a
        # late cleanup crash (e.g. HEASoft munmap_chunk SIGABRT) is harmless.
        if returncode != 0 and product_errors:
            product_errors.append(f'XSELECT returned {returncode}')

        _write_external_xselect_log(
            outputs.log_file,
            executable=executable,
            commands=commands,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            error='; '.join(product_errors) if product_errors else None,
        )
        if product_errors:
            raise XSelectExecutionError(
                f'XSELECT failed for {event.name}: {"; ".join(product_errors)}',
                log_path=outputs.log_file,
                returncode=returncode,
            )

        for final_path in selected_paths.values():
            if final_path.exists():
                final_path.unlink()
            shutil.move(str(workdir / final_path.name), str(final_path))

    return XSelectRunResult(
        event_path=event,
        outputs=outputs,
        session_name=session_name,
        returncode=returncode if returncode is not None else 0,
        commands=commands,
        requested_lc_binsize_s=requested_binsize,
        effective_lc_binsize_s=effective_binsize,
        event_timedel_s=event_timedel,
        image_binsize=image_binsize,
    )


def extract_spectrum_with_xselect(
    event_path: str | Path,
    output_dir: str | Path,
    **kwargs: Any,
) -> Path:
    """Convenience wrapper returning one XSELECT-extracted PHA path.

    All keyword arguments are forwarded to
    :func:`extract_products_with_xselect`, except that ``products`` is fixed to
    ``('spectrum',)``.
    """
    if 'products' in kwargs:
        raise TypeError('extract_spectrum_with_xselect fixes products to spectrum')
    result = extract_products_with_xselect(
        event_path,
        output_dir,
        products=('spectrum',),
        **kwargs,
    )
    if result.spectrum is None:  # pragma: no cover - protected by products above
        raise RuntimeError('XSELECT did not return a spectrum path')
    return result.spectrum


def _new_event_like(ev: EventData, **updates: Any) -> EventData:
    payload: dict[str, Any] = {
        'path': ev.path,
        'time': ev.time,
        'timezero': ev.timezero,
        'timezero_obj': ev.timezero_obj,
        'telescop': ev.telescop,
        'pi': ev.pi,
        'channel': ev.channel,
        'x': ev.x,
        'y': ev.y,
        'gti_start': ev.gti_start,
        'gti_stop': ev.gti_stop,
        'gti_start_obj': ev.gti_start_obj,
        'gti_stop_obj': ev.gti_stop_obj,
        'gti': ev.gti,
        'raw_columns': ev.raw_columns,
        'colmap': ev.colmap,
        'energy': ev.energy,
        'ebounds': ev.ebounds,
        'header': ev.header,
        'meta': ev.meta,
        'columns': ev.columns,
        'headers_dump': ev.headers_dump,
    }
    payload.update(updates)
    ev_cls = type(ev)
    return ev_cls(**payload)


class XSelectSession:
    """Interactive-style session holding original events and current filters.

    Usage:
      sess = XSelectSession('events.evt')
      sess.apply_time(tmin=0, tmax=100)
      sess.apply_region({'type':'circle','x':10,'y':10,'r':5})
      sess.clear_region()
      ev = sess.current  # EventData

    Implementation notes:
    - The session stores `original` (the unfiltered EventData) and `current`.
    - Filters are tracked in `state` and any change recomputes `current`
      by calling the module-level `select_events` against `original`.
    - If the session is created from an `EventData` without a file `path`,
      `original` will be that object; in that case the session cannot recover
      events removed earlier by other code unless the caller provided an
      unfiltered `EventData`.
    """

    def __init__(self, source: str | Path | EventData, *, keep_original: bool = True):
        # read original events when a path is provided; otherwise use provided EventData
        if isinstance(source, EventData):
            self.provided_ev = source
            self.path = getattr(source, 'path', None)
            if self.path is not None and keep_original:
                try:
                    self.original = read_evt(self.path)
                except Exception:
                    # fallback to provided object
                    self.original = source
            else:
                self.original = source
        else:
            self.path = Path(source)
            self.original = read_evt(self.path)

        # current view and state
        self.current = self.original
        self.state: dict = {
            'tmin': None, 'tmax': None,
            'pi_min': None, 'pi_max': None,
            'region': None, 'invert_region': False,
            'expr': None,
        }

    def recompute(self):
        """Recompute `current` from `original` using tracked state."""
        invert_region_val = self.state.get('invert_region', False)
        if invert_region_val is None:
            invert_region_val = False
        self.current = select_events(
            self.original,
            tmin=self.state.get('tmin'),
            tmax=self.state.get('tmax'),
            pi_min=self.state.get('pi_min'),
            pi_max=self.state.get('pi_max'),
            region=self.state.get('region'),
            invert_region=bool(invert_region_val),
            expr=self.state.get('expr')
        )

        # apply helpers
    def apply_time(self, *, tmin: Optional[float] = None, tmax: Optional[float] = None):
        """应用时间过滤；支持 float（相对时间秒）或 astropy/jinwu Time 对象。

        参数
        ----
        tmin/tmax : float | Time | TimeDelta, optional
            - float: 相对时间（秒），直接与事件 time 比较
            - Time 对象: 自动转换为相对时间（使用 original.timezero_obj）
            - TimeDelta: 取秒数作为相对时间
        """
        tmin_f = _coerce_time_bound(tmin, self.original)
        tmax_f = _coerce_time_bound(tmax, self.original)
        self.state['tmin'] = tmin_f
        self.state['tmax'] = tmax_f
        self.recompute()

    def apply_energy(self, *, pi_min: Optional[int] = None, pi_max: Optional[int] = None):
        self.state['pi_min'] = pi_min
        self.state['pi_max'] = pi_max
        self.recompute()

    def apply_region(self, region: dict | str | Path | None, *, invert: bool = False):
        self.state['region'] = region
        self.state['invert_region'] = bool(invert)
        self.recompute()

    def apply_expr(self, expr: Optional[str]):
        self.state['expr'] = expr
        self.recompute()

    # clear helpers
    def clear_region(self):
        self.state['region'] = None
        self.state['invert_region'] = False
        self.recompute()

    def clear_time(self):
        self.state['tmin'] = None
        self.state['tmax'] = None
        self.recompute()

    def clear_energy(self):
        self.state['pi_min'] = None
        self.state['pi_max'] = None
        self.recompute()

    def clear_all(self):
        self.state = {k: None for k in ('tmin', 'tmax', 'pi_min', 'pi_max', 'region', 'expr')}
        self.state['invert_region'] = False
        self.current = self.original

    # convenience accessors
    @property
    def meta(self):
        return getattr(self.current, 'meta', None)

    @property
    def header(self):
        return getattr(self.current, 'header', None)
    
    # Extract methods for convenience
    def extract_spectrum(self, **kwargs):
        """从当前过滤后的事件提取能谱。"""
        return extract_spectrum(self.current, **kwargs)
    
    def extract_curve(self, binsize: float, **kwargs):
        """从当前过滤后的事件提取光变曲线。"""
        return extract_curve(self.current, binsize=binsize, **kwargs)
    
    def extract_image(self, **kwargs):
        """从当前过滤后的事件提取图像。"""
        return extract_image(self.current, **kwargs)

    # Save helpers
    def save_current(self, outpath: str | Path, *, kind: str = 'evt', overwrite: bool = False, **kwargs) -> Path:
        """将当前过滤后的事件或派生产品保存到文件。

        参数
        - outpath: 输出文件路径
        - kind: 'evt' (保存事件表), 'lc' (提取并保存光变曲线),
                 'pha' (提取并保存能谱), 'img' (提取并保存图像)
        - overwrite: 是否覆盖已有文件
        - kwargs: 传递给底层提取/写出函数的其它参数

        等价于在 `self.current` 上调用 `save`，但通过 session 接口更直观。
        """
        cur = self.current
        # EventData.save 已经根据 kind 选择提取和写出逻辑
        return cur.save(outpath, kind=kind, overwrite=overwrite, **kwargs)


def write_curve(lc: LightcurveData, outpath: str | Path, *, overwrite: bool = False) -> Path:
    """Write a LightcurveData to a FITS file.

    Writes a binary table HDU containing at least `TIME` and `RATE` or
    `COUNTS` depending on `lc.is_rate`. Mirrors xselect's basic write-curve
    behavior: include primary header with TELESCOP/INSTRUME/TSTART/TSTOP/EXPOSURE.
    """
    outp = Path(outpath)
    if outp.exists() and not overwrite:
        raise FileExistsError(str(outp))

    from .data import LightcurveData
    if not isinstance(lc, LightcurveData):
        raise TypeError('lc must be LightcurveData')

    cols = []
    time = np.asarray(lc.time, dtype=float)
    cols.append(fits.Column(name='TIME', format='D', array=time))
    if lc.is_rate:
        val = np.asarray(lc.value, dtype=float)
        cols.append(fits.Column(name='RATE', format='E', array=val))
        if lc.error is not None:
            cols.append(fits.Column(name='ERROR', format='E', array=np.asarray(lc.error, dtype=float)))
    else:
        val = np.asarray(lc.value, dtype=float)
        cols.append(fits.Column(name='COUNTS', format='E', array=val))
        if lc.error is not None:
            cols.append(fits.Column(name='ERROR', format='E', array=np.asarray(lc.error, dtype=float)))

    hdu_tab = fits.BinTableHDU.from_columns(cols, name='LIGHTCURVE')
    hdr = hdu_tab.header
    hdr['EXTNAME'] = 'LIGHTCURVE'
    hdr['TELESCOP'] = lc.meta.telescop if (lc.meta is not None and getattr(lc.meta, 'telescop', None)) else hdr.get('TELESCOP', 'UNKNOWN')
    hdr['INSTRUME'] = lc.meta.instrume if (lc.meta is not None and getattr(lc.meta, 'instrume', None)) else hdr.get('INSTRUME', 'UNKNOWN')
    if lc.exposure is not None:
        try:
            hdr['EXPOSURE'] = float(lc.exposure)
        except Exception:
            pass
    if lc.meta is not None:
        if getattr(lc.meta, 'tstart', None) is not None:
            hdr['TSTART'] = float(lc.meta.tstart)
        if getattr(lc.meta, 'tstop', None) is not None:
            hdr['TSTOP'] = float(lc.meta.tstop)

    prih = fits.PrimaryHDU()
    try:
        if lc.meta is not None:
            if getattr(lc.meta, 'instrume', None):
                prih.header['INSTRUME'] = lc.meta.instrume
            if getattr(lc.meta, 'telescop', None):
                prih.header['TELESCOP'] = lc.meta.telescop
            if getattr(lc.meta, 'tstart', None) is not None:
                prih.header['TSTART'] = float(lc.meta.tstart)
            if getattr(lc.meta, 'tstop', None) is not None:
                prih.header['TSTOP'] = float(lc.meta.tstop)
    except Exception:
        pass

    hdul = fits.HDUList([prih, hdu_tab])
    hdul.writeto(outp, overwrite=overwrite)
    return outp


def write_image(img: np.ndarray, xedges: np.ndarray, yedges: np.ndarray, outpath: str | Path, *, overwrite: bool = False) -> Path:
    """Write a 2D image (numpy array) to a FITS Primary HDU.

    - `img` is a 2D array with shape (ny, nx) matching histogram2d output.
    - `xedges` and `yedges` are the bin edges used to build the image;
      helpful header keywords will be written (XMIN/XMAX/YMIN/YMAX/NX/NY).
    """
    outp = Path(outpath)
    if outp.exists() and not overwrite:
        raise FileExistsError(str(outp))

    data = np.asarray(img)
    # Primary image HDU
    prih = fits.PrimaryHDU(data=data.astype(np.float32))
    hdr = prih.header
    hdr['BUNIT'] = 'COUNT'
    try:
        hdr['NXPIX'] = int(data.shape[1])
        hdr['NYPIX'] = int(data.shape[0])
    except Exception:
        pass
    try:
        hdr['XMIN'] = float(xedges[0])
        hdr['XMAX'] = float(xedges[-1])
        hdr['YMIN'] = float(yedges[0])
        hdr['YMAX'] = float(yedges[-1])
    except Exception:
        pass

    hdul = fits.HDUList([prih])
    hdul.writeto(outp, overwrite=overwrite)
    return outp


# --- xselect.mdb integration (lazy load + optional persistent cache) ---
_MDB_TREE = None


def _get_mdb_tree(use_cache: bool = True):
    """Lazy-load parsed xselect.mdb into module cache. Returns the parsed tree."""
    global _MDB_TREE
    if _MDB_TREE is not None:
        return _MDB_TREE
    # guess path relative to package
    try:
        base = Path(__file__).resolve().parents[1]
        mdb_path = base / 'data' / 'xselect.mdb'
        cache_path = xselect_mdb.default_cache_path()
        if mdb_path.exists():
            _MDB_TREE = xselect_mdb.load_mdb(str(mdb_path), use_cache=use_cache, cache_path=cache_path)
            return _MDB_TREE
    except Exception:
        pass
    # fallback: try to locate via package import path
    try:
        import importlib.resources as _ir
        import io
        # attempt to read resource if installed as package
        with _ir.open_text('jinwu.data', 'xselect.mdb') as f:
            lines = f.readlines()
        _MDB_TREE = xselect_mdb._parse_lines(lines)
        return _MDB_TREE
    except Exception:
        _MDB_TREE = {}
        return _MDB_TREE


def _infer_adjustgti_timepixr_and_frame(ev: EventData, tree) -> tuple[bool, Optional[float], Optional[float]]:
    """Wrapper: call into `xselect_mdb.infer_adjustgti_timepixr_and_frame` for inference."""
    return xselect_mdb.infer_adjustgti_timepixr_and_frame(tree, getattr(ev, 'header', None), getattr(ev, 'meta', None))


def select_events(path: str | Path | EventData, *, tmin: Optional[float] = None, tmax: Optional[float] = None,
                  pi_min: Optional[int] = None, pi_max: Optional[int] = None,
                  region: Optional[dict | str | Path] = None, invert_region: bool = False,
                  expr: Optional[str] = None) -> EventData:
    """读取并按简单条件筛选事件表。

    参数
    - path: 事件 FITS 文件路径
    - tmin/tmax: 时间区间（闭区间）
    - pi_min/pi_max: PI/CHANNEL 范围（闭区间），若文件中没有 PI 列会尝试 CHANNEL

    返回
    - EventData（筛选后的副本，不会修改源文件）
    """
    ev = path if isinstance(path, EventData) else read_evt(path)

    # apply time, energy, region in xselect-like order: TIME -> ENERGY -> REGION
    ev2 = ev
    if (tmin is not None) or (tmax is not None):
        ev2 = filter_time(ev2, tmin=tmin, tmax=tmax)
    if (pi_min is not None) or (pi_max is not None):
        ev2 = filter_energy(ev2, pi_min=pi_min, pi_max=pi_max)

    # optional ftselect-like expression filtering (applies after energy filter)
    if expr is not None and str(expr).strip() != '':
        mask = exprmod.expression_to_mask(ev2, expr)
        t = np.asarray(ev2.time, dtype=float)
        new_time = t[mask]
        new_pi = None if ev2.pi is None else np.asarray(ev2.pi, dtype=int)[mask]
        new_ch = None if ev2.channel is None else np.asarray(ev2.channel, dtype=int)[mask]
        new_x = None if ev2.x is None else np.asarray(ev2.x)[mask]
        new_y = None if ev2.y is None else np.asarray(ev2.y)[mask]
        new_energy = None if ev2.energy is None else np.asarray(ev2.energy)[mask]
        ev2 = _new_event_like(
            ev2,
            time=new_time,
            pi=new_pi,
            channel=new_ch,
            x=new_x,
            y=new_y,
            energy=new_energy,
            raw_columns=None,
        )

    if region is not None:
        # region may be a dict (inline), a path to one or more region files, or a list of shapes
        shapes = []
        if isinstance(region, dict):
            shapes = [region]
        elif isinstance(region, (str, Path)):
            # Allow multiple files separated by whitespace (xselect permits multiple region files)
            s = str(region)
            parts = s.split()
            files = [Path(p) for p in parts]
            for f in files:
                if not f.exists():
                    raise FileNotFoundError(f"Region file not found: {f}")
                shapes.extend(regionmod.parse_ds9_region_file(f))
        else:
            # assume it's a list-like of shapes or file paths
            try:
                for item in region:
                    if isinstance(item, dict):
                        shapes.append(item)
                    elif isinstance(item, (str, Path)):
                        p = Path(item)
                        if not p.exists():
                            raise FileNotFoundError(f"Region file not found: {p}")
                        shapes.extend(regionmod.parse_ds9_region_file(p))
                    else:
                        shapes.append(item)
            except Exception:
                shapes = [region]

        # Determine preferred X/Y columns based on mission DB and file metadata.
        try:
            tree = _get_mdb_tree()
            mission = None
            instr = None
            mode = None
            if ev2.meta is not None:
                mission = getattr(ev2.meta, 'telescop', None)
                instr = getattr(ev2.meta, 'instrume', None)
            if mission is None and ev2.header is not None:
                mission = ev2.header.get('TELESCOP')
            if instr is None and ev2.header is not None:
                instr = ev2.header.get('INSTRUME')
            defaults = xselect_mdb.get_defaults(tree, str(mission).upper() if mission is not None else '', str(instr).upper() if instr is not None else None, mode)
        except Exception:
            defaults = {}

        # If defaults provide explicit column names, prefer them when parsing/ applying regions.
        # We'll attach preferred column names into shapes if they are simple dict regions lacking explicit column keys.
        col_x = defaults.get('x') or defaults.get('rawx') or defaults.get('detx')
        col_y = defaults.get('y') or defaults.get('rawy') or defaults.get('dety')
        # Normalize None
        col_x = None if col_x in (None, '') else col_x
        col_y = None if col_y in (None, '') else col_y

        # If region shapes don't include an explicit 'coord' or reference frame, set preferred columns
        for shp in shapes:
            if isinstance(shp, dict):
                if ('coord' not in shp) and (col_x is not None and col_y is not None):
                    # annotate so region apply can interpret which columns to use when needed
                    shp.setdefault('xcol', col_x)
                    shp.setdefault('ycol', col_y)

        out = regionmod.apply_region_mask_to_events(ev2, cast(list, shapes), invert=invert_region)
        return out
    return ev2


def _coerce_time_bound(val: Optional[float | Any], ev: Optional[EventData] = None) -> Optional[float]:
    """将 tmin/tmax 归一化为 float 秒（相对时间），支持 astropy/jinwu 的 Time/TimeDelta。

    - 若为 None，直接返回 None；
    - 若为 float/int，直接视为相对时间秒数；
    - 若为 Time 对象且提供了 ev.timezero_obj，则转换为相对时间：
      相对时间 = (Time - timezero_obj).sec
    - 若为 TimeDelta，则直接取 .sec
    """
    if val is None:
        return None
    
    # 检查是否为 Time 对象（非 TimeDelta）
    # 优先检查是否有 timezero_obj 可用于转换
    if ev is not None and getattr(ev, 'timezero_obj', None) is not None:
        timezero_obj = cast(Any, ev.timezero_obj)
        # 尝试判断 val 是否为 Time 对象（有 jd 属性但非 TimeDelta）
        has_jd = hasattr(val, 'jd')
        is_timedelta = hasattr(val, 'to_value') and not has_jd
        
        if has_jd and not is_timedelta:
            # val 是 Time 对象，转换为相对时间
            try:
                # 相对时间 = (val - timezero_obj) 的秒数
                diff = cast(Any, val) - timezero_obj
                if hasattr(diff, 'sec'):
                    return float(diff.sec)
                elif hasattr(diff, 'to_value'):
                    return float(cast(Any, diff).to_value('sec'))
            except Exception:
                pass
    
    # astropy.time.TimeDelta 或其他有 to_value 的对象
    try:
        if hasattr(val, 'to_value'):
            try:
                return float(cast(Any, val).to_value('sec'))
            except Exception:
                pass
        # 某些 Time-like 对象可能有 .sec 属性
        sec_attr = getattr(val, 'sec', None)
        if sec_attr is not None and not isinstance(sec_attr, (list, tuple, np.ndarray)):
            try:
                return float(sec_attr)
            except Exception:
                pass
    except Exception:
        pass
    
    # 回退：尝试当作标量秒数（相对时间）
    try:
        return float(val)  # type: ignore[arg-type]
    except Exception as exc:
        raise TypeError("tmin/tmax must be float seconds (relative time) or Time-like object") from exc


def filter_time(ev: EventData, *, tmin: Optional[float] = None, tmax: Optional[float] = None) -> EventData:
    """按时间范围过滤 EventData（闭区间）。

    参数
    ----
    ev : EventData
        事件数据（time 为相对时间）
    tmin/tmax : float | Time | TimeDelta, optional
        时间范围限制：
        - float: 相对时间（秒），直接与 ev.time 比较
        - Time 对象: 自动转换为相对时间（需要 ev.timezero_obj）
        - TimeDelta: 取秒数作为相对时间
    
    返回
    ----
    EventData
        过滤后的副本（不修改原数据）
    
    示例
    ----
    >>> # 使用相对时间（秒）
    >>> ev_filtered = filter_time(ev, tmin=0, tmax=1000)
    >>> # 使用 Time 对象
    >>> from astropy.time import Time
    >>> t0 = Time('2024-01-01T00:00:00', format='isot')
    >>> t1 = Time('2024-01-01T00:10:00', format='isot')
    >>> ev_filtered = filter_time(ev, tmin=t0, tmax=t1)
    """
    t = np.asarray(ev.time, dtype=float)
    if t.size == 0:
        return ev
    mask = np.ones(t.size, dtype=bool)
    
    # 转换时间边界（支持 Time 对象）
    tmin_f = _coerce_time_bound(tmin, ev)
    tmax_f = _coerce_time_bound(tmax, ev)
    
    if tmin_f is not None:
        mask &= (t >= tmin_f)
    if tmax_f is not None:
        mask &= (t <= tmax_f)
    
    new_time = t[mask]
    new_pi = None if ev.pi is None else np.asarray(ev.pi)[mask]
    new_ch = None if ev.channel is None else np.asarray(ev.channel)[mask]
    new_x = None if ev.x is None else np.asarray(ev.x)[mask]
    new_y = None if ev.y is None else np.asarray(ev.y)[mask]
    new_energy = None if ev.energy is None else np.asarray(ev.energy)[mask]
    
    return _new_event_like(
        ev,
        time=new_time,
        pi=new_pi,
        channel=new_ch,
        x=new_x,
        y=new_y,
        energy=new_energy,
        raw_columns=None,
    )


def filter_energy(ev: EventData, *, pi_min: Optional[int] = None, pi_max: Optional[int] = None) -> EventData:
    """Filter EventData by PI/CHANNEL range (inclusive). Prefer PI column if present."""
    # prefer PI then CHANNEL
    if ev.pi is not None:
        arr = np.asarray(ev.pi, dtype=int)
    elif ev.channel is not None:
        arr = np.asarray(ev.channel, dtype=int)
    else:
        # nothing to filter
        return ev
    if arr.size == 0:
        return ev
    mask = np.ones(arr.size, dtype=bool)
    if pi_min is not None:
        mask &= (arr >= int(pi_min))
    if pi_max is not None:
        mask &= (arr <= int(pi_max))
    new_time = np.asarray(ev.time, dtype=float)[mask]
    new_pi = None if ev.pi is None else np.asarray(ev.pi, dtype=int)[mask]
    new_ch = None if ev.channel is None else np.asarray(ev.channel, dtype=int)[mask]
    new_x = None if ev.x is None else np.asarray(ev.x)[mask]
    new_y = None if ev.y is None else np.asarray(ev.y)[mask]
    new_energy = None if ev.energy is None else np.asarray(ev.energy)[mask]
    return _new_event_like(
        ev,
        time=new_time,
        pi=new_pi,
        channel=new_ch,
        x=new_x,
        y=new_y,
        energy=new_energy,
        raw_columns=None,
    )


def merge_gti(gti_start: Optional[np.ndarray], gti_stop: Optional[np.ndarray], *, tol: float = 1e-9) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """兼容包装：委托到 `gtimod.merge_gti`。"""
    return gtimod.merge_gti(gti_start, gti_stop, tol=tol)


def trim_events_to_gti(ev: EventData, *, tol: float = 1e-9) -> EventData:
    """根据 EventData 中的 GTI 裁剪事件。如果没有 GTI，返回原始对象副本。

    返回的新 `EventData` 会把 `gti_start/gti_stop` 规范化为合并后的区间。
    """
    if ev.gti_start is None or ev.gti_stop is None:
        return ev
    ms, me = merge_gti(ev.gti_start, ev.gti_stop, tol=tol)
    if ms is None or me is None:
        return ev

    t = np.asarray(ev.time, dtype=float)
    
    # 重新计算合并后 GTI 的 Time 对象
    gti_start_obj_new = None
    gti_stop_obj_new = None
    if ev.timezero_obj is not None and ms is not None and me is not None:
        try:
            from astropy.time import TimeDelta
            gti_start_obj_new = ev.timezero_obj + TimeDelta(ms, format='sec')
            gti_stop_obj_new = ev.timezero_obj + TimeDelta(me, format='sec')
        except Exception:
            pass
    
    if t.size == 0:
        # empty events, but update GTI fields
        return _new_event_like(
            ev,
            time=t,
            gti_start=ms,
            gti_stop=me,
            gti_start_obj=gti_start_obj_new,
            gti_stop_obj=gti_stop_obj_new,
            gti=[(float(s), float(e)) for s, e in zip(ms, me)],
            raw_columns=None,
        )

    mask = np.zeros(t.size, dtype=bool)
    for s, e in zip(ms, me):
        mask |= (t >= float(s)) & (t <= float(e))

    new_time = t[mask]
    new_pi = None if ev.pi is None else np.asarray(ev.pi)[mask]
    new_ch = None if ev.channel is None else np.asarray(ev.channel)[mask]
    new_x = None if ev.x is None else np.asarray(ev.x)[mask]
    new_y = None if ev.y is None else np.asarray(ev.y)[mask]
    new_energy = None if ev.energy is None else np.asarray(ev.energy)[mask]

    return _new_event_like(
        ev,
        time=new_time,
        pi=new_pi,
        channel=new_ch,
        x=new_x,
        y=new_y,
        gti_start=ms,
        gti_stop=me,
        gti_start_obj=gti_start_obj_new,
        gti_stop_obj=gti_stop_obj_new,
        gti=[(float(s), float(e)) for s, e in zip(ms, me)],
        energy=new_energy,
        raw_columns=None,
    )


def _read_column_from_evt(path: str | Path, colname: str):
    """从事件 FITS 文件中读取指定列，若不存在返回 None。"""
    p = Path(path)
    try:
        with fits.open(p) as h:
            hevt = None
            for ext in h:
                d = getattr(ext, 'data', None)
                if d is None:
                    continue
                cols = getattr(d, 'columns', None)
                if cols is not None and 'TIME' in cols.names:
                    hevt = ext
                    break
            if hevt is None:
                return None
            d = cast(Any, hevt).data
            if colname in d.columns.names:
                return np.asarray(d[colname])
    except Exception:
        return None
    return None

def clear_region(ev_or_path: EventData | str | Path, *, tmin: Optional[float] = None, tmax: Optional[float] = None,
                 pi_min: Optional[int] = None, pi_max: Optional[int] = None) -> EventData:
    """Remove any region selection by re-reading the original event file and
    re-applying only time/energy filters.

    Notes:
    - If `ev_or_path` is an `EventData` and has a `path`, the full original
      events will be re-read from that file. If `ev_or_path` is a path string
      or `Path`, that file will be used.
    - If `tmin`/`tmax` or `pi_min`/`pi_max` are not provided and an
      `EventData` was passed, the function will infer them from the current
      `EventData` (i.e. keep the current time/energy constraints).
    - If no file path can be determined (in-memory `EventData` without
      `path`), a `ValueError` is raised because we cannot recover events
      excluded by the region filter.
    """
    # determine source path and optional current EventData
    ev = ev_or_path if isinstance(ev_or_path, EventData) else None
    if ev is not None:
        path = ev.path
    elif isinstance(ev_or_path, (str, Path)):
        path = Path(ev_or_path)
    else:
        path = None
    if path is None:
        raise ValueError('clear_region requires a file path (pass EventData with .path or a file path)')

    # infer filters from provided EventData when arguments absent
    if ev is not None:
        if tmin is None and getattr(ev, 'time', None) is not None and len(ev.time) > 0:
            tmin = float(np.min(ev.time))
            tmax = float(np.max(ev.time)) if tmax is None else tmax
        if pi_min is None and getattr(ev, 'pi', None) is not None:
            arr = np.asarray(ev.pi, dtype=int)
            if arr.size > 0:
                pi_min = int(arr.min())
                pi_max = int(arr.max()) if pi_max is None else pi_max

    # re-read full event file and apply time/energy filters
    full = read_evt(path)
    return select_events(full, tmin=tmin, tmax=tmax, pi_min=pi_min, pi_max=pi_max)


def clear_time(ev_or_path: EventData | str | Path, *, region: Optional[dict | str | Path] = None,
               pi_min: Optional[int] = None, pi_max: Optional[int] = None) -> EventData:
    """Remove any time selection by re-reading the original event file and
    re-applying only region/energy filters.

    If `region` is not provided, the region cannot be inferred from a
    filtered `EventData` — in that case the returned EventData will have no
    region applied (only energy filters if provided or inferred).
    """
    ev = ev_or_path if isinstance(ev_or_path, EventData) else None
    if ev is not None:
        path = ev.path
    elif isinstance(ev_or_path, (str, Path)):
        path = Path(ev_or_path)
    else:
        path = None
    if path is None:
        raise ValueError('clear_time requires a file path (pass EventData with .path or a file path)')

    # infer energy bounds from EventData if not supplied
    if ev is not None and pi_min is None and getattr(ev, 'pi', None) is not None:
        arr = np.asarray(ev.pi, dtype=int)
        if arr.size > 0:
            pi_min = int(arr.min())
            pi_max = int(arr.max()) if pi_max is None else pi_max

    full = read_evt(path)
    return select_events(full, region=region, pi_min=pi_min, pi_max=pi_max)


def clear_energy(ev_or_path: EventData | str | Path, *, region: Optional[dict | str | Path] = None,
                 tmin: Optional[float] = None, tmax: Optional[float] = None) -> EventData:
    """Remove any energy selection by re-reading the original event file and
    re-applying only region/time filters.
    """
    ev = ev_or_path if isinstance(ev_or_path, EventData) else None
    if ev is not None:
        path = ev.path
    elif isinstance(ev_or_path, (str, Path)):
        path = Path(ev_or_path)
    else:
        path = None
    if path is None:
        raise ValueError('clear_energy requires a file path (pass EventData with .path or a file path)')

    # infer time window from EventData if not supplied
    if ev is not None and (tmin is None and getattr(ev, 'time', None) is not None and len(ev.time) > 0):
        tmin = float(np.min(ev.time))
        tmax = float(np.max(ev.time)) if tmax is None else tmax

    full = read_evt(path)
    return select_events(full, tmin=tmin, tmax=tmax, region=region)


def clear_all(ev_or_path: EventData | str | Path) -> EventData:
    """Return the complete unfiltered EventData by re-reading the source file.

    If `ev_or_path` is an `EventData` with a `path` attribute, that file will
    be re-read; if a path is provided it will be used directly. If neither is
    available, a `ValueError` is raised because the original full event set
    cannot be reconstructed from an in-memory, filtered `EventData`.
    """
    ev = ev_or_path if isinstance(ev_or_path, EventData) else None
    if ev is not None:
        path = ev.path
    elif isinstance(ev_or_path, (str, Path)):
        path = Path(ev_or_path)
    else:
        path = None
    if path is None:
        raise ValueError('clear_all requires a file path (pass EventData with .path or a file path)')
    return read_evt(path)


def filter_region(ev_or_path: EventData | str | Path, region: dict) -> EventData:
    """按给定 region（dict）过滤事件并返回新的 EventData。

    region 支持基本形式：
      - {'type':'circle', 'x': X, 'y': Y, 'r': R}
      - {'type':'annulus', 'x': X, 'y': Y, 'r_in': R1, 'r_out': R2}
      - {'type':'box', 'x': Xc, 'y': Yc, 'width': W, 'height': H}

    如果传入的是文件路径，会从 FITS 中读取必要列（默认尝试 'X'/'Y', 'RAWX'/'RAWY', 'DETX'/'DETY'）。
    """
    # normalize to EventData
    ev = None
    if isinstance(ev_or_path, (str, Path)):
        ev = read_evt(ev_or_path)
    else:
        ev = ev_or_path

    # discover X/Y columns; consult xselect.mdb defaults for instrument-specific mappings
    xs = None
    ys = None
    try:
        tree = _get_mdb_tree()
        mission = None
        instr = None
        mode = None
        if isinstance(ev_or_path, (str, Path)):
            tmp_ev = read_evt(ev_or_path)
            hdr = tmp_ev.header
        else:
            hdr = getattr(ev_or_path, 'header', None)
        if hdr is not None:
            mission = hdr.get('TELESCOP')
            instr = hdr.get('INSTRUME')
        defaults = xselect_mdb.get_defaults(tree, str(mission).upper() if mission is not None else '', str(instr).upper() if instr is not None else None, mode)
    except Exception:
        defaults = {}

    # order of preference: explicit mdb mappings, then common column names
    x_candidates = []
    y_candidates = []
    if defaults.get('x'):
        x_candidates.append(str(defaults.get('x')))
    if defaults.get('rawx'):
        x_candidates.append(str(defaults.get('rawx')))
    if defaults.get('detx'):
        x_candidates.append(str(defaults.get('detx')))
    x_candidates.extend(['X', 'X_IMAGE', 'RAWX', 'DETX', 'DET_X', 'XDET'])

    if defaults.get('y'):
        y_candidates.append(str(defaults.get('y')))
    if defaults.get('rawy'):
        y_candidates.append(str(defaults.get('rawy')))
    if defaults.get('dety'):
        y_candidates.append(str(defaults.get('dety')))
    y_candidates.extend(['Y', 'Y_IMAGE', 'RAWY', 'DETY', 'DET_Y', 'YDET'])

    xcol_used = None
    ycol_used = None
    for xc in x_candidates:
        try:
            xs = _read_column_from_evt(ev.path, xc)
        except Exception:
            xs = None
        if xs is not None:
            xcol_used = xc
            break
    for yc in y_candidates:
        try:
            ys = _read_column_from_evt(ev.path, yc)
        except Exception:
            ys = None
        if ys is not None:
            ycol_used = yc
            break

    # If EventData has attributes (unlikely), prefer them
    if hasattr(ev, 'x') and hasattr(ev, 'y'):
        xs = np.asarray(getattr(ev, 'x'))
        ys = np.asarray(getattr(ev, 'y'))

    if xs is None or ys is None:
        raise ValueError('Event file lacks X/Y columns required for region filtering')

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    typ = region.get('type', 'circle').lower()
    mask = np.zeros(xs.size, dtype=bool)
    if typ == 'circle':
        cx = float(region.get('x', 0.0))
        cy = float(region.get('y', 0.0))
        r = float(region['r'])
        dx = xs - cx
        dy = ys - cy
        mask = (dx * dx + dy * dy) <= (r * r)
    elif typ == 'annulus':
        cx = float(region.get('x', 0.0))
        cy = float(region.get('y', 0.0))
        rin = float(region['r_in'])
        rout = float(region['r_out'])
        dx = xs - cx
        dy = ys - cy
        rr = dx * dx + dy * dy
        mask = (rr >= rin * rin) & (rr <= rout * rout)
    elif typ == 'box':
        cx = float(region.get('x', 0.0))
        cy = float(region.get('y', 0.0))
        w = float(region.get('width', region.get('w', 1.0)))
        h = float(region.get('height', region.get('h', 1.0)))
        mask = (np.abs(xs - cx) <= (w / 2.0)) & (np.abs(ys - cy) <= (h / 2.0))
    else:
        raise ValueError(f'Unsupported region type: {typ}')

    # apply mask to build new EventData
    t = np.asarray(ev.time, dtype=float)
    new_time = t[mask]
    new_pi = None if ev.pi is None else np.asarray(ev.pi)[mask]
    new_ch = None if ev.channel is None else np.asarray(ev.channel)[mask]
    new_x = None if ev.x is None else np.asarray(ev.x)[mask]
    new_y = None if ev.y is None else np.asarray(ev.y)[mask]
    new_energy = None if ev.energy is None else np.asarray(ev.energy)[mask]

    return _new_event_like(
        ev,
        time=new_time,
        pi=new_pi,
        channel=new_ch,
        x=new_x,
        y=new_y,
        energy=new_energy,
        raw_columns=None,
    )


def extract_spectrum(ev_or_path: EventData | str | Path, *, region: dict | None = None,
                     tmin: Optional[float] = None, tmax: Optional[float] = None,
                     channel_col: str = 'pi', ch_min: Optional[int] = None, ch_max: Optional[int] = None,
                     nbins: Optional[int] = None) -> PhaData:
    """从事件（或事件文件）提取 PHA。支持时间与区域筛选。
    返回 PhaData。
    """
    # Delegate to jinwu.ftools.fextract.extract which implements the extraction
    # behavior in pure Python and mirrors the common fextract options.
    return ftools.fextract.extract(ev_or_path, region=region, tmin=tmin, tmax=tmax,
                                    ch_min=ch_min, ch_max=ch_max, nbins=nbins, channel_col=channel_col)


def _exposure_per_bins(ms: np.ndarray, me: np.ndarray, bins: np.ndarray) -> np.ndarray:
    return gtimod.exposure_per_bins(ms, me, bins)


def extract_curve(ev_or_path: EventData | str | Path, *, binsize: float, tmin: Optional[float] = None,
                  tmax: Optional[float] = None) -> 'LightcurveData':
    """从事件生成光变曲线（counts per bin + per-bin exposure），返回 LightcurveData。

    binsize: bin 宽度（秒）
    """
    if isinstance(ev_or_path, (str, Path)):
        ev = read_evt(ev_or_path)
    else:
        ev = ev_or_path

    # time filter
    if tmin is not None or tmax is not None:
        ev = select_events(ev.path if isinstance(ev_or_path, (str, Path)) else ev.path, tmin=tmin, tmax=tmax)

    t = np.asarray(ev.time, dtype=float)
    if t.size == 0:
        from .data import LightcurveData
        # 空事件集时返回空的 LightcurveData，保持字段兼容
        empty = np.array([])
        return LightcurveData(
            path=ev.path,
            time=empty,
            value=empty,
            error=None,
            dt=binsize,
            # 时间字段
            timezero=getattr(ev, 'timezero', 0.0),
            timezero_obj=getattr(ev, 'timezero_obj', None),
            bin_lo=empty,
            bin_hi=empty,
            tstart=None,
            tseg=None,
            # 数据字段
            is_rate=False,
            counts=empty,
            rate=None,
            counts_err=None,
            rate_err=None,
            err_dist='poisson',
            # GTI 与质量
            gti_start=ev.gti_start,
            gti_stop=ev.gti_stop,
            quality=None,
            fracexp=None,
            backscal=None,
            areascal=None,
            # 曝光
            exposure=0.0,
            bin_exposure=None,
            # 时间系统元数据
            telescop=getattr(ev.meta, 'telescop', None) if ev.meta else None,
            timesys=getattr(ev.meta, 'timesys', None) if ev.meta else None,
            mjdref=getattr(ev.meta, 'mjdref', None) if ev.meta else None,
            # 其他
            region=None,
            header=ev.header,
            meta=ev.meta,
            headers_dump=ev.headers_dump,
            columns=("TIME", "COUNTS"),
            ratio=None,
        )

    tmin_eff = float(t.min())
    tmax_eff = float(t.max())
    nbins = max(1, int(np.ceil((tmax_eff - tmin_eff) / float(binsize))))
    edges = tmin_eff + np.arange(nbins + 1) * float(binsize)
    counts, _ = np.histogram(t, bins=edges)
    counts = counts.astype(float)

    # exposure per bin: use GTI if present
    if ev.gti_start is not None and ev.gti_stop is not None:
        ms, me = merge_gti(ev.gti_start, ev.gti_stop)
        if ms is None or me is None:
            expo = np.full(nbins, float(binsize))
        else:
            # attempt to apply adjustgti/frame alignment based on xselect.mdb defaults
            try:
                tree = _get_mdb_tree()
                # infer mission/instrument/mode from metadata/header
                adj, tp, frame_dt = _infer_adjustgti_timepixr_and_frame(ev, tree)
                if adj:
                    if frame_dt is None:
                        warnings.warn('adjustgti requested by xselect.mdb but frame_dt could not be inferred; skipping adjustgti')
                    else:
                        ms_adj, me_adj = gtimod.adjust_gti_to_frame(ms, me, frame_dt, timepixr=float(tp or 0.0))
                        if ms_adj is not None and me_adj is not None:
                            ms, me = ms_adj, me_adj
            except Exception:
                # conservative: if mdb parsing fails, proceed without adjust
                pass
            expo = _exposure_per_bins(ms, me, edges)
    else:
        expo = np.full(nbins, float(binsize))

    from .data import LightcurveData
    # 为了与当前 LightcurveData 设计兼容，这里同时填充 counts/counts_err、bin_lo/bin_hi、GTI 等字段
    # 从 EventData 继承时间参考和元数据
    return LightcurveData(
        path=ev.path,
        time=edges[:-1],
        value=counts,
        error=np.sqrt(counts),
        dt=binsize,
        # 时间字段
        timezero=getattr(ev, 'timezero', 0.0),
        timezero_obj=getattr(ev, 'timezero_obj', None),
        bin_lo=edges[:-1],
        bin_hi=edges[1:],
        tstart=float(edges[0]) if edges.size > 0 else None,
        tseg=float(edges[-1] - edges[0]) if edges.size > 0 else None,
        # 数据字段
        is_rate=False,
        counts=counts,
        rate=None,
        counts_err=np.sqrt(counts),
        rate_err=None,
        err_dist='poisson',
        # GTI 与质量
        gti_start=ev.gti_start,
        gti_stop=ev.gti_stop,
        quality=None,
        fracexp=None,
        backscal=None,
        areascal=None,
        # 曝光
        exposure=float(np.sum(expo)),
        bin_exposure=expo,
        # 时间系统元数据
        telescop=getattr(ev.meta, 'telescop', None) if ev.meta else None,
        timesys=getattr(ev.meta, 'timesys', None) if ev.meta else None,
        mjdref=getattr(ev.meta, 'mjdref', None) if ev.meta else None,
        # 其他
        region=None,
        header=ev.header,
        meta=ev.meta,
        headers_dump=ev.headers_dump,
        columns=("TIME", "COUNTS"),
        ratio=None,
    )


def extract_image(ev_or_path: EventData | str | Path, *, xcol: str | None = None, ycol: str | None = None,
                  bins: tuple[int, int] = (64, 64), xrange: tuple[float, float] | None = None, yrange: tuple[float, float] | None = None,
                  tmin: Optional[float] = None, tmax: Optional[float] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从事件生成 2D 图像（numpy ndarray, xedges, yedges）。

    - xcol/ycol 可指定 FITS 中的列名；否则尝试常见列名。
    - bins: (nx, ny)
    - xrange/yrange: 可选范围，否则基于数据范围自动。
    """
    if isinstance(ev_or_path, (str, Path)):
        ev = read_evt(ev_or_path)
    else:
        ev = ev_or_path

    # time filter
    if tmin is not None or tmax is not None:
        ev = select_events(ev.path if isinstance(ev_or_path, (str, Path)) else ev.path, tmin=tmin, tmax=tmax)

    # find columns
    if xcol is None:
        x_candidates = ['X', 'X_IMAGE', 'RAWX', 'DETX']
        xval = None
        for xc in x_candidates:
            xval = _read_column_from_evt(ev.path, xc)
            if xval is not None:
                xcol = xc
                break
    else:
        xval = _read_column_from_evt(ev.path, xcol)

    if ycol is None:
        y_candidates = ['Y', 'Y_IMAGE', 'RAWY', 'DETY']
        yval = None
        for yc in y_candidates:
            yval = _read_column_from_evt(ev.path, yc)
            if yval is not None:
                ycol = yc
                break
    else:
        yval = _read_column_from_evt(ev.path, ycol)

    if xval is None or yval is None:
        raise ValueError('Cannot find X/Y columns for image extraction')

    x = np.asarray(xval, dtype=float)
    y = np.asarray(yval, dtype=float)

    # apply optional time filter to arrays (if select_events changed ev.time)
    t = np.asarray(ev.time, dtype=float)
    if t.size != x.size:
        # align by reading time from FITS directly and filtering
        tcol = _read_column_from_evt(ev.path, 'TIME')
        if tcol is not None:
            t = np.asarray(tcol, dtype=float)
    # Build mask if select_events applied
    if tmin is not None or tmax is not None:
        mask = np.ones(x.size, dtype=bool)
        if tmin is not None:
            mask &= (t >= float(tmin))
        if tmax is not None:
            mask &= (t <= float(tmax))
        x = x[mask]
        y = y[mask]

    if xrange is None:
        xmin, xmax = float(x.min()) if x.size else 0.0, float(x.max()) if x.size else 1.0
    else:
        xmin, xmax = float(xrange[0]), float(xrange[1])
    if yrange is None:
        ymin, ymax = float(y.min()) if y.size else 0.0, float(y.max()) if y.size else 1.0
    else:
        ymin, ymax = float(yrange[0]), float(yrange[1])

    nx, ny = int(bins[0]), int(bins[1])
    xedges = np.linspace(xmin, xmax, nx + 1)
    yedges = np.linspace(ymin, ymax, ny + 1)
    img, xe, ye = np.histogram2d(y, x, bins=[yedges, xedges])
    # note: histogram2d returns array shape (ny, nx) with first axis y
    return img, xedges, yedges


def _estimate_exposure_from_eventdata(ev: EventData) -> float:
    """尝试从 EventData 中估算曝光：优先使用合并后的 GTI 累计时长，其次使用 meta 中的 TSTART/TSTOP，最后用事件跨度。

    该函数会调用 `merge_gti` 来规范化 GTI 区间。
    """
    if ev.gti_start is not None and ev.gti_stop is not None:
        ms, me = merge_gti(ev.gti_start, ev.gti_stop)
        if ms is not None and me is not None and ms.size > 0:
            return float(np.sum(me - ms))
    if ev.meta is not None and isinstance(ev.meta, OgipMeta):
        if ev.meta.tstart is not None and ev.meta.tstop is not None:
            return float(ev.meta.tstop - ev.meta.tstart)
    # fallback: use span of events
    if ev.time.size:
        return float(ev.time.max() - ev.time.min())
    return 0.0


def accumulate_spectrum_from_events(ev: EventData, *, channel_col: str = 'pi',
                                    ch_min: Optional[int] = None, ch_max: Optional[int] = None,
                                    nbins: Optional[int] = None) -> PhaData:
    """从 EventData 累积 PHA（计数直方）。

    参数
    - ev: 已读取或筛选的 EventData
    - channel_col: 首选的能道列，'pi' 或 'channel'
    - ch_min/ch_max: 道号截断范围（包含）
    - nbins: 如果提供，输出为 [0..nbins-1] 的道计数（否则基于数据自动）

    返回
    - PhaData 实例（未自动写入磁盘）
    """
    if channel_col not in ('pi', 'channel'):
        raise ValueError("channel_col must be 'pi' or 'channel'")

    arr = None
    if channel_col == 'pi' and ev.pi is not None:
        arr = np.asarray(ev.pi, dtype=int)
    elif channel_col == 'channel' and ev.channel is not None:
        arr = np.asarray(ev.channel, dtype=int)
    else:
        # try the other one
        if ev.pi is not None:
            arr = np.asarray(ev.pi, dtype=int)
        elif ev.channel is not None:
            arr = np.asarray(ev.channel, dtype=int)
        else:
            raise ValueError('EventData contains no PI or CHANNEL column')

    if arr.size == 0:
        # empty spectrum
        channels = np.array([], dtype=int)
        counts = np.array([], dtype=float)
        exposure = _estimate_exposure_from_eventdata(ev)
        return PhaData(path=ev.path, channels=channels, counts=counts,
                   stat_err=None, exposure=exposure, backscal=None, areascal=None,
                   quality=None, grouping=None, ebounds=None, header=ev.header, meta=ev.meta,
                   headers_dump=ev.headers_dump, columns=())

    # apply optional truncation
    if ch_min is not None:
        arr = arr[arr >= int(ch_min)]
    if ch_max is not None:
        arr = arr[arr <= int(ch_max)]

    if nbins is None:
        # choose bins from min..max inclusive
        ch_lo = int(arr.min())
        ch_hi = int(arr.max())
        nbins = ch_hi - ch_lo + 1
        bins = np.arange(ch_lo, ch_hi + 2, dtype=int)
        channels = np.arange(ch_lo, ch_hi + 1, dtype=int)
        counts, _ = np.histogram(arr, bins=bins)
    else:
        # fixed nbins: assume channels start at 0
        bins = np.arange(0, int(nbins) + 1, dtype=int)
        channels = np.arange(0, int(nbins), dtype=int)
        counts, _ = np.histogram(arr, bins=bins)

    # poisson stat error (sqrt) as default
    stat_err = np.sqrt(counts.astype(float))
    exposure = _estimate_exposure_from_eventdata(ev)

    pha = PhaData(
        path=ev.path,
        channels=channels, counts=counts.astype(float), stat_err=stat_err,
        exposure=exposure, backscal=None, areascal=None,
        quality=None, grouping=None, ebounds=None,
        header=ev.header, meta=ev.meta, headers_dump=ev.headers_dump, columns=('CHANNEL','COUNTS')
    )
    return pha


def write_pha(pha: PhaData, outpath: str | Path, *, overwrite: bool = False) -> Path:
    """写出 PHA，统一委托到 core.io.PhaWriter。"""
    from .io import write_pha as _io_write_pha

    return _io_write_pha(pha, outpath, overwrite=overwrite)
