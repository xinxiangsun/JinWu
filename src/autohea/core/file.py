"""
General-purpose OGIP FITS readers with numpy-first outputs.

This module consolidates robust, reusable readers for common high-energy
astrophysics FITS products (ARF, RMF, PHA spectrum, lightcurve, events),
returning standardized numpy arrays and light dataclasses for downstream use.

Design goals
------------
- Self-contained: only depends on astropy.io.fits and numpy; can interoperate
  with external libs like gdt but not require them.
- OGIP-friendly: understands common extensions/columns and gracefully handles
  missing optional fields.
- Numpy-first: outputs are plain ndarrays or small dataclasses with ndarrays.
- Practical utilities: band/channel filtering, RMF sparse-to-dense rebuild.

Conventions
-----------
- Energy band of interest is often 0.5–4.0 keV. ARF uses bins 81–780 to define
  band edges; PHA/RMF channel selection typically uses EBOUNDS overlap and the
  1024-channel range 51–399.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, cast, Union, Literal

import numpy as np
from astropy.io import fits

__all__ = [
    # Data containers
    "EnergyBand", "ChannelBand",
    "ArfData", "RmfData", "PhaData", "LightcurveData", "EventData",
    # Readers
    "OgipArfReader", "OgipRmfReader", "OgipPhaReader", "OgipLightcurveReader", "OgipEventReader",
    # Backward-compat aliases
    "ArfReader", "RmfReader", "RspReader",
    # Utilities
    "band_from_arf_bins", "channel_mask_from_ebounds",
    # Unified entry
    "OgipUnifiedReader", "OgipAnyData", "OgipData", "guess_ogip_kind", "read_ogip",
]


# ---------- Data containers ----------

@dataclass
class EnergyBand:
    emin: float
    emin_unit: str
    emax: float
    emax_unit: str

@dataclass
class ChannelBand:
    ch_lo: int
    ch_hi: int

@dataclass
class ArfData:
    energ_lo: np.ndarray  # shape (N,)
    energ_hi: np.ndarray  # shape (N,)
    specresp: np.ndarray  # shape (N,), cm^2
    header: Dict[str, Any]

@dataclass
class RmfData:
    energ_lo: np.ndarray  # shape (N_E,)
    energ_hi: np.ndarray  # shape (N_E,)
    # OGIP sparse representation
    n_grp: Optional[np.ndarray]  # shape (N_E,) or None
    f_chan: Optional[np.ndarray]  # variable-length sequences concatenated
    n_chan: Optional[np.ndarray]
    matrix: np.ndarray  # object array of per-row matrices, or 2D dense if rebuilt
    # EBOUNDS
    channel: Optional[np.ndarray]  # shape (N_C,) CHANNEL indices
    e_min: Optional[np.ndarray]    # shape (N_C,)
    e_max: Optional[np.ndarray]    # shape (N_C,)
    header: Dict[str, Any]

    def rebuild_dense(self) -> np.ndarray:
        """Rebuild a dense redistribution matrix of shape (N_E, N_C) if sparse columns present.

        Returns
        -------
        np.ndarray
            Dense matrix with probability per channel for each energy bin.
        """
        if self.channel is None or self.e_min is None or self.e_max is None:
            # No EBOUNDS; best effort: try to stack MATRIX if already dense-like
            if self.matrix.ndim == 2:
                return self.matrix
            raise ValueError("Cannot rebuild dense RMF without EBOUNDS and sparse definition")

        n_e = self.energ_lo.size
        n_c = int(self.channel.size)
        out = np.zeros((n_e, n_c), dtype=float)

        # If OGIP sparse columns exist, use them
        if (self.n_grp is not None) and (self.f_chan is not None) and (self.n_chan is not None):
            # MATRIX may be a variable-length vector per energy row; flatten sequentially per groups
            # We'll iterate per row, then within groups allocate consecutive channels
            idx_mat = 0
            for i in range(n_e):
                ng = int(self.n_grp[i]) if np.ndim(self.n_grp) > 0 else int(self.n_grp)
                # Each row's MATRIX may be stored in self.matrix[i]
                row_vals = self.matrix[i]
                # Some files store per-row vector as ndarray already
                if isinstance(row_vals, np.ndarray):
                    # Use per-row cursor
                    cursor = 0
                    # Flatten per-row f_chan/n_chan sequences: sometimes these are ragged; read via FITS tables
                    # Here, we assume f_chan/n_chan are stored per-row like MATRIX
                    fcs = np.atleast_1d(self.f_chan[i]) if np.ndim(self.f_chan) > 1 else np.atleast_1d(self.f_chan)
                    ncs = np.atleast_1d(self.n_chan[i]) if np.ndim(self.n_chan) > 1 else np.atleast_1d(self.n_chan)
                    for g in range(ng):
                        start = int(fcs[g])
                        width = int(ncs[g])
                        out[i, start:start+width] = row_vals[cursor:cursor+width]
                        cursor += width
                else:
                    # Fallback: if entire matrix is rectangular 2D already
                    if np.ndim(self.matrix) == 2 and self.matrix.shape[1] == n_c:
                        out[i, :] = self.matrix[i, :]
                    else:
                        raise ValueError("Unsupported RMF MATRIX layout")
            return out
        # Else, MATRIX might already be a (N_E, N_C) dense matrix
        if self.matrix.ndim == 2 and self.matrix.shape[1] == n_c:
            return self.matrix
        raise ValueError("RMF does not contain recognizable sparse columns and MATRIX is not dense")

@dataclass
class PhaData:
    channels: np.ndarray  # shape (N,)
    counts: np.ndarray    # shape (N,)
    stat_err: Optional[np.ndarray]
    exposure: float
    backscal: Optional[float]
    areascal: Optional[float]
    quality: Optional[np.ndarray]
    grouping: Optional[np.ndarray]
    ebounds: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]  # (CHANNEL, E_MIN, E_MAX)
    header: Dict[str, Any]

@dataclass
class LightcurveData:
    time: np.ndarray   # bin left edges or centers
    value: np.ndarray  # RATE or COUNTS
    error: Optional[np.ndarray]
    dt: Optional[float]
    exposure: Optional[float]
    is_rate: bool
    header: Dict[str, Any]

@dataclass
class EventData:
    time: np.ndarray   # events times (s)
    pi: Optional[np.ndarray]       # PI or CHANNEL if present
    channel: Optional[np.ndarray]
    gti_start: Optional[np.ndarray]
    gti_stop: Optional[np.ndarray]
    header: Dict[str, Any]


# Unified data union and wrapper
OgipData = Union[ArfData, RmfData, PhaData, LightcurveData, EventData]

@dataclass
class OgipAnyData:
    kind: Literal['arf', 'rmf', 'pha', 'lc', 'evt']
    data: OgipData
    path: Path


# ---------- Utilities ----------

def band_from_arf_bins(arf_path: str | Path, bin_lo: int = 81, bin_hi: int = 780) -> EnergyBand:
    with fits.open(arf_path) as h:
        hd = cast(Any, h["SPECRESP"])  # BinTableHDU
        d = hd.data
        elo = np.asarray(d["ENERG_LO"], float)
        ehi = np.asarray(d["ENERG_HI"], float)
    i0 = max(0, int(bin_lo) - 1)
    i1 = min(ehi.size - 1, int(bin_hi) - 1)
    emin = float(elo[i0])
    emax = float(ehi[i1])
    return EnergyBand(emin=emin, emin_unit="keV", emax=emax, emax_unit="keV")


def channel_mask_from_ebounds(
    ebounds: Tuple[np.ndarray, np.ndarray, np.ndarray],
    band: EnergyBand,
    ch_band: Optional[ChannelBand] = None,
) -> np.ndarray:
    ch, e_lo, e_hi = ebounds
    mask = (e_hi > float(band.emin)) & (e_lo < float(band.emax))
    if ch_band is not None:
        mask &= (ch >= int(ch_band.ch_lo)) & (ch <= int(ch_band.ch_hi))
    return mask


# ---------- Readers ----------

class OgipArfReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        self._data: Optional[ArfData] = None

    def read(self) -> ArfData:
        with fits.open(self.path) as h:
            hd = cast(Any, h["SPECRESP"])  # BinTableHDU
            d = hd.data
            energ_lo = np.asarray(d["ENERG_LO"], float)
            energ_hi = np.asarray(d["ENERG_HI"], float)
            specresp = np.asarray(d["SPECRESP"], float)
            header = dict(cast(Any, hd.header))
        self._data = ArfData(energ_lo, energ_hi, specresp, header)
        return self._data


class OgipRmfReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        self._data: Optional[RmfData] = None

    def read(self) -> RmfData:
        with fits.open(self.path) as h:
            hm = cast(Any, h["MATRIX"])  # BinTableHDU
            dm = hm.data
            energ_lo = np.asarray(dm["ENERG_LO"], float)
            energ_hi = np.asarray(dm["ENERG_HI"], float)
            header = dict(cast(Any, hm.header))

            matrix = np.asarray(dm["MATRIX"], dtype=object)
            n_grp = dm["N_GRP"] if "N_GRP" in dm.columns.names else None
            f_chan = dm["F_CHAN"] if "F_CHAN" in dm.columns.names else None
            n_chan = dm["N_CHAN"] if "N_CHAN" in dm.columns.names else None

            channel = e_min = e_max = None
            if "EBOUNDS" in h:
                de = cast(Any, h["EBOUNDS"]).data
                channel = np.asarray(de["CHANNEL"], int)
                e_min = np.asarray(de["E_MIN"], float)
                e_max = np.asarray(de["E_MAX"], float)

        self._data = RmfData(
            energ_lo=energ_lo,
            energ_hi=energ_hi,
            n_grp=(np.asarray(n_grp) if n_grp is not None else None),
            f_chan=(np.asarray(f_chan) if f_chan is not None else None),
            n_chan=(np.asarray(n_chan) if n_chan is not None else None),
            matrix=matrix,
            channel=channel,
            e_min=e_min,
            e_max=e_max,
            header=header,
        )
        return self._data


class OgipPhaReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        self._data: Optional[PhaData] = None

    def read(self) -> PhaData:
        with fits.open(self.path) as h:
            hs = cast(Any, h["SPECTRUM"])  # BinTableHDU
            ds = hs.data
            channels = np.asarray(ds["CHANNEL"], int)
            counts = np.asarray(ds["COUNTS"], float)
            stat_err = np.asarray(ds["STAT_ERR"], float) if "STAT_ERR" in ds.columns.names else None
            header_map = cast(Any, hs.header)
            exposure = float(header_map.get("EXPOSURE", header_map.get("EXPTIME", np.nan)))
            backscal = float(header_map.get("BACKSCAL")) if "BACKSCAL" in header_map else None
            areascal = float(header_map.get("AREASCAL")) if "AREASCAL" in header_map else None
            quality = np.asarray(ds["QUALITY"], int) if "QUALITY" in ds.columns.names else None
            grouping = np.asarray(ds["GROUPING"], int) if "GROUPING" in ds.columns.names else None
            ebounds = None
            if "EBOUNDS" in h:
                de = cast(Any, h["EBOUNDS"]).data
                ebounds = (
                    np.asarray(de["CHANNEL"], int),
                    np.asarray(de["E_MIN"], float),
                    np.asarray(de["E_MAX"], float),
                )
            header = dict(header_map)
        self._data = PhaData(channels, counts, stat_err, exposure, backscal, areascal, quality, grouping, ebounds, header)
        return self._data

    def select_by_band(
        self,
        band: EnergyBand,
        rmf_chan_band: Optional[ChannelBand] = ChannelBand(51, 399),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (channels, counts) filtered by energy band via EBOUNDS. If no EBOUNDS,
        fall back to channel range (e.g., 51–399)."""
        if self._data is None:
            _ = self.read()
        d = self._data
        assert d is not None
        if d.ebounds is not None:
            mask = channel_mask_from_ebounds(d.ebounds, band, rmf_chan_band)
            # EBOUNDS and SPECTRUM channels should align; ensure alignment via channel indices
            ch_all = d.ebounds[0]
            # Build lookup from channel to spectrum index
            idx_map = {int(c): i for i, c in enumerate(d.channels)}
            sel_channels = ch_all[mask]
            sel_idx = np.array([idx_map[c] for c in sel_channels if c in idx_map], dtype=int)
            return d.channels[sel_idx], d.counts[sel_idx]
        # Fallback to channel-band only
        if rmf_chan_band is None:
            return d.channels, d.counts
        m = (d.channels >= rmf_chan_band.ch_lo) & (d.channels <= rmf_chan_band.ch_hi)
        return d.channels[m], d.counts[m]


class OgipLightcurveReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        self._data: Optional[LightcurveData] = None

    def read(self) -> LightcurveData:
        with fits.open(self.path) as h:
            # Try common names: 'RATE' ext, or first table with TIME
            hdu = None
            for ext in h:
                ext_any = cast(Any, ext)
                if getattr(ext_any, "data", None) is None:
                    continue
                names = getattr(ext_any.data, "columns", None)
                if names is not None and ("TIME" in names.names):
                    hdu = ext_any
                    break
            if hdu is None:
                raise ValueError("No suitable lightcurve HDU with TIME column found")
            d = cast(Any, hdu.data)
            time = np.asarray(d["TIME"], float)
            val = err = None
            is_rate = False
            if "RATE" in d.columns.names:
                val = np.asarray(d["RATE"], float)
                err = np.asarray(d["ERROR"], float) if "ERROR" in d.columns.names else None
                is_rate = True
            elif "COUNTS" in d.columns.names:
                val = np.asarray(d["COUNTS"], float)
                err = np.asarray(d["ERROR"], float) if "ERROR" in d.columns.names else None
                is_rate = False
            else:
                raise ValueError("Lightcurve HDU lacks RATE/COUNTS column")
            header = dict(cast(Any, hdu.header))
            # Infer dt if possible
            dt = float(np.median(np.diff(time))) if time.size >= 2 else None
            exposure_val = header.get("EXPOSURE", header.get("EXPTIME", np.nan))
            exposure = float(exposure_val) if np.isfinite(exposure_val) else None
        self._data = LightcurveData(time=time, value=val, error=err, dt=dt, exposure=exposure, is_rate=is_rate, header=header)
        return self._data


class OgipEventReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        self._data: Optional[EventData] = None

    def read(self) -> EventData:
        with fits.open(self.path) as h:
            # EVENTS
            # Find first binary table HDU with TIME column
            hevt = None
            for ext in h:
                ext_any = cast(Any, ext)
                if getattr(ext_any, "data", None) is None:
                    continue
                names = getattr(ext_any.data, "columns", None)
                if names is not None and ("TIME" in names.names):
                    hevt = ext_any
                    break
            if hevt is None:
                raise ValueError("No EVENTS-like HDU with TIME column found")
            de = cast(Any, hevt.data)
            time = np.asarray(de["TIME"], float)
            pi = np.asarray(de["PI"], int) if "PI" in de.columns.names else None
            channel = np.asarray(de["CHANNEL"], int) if "CHANNEL" in de.columns.names else None
            header = dict(cast(Any, hevt.header))

            # GTI
            gti_start = gti_stop = None
            if "GTI" in h:
                hg = cast(Any, h["GTI"]).data
                gti_start = np.asarray(hg["START"], float)
                gti_stop = np.asarray(hg["STOP"], float)

        self._data = EventData(time=time, pi=pi, channel=channel, gti_start=gti_start, gti_stop=gti_stop, header=header)
        return self._data


# ---------- Backward-compat aliases ----------

class ArfReader(OgipArfReader):
    """Backward-compatible alias for ARF reader."""
    pass


class RmfReader(OgipRmfReader):
    """Backward-compatible alias for RMF reader."""
    pass


class RspReader(OgipRmfReader):
    """Alias for response reader; RSP often stores MATRIX+EBOUNDS akin to RMF."""
    pass


class LightcurveReader(OgipLightcurveReader):
    """Backward-compatible alias for lightcurve reader."""
    pass


# ---------- Unified reader ----------

def guess_ogip_kind(path: str | Path) -> Literal['arf', 'rmf', 'pha', 'lc', 'evt']:
    p = Path(path)
    name = p.name.lower()
    # quick filename-based hints
    if name.endswith('.arf'):
        return 'arf'
    if name.endswith('.rmf') or name.endswith('.rsp'):
        return 'rmf'
    if name.endswith('.pha') or name.endswith('.pi'):
        return 'pha'
    # Content-based fallback
    with fits.open(p) as h:
        # Prefer extension name checks
        extnames = {getattr(x, 'name', '').upper() for x in h}
        if 'SPECRESP' in extnames and 'MATRIX' not in extnames:
            return 'arf'
        if 'MATRIX' in extnames:
            return 'rmf'
        if 'SPECTRUM' in extnames:
            return 'pha'
        if 'EVENTS' in extnames or any(('TIME' in getattr(getattr(x, 'data', None), 'columns', ()).names) for x in h if getattr(x, 'data', None) is not None):
            # Ambiguity between LC and EVT: try RATE/COUNTS columns
            for x in h:
                d = getattr(x, 'data', None)
                cols = getattr(d, 'columns', None)
                if cols is None:
                    continue
                names = getattr(cols, 'names', ())
                if 'RATE' in names or 'COUNTS' in names:
                    return 'lc'
            return 'evt'
    # default to PHA if uncertain
    return 'pha'


class OgipUnifiedReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        self.kind: Optional[Literal['arf','rmf','pha','lc','evt']] = None
        self._data: Optional[OgipAnyData] = None

    def read(self, kind: Optional[Literal['arf','rmf','pha','lc','evt']] = None) -> OgipAnyData:
        k = kind or guess_ogip_kind(self.path)
        self.kind = k
        if k == 'arf':
            data = OgipArfReader(self.path).read()
        elif k == 'rmf':
            data = OgipRmfReader(self.path).read()
        elif k == 'pha':
            data = OgipPhaReader(self.path).read()
        elif k == 'lc':
            data = OgipLightcurveReader(self.path).read()
        elif k == 'evt':
            data = OgipEventReader(self.path).read()
        else:
            raise ValueError(f"Unknown OGIP kind: {k}")
        self._data = OgipAnyData(kind=k, data=data, path=self.path)
        return self._data


def read_ogip(path: str | Path, kind: Optional[Literal['arf','rmf','pha','lc','evt']] = None) -> OgipAnyData:
    """One-shot unified reader.

    Parameters
    ----------
    path : str | Path
        FITS path. Kind will be inferred if not provided.
    kind : Literal['arf','rmf','pha','lc','evt'] | None
        Optional explicit kind.

    Returns
    -------
    OgipAnyData
        Wrapper with kind and data.
    """
    return OgipUnifiedReader(path).read(kind)



