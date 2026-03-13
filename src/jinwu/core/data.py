from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union, cast

import numpy as np

from .base import ArfBase, RmfBase, PhaBase, LightcurveDataBase, EventDataBase, FitsHeaderDump
from .ogip import OgipFitsBase, ValidationReport


def _validate_time_series_like(obj: Any) -> ValidationReport:
    rpt = OgipFitsBase.validate(obj)
    for k in obj.REQUIRED_KEYS:
        if k not in obj.header:
            rpt.add('WARN', 'MISSING_KEY', f"Required key '{k}' not found (OGIP-93-003).")
    cols = getattr(obj, 'columns', ()) or ()
    colset = {c.upper() for c in cols}
    for group in obj.REQUIRED_COLUMNS_ANY:
        if not any(c.upper() in colset for c in group):
            rpt.add('ERROR', 'MISSING_COLUMN', f"Missing required column group: {group}")
    hdr = obj.header or {}
    if not (('MJDREF' in hdr) or (('MJDREFI' in hdr) or ('MJDREFF' in hdr))):
        rpt.add('WARN', 'MISSING_MJDREF', 'MJDREF / (MJDREFI+MJDREFF) not found in header; absolute times may be ambiguous.')
    try:
        timeunit = obj.get_keyword_ci('TIMEUNIT')
    except Exception:
        timeunit = (obj.header or {}).get('TIMEUNIT')
    if timeunit is not None and str(timeunit).upper() not in ('S', 'SEC', 'SECOND', 'SECONDS'):
        rpt.add('INFO', 'UNUSUAL_TIMEUNIT', f"TIMEUNIT='{timeunit}'")
    return rpt


class ArfData(ArfBase):
    """Concrete ARF data class with local behavior implementation."""

    def validate(self) -> ValidationReport:  # type: ignore[override]
        rpt = OgipFitsBase.validate(self)
        hdr = self.header
        for group in self.REQUIRED_KEYS_ANY:
            if not any(g in hdr for g in group):
                rpt.add('WARN', 'MISSING_KEY', f"Missing one of required keys {group} (CAL/GEN/92-002).")
        colset = {c.upper() for c in self.columns}
        for c in ["ENERG_LO", "ENERG_HI", "SPECRESP"]:
            if c not in colset:
                rpt.add('ERROR', 'MISSING_COLUMN', f"ARF missing column {c}")
        rpt.ok = len(rpt.errors()) == 0
        return rpt

    def plot(
        self,
        ax: Optional[Any] = None,
        *,
        energy_unit: str = 'keV',
        yscale: str = 'linear',
        title: Optional[str] = None,
        **kwargs,
    ):
        import matplotlib.pyplot as _plt

        mid_e = 0.5 * (np.asarray(self.energ_lo) + np.asarray(self.energ_hi))
        ax = ax or _plt.gca()
        assert ax is not None
        kwargs.setdefault('lw', 1.0)
        ax.plot(mid_e, self.specresp, **kwargs)
        ax.set_xlabel(f"Energy ({energy_unit})")
        ax.set_ylabel("Effective Area (cm$^2$)")
        ax.set_yscale(yscale)
        if title is None:
            title = Path(str(getattr(self, 'path', ''))).name or 'ARF'
        ax.set_title(title)
        ax.grid(alpha=0.3, ls='--')
        return ax

    def rebin(self, factor: int) -> 'ArfData':
        if factor <= 0:
            raise ValueError('factor must be > 0')
        from ..ftools.ftrbnrmf import rebin_arf

        elo = np.asarray(self.energ_lo, dtype=float)
        ehi = np.asarray(self.energ_hi, dtype=float)
        area = np.asarray(self.specresp, dtype=float)
        n = elo.size
        groups = [(i, min(i + factor, n)) for i in range(0, n, factor)]
        new_elo = np.array([elo[s] for s, _ in groups], dtype=float)
        new_ehi = np.array([ehi[e - 1] for _, e in groups], dtype=float)
        new_area = rebin_arf(elo, ehi, area, new_elo, new_ehi)
        return ArfData(
            path=self.path,
            energ_lo=new_elo,
            energ_hi=new_ehi,
            specresp=new_area,
            columns=self.columns,
            header=self.header,
            meta=self.meta,
            headers_dump=self.headers_dump,
        )


class RmfData(RmfBase):
    """Concrete RMF data class with local behavior implementation."""

    def rebuild_dense(self) -> np.ndarray:
        if self.channel is None or self.e_min is None or self.e_max is None:
            if self.matrix.ndim == 2:
                return self.matrix
            raise ValueError("Cannot rebuild dense RMF without EBOUNDS and sparse definition")

        n_e = self.energ_lo.size
        n_c = int(self.channel.size)
        out = np.zeros((n_e, n_c), dtype=float)

        if (self.n_grp is not None) and (self.f_chan is not None) and (self.n_chan is not None):
            for i in range(n_e):
                ng = int(self.n_grp[i]) if np.ndim(self.n_grp) > 0 else int(self.n_grp)
                row_vals = self.matrix[i]
                if isinstance(row_vals, np.ndarray):
                    cursor = 0
                    fcs = np.atleast_1d(self.f_chan[i]) if np.ndim(self.f_chan) > 1 else np.atleast_1d(self.f_chan)
                    ncs = np.atleast_1d(self.n_chan[i]) if np.ndim(self.n_chan) > 1 else np.atleast_1d(self.n_chan)
                    for g in range(ng):
                        start = int(fcs[g])
                        width = int(ncs[g])
                        out[i, start:start + width] = row_vals[cursor:cursor + width]
                        cursor += width
                else:
                    if np.ndim(self.matrix) == 2 and self.matrix.shape[1] == n_c:
                        out[i, :] = self.matrix[i, :]
                    else:
                        raise ValueError("Unsupported RMF MATRIX layout")
            return out

        if self.matrix.ndim == 2 and self.matrix.shape[1] == n_c:
            return self.matrix
        raise ValueError("RMF does not contain recognizable sparse columns and MATRIX is not dense")

    def validate(self) -> ValidationReport:  # type: ignore[override]
        rpt = OgipFitsBase.validate(self)
        hdr = self.header
        for group in self.REQUIRED_KEYS_ANY:
            if not any(g in hdr for g in group):
                rpt.add('WARN', 'MISSING_KEY', f"Missing one of required keys {group} (CAL/GEN/92-002).")
        colset = {c.upper() for c in self.columns}
        for c in ["ENERG_LO", "ENERG_HI", "MATRIX"]:
            if c not in colset:
                rpt.add('ERROR', 'MISSING_COLUMN', f"RMF missing column {c}")
        if self.channel is not None:
            for c in ["CHANNEL", "E_MIN", "E_MAX"]:
                if c not in colset:
                    rpt.add('WARN', 'MISSING_COLUMN', f"EBOUNDS expected column {c}")
        rpt.ok = len(rpt.errors()) == 0
        return rpt

    @property
    def dense_matrix(self) -> np.ndarray:
        return self.rebuild_dense()

    def plot(
        self,
        ax: Optional[Any] = None,
        *,
        kind: str = 'matrix',
        row: int = 0,
        yscale: str = 'linear',
        cmap: str = 'viridis',
        title: Optional[str] = None,
        **kwargs,
    ):
        import matplotlib.pyplot as _plt

        ax = ax or _plt.gca()
        assert ax is not None
        if kind == 'matrix':
            dm = self.dense_matrix
            im = ax.imshow(dm, aspect='auto', origin='lower', cmap=cmap, **kwargs)
            ax.set_xlabel('Channel')
            ax.set_ylabel('Energy bin')
            if title is None:
                title = Path(str(getattr(self, 'path', ''))).name or 'RMF'
            ax.set_title(title)
            ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Prob.')
        else:
            dm = self.dense_matrix
            row = max(0, min(dm.shape[0] - 1, int(row)))
            kwargs.setdefault('lw', 1.0)
            ax.plot(dm[row], **kwargs)
            ax.set_xlabel('Channel')
            ax.set_ylabel('Probability')
            ax.set_yscale(yscale)
            if title is None:
                title = f"RMF row {row}"
            ax.set_title(title)
        ax.grid(alpha=0.3, ls='--')
        return ax

    def rebin(self, factor: int) -> 'RmfData':
        if factor <= 0:
            raise ValueError('factor must be > 0')
        dense = self.rebuild_dense()
        n_e, _ = dense.shape
        groups = [(i, min(i + factor, n_e)) for i in range(0, n_e, factor)]
        new_rows = []
        new_elo = []
        new_ehi = []
        for s, e in groups:
            new_rows.append(np.sum(dense[s:e, :], axis=0))
            new_elo.append(float(self.energ_lo[s]))
            new_ehi.append(float(self.energ_hi[e - 1]))
        new_matrix = np.vstack(new_rows)
        return RmfData(
            path=self.path,
            energ_lo=np.asarray(new_elo, dtype=float),
            energ_hi=np.asarray(new_ehi, dtype=float),
            n_grp=None,
            f_chan=None,
            n_chan=None,
            matrix=new_matrix,
            channel=self.channel,
            e_min=self.e_min,
            e_max=self.e_max,
            columns=self.columns,
            header=self.header,
            meta=self.meta,
            headers_dump=self.headers_dump,
        )


class PhaData(PhaBase):
    """Concrete PHA data class with local behavior implementation."""

    def validate(self) -> ValidationReport:  # type: ignore[override]
        rpt = OgipFitsBase.validate(self)
        for k in self.REQUIRED_KEYS:
            if k not in self.header:
                rpt.add('WARN', 'MISSING_KEY', f"Required key '{k}' not found (OGIP-92-007).")
        colset = {c.upper() for c in self.columns}
        for c in self.REQUIRED_COLUMNS:
            if c not in colset:
                rpt.add('ERROR', 'MISSING_COLUMN', f"PHA missing column {c}")
        exp_val = self.header.get('EXPOSURE', self.header.get('EXPTIME', None))
        if exp_val is not None:
            try:
                if float(exp_val) <= 0:
                    rpt.add('WARN', 'BAD_EXPOSURE', f"Non-positive exposure value: {exp_val}")
            except Exception:
                rpt.add('WARN', 'BAD_EXPOSURE', f"Exposure not numeric: {exp_val}")
        rpt.ok = len(rpt.errors()) == 0
        return rpt

    @property
    def count_rate(self) -> Optional[np.ndarray]:
        if self.exposure and self.exposure > 0:
            return self.counts / self.exposure
        return None

    def plot(self, **kwargs):
        from .plot import plot_spectrum

        return plot_spectrum(self, **kwargs)

    def slice(
        self,
        *,
        emin: Optional[float] = None,
        emax: Optional[float] = None,
        ch_lo: Optional[int] = None,
        ch_hi: Optional[int] = None,
    ) -> 'PhaData':
        from .ops import slice_pha

        return cast(PhaData, slice_pha(cast(Any, self), emin=emin, emax=emax, ch_lo=ch_lo, ch_hi=ch_hi))

    def rebin(self, *, factor: Optional[int] = None, min_counts: Optional[float] = None) -> 'PhaData':
        from .ops import rebin_pha

        return cast(PhaData, rebin_pha(cast(Any, self), factor=factor, min_counts=min_counts))

    def grppha(
        self,
        *,
        min_counts: Optional[float] = None,
        groupfile: Optional[str] = None,
        rebin: bool = False,
        outfile: Optional[str] = None,
        overwrite: bool = False,
    ) -> 'PhaData':
        from ..ftools.grppha import grppha as _grppha

        return cast(PhaData, _grppha(
            cast(Any, self),
            min_counts=min_counts,
            groupfile=groupfile,
            rebin=rebin,
            outfile=outfile,
            overwrite=overwrite,
        ))

    def group_by_min_counts(self, min_counts: float) -> np.ndarray:
        from ..ftools.grppha import compute_grouping_by_min_counts

        return compute_grouping_by_min_counts(self.counts, min_counts)


class LightcurveData(LightcurveDataBase):
    """Concrete lightcurve data class with local behavior implementation."""

    def validate(self) -> ValidationReport:  # type: ignore[override]
        rpt = _validate_time_series_like(self)
        colset = {c.upper() for c in self.columns}
        if "TIME" not in colset:
            rpt.add('ERROR', 'MISSING_COLUMN', "Lightcurve missing TIME column")
        if not any(x in colset for x in ["RATE", "COUNTS"]):
            rpt.add('ERROR', 'MISSING_COLUMN', "Lightcurve missing RATE/COUNTS column")
        if self.time is None or len(self.time) == 0:
            rpt.add('ERROR', 'MISSING_TIME', "time array is empty or None")
        if self.value is None or len(self.value) == 0:
            rpt.add('ERROR', 'MISSING_VALUE', "value array is empty or None")
        if self.time is not None and self.value is not None and len(self.time) != len(self.value):
            rpt.add('ERROR', 'LENGTH_MISMATCH', f"time ({len(self.time)}) and value ({len(self.value)}) length mismatch")
        if self.gti_start is not None and self.gti_stop is not None:
            if len(self.gti_start) != len(self.gti_stop):
                rpt.add('ERROR', 'GTI_MISMATCH', "gti_start and gti_stop must have same length")
            if len(self.gti_start) > 0 and not np.all(self.gti_start < self.gti_stop):
                rpt.add('ERROR', 'GTI_ORDER', "gti_start must be < gti_stop")
        if self.time is not None and len(self.time) > 1 and not np.all(np.diff(self.time) > 0):
            rpt.add('WARN', 'TIME_NOT_SORTED', "time array is not strictly increasing")
        if self.exposure is not None and self.exposure <= 0:
            rpt.add('WARN', 'BAD_EXPOSURE', f"exposure must be > 0, got {self.exposure}")
        try:
            _, _, width = self._resolve_bin_geometry()
            if self.time is not None and len(self.time) != len(width):
                rpt.add('ERROR', 'BIN_LENGTH_MISMATCH', "time and bin geometry length mismatch")
            if np.any(~np.isfinite(width)) or np.any(width <= 0):
                rpt.add('ERROR', 'BAD_BIN_WIDTH', "bin widths must be finite and > 0")
        except Exception as e:
            rpt.add('WARN', 'BIN_GEOMETRY_UNRESOLVED', f"failed to resolve bin geometry: {e}")
        if self.bin_exposure is not None and self.time is not None and len(self.bin_exposure) != len(self.time):
            rpt.add('ERROR', 'BIN_EXPOSURE_MISMATCH', "bin_exposure length must match time length")
        rpt.ok = len(rpt.errors()) == 0
        return rpt

    @property
    def n(self) -> int:
        return len(self.time) if self.time is not None else 0

    @property
    def absolute_time(self) -> np.ndarray:
        if self.time is None:
            return np.asarray([], dtype=float)
        return np.asarray(self.time, dtype=float) + float(self.timezero)

    @property
    def gti(self) -> Optional[list[tuple[float, float]]]:
        if self.gti_start is None or self.gti_stop is None:
            return None
        return [(float(s), float(e)) for s, e in zip(self.gti_start, self.gti_stop)]

    def get_time_object(self, index: Optional[int] = None) -> Optional[Any]:
        if self.timezero_obj is None:
            return None
        from astropy.time import TimeDelta

        absolute_times = self.absolute_time
        dt = TimeDelta(absolute_times[index], format='sec') if index is not None else TimeDelta(absolute_times, format='sec')
        return self.timezero_obj + dt

    @property
    def bin_centers(self) -> Optional[np.ndarray]:
        try:
            left, right, _ = self._resolve_bin_geometry()
            return 0.5 * (left + right)
        except Exception:
            return None

    def _resolve_bin_geometry(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.time is None:
            return np.asarray([], float), np.asarray([], float), np.asarray([], float)
        t = np.asarray(self.time, dtype=float)
        n = t.size
        if n == 0:
            return np.asarray([], float), np.asarray([], float), np.asarray([], float)
        if self.bin_lo is not None and self.bin_hi is not None:
            lo = np.asarray(self.bin_lo, dtype=float)
            hi = np.asarray(self.bin_hi, dtype=float)
            if lo.shape == hi.shape == t.shape:
                return lo, hi, hi - lo
        if self.bin_width is not None:
            bw = np.asarray(self.bin_width, dtype=float)
            if bw.shape == t.shape:
                return t - 0.5 * bw, t + 0.5 * bw, bw
        if self.dt is not None:
            dt_arr = np.asarray(self.dt, dtype=float)
            if dt_arr.ndim == 0:
                dt_val = float(dt_arr)
                if np.isfinite(dt_val) and dt_val > 0:
                    bw = np.full_like(t, dt_val, dtype=float)
                    return t - 0.5 * bw, t + 0.5 * bw, bw
            elif dt_arr.shape == t.shape:
                bw = dt_arr
                return t - 0.5 * bw, t + 0.5 * bw, bw
        est = float(np.median(np.diff(t))) if n >= 2 else 1.0
        bw = np.full_like(t, est, dtype=float)
        return t - 0.5 * bw, t + 0.5 * bw, bw

    @property
    def mean_rate(self) -> Optional[float]:
        if self.rate is None or len(self.rate) == 0:
            return None
        return float(np.mean(self.rate))

    @property
    def mean_counts(self) -> Optional[float]:
        if self.counts is None or len(self.counts) == 0:
            return None
        return float(np.mean(self.counts))

    @property
    def total_counts(self) -> Optional[float]:
        if self.counts is None:
            return None
        return float(np.sum(self.counts))

    @property
    def _legacy_value(self) -> Optional[np.ndarray]:
        import warnings

        warnings.warn("LightcurveData.value is deprecated; use counts or rate instead.", DeprecationWarning, stacklevel=2)
        if self.counts is not None:
            return self.counts
        if self.rate is not None and self.dt is not None and np.any(self.dt > 0):
            return self.rate * (self.dt if isinstance(self.dt, np.ndarray) else self.dt)
        return None

    @property
    def _legacy_error(self) -> Optional[np.ndarray]:
        import warnings

        warnings.warn("LightcurveData.error is deprecated; use counts_err or rate_err instead.", DeprecationWarning, stacklevel=2)
        if self.counts_err is not None:
            return self.counts_err
        if self.rate_err is not None and self.dt is not None:
            return self.rate_err * self.dt
        return None

    @property
    def _legacy_is_rate(self) -> bool:
        import warnings

        warnings.warn("LightcurveData.is_rate is deprecated; check counts/rate fields instead.", DeprecationWarning, stacklevel=2)
        return self.rate is not None and (self.counts is None or self.counts.sum() == 0)

    def apply_gti(self, inplace: bool = False) -> 'LightcurveData':
        if self.gti is None:
            return self if inplace else self._copy()
        from stingray.gti import create_gti_mask

        gti_arr = np.array(self.gti)
        bin_lo, _, _ = self._resolve_bin_geometry()
        mask = create_gti_mask(bin_lo, gti_arr, dt=self.dt if isinstance(self.dt, float) else None)
        return self.apply_mask(np.asarray(mask, dtype=bool), inplace=inplace)

    def split_by_gti(self, min_points: int = 1) -> list['LightcurveData']:
        if self.gti is None or len(self.gti) == 0:
            return [self]
        from stingray.gti import create_gti_mask

        result = []
        for start, stop in self.gti:
            gti_arr = np.array([[start, stop]])
            bin_lo, _, _ = self._resolve_bin_geometry()
            mask = create_gti_mask(bin_lo, gti_arr, dt=self.dt if isinstance(self.dt, float) else None)
            mask = np.asarray(mask, dtype=bool)
            if np.sum(mask) < min_points:
                continue
            lc_segment = self.apply_mask(mask, inplace=False)
            lc_segment.gti_start = np.array([start])
            lc_segment.gti_stop = np.array([stop])
            result.append(lc_segment)
        return result if result else [self]

    def apply_mask(self, mask: np.ndarray, inplace: bool = False, filtered_attrs: Optional[list[str]] = None) -> 'LightcurveData':
        all_array_attrs = [
            'bin_lo', 'bin_hi', 'bin_width', 'counts', 'rate', 'counts_err', 'rate_err',
            'quality', 'fracexp', 'bin_exposure'
        ]
        if filtered_attrs is None:
            filtered_attrs = all_array_attrs
        lc_new = self if inplace else self._copy()
        if lc_new.time is not None:
            lc_new.time = np.asanyarray(lc_new.time)[mask]
        if lc_new.value is not None:
            lc_new.value = np.asanyarray(lc_new.value)[mask]
        if lc_new.error is not None:
            lc_new.error = np.asanyarray(lc_new.error)[mask]
        for attr in all_array_attrs:
            val = getattr(lc_new, attr, None)
            if val is None:
                continue
            if attr not in filtered_attrs:
                setattr(lc_new, attr, None)
            else:
                try:
                    setattr(lc_new, attr, np.asanyarray(val)[mask])
                except (IndexError, TypeError):
                    pass
        try:
            lo, hi, bw = lc_new._resolve_bin_geometry()
            if lo.size > 0:
                lc_new.tstart = float(lo[0])
                lc_new.tseg = float(hi[-1] - lo[0])
                lc_new.bin_lo = lo
                lc_new.bin_hi = hi
                lc_new.bin_width = bw
                if np.allclose(bw, float(np.median(bw)), rtol=1e-8, atol=1e-12):
                    lc_new.binning = 'uniform'
                else:
                    lc_new.binning = 'variable'
        except Exception:
            pass
        return lc_new

    def join(self, other: 'LightcurveData', skip_checks: bool = False) -> 'LightcurveData':
        import copy
        import warnings

        if self.mjdref is not None and other.mjdref is not None and self.mjdref != other.mjdref:
            warnings.warn(
                f"MJDref mismatch: self={self.mjdref}, other={other.mjdref}. Converting other to self's mjdref.",
                UserWarning,
            )
            other = copy.deepcopy(other)
            assert other.mjdref is not None and self.mjdref is not None
            time_offset = (other.mjdref - self.mjdref) * 86400.0
            o_lo, o_hi, o_bw = other._resolve_bin_geometry()
            other.bin_lo = o_lo + time_offset
            other.bin_hi = o_hi + time_offset
            other.bin_width = o_bw
            if other.gti_start is not None:
                other.gti_start = other.gti_start + time_offset
            if other.gti_stop is not None:
                other.gti_stop = other.gti_stop + time_offset
            other.mjdref = self.mjdref
        if self.tstart is not None and other.tstart is not None and self.tstart < other.tstart:
            first_lc, second_lc = self, other
        else:
            first_lc, second_lc = other, self
        self_lo, _, _ = self._resolve_bin_geometry()
        other_lo, _, _ = other._resolve_bin_geometry()
        if len(np.intersect1d(self_lo, other_lo)) > 0:
            warnings.warn("The two light curves have overlapping time ranges. In overlapping regions, counts will be summed.", UserWarning)
        lo1, hi1, bw1 = first_lc._resolve_bin_geometry()
        lo2, hi2, bw2 = second_lc._resolve_bin_geometry()
        new_bin_lo = np.concatenate([lo1, lo2])
        new_bin_hi = np.concatenate([hi1, hi2])
        new_bin_width = np.concatenate([bw1, bw2])
        new_counts = np.concatenate([first_lc.counts, second_lc.counts]) if (first_lc.counts is not None and second_lc.counts is not None) else None
        new_rate = np.concatenate([first_lc.rate, second_lc.rate]) if (first_lc.rate is not None and second_lc.rate is not None) else None
        new_counts_err = np.concatenate([first_lc.counts_err, second_lc.counts_err]) if (first_lc.counts_err is not None and second_lc.counts_err is not None) else None
        new_rate_err = np.concatenate([first_lc.rate_err, second_lc.rate_err]) if (first_lc.rate_err is not None and second_lc.rate_err is not None) else None
        if first_lc.gti is not None and second_lc.gti is not None:
            from stingray.gti import join_gtis

            new_gti = join_gtis(np.array(first_lc.gti), np.array(second_lc.gti))
            new_gti_start = new_gti[:, 0]
            new_gti_stop = new_gti[:, 1]
        else:
            new_gti_start = None
            new_gti_stop = None
        lc_cls = type(self)
        lc_new = lc_cls(
            path=self.path,
            bin_lo=new_bin_lo,
            bin_hi=new_bin_hi,
            bin_width=new_bin_width,
            counts=new_counts,
            rate=new_rate,
            counts_err=new_counts_err,
            rate_err=new_rate_err,
            dt=self.dt,
            tstart=float(new_bin_lo[0]),
            gti_start=new_gti_start,
            gti_stop=new_gti_stop,
            mjdref=self.mjdref,
            timesys=self.timesys,
            exposure=self.exposure,
            header=self.header,
            meta=self.meta,
            headers_dump=self.headers_dump,
            columns=self.columns,
        )
        lc_new.binning = 'variable' if (new_bin_width.size > 1 and not np.allclose(new_bin_width, np.median(new_bin_width))) else 'uniform'
        lc_new.tseg = float(new_bin_hi[-1] - new_bin_lo[0]) if len(new_bin_hi) > 0 else None
        return lc_new

    def truncate(self, tmin: Optional[float] = None, tmax: Optional[float] = None) -> 'LightcurveData':
        lo, hi, _ = self._resolve_bin_geometry()
        tmin_val = tmin if tmin is not None else (lo[0] if len(lo) > 0 else 0.0)
        tmax_val = tmax if tmax is not None else (hi[-1] if len(hi) > 0 else float(np.inf))
        mask = (lo >= tmin_val) & (hi <= tmax_val)
        lc_truncated = self.apply_mask(mask, inplace=False)
        if lc_truncated.gti is not None:
            gti_filtered = [
                (max(float(s), tmin_val), min(float(e), tmax_val))
                for s, e in lc_truncated.gti
                if min(float(e), tmax_val) > max(float(s), tmin_val)
            ]
            if gti_filtered:
                lc_truncated.gti_start = np.array([s for s, _ in gti_filtered])
                lc_truncated.gti_stop = np.array([e for _, e in gti_filtered])
            else:
                lc_truncated.gti_start = None
                lc_truncated.gti_stop = None
        return lc_truncated

    def sort(self, inplace: bool = False) -> 'LightcurveData':
        lo, _, _ = self._resolve_bin_geometry()
        sort_idx = np.argsort(lo)
        mask = np.zeros(len(lo), dtype=bool)
        mask[sort_idx] = True
        return self.apply_mask(mask, inplace=inplace)

    def _copy(self) -> 'LightcurveData':
        import copy

        return copy.copy(self)

    def slice(self, tmin: Optional[Union[float, Any]] = None, tmax: Optional[Union[float, Any]] = None) -> 'LightcurveData':
        from .ops import slice_lightcurve

        return cast(LightcurveData, slice_lightcurve(cast(Any, self), tmin=tmin, tmax=tmax))

    def rebin(
        self,
        binsize: float,
        method: Literal['auto', 'sum', 'mean'] = 'auto',
        *,
        align_ref: Optional[float] = None,
        empty_bin: Literal['zero', 'nan'] = 'zero',
    ) -> 'LightcurveData':
        from .ops import rebin_lightcurve

        return cast(LightcurveData, rebin_lightcurve(cast(Any, self), binsize=binsize, method=method, align_ref=align_ref, empty_bin=empty_bin))

    def __sub__(
        self,
        other: 'LightcurveData',
        *,
        ratio: Optional[float] = None,
        use_exposure_weighted_ratio: bool = True,
    ) -> 'LightcurveData':
        from jinwu.core.datasets import netdata

        result = netdata(cast(Any, self), cast(Any, other), ratio=ratio, use_exposure_weighted_ratio=use_exposure_weighted_ratio)
        if getattr(result, "kind", None) != "lc":
            raise TypeError(f"netdata must return LightcurveData-like object, got {type(result).__name__}")
        return cast(LightcurveData, result)

    def __add__(self, other: 'LightcurveData'):
        from jinwu.core.datasets import LightcurveDataset

        return LightcurveDataset(data=cast(Any, [self, other]))

    def plot(self, **kwargs):
        from .plot import plot_lightcurve

        return plot_lightcurve(self, **kwargs)


class EventData(EventDataBase):
    """Concrete event data class with local behavior implementation."""

    def validate(self) -> ValidationReport:  # type: ignore[override]
        rpt = _validate_time_series_like(self)
        colset = {c.upper() for c in self.columns}
        if "TIME" not in colset:
            rpt.add('ERROR', 'MISSING_COLUMN', "EVENTS missing TIME column")
        rpt.ok = len(rpt.errors()) == 0
        return rpt

    @property
    def n(self) -> int:
        return len(self.time) if self.time is not None else 0

    @property
    def duration(self) -> Optional[float]:
        if self.time.size:
            return float(self.time.max() - self.time.min())
        return None

    @property
    def absolute_time(self) -> np.ndarray:
        return self.time + self.timezero

    @property
    def gti_exposure(self) -> Optional[float]:
        if self.gti_start is None or self.gti_stop is None:
            return None
        return float(np.sum(self.gti_stop - self.gti_start))

    def get_time_object(self, index: Optional[int] = None) -> Optional[Any]:
        if self.timezero_obj is None:
            return None
        from astropy.time import TimeDelta

        absolute_times = self.absolute_time
        dt = TimeDelta(absolute_times[index], format='sec') if index is not None else TimeDelta(absolute_times, format='sec')
        return self.timezero_obj + dt

    def get_energy(self, rmf: Optional[RmfData] = None) -> Optional[np.ndarray]:
        if getattr(self, 'energy', None) is not None:
            return self.energy
        if rmf is not None:
            try:
                dm = np.asarray(rmf.dense_matrix) if hasattr(rmf, 'dense_matrix') else np.asarray(rmf.rebuild_dense())
                if getattr(rmf, 'energ_lo', None) is not None and getattr(rmf, 'energ_hi', None) is not None:
                    e_centers = 0.5 * (np.asarray(rmf.energ_lo) + np.asarray(rmf.energ_hi))
                elif getattr(rmf, 'e_min', None) is not None and getattr(rmf, 'e_max', None) is not None:
                    e_centers = 0.5 * (np.asarray(rmf.e_min) + np.asarray(rmf.e_max))
                else:
                    e_centers = None
                if dm.ndim == 2:
                    if e_centers is not None and dm.shape[0] == e_centers.size and dm.shape[1] != e_centers.size:
                        dm_t = dm.T
                    else:
                        dm_t = dm
                else:
                    dm_t = np.asarray(dm)
                if getattr(self, 'channel', None) is not None:
                    ch_ev = np.asarray(self.channel).astype(int)
                elif getattr(self, 'pi', None) is not None:
                    ch_ev = np.asarray(self.pi).astype(int)
                else:
                    ch_ev = None
                if (ch_ev is not None) and (e_centers is not None):
                    from ..ftools.rmf_mapping import map_channels_to_energy

                    mapped = map_channels_to_energy(dm_t, e_centers, ch_ev, method='expected')
                    self.energy = np.asarray(mapped, dtype=float)
                    return self.energy
            except Exception:
                pass
        if getattr(self, 'ebounds', None) is not None:
            ebounds_tuple = self.ebounds
            if ebounds_tuple is not None and len(ebounds_tuple) == 3:
                ch, e_lo, e_hi = ebounds_tuple
                emid = 0.5 * (np.asarray(e_lo) + np.asarray(e_hi))
                if getattr(self, 'channel', None) is not None:
                    ch_ev = np.asarray(self.channel)
                elif getattr(self, 'pi', None) is not None:
                    ch_ev = np.asarray(self.pi)
                else:
                    return None
                cmap = {int(c): float(e) for c, e in zip(np.asarray(ch, int), emid)}
                mapped = np.array([cmap.get(int(cc), np.nan) for cc in ch_ev], dtype=float)
                self.energy = mapped
                return self.energy
        return None

    def plot(
        self,
        ax: Optional[Any] = None,
        *,
        bins: Union[int, Tuple[int, int]] = 300,
        cmap: str = 'viridis',
        title: Optional[str] = None,
        invert_ra: bool = True,
        show_grid: bool = True,
        show_colorbar: bool = True,
        **kwargs,
    ):
        import matplotlib.pyplot as _plt

        def _get_raw_col(name: Optional[str]) -> Optional[np.ndarray]:
            if name is None or self.raw_columns is None:
                return None
            if name in self.raw_columns:
                return np.asarray(self.raw_columns[name], float)
            return None

        def _find_name(cands: Tuple[str, ...]) -> Optional[str]:
            if self.raw_columns is None:
                return None
            upper_to_orig = {str(k).upper(): str(k) for k in self.raw_columns.keys()}
            for c in cands:
                if c.upper() in upper_to_orig:
                    return upper_to_orig[c.upper()]
            return None

        def _to_ra_dec_from_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            hdr = self.header or {}
            try:
                from astropy.wcs import WCS

                w = WCS(hdr)
                if getattr(w, 'has_celestial', False):
                    world = np.asarray(w.all_pix2world(np.column_stack([x, y]), 1), float)
                    ra_w = np.asarray(world[..., 0], float)
                    dec_w = np.asarray(world[..., 1], float)
                    if np.isfinite(ra_w).any() and np.isfinite(dec_w).any():
                        return ra_w, dec_w
            except Exception:
                pass

            def _pick_float(keys: Tuple[str, ...], default: Optional[float] = None) -> Optional[float]:
                for k in keys:
                    if k in hdr:
                        try:
                            return float(hdr[k])
                        except Exception:
                            continue
                return default

            crval1 = _pick_float(('TCRVL11', 'CRVAL1', 'TCRVL1'))
            crval2 = _pick_float(('TCRVL12', 'CRVAL2', 'TCRVL2'))
            crpix1 = _pick_float(('TCRPX11', 'CRPIX1', 'TCRPX1'), default=0.0)
            crpix2 = _pick_float(('TCRPX12', 'CRPIX2', 'TCRPX2'), default=0.0)
            cdelt1 = _pick_float(('TCDLT11', 'CDELT1', 'TCDLT1'))
            cdelt2 = _pick_float(('TCDLT12', 'CDELT2', 'TCDLT2'))
            if (crval1 is None) or (crval2 is None) or (cdelt1 is None) or (cdelt2 is None):
                raise ValueError('Cannot build RA/DEC: missing RA/DEC columns and WCS keywords')
            cos_dec0 = np.cos(np.deg2rad(crval2))
            if np.abs(cos_dec0) < 1e-8:
                cos_dec0 = 1.0
            crpix1_f = float(0.0 if crpix1 is None else crpix1)
            crpix2_f = float(0.0 if crpix2 is None else crpix2)
            ra_l = crval1 + (x - crpix1_f) * float(cdelt1) / cos_dec0
            dec_l = crval2 + (y - crpix2_f) * float(cdelt2)
            return np.asarray(ra_l, float), np.asarray(dec_l, float)

        ra_name = self.colmap.get('ra') if self.colmap is not None else None
        dec_name = self.colmap.get('dec') if self.colmap is not None else None
        ra = _get_raw_col(ra_name)
        dec = _get_raw_col(dec_name)
        if (ra is None) or (dec is None):
            if ra is None:
                ra = _get_raw_col(_find_name(('RA', 'RA_OBJ', 'RAX', 'RA_DEG')))
            if dec is None:
                dec = _get_raw_col(_find_name(('DEC', 'DEC_OBJ', 'DECX', 'DEC_DEG')))
        if (ra is None) or (dec is None):
            x = np.asarray(self.x, float) if self.x is not None else None
            y = np.asarray(self.y, float) if self.y is not None else None
            if (x is None) or (y is None):
                key_x = self.colmap.get('x') if self.colmap is not None else None
                key_y = self.colmap.get('y') if self.colmap is not None else None
                x = _get_raw_col(key_x)
                y = _get_raw_col(key_y)
            if (x is None) or (y is None):
                raise ValueError('No RA/DEC columns and no X/Y columns available for sky plotting')
            ra, dec = _to_ra_dec_from_xy(np.asarray(x, float), np.asarray(y, float))
        ra = np.asarray(ra, float)
        dec = np.asarray(dec, float)
        mask = np.isfinite(ra) & np.isfinite(dec)
        if not np.any(mask):
            raise ValueError('No finite RA/DEC values available for plotting')
        ra = ra[mask]
        dec = dec[mask]
        ax = ax or _plt.gca()
        assert ax is not None
        h = ax.hist2d(ra, dec, bins=bins, cmap=cmap, **kwargs)
        if show_colorbar:
            _plt.colorbar(h[3], ax=ax, label='Counts')
        ax.set_xlabel('Right ascension (deg)')
        ax.set_ylabel('Declination (deg)')
        if invert_ra:
            ax.invert_xaxis()
        if title is None:
            fname = Path(str(getattr(self, 'path', ''))).name
            title = f'{fname} sky image' if fname else 'Sky image'
        ax.set_title(title)
        if show_grid:
            ax.grid(alpha=0.3, ls='--')
        return ax

    def slice(
        self,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        *,
        pi_min: Optional[int] = None,
        pi_max: Optional[int] = None,
        ch_min: Optional[int] = None,
        ch_max: Optional[int] = None,
    ) -> 'EventData':
        from .ops import slice_events

        return cast(EventData, slice_events(cast(Any, self), tmin=tmin, tmax=tmax, pi_min=pi_min, pi_max=pi_max, ch_min=ch_min, ch_max=ch_max))

    def rebin(self, binsize: float, *, tmin: Optional[float] = None, tmax: Optional[float] = None) -> 'LightcurveData':
        from .ops import rebin_events_to_lightcurve

        return cast(LightcurveData, rebin_events_to_lightcurve(cast(Any, self), binsize=binsize, tmin=tmin, tmax=tmax))

    def xselect(self) -> 'Any':
        from .xselect import XSelectSession

        if self._xselect_session is None:
            self._xselect_session = cast(Any, XSelectSession(cast(Any, self)))
        return self._xselect_session

    def filter_time(self, tmin: Optional[float] = None, tmax: Optional[float] = None) -> 'EventData':
        session = self.xselect()
        session.apply_time(tmin=tmin, tmax=tmax)
        cur = getattr(session, 'current', None)
        return cur if cur is not None else self

    def filter_region(self, region) -> 'EventData':
        session = self.xselect()
        session.apply_region(region)
        cur = getattr(session, 'current', None)
        return cur if cur is not None else self

    def filter_energy(self, pi_min: Optional[int] = None, pi_max: Optional[int] = None) -> 'EventData':
        session = self.xselect()
        session.apply_energy(pi_min=pi_min, pi_max=pi_max)
        cur = getattr(session, 'current', None)
        return cur if cur is not None else self

    def _current_for_products(self) -> 'EventData':
        sess = getattr(self, '_xselect_session', None)
        if sess is not None:
            cur = getattr(sess, 'current', None)
            if cur is not None:
                return cur
        return self

    def extract_spectrum(self, **kwargs) -> PhaData:
        from . import xselect

        src = self._current_for_products()
        if src is not self:
            return src.extract_spectrum(**kwargs)
        return cast(PhaData, xselect.extract_spectrum(cast(Any, self), **kwargs))

    def extract_curve(self, binsize: float, **kwargs) -> 'LightcurveData':
        from . import xselect

        src = self._current_for_products()
        if src is not self:
            return src.extract_curve(binsize=binsize, **kwargs)
        return cast(LightcurveData, xselect.extract_curve(cast(Any, self), binsize=binsize, **kwargs))

    def extract_image(self, **kwargs):
        from . import xselect

        src = self._current_for_products()
        if src is not self:
            return src.extract_image(**kwargs)
        return xselect.extract_image(cast(Any, self), **kwargs)

    def save(self, outpath: str | Path, kind: str = 'evt', overwrite: bool = False, **kwargs) -> Path:
        src = self._current_for_products()
        if src is not self:
            return src.save(outpath, kind=kind, overwrite=overwrite, **kwargs)
        outp = Path(outpath)
        if kind == 'evt':
            from astropy.io import fits
            from astropy.table import Table
            if getattr(self, 'raw_columns', None):
                try:
                    t = Table(self.raw_columns)
                except Exception:
                    cols = {'TIME': self.time}
                    if getattr(self, 'x', None) is not None:
                        cols['X'] = getattr(self, 'x')
                    if getattr(self, 'y', None) is not None:
                        cols['Y'] = getattr(self, 'y')
                    if getattr(self, 'energy', None) is not None:
                        cols['ENERGY'] = getattr(self, 'energy')
                    t = Table(cols)
            else:
                cols = {'TIME': self.time}
                if getattr(self, 'x', None) is not None:
                    cols['X'] = getattr(self, 'x')
                if getattr(self, 'y', None) is not None:
                    cols['Y'] = getattr(self, 'y')
                if getattr(self, 'energy', None) is not None:
                    cols['ENERGY'] = getattr(self, 'energy')
                t = Table(cols)
            t.write(outp, format='fits', overwrite=overwrite)
            try:
                with fits.open(outp, mode='update') as hdul:
                    if getattr(self, 'headers_dump', None) is not None and isinstance(self.headers_dump, FitsHeaderDump):
                        if getattr(self.headers_dump, 'primary', None) is not None:
                            for k, v in (self.headers_dump.primary or {}).items():
                                try:
                                    prih = cast(Any, hdul[0])
                                    prih.header[k] = v
                                except Exception:
                                    continue
                    if getattr(self, 'header', None) is not None and len(hdul) > 1:
                        tbl = cast(Any, hdul[1])
                        for k, v in (self.header or {}).items():
                            if k in ('TFIELDS', 'TTYPE1', 'TFORM1', 'XTENSION', 'BITPIX', 'NAXIS'):
                                continue
                            try:
                                tbl.header[k] = v
                            except Exception:
                                continue
                    hdul.flush()
            except Exception:
                pass
            return outp
        if kind == 'lc':
            lc = self.extract_curve(**kwargs)
            from . import xselect

            return xselect.write_curve(cast(Any, lc), outp, overwrite=overwrite)
        if kind == 'pha':
            pha = self.extract_spectrum(**kwargs)
            from . import xselect

            return xselect.write_pha(cast(Any, pha), outp, overwrite=overwrite)
        if kind == 'img':
            img, xedges, yedges = self.extract_image(**kwargs)
            from . import xselect

            return xselect.write_image(img, xedges, yedges, outp, overwrite=overwrite)
        raise ValueError('unknown kind: ' + str(kind))

    def clear_region(self, *, use_original: bool = True) -> 'EventData':
        sess = getattr(self, '_xselect_session', None)
        if sess is not None:
            sess.clear_region()
            return self._current_for_products()
        src = None
        if use_original and getattr(self, '_original_events', None) is not None:
            src = getattr(self, '_original_events')
        elif getattr(self, 'path', None) is not None:
            from .io import read_evt

            src = read_evt(self.path)
        else:
            raise ValueError('No original events available to clear region')
        tmin = None
        tmax = None
        time_arr = getattr(self, 'time', None)
        if time_arr is not None and isinstance(time_arr, np.ndarray) and time_arr.size:
            t = np.asarray(time_arr, dtype=float)
            tmin = float(t.min())
            tmax = float(t.max())
        pi_min = None
        pi_max = None
        if getattr(self, 'pi', None) is not None:
            arr = np.asarray(self.pi, dtype=int)
            if arr.size:
                pi_min = int(arr.min())
                pi_max = int(arr.max())
        elif getattr(self, 'channel', None) is not None:
            arr = np.asarray(self.channel, dtype=int)
            if arr.size:
                pi_min = int(arr.min())
                pi_max = int(arr.max())
        from . import xselect as _xsel

        return cast(EventData, _xsel.select_events(cast(Any, src), tmin=tmin, tmax=tmax, pi_min=pi_min, pi_max=pi_max))

    def clear_time(self, *, use_original: bool = True) -> 'EventData':
        sess = getattr(self, '_xselect_session', None)
        if sess is not None:
            sess.clear_time()
            return self._current_for_products()
        src = None
        if use_original and getattr(self, '_original_events', None) is not None:
            src = getattr(self, '_original_events')
        elif getattr(self, 'path', None) is not None:
            from .io import read_evt

            src = read_evt(self.path)
        else:
            raise ValueError('No original events available to clear time')
        pi_min = None
        pi_max = None
        if getattr(self, 'pi', None) is not None:
            arr = np.asarray(self.pi, dtype=int)
            if arr.size:
                pi_min = int(arr.min())
                pi_max = int(arr.max())
        elif getattr(self, 'channel', None) is not None:
            arr = np.asarray(self.channel, dtype=int)
            if arr.size:
                pi_min = int(arr.min())
                pi_max = int(arr.max())
        from . import xselect as _xsel

        return cast(EventData, _xsel.select_events(cast(Any, src), region=None, pi_min=pi_min, pi_max=pi_max))

    def clear_energy(self, *, use_original: bool = True) -> 'EventData':
        sess = getattr(self, '_xselect_session', None)
        if sess is not None:
            sess.clear_energy()
            return self._current_for_products()
        src = None
        if use_original and getattr(self, '_original_events', None) is not None:
            src = getattr(self, '_original_events')
        elif getattr(self, 'path', None) is not None:
            from .io import read_evt

            src = read_evt(self.path)
        else:
            raise ValueError('No original events available to clear energy')
        tmin = None
        tmax = None
        time_arr = getattr(self, 'time', None)
        if time_arr is not None and isinstance(time_arr, np.ndarray) and time_arr.size:
            t = np.asarray(time_arr, dtype=float)
            tmin = float(t.min())
            tmax = float(t.max())
        from . import xselect as _xsel

        return cast(EventData, _xsel.select_events(cast(Any, src), tmin=tmin, tmax=tmax, region=None))

    def clear_all(self, *, use_original: bool = True) -> 'EventData':
        sess = getattr(self, '_xselect_session', None)
        if sess is not None:
            sess.clear_all()
            return self._current_for_products()
        if use_original and getattr(self, '_original_events', None) is not None:
            return getattr(self, '_original_events')
        if getattr(self, 'path', None) is not None:
            from .io import read_evt

            return read_evt(self.path)
        raise ValueError('No original events available to clear all')

__all__ = [
    'ArfData',
    'RmfData',
    'PhaData',
    'LightcurveData',
    'EventData',
]
