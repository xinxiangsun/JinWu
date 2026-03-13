from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union, cast, overload

import numpy as np
from astropy.io import fits

from .base import ChannelBand, EnergyBand, FitsHeaderDump, HduHeader, OgipMeta, RegionArea
from .data import ArfData, EventData, LightcurveData, PhaData, RmfData
from .ogip import ValidationReport


def band_from_arf_bins(arf_path: str | Path, bin_lo: int = 81, bin_hi: int = 780) -> EnergyBand:
    with fits.open(arf_path) as h:
        hd = cast(Any, h["SPECRESP"])
        d = hd.data
        elo = np.asarray(d["ENERG_LO"], float)
        ehi = np.asarray(d["ENERG_HI"], float)
    i0 = max(0, int(bin_lo) - 1)
    i1 = min(ehi.size - 1, int(bin_hi) - 1)
    emin = float(elo[i0])
    emax = float(ehi[i1])
    return EnergyBand(emin=emin, emin_unit="keV", emax=emax, emax_unit="keV")


def channel_mask_from_ebounds(
    ebounds: tuple[np.ndarray, np.ndarray, np.ndarray],
    band: EnergyBand,
    ch_band: Optional[ChannelBand] = None,
) -> np.ndarray:
    ch, e_lo, e_hi = ebounds
    mask = (e_hi > float(band.emin)) & (e_lo < float(band.emax))
    if ch_band is not None:
        mask &= (ch >= int(ch_band.ch_lo)) & (ch <= int(ch_band.ch_hi))
    return mask


def _combine_mjdref(header: Dict[str, Any]) -> Optional[float]:
    if header is None:
        return None
    if ("MJDREFI" in header) or ("MJDREFF" in header):
        mjdi = float(header.get("MJDREFI", 0.0))
        mjdf = float(header.get("MJDREFF", 0.0))
        return mjdi + mjdf
    if "MJDREF" in header:
        try:
            return float(header["MJDREF"])
        except Exception:
            return None
    return None


def _first_non_empty(keys: list[str], *headers: Dict[str, Any]) -> Optional[Any]:
    for key in keys:
        for hdr in headers:
            if hdr is None:
                continue
            if key in hdr and hdr[key] not in (None, "", " "):
                return hdr[key]
    return None


def _collect_headers_dump(hdul: fits.HDUList) -> FitsHeaderDump:
    hdr0 = cast(Any, getattr(hdul[0], 'header', {})) if len(hdul) > 0 else {}
    primary = dict(hdr0) if hdr0 else {}
    exts: list[HduHeader] = []
    for hdu in hdul[1:]:
        hdu_any = cast(Any, hdu)
        name = str(getattr(hdu, 'name', '') or '')
        ver_val = cast(Any, hdu_any.header).get('EXTVER', None)
        try:
            ver = int(ver_val) if ver_val is not None else None
        except Exception:
            ver = None
        exts.append(HduHeader(name=name, ver=ver, header=dict(cast(Any, hdu_any.header))))
    return FitsHeaderDump(primary=primary, extensions=exts)


def _build_meta(hdul: fits.HDUList, prefer_header: Optional[Dict[str, Any]]) -> OgipMeta:
    dump = _collect_headers_dump(hdul)
    primary = dump.primary
    other_ext_headers = [x.header for x in dump.extensions]
    telescop = _first_non_empty(["TELESCOP"], *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr))
    instrume = _first_non_empty(["INSTRUME"], *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr))
    detnam = _first_non_empty(["DETNAM", "DETNAME"], *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr))
    timesys = _first_non_empty(["TIMESYS"], *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr))
    timeunit = _first_non_empty(["TIMEUNIT"], *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr))

    mjdref = None
    for hdr in [prefer_header, primary] + other_ext_headers:
        if hdr is None:
            continue
        mjdref = _combine_mjdref(hdr)
        if mjdref is not None:
            break

    tstart = None
    tstop = None
    for hdr in [prefer_header, primary] + other_ext_headers:
        if hdr is None:
            continue
        if (tstart is None) and ("TSTART" in hdr):
            try:
                tstart = float(hdr["TSTART"])
            except Exception:
                pass
        if (tstop is None) and ("TSTOP" in hdr):
            try:
                tstop = float(hdr["TSTOP"])
            except Exception:
                pass
        if (tstart is not None) and (tstop is not None):
            break

    obj = _first_non_empty(["OBJECT"], *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr))
    obs_id = _first_non_empty(["OBS_ID", "OBS_ID"], *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr))

    binsize_val = _first_non_empty(
        ["BIN_SIZE", "BINSIZE", "TIMEDEL", "DELTAT", "TBIN", "TIMEBIN"],
        *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr),
    )
    try:
        binsize = float(binsize_val) if binsize_val is not None else None
    except Exception:
        binsize = None

    timezero_val = _first_non_empty(["TIMEZERO"], *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr))
    try:
        timezero = float(timezero_val) if timezero_val is not None else None
    except Exception:
        timezero = None

    trefpos_val = _first_non_empty(["TREFPOS", "TREFDIR"], *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr))
    try:
        trefpos = str(trefpos_val) if trefpos_val is not None else None
    except Exception:
        trefpos = None

    dateobs_val = _first_non_empty(["DATE-OBS", "DATE_OBS"], *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr))
    try:
        dateobs = str(dateobs_val) if dateobs_val is not None else None
    except Exception:
        dateobs = None

    return OgipMeta(
        telescop=str(telescop) if telescop is not None else None,
        instrume=str(instrume) if instrume is not None else None,
        detnam=str(detnam) if detnam is not None else None,
        timesys=str(timesys) if timesys is not None else None,
        timeunit=str(timeunit) if timeunit is not None else None,
        mjdref=float(mjdref) if mjdref is not None else None,
        tstart=float(tstart) if tstart is not None else None,
        tstop=float(tstop) if tstop is not None else None,
        object=str(obj) if obj is not None else None,
        obs_id=str(obs_id) if obs_id is not None else None,
        binsize=binsize,
        timezero=timezero,
        trefpos=trefpos,
        dateobs=dateobs,
    )


def _mission_timezero_object(telescop: Optional[str], timezero: float, *, allow_unix_fallback: bool = False):
    if telescop is None:
        return None
    try:
        from .time import Time

        format_map = {
            'FERMI': 'fermi',
            'EP': 'ep',
            'LEIA': 'leia',
            'GECAM': 'gecam',
            'HXMT': 'hxmt',
            'SWIFT': 'swift',
            'GRID': 'grid',
            'MAXI': 'maxi',
            'SUZAKU': 'suzaku',
            'XMM': 'newton',
            'NEWTON': 'newton',
            'XRISM': 'xrism',
        }
        met_format = None
        for key, fmt in format_map.items():
            if key in telescop:
                met_format = fmt
                break
        if met_format is not None:
            return Time(timezero, format=met_format)
        if allow_unix_fallback:
            try:
                return Time(timezero, format='unix', scale='utc')
            except Exception:
                return None
        raise RuntimeError("未知望远镜类型,请将header中的相关关键字发送给作者以方便添加" + telescop)
    except ImportError:
        return None


def _extract_gti(hdul: fits.HDUList) -> Optional[list[tuple[float, float]]]:
    if hdul is None:
        return None
    for hdu in hdul:
        hdr = getattr(hdu, 'header', {})
        name = (hdr.get('EXTNAME') or '').upper() if hdr else ''
        if name == 'GTI' or getattr(hdu, 'name', '').upper() == 'GTI':
            data = getattr(hdu, 'data', None)
            if data is None:
                continue
            cols = getattr(data, 'columns', None)
            colnames = [n.upper() for n in (cols.names if cols is not None else [])]
            start_col = None
            stop_col = None
            for n in ['START', 'TSTART']:
                if n in colnames:
                    start_col = n
                    break
            for n in ['STOP', 'TSTOP']:
                if n in colnames:
                    stop_col = n
                    break
            if start_col and stop_col:
                arr_start = data[start_col]
                arr_stop = data[stop_col]
                try:
                    return [(float(s), float(e)) for s, e in zip(arr_start, arr_stop)]
                except (TypeError, ValueError):
                    return None
    return None


def _load_regions(hdul: fits.HDUList) -> Optional[RegionArea]:
    reg_hdu = None
    for ext in hdul:
        name = (getattr(ext, 'name', '') or '').upper()
        if name == 'REG00101':
            reg_hdu = ext
            break
    if reg_hdu is None or getattr(reg_hdu, 'data', None) is None:
        return None
    reg_any = cast(Any, reg_hdu)
    if getattr(reg_any, 'data', None) is None:
        return None
    data = reg_any.data
    cols = getattr(data, 'columns', None)
    colnames = [str(n).upper() for n in (cols.names if cols is not None else [])]

    def _get_col(name_variants: list[str]) -> Optional[str]:
        for nn in name_variants:
            if nn in colnames:
                return nn
        return None

    shape_col = _get_col(['SHAPE'])
    component_col = _get_col(['COMPONENT'])
    r_col = _get_col(['R', 'RADIUS', 'R0'])
    rin_col = _get_col(['R_IN', 'RIN', 'R1'])
    rout_col = _get_col(['R_OUT', 'ROUT', 'R2'])
    nrows = len(data)
    rows_info: list[Dict[str, Any]] = []

    def _as_float(v: Any) -> Optional[float]:
        try:
            return float(v)
        except Exception:
            try:
                return float(v[0])
            except Exception:
                return None

    for i in range(nrows):
        shape_val = ''
        if shape_col and shape_col in colnames:
            try:
                shape_val = str(data[shape_col][i]).upper().strip()
            except Exception:
                shape_val = ''
        comp_val = None
        if component_col and component_col in colnames:
            try:
                comp_val = int(data[component_col][i])
            except Exception:
                tmp = _as_float(data[component_col][i])
                comp_val = int(tmp) if tmp is not None else None
        area = None
        if (shape_val == 'CIRCLE') or (shape_col is None and r_col is not None and (rin_col is None or rout_col is None)):
            rv = data[r_col][i] if r_col else None
            r = _as_float(rv) if rv is not None else None
            if r is not None and r > 0:
                area = np.pi * (r ** 2)
        elif shape_val == 'ANNULUS' or (rin_col is not None and rout_col is not None):
            rin = _as_float(data[rin_col][i]) if rin_col else None
            rout = _as_float(data[rout_col][i]) if rout_col else None
            if rin is None and rout is None and r_col:
                rv = data[r_col][i]
                try:
                    rin = float(rv[0])
                    rout = float(rv[1])
                except Exception:
                    rin = None
                    rout = None
            if (rin is not None) and (rout is not None) and rout > rin:
                area = np.pi * (rout ** 2 - rin ** 2)
        rows_info.append({'shape': shape_val, 'area': area, 'component': comp_val, 'role': 'unknown'})
    if not rows_info:
        return None
    annuli = [r for r in rows_info if r['shape'] == 'ANNULUS' and r['area']]
    circles = [r for r in rows_info if r['shape'] == 'CIRCLE' and r['area']]
    if component_col and any(r['component'] is not None for r in rows_info):
        for r in rows_info:
            comp = r['component']
            if comp == 1:
                r['role'] = 'source'
            elif comp is not None:
                r['role'] = 'background'
    if not component_col:
        if annuli:
            for r in annuli:
                if r['role'] == 'unknown':
                    r['role'] = 'background'
            if circles:
                for r in circles:
                    if r['role'] == 'unknown':
                        r['role'] = 'source'
        elif len(circles) >= 1:
            sorted_c = sorted(circles, key=lambda x: x['area'] or 0)
            if sorted_c:
                if sorted_c[0]['role'] == 'unknown':
                    sorted_c[0]['role'] = 'source'
                for r in sorted_c[1:]:
                    if r['role'] == 'unknown':
                        r['role'] = 'background'
    if len(rows_info) == 1 and rows_info[0]['role'] == 'unknown':
        rows_info[0]['role'] = 'source'

    def _normalize_role(role: str) -> Literal['src', 'bkg', 'unk']:
        if role == 'source':
            return 'src'
        if role == 'background':
            return 'bkg'
        return 'unk'

    regions = [
        RegionArea(role=_normalize_role(cast(Any, r['role'])), shape=r['shape'] or None, area=r['area'], component=r['component'])
        for r in rows_info
    ]
    if not regions:
        return None
    src_region = next((r for r in regions if r.role == 'src'), None)
    if src_region is not None:
        return src_region
    bkg_region = next((r for r in regions if r.role == 'bkg'), None)
    if bkg_region is not None:
        return bkg_region
    return regions[0]


class OgipArfReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        self._data: Optional[ArfData] = None

    def read(self) -> ArfData:
        with fits.open(self.path) as h:
            hd = cast(Any, h["SPECRESP"])
            d = hd.data
            columns = tuple(getattr(d.columns, 'names', ()) or ())
            energ_lo = np.asarray(d["ENERG_LO"], float)
            energ_hi = np.asarray(d["ENERG_HI"], float)
            specresp = np.asarray(d["SPECRESP"], float)
            header = dict(cast(Any, hd.header))
            headers_dump = _collect_headers_dump(h)
            meta = _build_meta(h, header)

        self._data = ArfData(
            path=self.path,
            energ_lo=energ_lo,
            energ_hi=energ_hi,
            specresp=specresp,
            columns=columns,
            header=header,
            meta=meta,
            headers_dump=headers_dump,
        )
        return self._data

    def validate(self) -> ValidationReport:
        if self._data is None:
            self.read()
        assert self._data is not None
        return self._data.validate()


class OgipRmfReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        self._data: Optional[RmfData] = None

    def read(self) -> RmfData:
        with fits.open(self.path) as h:
            hm = cast(Any, h["MATRIX"])
            dm = hm.data
            matrix_columns = tuple(getattr(dm.columns, 'names', ()) or ())
            energ_lo = np.asarray(dm["ENERG_LO"], float)
            energ_hi = np.asarray(dm["ENERG_HI"], float)
            header = dict(cast(Any, hm.header))
            headers_dump = _collect_headers_dump(h)
            meta = _build_meta(h, header)

            matrix = np.asarray(dm["MATRIX"], dtype=object)
            n_grp = np.asarray(dm["N_GRP"]) if "N_GRP" in matrix_columns else None
            f_chan = np.asarray(dm["F_CHAN"]) if "F_CHAN" in matrix_columns else None
            n_chan = np.asarray(dm["N_CHAN"]) if "N_CHAN" in matrix_columns else None

            channel = None
            e_min = None
            e_max = None
            ebounds_columns: tuple[str, ...] = ()
            if "EBOUNDS" in h:
                de = cast(Any, h["EBOUNDS"]).data
                ebounds_columns = tuple(getattr(de.columns, 'names', ()) or ())
                channel = np.asarray(de["CHANNEL"], int)
                e_min = np.asarray(de["E_MIN"], float)
                e_max = np.asarray(de["E_MAX"], float)

        columns = matrix_columns + tuple(name for name in ebounds_columns if name not in matrix_columns)
        self._data = RmfData(
            path=self.path,
            energ_lo=energ_lo,
            energ_hi=energ_hi,
            n_grp=n_grp,
            f_chan=f_chan,
            n_chan=n_chan,
            matrix=matrix,
            channel=channel,
            e_min=e_min,
            e_max=e_max,
            columns=columns,
            header=header,
            meta=meta,
            headers_dump=headers_dump,
        )
        return self._data

    def validate(self) -> ValidationReport:
        if self._data is None:
            self.read()
        assert self._data is not None
        return self._data.validate()


class OgipPhaReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        self._data: Optional[PhaData] = None

    def read(self) -> PhaData:
        with fits.open(self.path) as h:
            hs = cast(Any, h["SPECTRUM"])
            ds = hs.data
            spectrum_columns = tuple(getattr(ds.columns, 'names', ()) or ())
            channels = np.asarray(ds["CHANNEL"], int)
            counts = np.asarray(ds["COUNTS"], float)
            stat_err = np.asarray(ds["STAT_ERR"], float) if "STAT_ERR" in spectrum_columns else None
            header_map = cast(Any, hs.header)
            exposure = float(header_map.get("EXPOSURE", header_map.get("EXPTIME", np.nan)))
            backscal = float(header_map.get("BACKSCAL")) if "BACKSCAL" in header_map else None
            areascal = float(header_map.get("AREASCAL")) if "AREASCAL" in header_map else None
            quality = np.asarray(ds["QUALITY"], int) if "QUALITY" in spectrum_columns else None
            grouping = np.asarray(ds["GROUPING"], int) if "GROUPING" in spectrum_columns else None

            ebounds = None
            ebounds_columns: tuple[str, ...] = ()
            if "EBOUNDS" in h:
                de = cast(Any, h["EBOUNDS"]).data
                ebounds_columns = tuple(getattr(de.columns, 'names', ()) or ())
                ebounds = (
                    np.asarray(de["CHANNEL"], int),
                    np.asarray(de["E_MIN"], float),
                    np.asarray(de["E_MAX"], float),
                )

            header = dict(header_map)
            headers_dump = _collect_headers_dump(h)
            meta = _build_meta(h, header)

        respfile = None
        ancrfile = None
        try:
            if 'RESPFILE' in header and header.get('RESPFILE') not in (None, '', ' '):
                respfile = str(header.get('RESPFILE'))
        except Exception:
            respfile = None
        try:
            if 'ANCRFILE' in header and header.get('ANCRFILE') not in (None, '', ' '):
                ancrfile = str(header.get('ANCRFILE'))
        except Exception:
            ancrfile = None

        columns = spectrum_columns + tuple(name for name in ebounds_columns if name not in spectrum_columns)
        self._data = PhaData(
            path=self.path,
            channels=channels,
            counts=counts,
            stat_err=stat_err,
            exposure=exposure,
            backscal=backscal,
            areascal=areascal,
            respfile=respfile,
            ancrfile=ancrfile,
            quality=quality,
            grouping=grouping,
            ebounds=ebounds,
            columns=columns,
            header=header,
            meta=meta,
            headers_dump=headers_dump,
        )
        return self._data

    def validate(self) -> ValidationReport:
        if self._data is None:
            self.read()
        assert self._data is not None
        return self._data.validate()

    def select_by_band(
        self,
        band: EnergyBand,
        rmf_chan_band: Optional[ChannelBand] = ChannelBand(51, 399),
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._data is None:
            _ = self.read()
        d = self._data
        assert d is not None
        if d.ebounds is not None:
            mask = channel_mask_from_ebounds(d.ebounds, band, rmf_chan_band)
            ch_all = d.ebounds[0]
            idx_map = {int(c): i for i, c in enumerate(d.channels)}
            sel_channels = ch_all[mask]
            sel_idx = np.array([idx_map[c] for c in sel_channels if c in idx_map], dtype=int)
            return d.channels[sel_idx], d.counts[sel_idx]
        if rmf_chan_band is None:
            return d.channels, d.counts
        mask = (d.channels >= rmf_chan_band.ch_lo) & (d.channels <= rmf_chan_band.ch_hi)
        return d.channels[mask], d.counts[mask]


class OgipLightcurveReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        self._data: Optional[LightcurveData] = None

    def read(self) -> LightcurveData:
        with fits.open(self.path) as h:
            hdu = None
            for ext in h:
                ext_any = cast(Any, ext)
                if getattr(ext_any, 'data', None) is None:
                    continue
                names = getattr(ext_any.data, 'columns', None)
                if names is not None and ('TIME' in [n.upper() for n in names.names]):
                    hdu = ext_any
                    break
            if hdu is None:
                raise ValueError('No suitable lightcurve HDU with TIME column found')
            d = cast(Any, hdu.data)
            col_names_upper = [n.upper() for n in d.columns.names]
            header = dict(cast(Any, hdu.header))
            time_raw = np.asarray(d['TIME'], dtype=float)
            timezero_raw = 0.0
            if 'TIMEZERO' in header:
                try:
                    timezero_raw = float(header['TIMEZERO'])
                except (ValueError, TypeError):
                    timezero_raw = 0.0
            time_offset = float(time_raw[0]) if len(time_raw) > 0 else 0.0
            time = time_raw - time_offset
            timezero = timezero_raw + time_offset
            telescop = None
            primary_header = dict(cast(Any, h[0]).header) if len(h) > 0 else {}
            for hdr in [header, primary_header]:
                if 'TELESCOP' in hdr:
                    telescop = str(hdr['TELESCOP']).strip().upper()
                    break
            timezero_obj = _mission_timezero_object(telescop, timezero, allow_unix_fallback=False)
            dt = None
            if 'TIMEDEL' in header:
                try:
                    dt = float(header['TIMEDEL'])
                except (ValueError, TypeError):
                    pass
            if dt is None and time.size >= 2:
                dt = float(np.median(np.diff(time)))
            if dt is None:
                raise ValueError('binsize未被正确加载')
            bin_lo = time.copy()
            bin_hi = time + dt
            rate = None
            counts = None
            is_rate = False
            value = None
            if 'RATE' in col_names_upper:
                rate = np.asarray(d['RATE'], dtype=float)
                value = rate
                is_rate = True
                if 'COUNTS' not in col_names_upper and dt > 0:
                    counts = rate * dt
            if 'COUNTS' in col_names_upper:
                counts = np.asarray(d['COUNTS'], dtype=float)
                if value is None:
                    value = counts
                    is_rate = False
                if rate is None and dt > 0:
                    rate = counts / dt
            if value is None:
                raise ValueError('Lightcurve HDU lacks RATE/COUNTS column')
            error = None
            rate_err = None
            counts_err = None
            if 'ERROR' in col_names_upper:
                error = np.asarray(d['ERROR'], dtype=float)
                if is_rate:
                    rate_err = error
                    if counts is not None and dt > 0:
                        counts_err = error * dt
                else:
                    counts_err = error
                    if rate is not None and dt > 0:
                        rate_err = error / dt
            fracexp = np.asarray(d['FRACEXP'], dtype=float) if 'FRACEXP' in col_names_upper else None
            quality = np.asarray(d['QUALITY'], dtype=int) if 'QUALITY' in col_names_upper else None
            backscal_col = d['BACKSCAL'] if 'BACKSCAL' in col_names_upper else (d['BACK_SCAL'] if 'BACK_SCAL' in col_names_upper else None)
            areascal_col = d['AREASCAL'] if 'AREASCAL' in col_names_upper else (d['AREA_SCAL'] if 'AREA_SCAL' in col_names_upper else None)
            gti_start = None
            gti_stop = None
            try:
                gti_list = _extract_gti(h)
                if gti_list is not None:
                    gti_start = np.array([s for s, _ in gti_list], dtype=float)
                    gti_stop = np.array([e for _, e in gti_list], dtype=float)
            except Exception:
                pass
            tstart = timezero
            if len(time) > 0:
                tstop = timezero + time[-1] + dt
                tseg = float(tstop - tstart)
            else:
                tseg = None
            headers_dump = _collect_headers_dump(h)
            meta = _build_meta(h, header)
            if timezero_obj is None:
                raise ValueError('无法构建 timezero_obj，请检查 TELESCOP/MJDREF/时间字段')
            exposure = None
            if 'EXPOSURE' in header:
                try:
                    exposure = float(header['EXPOSURE'])
                except (ValueError, TypeError):
                    pass
            if exposure is None and 'EXPTIME' in header:
                try:
                    exposure = float(header['EXPTIME'])
                except (ValueError, TypeError):
                    pass
            bin_exposure = None
            bin_width = None
            if (bin_lo is not None) and (bin_hi is not None) and (len(bin_lo) == len(bin_hi)):
                try:
                    bin_width = np.asarray(bin_hi, dtype=float) - np.asarray(bin_lo, dtype=float)
                except Exception:
                    bin_width = None
            if bin_width is None and len(time) > 0:
                dt_arr = np.asarray(dt, dtype=float) if dt is not None else np.asarray([], dtype=float)
                if dt_arr.ndim == 0 and dt_arr.size != 0 and np.isfinite(float(dt_arr)) and float(dt_arr) > 0:
                    bin_width = np.full(len(time), float(dt_arr), dtype=float)
                elif dt_arr.ndim == 1 and dt_arr.size == len(time):
                    bin_width = dt_arr
            if fracexp is not None and len(time) > 0:
                fracexp_arr = np.asarray(fracexp, dtype=float)
                if fracexp_arr.shape == (len(time),):
                    fracexp_arr = np.where(np.isfinite(fracexp_arr), fracexp_arr, 1.0)
                    fracexp_arr = np.clip(fracexp_arr, 0.0, 1.0)
                    if bin_width is not None:
                        bin_exposure = fracexp_arr * np.asarray(bin_width, dtype=float)
            if bin_exposure is None and bin_width is not None:
                bin_exposure = np.asarray(bin_width, dtype=float)
            if bin_exposure is None and exposure is not None and len(time) > 0:
                bin_exposure = np.full(len(time), exposure / len(time), dtype=float)
            err_dist = 'poisson' if counts is not None else ('gauss' if rate is not None else None)
            try:
                region = _load_regions(h)
            except Exception:
                region = None
            timesys = str(header['TIMESYS']) if 'TIMESYS' in header else (meta.timesys if meta and meta.timesys else None)
            mjdref = meta.mjdref if meta and meta.mjdref else None
            self._data = LightcurveData(
                path=self.path,
                time=time,
                timezero=timezero,
                timezero_obj=timezero_obj,
                dt=dt,
                bin_lo=bin_lo,
                bin_hi=bin_hi,
                tstart=tstart,
                tseg=tseg,
                value=value,
                error=error,
                is_rate=is_rate,
                counts=counts,
                rate=rate,
                counts_err=counts_err,
                rate_err=rate_err,
                err_dist=err_dist,
                gti_start=gti_start,
                gti_stop=gti_stop,
                quality=quality,
                fracexp=fracexp,
                exposure=exposure,
                bin_exposure=bin_exposure,
                backscal=backscal_col,
                areascal=areascal_col,
                telescop=telescop,
                timesys=timesys,
                mjdref=mjdref,
                region=region,
                header=header,
                meta=meta,
                headers_dump=headers_dump,
                columns=tuple(d.columns.names),
            )
        return self._data

    def validate(self) -> ValidationReport:
        if self._data is None:
            self.read()
        assert self._data is not None
        return self._data.validate()


class OgipEventReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        self._data: Optional[EventData] = None

    def read(self) -> EventData:
        with fits.open(self.path) as h:
            hevt = None
            for ext in h:
                ext_any = cast(Any, ext)
                if getattr(ext_any, 'data', None) is None:
                    continue
                names = getattr(ext_any.data, 'columns', None)
                if names is not None and ('TIME' in names.names):
                    hevt = ext_any
                    break
            if hevt is None:
                raise ValueError('No EVENTS-like HDU with TIME column found')
            de = cast(Any, hevt.data)
            colnames = list(getattr(de, 'columns').names) if getattr(de, 'columns', None) is not None else []
            raw_columns: Dict[str, np.ndarray] = {}
            for cn in colnames:
                try:
                    raw_columns[cn] = np.asarray(de[cn])
                except Exception:
                    try:
                        raw_columns[cn] = np.asarray([r[cn] for r in de])
                    except Exception:
                        raw_columns[cn] = np.asarray([])
            time_raw = np.asarray(raw_columns.get('TIME') if 'TIME' in raw_columns else de['TIME'], float)
            header = dict(cast(Any, hevt.header))
            timezero_raw = 0.0
            if 'TIMEZERO' in header:
                try:
                    timezero_raw = float(header['TIMEZERO'])
                except (ValueError, TypeError):
                    timezero_raw = 0.0
            time_offset = float(time_raw[0]) if len(time_raw) > 0 else 0.0
            time = time_raw - time_offset
            timezero = timezero_raw + time_offset
            telescop = None
            primary_header = dict(cast(Any, h[0]).header) if len(h) > 0 else {}
            for hdr in [header, primary_header]:
                if 'TELESCOP' in hdr:
                    telescop = str(hdr['TELESCOP']).strip().upper()
                    break
            timezero_obj = _mission_timezero_object(telescop, timezero, allow_unix_fallback=True) if telescop is not None and timezero != 0.0 else None
            pi = np.asarray(raw_columns['PI'], int) if 'PI' in raw_columns else None
            channel = np.asarray(raw_columns['CHANNEL'], int) if 'CHANNEL' in raw_columns else None
            headers_dump = _collect_headers_dump(h)
            meta = _build_meta(h, header)
            ebounds = None
            try:
                if 'EBOUNDS' in h:
                    de_eb = cast(Any, h['EBOUNDS']).data
                    ebounds = (
                        np.asarray(de_eb['CHANNEL'], int),
                        np.asarray(de_eb['E_MIN'], float),
                        np.asarray(de_eb['E_MAX'], float),
                    )
            except Exception:
                ebounds = None
            gti_start = None
            gti_stop = None
            gti_start_obj = None
            gti_stop_obj = None
            gti_list = None
            try:
                for hdu in h:
                    if getattr(hdu, 'name', '').upper() == 'GTI':
                        gti_data = getattr(hdu, 'data', None)
                        if gti_data is not None and 'START' in gti_data.columns.names and 'STOP' in gti_data.columns.names:
                            gti_start_raw = np.asarray(gti_data['START'], float)
                            gti_stop_raw = np.asarray(gti_data['STOP'], float)
                            gti_start = gti_start_raw - time_offset
                            gti_stop = gti_stop_raw - time_offset
                            gti_list = [(float(s), float(e)) for s, e in zip(gti_start, gti_stop)]
                            if timezero_obj is not None:
                                try:
                                    from astropy.time import TimeDelta

                                    gti_start_obj = timezero_obj + TimeDelta(gti_start, format='sec')
                                    gti_stop_obj = timezero_obj + TimeDelta(gti_stop, format='sec')
                                except Exception:
                                    gti_start_obj = gti_stop_obj = None
                            break
            except Exception:
                gti_start = gti_stop = None
                gti_start_obj = gti_stop_obj = None
                gti_list = None
            u2orig = {cn.upper(): cn for cn in colnames}

            def _find(*cands: str) -> Optional[str]:
                for c in cands:
                    if c is None:
                        continue
                    uc = c.upper()
                    if uc in u2orig:
                        return u2orig[uc]
                return None

            colmap: Dict[str, Optional[str]] = {}
            colmap['x'] = _find('X', 'XRAW', 'RAWX', 'DETX', 'DET_X', 'SKX', 'XDET')
            colmap['y'] = _find('Y', 'YRAW', 'RAWY', 'DETY', 'DET_Y', 'SKY', 'YDET')
            colmap['ra'] = _find('RA', 'RA_OBJ', 'RAX', 'RA_DEG')
            colmap['dec'] = _find('DEC', 'DEC_OBJ', 'DECX', 'DEC_DEG')
            colmap['energy'] = _find('ENERGY', 'E', 'ENERG', 'PHOTON_ENERGY')
            colmap['pha'] = _find('PHA', 'PI')
            key_x = colmap.get('x')
            key_y = colmap.get('y')
            key_energy = colmap.get('energy')
            xarr = np.asarray(raw_columns[key_x]) if (key_x is not None and key_x in raw_columns) else None
            yarr = np.asarray(raw_columns[key_y]) if (key_y is not None and key_y in raw_columns) else None
            energy = np.asarray(raw_columns[key_energy]) if (key_energy is not None and key_energy in raw_columns) else None
        self._data = EventData(
            path=self.path,
            time=time,
            timezero=timezero,
            timezero_obj=timezero_obj,
            telescop=telescop,
            pi=pi,
            channel=channel,
            x=xarr,
            y=yarr,
            gti_start=gti_start,
            gti_stop=gti_stop,
            gti_start_obj=gti_start_obj,
            gti_stop_obj=gti_stop_obj,
            gti=gti_list,
            raw_columns=raw_columns,
            colmap=colmap,
            energy=energy,
            ebounds=ebounds,
            header=header,
            meta=meta,
            headers_dump=headers_dump,
            columns=tuple(colnames),
        )
        return self._data

    def validate(self) -> ValidationReport:
        if self._data is None:
            self.read()
        assert self._data is not None
        return self._data.validate()


class ArfReader(OgipArfReader):
    pass


class RmfReader(OgipRmfReader):
    pass


class RspReader(OgipRmfReader):
    pass


class LightcurveReader(OgipLightcurveReader):
    pass


OgipData = Union[ArfData, RmfData, PhaData, LightcurveData, EventData]


def guess_ogip_kind(path) -> Literal['arf', 'rmf', 'pha', 'lc', 'evt']:
    p = Path(path)
    name = p.name.lower()
    if name.endswith('.arf'):
        return 'arf'
    if name.endswith('.rmf') or name.endswith('.rsp'):
        return 'rmf'
    if name.endswith('.pha') or name.endswith('.pi'):
        return 'pha'
    with fits.open(p) as h:
        extnames = {getattr(x, 'name', '').upper() for x in h}
        if 'SPECRESP' in extnames and 'MATRIX' not in extnames:
            return 'arf'
        if 'MATRIX' in extnames:
            return 'rmf'
        if 'SPECTRUM' in extnames:
            return 'pha'
        has_time = False
        for x in h:
            d = getattr(x, 'data', None)
            cols = getattr(d, 'columns', None)
            names = getattr(cols, 'names', ()) if cols is not None else ()
            if 'TIME' in names:
                has_time = True
                break
        if 'EVENTS' in extnames or has_time:
            for x in h:
                d = getattr(x, 'data', None)
                cols = getattr(d, 'columns', None)
                names = getattr(cols, 'names', ()) if cols is not None else ()
                if 'RATE' in names or 'COUNTS' in names:
                    return 'lc'
            return 'evt'
    return 'pha'


@overload
def readfits(path, kind: Literal['arf']) -> ArfData: ...


@overload
def readfits(path, kind: Literal['rmf']) -> RmfData: ...


@overload
def readfits(path, kind: Literal['pha']) -> PhaData: ...


@overload
def readfits(path, kind: Literal['lc']) -> LightcurveData: ...


@overload
def readfits(path, kind: Literal['evt']) -> EventData: ...


@overload
def readfits(path, kind: None = ...) -> OgipData: ...


def readfits(path, kind: Optional[Literal['arf', 'rmf', 'pha', 'lc', 'evt']] = None) -> OgipData:
    k = kind or guess_ogip_kind(path)
    if k == 'arf':
        return read_arf(path)
    if k == 'rmf':
        return read_rmf(path)
    if k == 'pha':
        return read_pha(path)
    if k == 'lc':
        return read_lc(path)
    if k == 'evt':
        return read_evt(path)
    raise ValueError(f"Unknown OGIP kind: {k}")


def read_arf(path) -> ArfData:
    return OgipArfReader(path).read()


def read_rmf(path) -> RmfData:
    return OgipRmfReader(path).read()


def read_pha(path) -> PhaData:
    return OgipPhaReader(path).read()


def read_lc(path) -> LightcurveData:
    return OgipLightcurveReader(path).read()


def read_evt(path) -> EventData:
    return OgipEventReader(path).read()


__all__ = [
    "OgipArfReader",
    "OgipRmfReader",
    "OgipPhaReader",
    "OgipLightcurveReader",
    "OgipEventReader",
    "ArfReader",
    "RmfReader",
    "RspReader",
    "LightcurveReader",
    "OgipData",
    "band_from_arf_bins",
    "channel_mask_from_ebounds",
    "guess_ogip_kind",
    "readfits",
    "read_arf",
    "read_rmf",
    "read_pha",
    "read_lc",
    "read_evt",
]
