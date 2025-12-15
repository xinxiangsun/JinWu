"""区域（region）解析与点内判定的轻量实现。

目标：用纯 Python 支持常见的 DS9/ASCII region 文件（circle, annulus, box, polygon, ellipse）
并提供对 EventData 的点内筛选函数。设计为不依赖大量外部工具：
- 优先使用 astropy.wcs（若可用）来做像素<->世界坐标变换；
- 优先使用 shapely（若可用）做几何测试；否则使用 numpy + matplotlib.path 进行点内判定。

此模块实现的功能覆盖 xselect 常用场景；对于极端或特殊的 region 语法，应使用已有的
`fregion`/`fregcon`工具链（可在系统中外调）。
"""
from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional
import re
from pathlib import Path
import numpy as np

try:
    from numba import njit
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False
    def njit(*args, **kwargs):
        """Dummy decorator when numba is not available."""
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

try:
    from astropy.wcs import WCS  # type: ignore
    _HAVE_WCS = True
except Exception:
    _HAVE_WCS = False

try:
    from shapely.geometry import Point, Polygon
    _HAVE_SHAPELY = True
except Exception:
    _HAVE_SHAPELY = False

try:
    from matplotlib.path import Path as MplPath
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False


def parse_ds9_region_line(line: str) -> Optional[Dict[str, Any]]:
    """解析 DS9 region 文件中的一行，返回 shape dict 或 None（注释/空行）。

    支持语法示例：
      circle(123,456,10)
      annulus(123,456,5,10)
      box(123,456,20,10,30)
      polygon(10,10, 20,10, 20,20, 10,20)
      ellipse(123,456,20,10,30)
    返回字典具有 'type' 键以及参数键。
    """
    s = line.strip()
    if not s or s.startswith('#') or s.startswith('!'):
        return None
    # detect optional leading coordinate system like "fk5; circle(...)" -> record coordsys
    coordsys = None
    if ';' in s and s.index(';') < s.find('('):
        parts = s.split(';', 1)
        coordsys = parts[0].strip().lower()
        s = parts[1].strip()
    m = re.match(r"^(\w+)\s*\((.*)\)", s)
    if not m:
        return None
    typ = m.group(1).lower()
    args = m.group(2)
    parts_raw = [p.strip() for p in re.split(r",|\s+", args) if p.strip()]
    parts = list(parts_raw)

    def _parse_token(tok: str):
        # handle sexagesimal RA/DEC like 12:34:56.78 using astropy if available
        if ':' in tok:
            try:
                from astropy.coordinates import Angle
                a = Angle(tok)
                return float(a.degree)
            except Exception:
                pass
        # handle units: trailing double-quote for arcsec or 'arcsec'/'deg'
        if tok.endswith('"') or tok.lower().endswith('arcsec'):
            try:
                v = float(re.sub(r'[^0-9.+-eE]', '', tok))
                return v  # mark as numeric in arcsec; caller will interpret
            except Exception:
                return tok
        if tok.lower().endswith('deg') or tok.lower().endswith('d'):
            try:
                v = float(re.sub(r'[^0-9.+-eE]', '', tok))
                return v
            except Exception:
                return tok
        # plain float
        try:
            return float(tok)
        except Exception:
            return tok

    vals = [_parse_token(p) for p in parts]
    # keep unit hints by inspecting raw tokens
    def _unit_of(tok: str) -> Optional[str]:
        s = str(tok).strip().lower()
        if s.endswith('"') or 'arcsec' in s:
            return 'arcsec'
        if s.endswith('deg') or s.endswith('d') or 'degree' in s:
            return 'deg'
        # pixel-like tokens often contain 'pix' or 'pixel'
        if 'pix' in s or 'pixel' in s:
            return 'pix'
        return None
    if typ == 'circle':
        if len(vals) >= 3:
            xval, yval, rval = vals[0], vals[1], vals[2]
            d = {'type': 'circle', 'x': xval, 'y': yval, 'r': rval}
            # record radius unit hint
            try:
                unit = _unit_of(parts_raw[2])
                if unit:
                    d['r_unit'] = unit
            except Exception:
                pass
            if coordsys:
                d['coordsys'] = coordsys
            # Try to record unit hints for radius if token was string-like
            try:
                # if radius token was non-float string originally, keep as-is
                pass
            except Exception:
                pass
            return d
    if typ == 'point':
        # point(x,y) - treat as very small circle (or exact point)
        if len(vals) >= 2:
            d = {'type': 'point', 'x': vals[0], 'y': vals[1]}
            if coordsys:
                d['coordsys'] = coordsys
            return d
    if typ == 'annulus':
        if len(vals) >= 4:
            d = {'type': 'annulus', 'x': vals[0], 'y': vals[1], 'r_in': vals[2], 'r_out': vals[3]}
            # unit hints
            try:
                u_in = _unit_of(parts_raw[2])
                u_out = _unit_of(parts_raw[3])
                if u_in:
                    d['r_in_unit'] = u_in
                if u_out:
                    d['r_out_unit'] = u_out
            except Exception:
                pass
            if coordsys:
                d['coordsys'] = coordsys
            return d
    if typ == 'box':
        # box(xc,yc,w,h,angle)
        if len(vals) >= 4:
            res = {'type': 'box', 'x': vals[0], 'y': vals[1], 'width': vals[2], 'height': vals[3]}
            try:
                # width/height unit hints
                u_w = _unit_of(parts_raw[2])
                u_h = _unit_of(parts_raw[3])
                if u_w:
                    res['width_unit'] = u_w
                if u_h:
                    res['height_unit'] = u_h
            except Exception:
                pass
            if len(vals) >= 5:
                res['angle'] = vals[4]
            if coordsys:
                res['coordsys'] = coordsys
            return res
    if typ == 'polygon':
        # polygon(x1,y1, x2,y2, ...)
        if len(vals) >= 6 and len(vals) % 2 == 0:
            pts = [(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]
            d = {'type': 'polygon', 'points': pts}
            if coordsys:
                d['coordsys'] = coordsys
            return d
    if typ == 'ellipse':
        # ellipse(xc,yc, a, b, angle)
        if len(vals) >= 5:
            d = {'type': 'ellipse', 'x': vals[0], 'y': vals[1], 'a': vals[2], 'b': vals[3], 'angle': vals[4]}
            try:
                ua = _unit_of(parts_raw[2])
                ub = _unit_of(parts_raw[3])
                if ua:
                    d['a_unit'] = ua
                if ub:
                    d['b_unit'] = ub
            except Exception:
                pass
            if coordsys:
                d['coordsys'] = coordsys
            return d
    # unsupported shape -> return raw
    return {'type': typ, 'args': vals}


def parse_ds9_region_file(path: str | Path) -> List[Dict[str, Any]]:
    """解析一个 region 文件（多行），返回 shape dict 列表（按顺序）。"""
    p = Path(path)
    shapes: List[Dict[str, Any]] = []
    with p.open('r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            try:
                r = parse_ds9_region_line(line)
            except Exception:
                r = None
            if r:
                shapes.append(r)
    return shapes


def _points_in_polygon_mpl(xs: np.ndarray, ys: np.ndarray, poly: List[Tuple[float, float]]) -> np.ndarray:
    pts = np.vstack([xs, ys]).T
    if not _HAVE_MPL:
        # fallback to ray-casting implementation
        return _points_in_polygon_numpy(xs, ys, poly)
    path = MplPath(np.array(poly))
    return path.contains_points(pts)


if _HAVE_NUMBA:
    @njit
    def _points_in_polygon_numpy(xs: np.ndarray, ys: np.ndarray, poly_x: np.ndarray, poly_y: np.ndarray) -> np.ndarray:
        """Numba-accelerated ray casting for polygon point-in-polygon test."""
        n_points = xs.size
        n_poly = poly_x.size
        inside = np.zeros(n_points, dtype=np.bool_)
        
        for k in range(n_points):
            x = xs[k]
            y = ys[k]
            count = 0
            
            for i in range(n_poly):
                j = (i + 1) % n_poly
                xi = poly_x[i]
                yi = poly_y[i]
                xj = poly_x[j]
                yj = poly_y[j]
                
                if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
                    count += 1
            
            inside[k] = (count % 2) == 1
        
        return inside
    
    def _points_in_polygon_wrapper(xs: np.ndarray, ys: np.ndarray, poly: List[Tuple[float, float]]) -> np.ndarray:
        """Wrapper to convert polygon list to arrays for numba."""
        px = np.array([p[0] for p in poly], dtype=np.float64)
        py = np.array([p[1] for p in poly], dtype=np.float64)
        return _points_in_polygon_numpy(xs, ys, px, py)
else:
    def _points_in_polygon_wrapper(xs: np.ndarray, ys: np.ndarray, poly: List[Tuple[float, float]]) -> np.ndarray:
        """Pure numpy ray casting algorithm for polygon point-in-polygon."""
        x = xs
        y = ys
        n = len(poly)
        inside = np.zeros_like(x, dtype=bool)
        px = np.array([p[0] for p in poly])
        py = np.array([p[1] for p in poly])
        for i in range(n):
            j = (i + 1) % n
            xi = px[i]
            yi = py[i]
            xj = px[j]
            yj = py[j]
            intersect = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi)
            inside ^= intersect
        return inside


if _HAVE_NUMBA:
    @njit
    def _circle_mask(xs: np.ndarray, ys: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray:
        """Numba-accelerated circle mask."""
        n = xs.size
        mask = np.zeros(n, dtype=np.bool_)
        r2 = r * r
        for i in range(n):
            dx = xs[i] - cx
            dy = ys[i] - cy
            mask[i] = (dx * dx + dy * dy) <= r2
        return mask
    
    @njit
    def _annulus_mask(xs: np.ndarray, ys: np.ndarray, cx: float, cy: float, rin: float, rout: float) -> np.ndarray:
        """Numba-accelerated annulus mask."""
        n = xs.size
        mask = np.zeros(n, dtype=np.bool_)
        rin2 = rin * rin
        rout2 = rout * rout
        for i in range(n):
            dx = xs[i] - cx
            dy = ys[i] - cy
            r2 = dx * dx + dy * dy
            mask[i] = (r2 >= rin2) and (r2 <= rout2)
        return mask
    
    @njit
    def _box_mask(xs: np.ndarray, ys: np.ndarray, cx: float, cy: float, w: float, h: float, angle: float) -> np.ndarray:
        """Numba-accelerated rotated box mask."""
        n = xs.size
        mask = np.zeros(n, dtype=np.bool_)
        theta = np.deg2rad(angle)
        ca = np.cos(-theta)
        sa = np.sin(-theta)
        hw = w / 2.0
        hh = h / 2.0
        for i in range(n):
            dx = xs[i] - cx
            dy = ys[i] - cy
            xr = dx * ca - dy * sa
            yr = dx * sa + dy * ca
            mask[i] = (abs(xr) <= hw) and (abs(yr) <= hh)
        return mask
else:
    def _circle_mask(xs: np.ndarray, ys: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray:
        """Pure numpy circle mask."""
        dx = xs - cx
        dy = ys - cy
        return (dx * dx + dy * dy) <= (r * r)
    
    def _annulus_mask(xs: np.ndarray, ys: np.ndarray, cx: float, cy: float, rin: float, rout: float) -> np.ndarray:
        """Pure numpy annulus mask."""
        dx = xs - cx
        dy = ys - cy
        rr = dx * dx + dy * dy
        return (rr >= rin * rin) & (rr <= rout * rout)
    
    def _box_mask(xs: np.ndarray, ys: np.ndarray, cx: float, cy: float, w: float, h: float, angle: float) -> np.ndarray:
        """Pure numpy rotated box mask."""
        theta = np.deg2rad(angle)
        ca = np.cos(-theta)
        sa = np.sin(-theta)
        dx = xs - cx
        dy = ys - cy
        xr = dx * ca - dy * sa
        yr = dx * sa + dy * ca
        return (np.abs(xr) <= (w / 2.0)) & (np.abs(yr) <= (h / 2.0))


def points_in_shape(xs: np.ndarray, ys: np.ndarray, shape: Dict[str, Any]) -> np.ndarray:
    """对于给定 shape dict，返回与 xs/ys 对应的布尔 mask 表示点是否在 shape 内。"""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    
    typ = shape.get('type', '').lower()
    if typ == 'circle':
        cx = float(shape['x'])
        cy = float(shape['y'])
        r = float(shape['r'])
        return _circle_mask(xs, ys, cx, cy, r)
    if typ == 'annulus':
        cx = float(shape['x'])
        cy = float(shape['y'])
        rin = float(shape['r_in'])
        rout = float(shape['r_out'])
        return _annulus_mask(xs, ys, cx, cy, rin, rout)
    if typ == 'box':
        cx = float(shape.get('x', 0.0))
        cy = float(shape.get('y', 0.0))
        w = float(shape.get('width', shape.get('w', 1.0)))
        h = float(shape.get('height', shape.get('h', 1.0)))
        ang = float(shape.get('angle', 0.0))
        return _box_mask(xs, ys, cx, cy, w, h, ang)
    if typ == 'polygon':
        pts = shape['points']
        if _HAVE_SHAPELY:
            poly = Polygon(pts)
            return np.array([poly.contains(Point(xx, yy)) for xx, yy in zip(xs, ys)], dtype=bool)
        else:
            return _points_in_polygon_wrapper(xs, ys, pts)
    if typ == 'ellipse':
        # approximate ellipse using affine transform (angle in degrees)
        cx = float(shape.get('x', 0.0))
        cy = float(shape.get('y', 0.0))
        a = float(shape.get('a', 1.0))
        b = float(shape.get('b', 1.0))
        ang = float(shape.get('angle', 0.0)) * np.pi / 180.0
        dx = xs - cx
        dy = ys - cy
        # rotate points by -angle
        ca = np.cos(-ang)
        sa = np.sin(-ang)
        xr = dx * ca - dy * sa
        yr = dx * sa + dy * ca
        return (xr * xr) / (a * a) + (yr * yr) / (b * b) <= 1.0
    if typ == 'point':
        cx = float(shape.get('x', 0.0))
        cy = float(shape.get('y', 0.0))
        # exact point: inside only if coordinates exactly match (within tiny tol)
        tol = float(shape.get('tol', 1e-6))
        return (np.abs(xs - cx) <= tol) & (np.abs(ys - cy) <= tol)
    # unsupported shape -> false mask
    return np.zeros_like(xs, dtype=bool)


def apply_region_mask_to_events(ev, shapes: List[Dict[str, Any]], invert: bool = False):
    """对 EventData 应用一组 shapes（OR 组合），返回新的 EventData（保留 GTI 等元信息）。

    shapes: list of shape dicts; 点满足任一 shape 即被视为 inside（逻辑 OR）。
    invert: 若为 True，则返回 outside 的事件。
    """
    # try to read X/Y from ev (prefer attributes), fall back to reading FITS columns
    if hasattr(ev, 'x') and hasattr(ev, 'y'):
        xs = np.asarray(getattr(ev, 'x'), dtype=float)
        ys = np.asarray(getattr(ev, 'y'), dtype=float)
    else:
        # lazy import to avoid circular imports
        from ..core.xselect import _read_column_from_evt
        # Allow shapes to suggest preferred column names via 'xcol'/'ycol'
        preferred_x = []
        preferred_y = []
        for sh in shapes:
            if isinstance(sh, dict):
                xc = sh.get('xcol')
                yc = sh.get('ycol')
                if xc:
                    preferred_x.append(str(xc))
                if yc:
                    preferred_y.append(str(yc))

        # candidate search order
        cand_x = []
        cand_y = []
        cand_x.extend(preferred_x)
        cand_x.extend(['X', 'X_IMAGE', 'RAWX', 'DETX'])
        cand_y.extend(preferred_y)
        cand_y.extend(['Y', 'Y_IMAGE', 'RAWY', 'DETY'])

        x = None
        y = None
        # Try to find a matching pair of columns
        for xc in cand_x:
            xv = _read_column_from_evt(ev.path, xc)
            if xv is None:
                continue
            for yc in cand_y:
                yv = _read_column_from_evt(ev.path, yc)
                if yv is None:
                    continue
                # found matching pair
                x = xv
                y = yv
                break
            if x is not None and y is not None:
                break
        if x is None or y is None:
            raise ValueError('Event file lacks X/Y columns required for region filtering')
        xs = np.asarray(x, dtype=float)
        ys = np.asarray(y, dtype=float)

    # If any shape uses a sky coordinate system, attempt to convert their coordinates
    # to detector pixel coordinates via teldef. This allows DS9 sky regions (fk5/fk4)
    # to be applied to event DET/X,Y coordinates.
    has_sky = any(('coordsys' in sh and str(sh['coordsys']).lower().startswith(('fk','icrs','eq'))) for sh in shapes)
    if has_sky:
        try:
            from .teldef import find_teldef_from_event
            tel = find_teldef_from_event(ev)
        except Exception:
            tel = None
    else:
        tel = None

    if len(shapes) == 0:
        mask = np.zeros(xs.size, dtype=bool)
    else:
        mask = np.zeros(xs.size, dtype=bool)
        # If astropy WCS available, try to get WCS from event file headers and use it
        wcs_obj = None
        if _HAVE_WCS:
            try:
                from astropy.wcs import WCS as _WCS
                from astropy.io import fits as _fits
                # try to find HDU with TIME column and WCS in its header
                hdu_with_time = None
                try:
                    with _fits.open(ev.path) as _h:
                        for _ext in _h:
                            _d = getattr(_ext, 'data', None)
                            _cols = getattr(_d, 'columns', None)
                            if _cols is not None and 'TIME' in _cols.names:
                                hdu_with_time = _ext
                                break
                        if hdu_with_time is None:
                            # try primary
                            hdu_with_time = _h[0]
                        hdr = hdu_with_time.header
                        wcs_obj = _WCS(hdr)
                except Exception:
                    wcs_obj = None
            except Exception:
                wcs_obj = None

        for sh in shapes:
            sh2 = sh
            # If shape specifies coordsys that appears to be sky-based, convert coords
            if 'coordsys' in sh and sh['coordsys'] and str(sh['coordsys']).lower().startswith(('fk','icrs','eq')):
                try:
                    typ = sh.get('type','').lower()
                    # use astropy WCS if available and valid
                    if wcs_obj is not None and getattr(wcs_obj, 'wcs', None) is not None:
                        if typ in ('circle','annulus','box','ellipse'):
                            ra = float(sh['x'])
                            dec = float(sh['y'])
                            # astropy expects arrays in shape (N, ndim)
                            px, py = wcs_obj.wcs_world2pix([[ra, dec]], 0)[0]
                            sh2 = dict(sh)
                            sh2['x'] = float(px)
                            sh2['y'] = float(py)
                            # radius/unit handling: consult shape unit hints (r_unit, width_unit, etc.)
                            def _to_pix(val, unit_key=None):
                                if val is None:
                                    return None
                                try:
                                    v = float(val)
                                except Exception:
                                    return None
                                unit = None
                                if unit_key is not None:
                                    unit = sh.get(unit_key)
                                # default DS9 behaviour: sky numeric radii are arcsec unless explicit deg
                                if unit is None:
                                    # heuristics: values > 1 likely arcsec; tiny values maybe degrees
                                    unit = 'arcsec' if v > 1e-3 else 'deg'
                                # fetch approximate pixel scale from WCS: use CDELT or CD
                                pixscale_deg = None
                                try:
                                    if getattr(wcs_obj.wcs, 'cdelt', None) is not None:
                                        pixscale_deg = abs(float(wcs_obj.wcs.cdelt[0]))
                                    else:
                                        cd = getattr(wcs_obj.wcs, 'cd', None)
                                        if cd is not None:
                                            pixscale_deg = float(abs(np.linalg.det(cd)) ** 0.5)
                                except Exception:
                                    pixscale_deg = None
                                if pixscale_deg is None:
                                    return v
                                if unit == 'arcsec':
                                    return (v / 3600.0) / pixscale_deg
                                if unit == 'deg':
                                    return v / pixscale_deg
                                if unit == 'pix' or unit == 'pixel':
                                    return v
                                return v

                            # apply conversions
                            if 'r' in sh2:
                                rp = _to_pix(sh.get('r'), 'r_unit')
                                if rp is not None:
                                    sh2['r'] = rp
                            if 'r_in' in sh2:
                                rinp = _to_pix(sh.get('r_in'), 'r_in_unit')
                                if rinp is not None:
                                    sh2['r_in'] = rinp
                            if 'r_out' in sh2:
                                routp = _to_pix(sh.get('r_out'), 'r_out_unit')
                                if routp is not None:
                                    sh2['r_out'] = routp
                            if 'width' in sh2:
                                wp = _to_pix(sh.get('width'), 'width_unit')
                                if wp is not None:
                                    sh2['width'] = wp
                            if 'height' in sh2:
                                hp = _to_pix(sh.get('height'), 'height_unit')
                                if hp is not None:
                                    sh2['height'] = hp
                        elif typ == 'polygon':
                            pts = sh.get('points', [])
                            newpts = []
                            for (ra, dec) in pts:
                                px, py = wcs_obj.wcs_world2pix([[float(ra), float(dec)]], 0)[0]
                                newpts.append((float(px), float(py)))
                            sh2 = dict(sh)
                            sh2['points'] = newpts
                            sh2.pop('coordsys', None)
                    elif tel is not None:
                        # fallback to teldef conversion (existing behavior)
                        if typ in ('circle','annulus','box','ellipse'):
                            ra = float(sh['x'])
                            dec = float(sh['y'])
                            xdet, ydet = tel.sky_to_det(ra, dec)
                            sh2 = dict(sh)
                            sh2['x'] = float(xdet)
                            sh2['y'] = float(ydet)
                            # convert using teldef.pixscale_deg when possible and consult unit hints
                            def _to_pix_tel(v, unit_key=None):
                                if v is None:
                                    return None
                                try:
                                    fv = float(v)
                                except Exception:
                                    return None
                                unit = sh.get(unit_key) if unit_key is not None else None
                                if unit is None:
                                    unit = 'arcsec' if fv > 1e-3 else 'deg'
                                if tel.pixscale_deg is None:
                                    return fv
                                if unit == 'arcsec':
                                    return (fv / 3600.0) / float(tel.pixscale_deg)
                                if unit == 'deg':
                                    return fv / float(tel.pixscale_deg)
                                return fv

                            if 'r' in sh2:
                                v = _to_pix_tel(sh.get('r'), 'r_unit')
                                if v is not None:
                                    sh2['r'] = v
                            if 'r_in' in sh2:
                                v = _to_pix_tel(sh.get('r_in'), 'r_in_unit')
                                if v is not None:
                                    sh2['r_in'] = v
                            if 'r_out' in sh2:
                                v = _to_pix_tel(sh.get('r_out'), 'r_out_unit')
                                if v is not None:
                                    sh2['r_out'] = v
                            if 'width' in sh2:
                                v = _to_pix_tel(sh.get('width'), 'width_unit')
                                if v is not None:
                                    sh2['width'] = v
                            if 'height' in sh2:
                                v = _to_pix_tel(sh.get('height'), 'height_unit')
                                if v is not None:
                                    sh2['height'] = v
                        elif typ == 'polygon':
                            pts = sh.get('points', [])
                            newpts = []
                            for (ra, dec) in pts:
                                xdet, ydet = tel.sky_to_det(float(ra), float(dec))
                                newpts.append((float(xdet), float(ydet)))
                            sh2 = dict(sh)
                            sh2['points'] = newpts
                            sh2.pop('coordsys', None)
                except Exception:
                    sh2 = sh
            try:
                m = points_in_shape(xs, ys, sh2)
            except Exception:
                m = np.zeros(xs.size, dtype=bool)
            mask |= m

    if invert:
        mask = ~mask

    # build new EventData
    from ..core.file import EventData
    t = np.asarray(ev.time, dtype=float)
    new_time = t[mask]
    new_pi = None if ev.pi is None else np.asarray(ev.pi)[mask]
    new_ch = None if ev.channel is None else np.asarray(ev.channel)[mask]
    return EventData(kind=ev.kind, path=ev.path, time=new_time, pi=new_pi, channel=new_ch,
                     gti_start=ev.gti_start, gti_stop=ev.gti_stop,
                     header=ev.header, meta=ev.meta, columns=ev.columns, headers_dump=ev.headers_dump)
