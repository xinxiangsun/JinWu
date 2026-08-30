"""简化 teldef 解析与坐标变换工具。

此模块实现一个轻量级 Teldef 类，用来把 RAW<->DET<->SKY 之间进行转换。
实现目标：提供对常见 teldef 要素（CRVAL/CRPIX/pixel scale / CD matrix）支持，
以及从事件头查找 teldef 文件的便捷接口。该实现并非完整复刻 coordfits 的全部
细节，但足以支持把 DS9 的 sky-region 转换到 detector 像素坐标以供 region 过滤。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Sequence
import re
import math
import numpy as np
from astropy.io import fits
from . import teldef_helpers as _th
try:
    from astropy.wcs import WCS
    _HAVE_WCS = True
except Exception:
    WCS = None  # type: ignore
    _HAVE_WCS = False

_FLOAT_RE = re.compile(r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
_FLOAT_RE_D = re.compile(r"([-+]?[0-9]*\.?[0-9]+(?:[eEdD][-+]?[0-9]+)?)")


class Teldef:
    """简化的 Teldef 表示。

    属性（常见）：
    - crval: (ra_deg, dec_deg) 参考天球坐标
    - crpix: (x_pix, y_pix) 参考像素
    - pixscale_deg: pixel scale in degrees/pixel (if CD not provided)
    - cd: optional 2x2 CD matrix (deg/pix)
    """
    def __init__(self, crval: Optional[Tuple[float, float]] = None, crpix: Optional[Tuple[float, float]] = None,
                 pixscale_deg: Optional[float] = None, cd: Optional[np.ndarray] = None):
        self.crval = crval
        self.crpix = crpix
        self.pixscale_deg = pixscale_deg
        self.cd = cd

    @classmethod
    def from_file(cls, path: str | Path) -> 'Teldef':
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(str(p))
        # If FITS with WCS, prefer WCS extraction
        if p.suffix.lower() in ('.fits', '.fit', '.fz') and _HAVE_WCS:
            try:
                with fits.open(p) as h:
                    # try to find an HDU with WCS header
                    for ext in h:
                        hdr = ext.header
                        try:
                            w = WCS(hdr)
                            # require at least one world axis
                            if w.wcs.naxis >= 2:
                                # extract CRVAL/CRPIX and CD or CDELT
                                crval = tuple(w.wcs.crval.tolist()) if getattr(w.wcs, 'crval', None) is not None else None
                                crpix = tuple(w.wcs.crpix.tolist()) if getattr(w.wcs, 'crpix', None) is not None else None
                                cd = None
                                if getattr(w.wcs, 'cd', None) is not None and w.wcs.cd is not None:
                                    cd = np.asarray(w.wcs.cd, dtype=float)
                                elif getattr(w.wcs, 'cdelt', None) is not None and w.wcs.cdelt is not None:
                                    # build diagonal CD from CDELT (deg/pix)
                                    cd = np.diag(np.asarray(w.wcs.cdelt, dtype=float)[:2])
                                pixscale = None
                                try:
                                    # approximate pixel scale from CD matrix determinant
                                    if cd is not None:
                                        # approximate using sqrt(|det|) per axis
                                        pixscale = float(abs(np.linalg.det(cd)) ** 0.5)
                                    else:
                                        pixscale = float(abs(w.wcs.cdelt[0]))
                                except Exception:
                                    pixscale = None
                                return cls(crval=crval, crpix=crpix, pixscale_deg=pixscale, cd=cd)
                        except Exception:
                            continue
            except Exception:
                # fallback to plain-text parsing below
                pass
        text = p.read_text(errors='ignore')
        # try to find CRVAL/CRPIX/PIXEL_SCALE or CD matrix
        crval = None
        crpix = None
        pixscale = None
        cd = None

        # common keywords attempts
        m = re.search(r'CRVAL\s*[:=]\s*' + _FLOAT_RE.pattern + r'\s*,\s*' + _FLOAT_RE.pattern, text, re.IGNORECASE)
        if m:
            crval = (float(m.group(1)), float(m.group(2)))
        else:
            # try separate lines
            m1 = re.search(r'RA_PNT\s*[:=]\s*' + _FLOAT_RE.pattern, text, re.IGNORECASE)
            m2 = re.search(r'DEC_PNT\s*[:=]\s*' + _FLOAT_RE.pattern, text, re.IGNORECASE)
            if m1 and m2:
                crval = (float(m1.group(1)), float(m2.group(1)))

        m = re.search(r'CRPIX\s*[:=]\s*' + _FLOAT_RE.pattern + r'\s*,\s*' + _FLOAT_RE.pattern, text, re.IGNORECASE)
        if m:
            crpix = (float(m.group(1)), float(m.group(2)))

        # pixel scale
        m = re.search(r'PIXEL[_ ]?SCALE\s*[:=]\s*' + _FLOAT_RE.pattern, text, re.IGNORECASE)
        if m:
            pixscale = float(m.group(1))
        else:
            m = re.search(r'PIXSIZE\s*[:=]\s*' + _FLOAT_RE.pattern, text, re.IGNORECASE)
            if m:
                pixscale = float(m.group(1))

        # CD matrix (cd1_1, cd1_2, cd2_1, cd2_2)
        cd_keys = ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']
        cd_vals = []
        for k in cd_keys:
            m = re.search(rf'{k}\s*[:=]\s*' + _FLOAT_RE.pattern, text, re.IGNORECASE)
            if m:
                cd_vals.append(float(m.group(1)))
            else:
                cd_vals = []
                break
        if cd_vals:
            cd = np.array([[cd_vals[0], cd_vals[1]], [cd_vals[2], cd_vals[3]]], dtype=float)

        # attempt to parse ALIGNM (3x3) and focal length / detector scales (SWIFT-style)
        align = None
        m_align = []
        for i in range(1,4):
            for j in range(1,4):
                key = f'ALIGNM{i}{j}'
                m = re.search(rf'{key}\s*[:=]\s*' + _FLOAT_RE_D.pattern, text, re.IGNORECASE)
                if m:
                    # allow Fortran D exponent
                    val = m.group(1).replace('D','E').replace('d','e')
                    try:
                        m_align.append(float(val))
                    except Exception:
                        m_align = []
                        break
                else:
                    m_align = []
                    break
            if not m_align:
                break
        if len(m_align) == 9:
            align = np.array(m_align, dtype=float).reshape((3,3))

        # focal length
        focal = None
        m = re.search(r'FOCALLEN\s*[:=]\s*' + _FLOAT_RE_D.pattern, text, re.IGNORECASE)
        if m:
            focal = float(m.group(1).replace('D','E').replace('d','e'))

        # detector scale and pixel definitions
        det_xscl = None
        det_yscl = None
        m = re.search(r'DET[_ ]?XSCL\s*[:=]\s*' + _FLOAT_RE_D.pattern, text, re.IGNORECASE)
        if m:
            det_xscl = float(m.group(1).replace('D','E').replace('d','e'))
        m = re.search(r'DET[_ ]?YSCL\s*[:=]\s*' + _FLOAT_RE_D.pattern, text, re.IGNORECASE)
        if m:
            det_yscl = float(m.group(1).replace('D','E').replace('d','e'))

        # DET first pixel and size
        det_xpix1 = None
        det_ypix1 = None
        det_xsiz = None
        det_ysiz = None
        m = re.search(r'DETXPIX1\s*[:=]\s*' + _FLOAT_RE_D.pattern, text, re.IGNORECASE)
        if m:
            det_xpix1 = float(m.group(1).replace('D','E').replace('d','e'))
        m = re.search(r'DETYPIX1\s*[:=]\s*' + _FLOAT_RE_D.pattern, text, re.IGNORECASE)
        if m:
            det_ypix1 = float(m.group(1).replace('D','E').replace('d','e'))
        m = re.search(r'DET[_ ]?XSIZ\s*[:=]\s*' + _FLOAT_RE_D.pattern, text, re.IGNORECASE)
        if m:
            det_xsiz = float(m.group(1).replace('D','E').replace('d','e'))
        m = re.search(r'DET[_ ]?YSIZ\s*[:=]\s*' + _FLOAT_RE_D.pattern, text, re.IGNORECASE)
        if m:
            det_ysiz = float(m.group(1).replace('D','E').replace('d','e'))

        # optical axis in detector coordinates
        optax_x = None
        optax_y = None
        m = re.search(r'OPTAXISX\s*[:=]\s*' + _FLOAT_RE_D.pattern, text, re.IGNORECASE)
        if m:
            optax_x = float(m.group(1).replace('D','E').replace('d','e'))
        m = re.search(r'OPTAXISY\s*[:=]\s*' + _FLOAT_RE_D.pattern, text, re.IGNORECASE)
        if m:
            optax_y = float(m.group(1).replace('D','E').replace('d','e'))

        # DET_ROTD rotation (degrees) and RAWFLIPY flag (boolean)
        det_rotd = None
        m = re.search(r'DET_ROTD\s*[:=]\s*' + _FLOAT_RE_D.pattern, text, re.IGNORECASE)
        if m:
            det_rotd = float(m.group(1).replace('D','E').replace('d','e'))
        rawflipy = False
        m = re.search(r'RAWFLIPY\s*[:=]\s*' + _FLOAT_RE_D.pattern, text, re.IGNORECASE)
        if m:
            try:
                rawflipy = bool(int(float(m.group(1).replace('D','E').replace('d','e'))))
            except Exception:
                rawflipy = False
        # CORINRAW - if present maps are in raw coords and should be pre-transformed
        corinraw = False
        if re.search(r'CORINRAW', text, re.IGNORECASE):
            corinraw = True

        # If we have ALIGNM and focal and detector scales, attach to returned object
        tel = cls(crval=crval, crpix=crpix, pixscale_deg=pixscale, cd=cd)
        if align is not None:
            tel.align = align
            tel.focal_length = focal
            tel.det_xscl = det_xscl
            tel.det_yscl = det_yscl
            tel.det_xpix1 = det_xpix1
            tel.det_ypix1 = det_ypix1
            tel.det_xsiz = det_xsiz
            tel.det_ysiz = det_ysiz
            tel.optaxis = (optax_x, optax_y)
            # attach optional transform flags
            tel.det_rotd = det_rotd
            tel.rawflipy = rawflipy
            tel.corinraw = corinraw
        return tel

    def attach_raw_map(self, mapx: object) -> None:
        """Attach a MapXform-like object as the teldef's raw_map.

        The provided object must implement `sample_delta_at_pixel(x,y)`.
        """
        self.raw_map = mapx

    def load_mapxform_from_fits(self, path: str | Path) -> None:
        """Attempt to load DELTAX/DELTAY image HDUs from a teldef FITS file and attach as MapXform.

        The function looks for HDUs with EXTNAME containing 'DELTAX' and 'DELTAY' (case-insensitive).
        If found, it reads arrays and header keywords to infer origin/scale (CRPIX/CDELT/CRVAL) and
        constructs a MapXform which is attached as `self.raw_map`.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(str(p))
        try:
            with fits.open(p) as hdul:
                deltax = None
                deltay = None
                deltax_hdr = None
                deltay_hdr = None
                for h in hdul:
                    extname = h.header.get('EXTNAME', '')
                    if extname is None:
                        extname = ''
                    en = extname.upper()
                    if 'DELTAX' in en and h.data is not None:
                        deltax = np.asarray(h.data, dtype=float)
                        deltax_hdr = h.header
                    if 'DELTAY' in en and h.data is not None:
                        deltay = np.asarray(h.data, dtype=float)
                        deltay_hdr = h.header
                if deltax is None or deltay is None:
                    # try primary+1 and primary+2 heuristic
                    if len(hdul) > 2 and hdul[1].data is not None and hdul[2].data is not None:
                        deltax = np.asarray(hdul[1].data, dtype=float)
                        deltay = np.asarray(hdul[2].data, dtype=float)
                        deltax_hdr = hdul[1].header
                        deltay_hdr = hdul[2].header
                if deltax is None or deltay is None:
                    return

                # infer origin and scale from header keywords or defaults
                def infer_origin_scale(hdr):
                    # prefer CRVAL/CRPIX/CDELT style
                    crpix1 = hdr.get('CRPIX1')
                    crval1 = hdr.get('CRVAL1')
                    cdelt1 = hdr.get('CDELT1')
                    crpix2 = hdr.get('CRPIX2')
                    crval2 = hdr.get('CRVAL2')
                    cdelt2 = hdr.get('CDELT2')
                    if crpix1 is not None and cdelt1 is not None and crval1 is not None:
                        origin_x = float(crval1)
                        scale_x = float(cdelt1)
                    else:
                        origin_x = 0.0
                        scale_x = 1.0
                    if crpix2 is not None and cdelt2 is not None and crval2 is not None:
                        origin_y = float(crval2)
                        scale_y = float(cdelt2)
                    else:
                        origin_y = 0.0
                        scale_y = 1.0
                    return origin_x, scale_x, origin_y, scale_y

                ox, sx, oy, sy = infer_origin_scale(deltax_hdr)
                # construct MapXform and attach
                from .teldef_helpers import MapXform
                mapx = MapXform(deltax, deltay, ox, sx, oy, sy)
                self.attach_raw_map(mapx)
        except Exception:
            # non-fatal: leave raw_map unset
            return

    def sky_to_det(self, ra_deg: float, dec_deg: float) -> Tuple[float, float]:
        """把天球坐标 (ra,dec) 转换为 detector 像素坐标 (x,y).

        使用简化的切平面投影（gnomonic）+ CD 矩阵 或 pixscale/CRPIX.
        要求已有 self.crval 和 self.crpix 或 pixscale.
        """
        if self.crval is None or self.crpix is None:
            raise ValueError('Teldef lacks CRVAL/CRPIX required for sky->det')
        ra0 = math.radians(self.crval[0])
        dec0 = math.radians(self.crval[1])
        ra = math.radians(ra_deg)
        dec = math.radians(dec_deg)
        dra = ra - ra0
        # normalize
        dra = (dra + math.pi) % (2 * math.pi) - math.pi
        sin_d0 = math.sin(dec0)
        cos_d0 = math.cos(dec0)
        sin_d = math.sin(dec)
        cos_d = math.cos(dec)
        denom = sin_d * sin_d0 + cos_d * cos_d0 * math.cos(dra)
        if denom == 0:
            raise ValueError('Gnomonic projection singular for given coordinates')
        x_rad = cos_d * math.sin(dra) / denom
        y_rad = (sin_d * cos_d0 - cos_d * sin_d0 * math.cos(dra)) / denom

        # x_rad/y_rad are in radians on tangent plane; convert to degrees
        x_deg = math.degrees(x_rad)
        y_deg = math.degrees(y_rad)

        if self.cd is not None:
            # apply inverse CD to get pixels: [dx_deg, dy_deg] = CD @ [dpx, dpy]
            # so pixels = inv(CD) @ [dx_deg, dy_deg]
            try:
                invcd = np.linalg.inv(self.cd)
                pix = invcd @ np.array([x_deg, y_deg], dtype=float)
                dx_pix, dy_pix = float(pix[0]), float(pix[1])
            except Exception:
                raise
        else:
            if self.pixscale_deg is None:
                raise ValueError('No CD or pixscale defined in teldef')
            dx_pix = x_deg / float(self.pixscale_deg)
            dy_pix = y_deg / float(self.pixscale_deg)

        x_pix = float(self.crpix[0]) + dx_pix
        y_pix = float(self.crpix[1]) + dy_pix
        return x_pix, y_pix

    def det_to_sky(self, x_pix: float, y_pix: float) -> Tuple[float, float]:
        """把 detector 像素 (x,y) 转换为 sky (ra,dec)（逆向投影）。"""
        if self.crval is None or self.crpix is None:
            raise ValueError('Teldef lacks CRVAL/CRPIX required for det->sky')
        dx = float(x_pix) - float(self.crpix[0])
        dy = float(y_pix) - float(self.crpix[1])
        if self.cd is not None:
            d_deg = self.cd @ np.array([dx, dy], dtype=float)
            x_deg, y_deg = float(d_deg[0]), float(d_deg[1])
        else:
            if self.pixscale_deg is None:
                raise ValueError('No CD or pixscale defined in teldef')
            x_deg = dx * float(self.pixscale_deg)
            y_deg = dy * float(self.pixscale_deg)

        x_rad = math.radians(x_deg)
        y_rad = math.radians(y_deg)
        ra0 = math.radians(self.crval[0])
        dec0 = math.radians(self.crval[1])
        denom = math.cos(dec0) - y_rad * math.sin(dec0)
        if abs(denom) < 1e-15:
            raise ValueError('Inverse gnomonic singular')
        ra = ra0 + math.atan2(x_rad, denom)
        dec = math.atan2(y_rad * math.cos(dec0) + math.sin(dec0), math.sqrt(denom * denom + x_rad * x_rad))
        return math.degrees(ra), math.degrees(dec)

    def sky_to_det_with_pointing(self, ra_deg: float, dec_deg: float, ra_pnt: float, dec_pnt: float) -> Tuple[float, float]:
        """Convert sky RA/Dec to detector pixels using ALIGNM/FOCALLEN teldef.

        This implements the common Teldef convention:
        - project RA/Dec into tangent plane about boresight (ra_pnt,dec_pnt)
        - convert tangent-plane radians to focal-plane mm via focal_length
        - rotate vector by inverse ALIGNM to detector frame
        - convert mm -> pixels using detector scales and optical axis
        """
        if getattr(self, 'align', None) is None or getattr(self, 'focal_length', None) is None:
            raise ValueError('Teldef missing ALIGNM/FOCALLEN')
        # compute tangent-plane coords in radians (gnomonic) relative to boresight
        ra0 = math.radians(ra_pnt)
        dec0 = math.radians(dec_pnt)
        ra = math.radians(ra_deg)
        dec = math.radians(dec_deg)
        dra = ra - ra0
        dra = (dra + math.pi) % (2 * math.pi) - math.pi
        sin_d0 = math.sin(dec0)
        cos_d0 = math.cos(dec0)
        sin_d = math.sin(dec)
        cos_d = math.cos(dec)
        denom = sin_d * sin_d0 + cos_d * cos_d0 * math.cos(dra)
        if denom == 0:
            raise ValueError('Gnomonic projection singular for given coordinates')
        x_rad = cos_d * math.sin(dra) / denom
        y_rad = (sin_d * cos_d0 - cos_d * sin_d0 * math.cos(dra)) / denom
        # focal plane mm
        mm_x = x_rad * float(self.focal_length)
        mm_y = y_rad * float(self.focal_length)
        mm_z = float(self.focal_length)
        vec_foc = np.array([mm_x, mm_y, mm_z], dtype=float)
        # rotate into detector frame: vec_det = inv(ALIGNM) @ vec_foc
        try:
            # alignment matrices in teldef are often orthonormal rotations; use transpose
            inv_align = self.align.T
            vec_det = inv_align @ vec_foc
        except Exception:
            raise
        # convert mm to pixels using detector scale and optical axis
        sx = float(self.det_xscl) if getattr(self, 'det_xscl', None) is not None else 1.0
        sy = float(self.det_yscl) if getattr(self, 'det_yscl', None) is not None else 1.0
        optx = float(self.optaxis[0]) if getattr(self, 'optaxis', None) is not None and self.optaxis[0] is not None else (float(self.det_xpix1) + (float(self.det_xsiz)-1)/2.0 if getattr(self,'det_xpix1',None) is not None and getattr(self,'det_xsiz',None) is not None else 0.0)
        opty = float(self.optaxis[1]) if getattr(self, 'optaxis', None) is not None and self.optaxis[1] is not None else (float(self.det_ypix1) + (float(self.det_ysiz)-1)/2.0 if getattr(self,'det_ypix1',None) is not None and getattr(self,'det_ysiz',None) is not None else 0.0)
        # detector coordinates in pixels
        x_pix = vec_det[0] / sx + optx
        y_pix = vec_det[1] / sy + opty
        return float(x_pix), float(y_pix)

    def det_to_sky_with_pointing(self, x_pix: float, y_pix: float, ra_pnt: float, dec_pnt: float) -> Tuple[float, float]:
        """Convert detector pixels to sky RA/Dec using ALIGNM/FOCALLEN teldef.

        Reverse of `sky_to_det_with_pointing`.
        """
        if getattr(self, 'align', None) is None or getattr(self, 'focal_length', None) is None:
            raise ValueError('Teldef missing ALIGNM/FOCALLEN')
        sx = float(self.det_xscl) if getattr(self, 'det_xscl', None) is not None else 1.0
        sy = float(self.det_yscl) if getattr(self, 'det_yscl', None) is not None else 1.0
        optx = float(self.optaxis[0]) if getattr(self, 'optaxis', None) is not None and self.optaxis[0] is not None else (float(self.det_xpix1) + (float(self.det_xsiz)-1)/2.0 if getattr(self,'det_xpix1',None) is not None and getattr(self,'det_xsiz',None) is not None else 0.0)
        opty = float(self.optaxis[1]) if getattr(self, 'optaxis', None) is not None and self.optaxis[1] is not None else (float(self.det_ypix1) + (float(self.det_ysiz)-1)/2.0 if getattr(self,'det_ypix1',None) is not None and getattr(self,'det_ysiz',None) is not None else 0.0)
        mm_x = (float(x_pix) - optx) * sx
        mm_y = (float(y_pix) - opty) * sy
        mm_z = float(self.focal_length)
        vec_det = np.array([mm_x, mm_y, mm_z], dtype=float)
        # rotate into focal/satellite frame (use transpose of ALIGNM)
        vec_foc = self.align.T @ vec_det
        # get tangent plane coords (radians)
        x_rad = vec_foc[0] / vec_foc[2]
        y_rad = vec_foc[1] / vec_foc[2]
        # inverse gnomonic to RA/Dec using boresight
        ra0 = math.radians(ra_pnt)
        dec0 = math.radians(dec_pnt)
        denom = math.cos(dec0) - y_rad * math.sin(dec0)
        if abs(denom) < 1e-15:
            raise ValueError('Inverse gnomonic singular')
        ra = ra0 + math.atan2(x_rad, denom)
        dec = math.atan2(y_rad * math.cos(dec0) + math.sin(dec0), math.sqrt(denom * denom + x_rad * x_rad))
        return math.degrees(ra), math.degrees(dec)

    def setSkyCoordCenterInTeldef(self, ra_deg: float, dec_deg: float, roll_deg: float = 0.0) -> None:
        """Set the nominal pointing quaternion and rotation matrix for this Teldef.

        This mirrors coordfits' `setSkyCoordCenterInTeldef`: it computes a
        quaternion `q0` that represents the nominal pointing (ra,dec,roll)
        and stores `rot0` as the equivalent 3x3 rotation matrix for faster
        repeated calculations.
        """
        # use helper to build quaternion from RA/Dec/Roll
        q0 = _th.radecroll_to_quat(float(ra_deg), float(dec_deg), float(roll_deg))
        self.q0 = q0
        # rotation matrix equivalent
        self.rot0 = _th.quat_to_rotmatrix(q0)

    def build_focal_to_pixel_xform(self) -> None:
        """Build and cache an XFORM2D mapping focal-plane mm coords to detector pixels.

        Stores result in `self.focal_to_pixel_xform`. Requires that `align`,
        `focal_length`, `det_xscl`, `det_yscl`, and `optaxis` are set on the Teldef.
        """
        if getattr(self, 'align', None) is None:
            raise ValueError('Teldef has no align matrix to build focal->pixel xform')
        if getattr(self, 'focal_length', None) is None:
            raise ValueError('Teldef missing focal_length')
        if getattr(self, 'det_xscl', None) is None or getattr(self, 'det_yscl', None) is None:
            raise ValueError('Teldef missing detector scale')
        if getattr(self, 'optaxis', None) is None:
            raise ValueError('Teldef missing optaxis')
        inv_align = np.asarray(self.align, dtype=float).T
        self.focal_to_pixel_xform = _th.build_focal_to_pixel_xform(inv_align,
                                                                   float(self.focal_length),
                                                                   float(self.det_xscl),
                                                                   float(self.det_yscl),
                                                                   self.optaxis)

    def convert_detector_to_sky(self, x_pix: float, y_pix: float,
                                ra_pnt: float | None = None, dec_pnt: float | None = None,
                                q: Optional[Sequence[float]] = None,
                                v: float = 0.0, vhat: Optional[Sequence[float]] = None) -> Tuple[float, float]:
        """Convert detector pixels to sky RA/Dec.

        Preferred usage is to provide `ra_pnt` and `dec_pnt` (boresight pointing)
        which will be used to perform inverse gnomonic projection after
        transforming detector->focal.

        If `q` (spacecraft quaternion) is provided without `ra_pnt`/`dec_pnt`,
        this code currently raises NotImplementedError — extracting RA/Dec from
        quaternion will be implemented in a later step.

        This method caches the last pointing used so that repeated conversions
        with the same pointing can use `repeat_detector_to_sky`.
        """
        # If ra/dec provided, delegate to existing method
        if ra_pnt is not None and dec_pnt is not None:
            # cache pointing
            self._last_pointing = (float(ra_pnt), float(dec_pnt))
            # if a det2sky fast-path exists, ensure aberration params are set
            if getattr(self, 'det2sky', None) is not None:
                if vhat is not None:
                    self.det2sky.aberration = (float(v), tuple(vhat))
                else:
                    self.det2sky.aberration = None
                return self.det2sky.apply(x_pix, y_pix)
            return self.det_to_sky_with_pointing(x_pix, y_pix, float(ra_pnt), float(dec_pnt))

        # If quaternion provided, not implemented yet
        if q is not None:
            # accept quaternion in any sequence form; convert to numpy array
            try:
                qq = np.asarray(q, dtype=float)
            except Exception:
                raise ValueError('Invalid quaternion provided')
            # convert quaternion to RA/Dec/Roll using helper
            ra_dec_roll = _th.quat_to_radecroll(qq)
            ra_pnt_calc, dec_pnt_calc, roll_calc = ra_dec_roll
            # cache and delegate (set aberration if provided)
            self._last_pointing = (float(ra_pnt_calc), float(dec_pnt_calc))
            if getattr(self, 'det2sky', None) is not None:
                if vhat is not None:
                    self.det2sky.aberration = (float(v), tuple(vhat))
                else:
                    self.det2sky.aberration = None
                return self.det2sky.apply(x_pix, y_pix)
            return self.det_to_sky_with_pointing(x_pix, y_pix, ra_pnt_calc, dec_pnt_calc)

        raise ValueError('Either ra_pnt/dec_pnt or q must be provided')

    def repeat_detector_to_sky(self, x_pix: float, y_pix: float) -> Tuple[float, float]:
        """Repeat last detector->sky conversion using cached pointing.

        Requires that `convert_detector_to_sky` was previously called with
        an explicit `ra_pnt`/`dec_pnt` and that cache is present.
        """
        pt = getattr(self, '_last_pointing', None)
        if pt is None:
            raise ValueError('No cached pointing available; call convert_detector_to_sky first')
        # if we have a cached det2sky fast-path, use it
        if getattr(self, 'det2sky_func', None) is not None and getattr(self, '_cached_det2sky_pointing', None) == pt:
            return self.det2sky_func(x_pix, y_pix)
        return self.det_to_sky_with_pointing(x_pix, y_pix, pt[0], pt[1])

    def _build_and_cache_det2sky(self, ra_pnt: float, dec_pnt: float) -> None:
        """Build a fast callable det2sky_func(x_pix,y_pix) for given pointing and cache it.

        The callable computes RA/Dec for a single detector pixel using the
        same math as `det_to_sky_with_pointing` but with precomputed constants
        to speed repeated conversions.
        """
        # precompute trig values for inverse gnomonic
        ra0 = math.radians(ra_pnt)
        dec0 = math.radians(dec_pnt)
        sin_d0 = math.sin(dec0)
        cos_d0 = math.cos(dec0)
        focal = float(self.focal_length)
        inv_align = self.align.T
        sx = float(self.det_xscl)
        sy = float(self.det_yscl)
        optx = float(self.optaxis[0]) if getattr(self, 'optaxis', None) is not None and self.optaxis[0] is not None else 0.0
        opty = float(self.optaxis[1]) if getattr(self, 'optaxis', None) is not None and self.optaxis[1] is not None else 0.0

        # construct Det2Sky object (encapsulates same math)
        from .teldef_helpers import Det2Sky, MapXform
        # if a raw_map (mapxform) exists on teldef, attach it (assume it is in pixel units)
        nonlinear = getattr(self, 'raw_map', None)
        # If a focal->pixel XFORM exists and the raw map is defined in a
        # different coordinate basis (e.g., CORINRAW or needs DET_ROTD),
        # pre-apply the xform to move the map into detector pixel coords.
        if nonlinear is not None and getattr(self, 'focal_to_pixel_xform', None) is not None:
            try:
                nonlinear = nonlinear.apply_xform2d(self.focal_to_pixel_xform)
            except Exception:
                # if pre-application fails, fall back to original map
                nonlinear = getattr(self, 'raw_map', None)
        # pass aberration placeholder if present on teldef
        aberr = getattr(self, 'aberration', None)
        det2sky_obj = Det2Sky(inv_align, focal, sx, sy, (optx, opty), ra_pnt, dec_pnt, nonlinear=nonlinear, aberration=aberr)
        self.det2sky = det2sky_obj
        self.det2sky_func = det2sky_obj.apply
        self._cached_det2sky_pointing = (ra_pnt, dec_pnt)

    def compute_det2sky_from_quaternion(self, q: Sequence[float]) -> None:
        """Compute and cache det2sky using a spacecraft quaternion.

        Stores `self.delta_q` (relative to `q0` if present) and builds the
        cached det2sky function for the pointing encoded by `q`.
        """
        qq = np.asarray(q, dtype=float)
        qq = qq / np.linalg.norm(qq)
        # compute delta_q relative to q0 if available
        if getattr(self, 'q0', None) is not None:
            self.delta_q = _th.quat_mul(qq, _th.quat_conjugate(self.q0))
        else:
            self.delta_q = None
        # derive RA/Dec/Roll from quaternion and build cached det2sky
        ra_pnt_calc, dec_pnt_calc, _roll = _th.quat_to_radecroll(qq)
        self._last_pointing = (float(ra_pnt_calc), float(dec_pnt_calc))
        self._build_and_cache_det2sky(float(ra_pnt_calc), float(dec_pnt_calc))
def find_teldef_from_event(ev) -> Optional[Teldef]:
    """尝试从 EventData 的 header 或 meta 中查找 teldef 文件并加载。

    查找顺序：header['TELDEF']、header['TELDEFNAME']、meta.detnam 对应的文件名。
    返回 Teldef 实例或 None（找不到或解析失败时）。
    """
    hdr = getattr(ev, 'header', None) or {}
    key_candidates = ['TELDEF', 'TELDEFNAME', 'TELDEFFILE', 'TELDEF_FILE']
    tel = None
    for k in key_candidates:
        v = hdr.get(k)
        if v:
            try:
                tel = Teldef.from_file(v)
                return tel
            except Exception:
                continue

    # try meta.detnam as a file name in working dir
    meta = getattr(ev, 'meta', None)
    detnam = getattr(meta, 'detnam', None) if meta is not None else None
    if detnam:
        # try to find a teldef file named detnam.teldef or TELDEF_{detnam}.txt
        candidates = [f"{detnam}.teldef", f"teldef_{detnam}.txt", detnam]
        for c in candidates:
            p = Path(c)
            if p.exists():
                try:
                    return Teldef.from_file(p)
                except Exception:
                    continue
    return None
