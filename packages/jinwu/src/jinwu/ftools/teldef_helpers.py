"""Teldef quaternion and rotation helpers.

提供最小的四元数与旋转矩阵工具集：构造、归一化、乘法、轴角转四元数、四元数->旋转矩阵、用四元数旋转向量等。
这些函数作为移植 coordfits 的基础工具，接口尽量简单、基于 numpy。
"""
from __future__ import annotations

import math
from typing import Sequence
import numpy as np


def quat_normalize(q: Sequence[float]) -> np.ndarray:
    """Normalize quaternion (w, x, y, z) and return numpy array."""
    a = np.asarray(q, dtype=float)
    norm = np.linalg.norm(a)
    if norm == 0:
        raise ValueError('Zero-length quaternion')
    return a / norm


def quat_mul(q1: Sequence[float], q2: Sequence[float]) -> np.ndarray:
    """Hamilton product q = q1 * q2 for quaternions in (w,x,y,z) order."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z], dtype=float)


def quat_conjugate(q: Sequence[float]) -> np.ndarray:
    """Return quaternion conjugate (w, -x, -y, -z)."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)


def quat_from_axis_angle(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    """Create quaternion from axis (x,y,z) and rotation angle in radians.

    The axis does not need to be normalized.
    Returns (w, x, y, z).
    """
    ax = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(ax)
    if norm == 0:
        raise ValueError('Zero rotation axis')
    u = ax / norm
    s = math.sin(angle_rad / 2.0)
    w = math.cos(angle_rad / 2.0)
    x, y, z = u * s
    return np.array([w, x, y, z], dtype=float)


def quat_to_rotmatrix(q: Sequence[float]) -> np.ndarray:
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix.

    Uses the convention that vectors are rotated as v' = R v where R is
    built from the quaternion q.
    """
    w, x, y, z = quat_normalize(q)
    # Rotation matrix elements (source: standard quaternion to matrix)
    R = np.empty((3, 3), dtype=float)
    R[0, 0] = 1 - 2*(y*y + z*z)
    R[0, 1] = 2*(x*y - z*w)
    R[0, 2] = 2*(x*z + y*w)
    R[1, 0] = 2*(x*y + z*w)
    R[1, 1] = 1 - 2*(x*x + z*z)
    R[1, 2] = 2*(y*z - x*w)
    R[2, 0] = 2*(x*z - y*w)
    R[2, 1] = 2*(y*z + x*w)
    R[2, 2] = 1 - 2*(x*x + y*y)
    return R


def rotate_vector_by_quat(q: Sequence[float], v: Sequence[float]) -> np.ndarray:
    """Rotate a 3-vector `v` by quaternion `q` (w,x,y,z)."""
    # Use matrix form for clarity / numeric stability
    R = quat_to_rotmatrix(q)
    return R @ np.asarray(v, dtype=float)


class XFORM2D:
    """Simple 2D linear transform with optional translation.

    Represents transform: [x_out, y_out]^T = M @ [x_in, y_in]^T + t
    where M is 2x2 matrix and t is 2-vector.
    """
    def __init__(self, matrix=None, offset=None):
        if matrix is None:
            matrix = np.eye(2, dtype=float)
        if offset is None:
            offset = np.zeros(2, dtype=float)
        self.matrix = np.asarray(matrix, dtype=float).reshape((2,2))
        self.offset = np.asarray(offset, dtype=float).reshape((2,))

    def apply(self, x: float, y: float) -> tuple[float, float]:
        v = self.matrix @ np.array([x, y], dtype=float) + self.offset
        return float(v[0]), float(v[1])

    def apply_array(self, arr: np.ndarray) -> np.ndarray:
        """Apply to Nx2 array of coordinates, returns Nx2 array."""
        a = np.asarray(arr, dtype=float)
        return (a @ self.matrix.T) + self.offset

    def inverse(self) -> 'XFORM2D':
        """Return inverse XFORM2D (such that inv.apply(*self.apply(x,y)) == (x,y))."""
        invm = np.linalg.inv(self.matrix)
        invoff = - (invm @ self.offset)
        return XFORM2D(matrix=invm, offset=invoff)


def rotmatrix_to_xform2d(R: np.ndarray, focal_length: float = 1.0) -> XFORM2D:
    """Construct a 2D perspective-like linear mapping from a 3x3 rotation matrix.

    This helper produces a small linear approximation suitable for small-angle
    tangential projections: maps focal-plane mm x,y (pre-rotation) into
    detector x,y by taking the first two rows of R and scaling by focal length.
    The operation is simplistic and is provided as a convenience; the full
    coordfits behavior uses XFORM2D with perspective projection and more
    complex handling — that will be implemented in later steps.
    """
    R = np.asarray(R, dtype=float).reshape((3,3))
    # map focal-plane coords (x_mm, y_mm, z_mm=focal) to detector mm via R @ v
    # approximate linear mapping from (x_mm, y_mm) -> detector_x_mm, detector_y_mm
    # by ignoring dependence on z_mm variations (assume z fixed = focal_length)
    M = R[:2, :2]  # 2x2 block
    off = R[:2, 2] * focal_length
    return XFORM2D(matrix=M * 1.0, offset=off)


def build_focal_to_pixel_xform(inv_align: np.ndarray, focal_length: float, det_xscl: float, det_yscl: float, optaxis: Sequence[float]) -> XFORM2D:
    """Build XFORM2D that maps focal-plane mm coords (x_mm, y_mm) to detector pixels.

    Given inv_align (3x3) which maps focal vector to detector vector (mm), and
    a fixed focal_length (z_mm for focal point), this constructs a linear mapping
    from [x_mm, y_mm] -> [x_pix, y_pix]:

        vec_det = inv_align @ [x_mm, y_mm, focal_length]
        x_pix = vec_det[0] / det_xscl + optx
        y_pix = vec_det[1] / det_yscl + opty

    Returns an XFORM2D implementing pixels = M @ [x_mm, y_mm] + offset
    """
    invA = np.asarray(inv_align, dtype=float).reshape((3,3))
    A = invA[:2, :2]
    b = invA[:2, 2] * float(focal_length)
    sx = float(det_xscl)
    sy = float(det_yscl)
    M = np.array([[A[0,0]/sx, A[0,1]/sx], [A[1,0]/sy, A[1,1]/sy]], dtype=float)
    off = np.array([b[0]/sx + float(optaxis[0]), b[1]/sy + float(optaxis[1])], dtype=float)
    return XFORM2D(matrix=M, offset=off)


class Det2Sky:
    """Class representing a detector->sky mapping for a fixed pointing.

    This encapsulates the math used in `Teldef._build_and_cache_det2sky` but
    as an object so it can store optional non-linear corrections and
    aberration parameters. The `apply` method returns (ra_deg, dec_deg).
    """
    def __init__(self, inv_align: np.ndarray, focal: float, det_xscl: float, det_yscl: float,
                 optaxis: Sequence[float], ra_pnt_deg: float, dec_pnt_deg: float,
                 nonlinear=None, aberration=None):
        self.inv_align = np.asarray(inv_align, dtype=float).reshape((3,3))
        self.focal = float(focal)
        self.det_xscl = float(det_xscl)
        self.det_yscl = float(det_yscl)
        self.optaxis = (float(optaxis[0]), float(optaxis[1]))
        self.ra_pnt = float(ra_pnt_deg)
        self.dec_pnt = float(dec_pnt_deg)
        self.nonlinear = nonlinear  # placeholder for non-linear raw->det corrections
        self.aberration = aberration  # placeholder for aberration params

        # precompute trig
        self._ra0 = math.radians(self.ra_pnt)
        self._dec0 = math.radians(self.dec_pnt)
        self._sin_dec0 = math.sin(self._dec0)
        self._cos_dec0 = math.cos(self._dec0)

    def apply(self, x_pix: float, y_pix: float) -> tuple[float, float]:
        # apply non-linear corrections if any
        x_p = float(x_pix)
        y_p = float(y_pix)
        if self.nonlinear is not None:
            try:
                dx_pix, dy_pix = self.nonlinear.sample_delta_at_pixel(x_p, y_p)
                x_p = x_p + dx_pix
                y_p = y_p + dy_pix
            except Exception:
                # on any interpolation error, fall back to uncorrected coords
                pass
        # convert to mm
        mm_x = (x_p - self.optaxis[0]) * self.det_xscl
        mm_y = (y_p - self.optaxis[1]) * self.det_yscl
        mm_z = self.focal
        vec_det = np.array([mm_x, mm_y, mm_z], dtype=float)
        vec_foc = self.inv_align @ vec_det
        x_rad = vec_foc[0] / vec_foc[2]
        y_rad = vec_foc[1] / vec_foc[2]
        denom = self._cos_dec0 - y_rad * self._sin_dec0
        if abs(denom) < 1e-15:
            raise ValueError('Inverse gnomonic singular')
        ra = self._ra0 + math.atan2(x_rad, denom)
        dec = math.atan2(y_rad * self._cos_dec0 + self._sin_dec0, math.sqrt(denom * denom + x_rad * x_rad))
        ra_deg = math.degrees(ra)
        dec_deg = math.degrees(dec)
        # apply aberration correction if requested
        if self.aberration is not None:
            try:
                v_kms, vhat = self.aberration
                ra_deg, dec_deg = apply_aberration_to_radec(ra_deg, dec_deg, float(v_kms), vhat)
            except Exception:
                pass
        return ra_deg, dec_deg

    def __call__(self, x_pix: float, y_pix: float) -> tuple[float, float]:
        return self.apply(x_pix, y_pix)


class MapXform:
    """Simple map-based distortion stored as deltax/deltay arrays.

    The map is defined by origin (x0,y0) and scale (sx,sy) such that map
    pixel indices i,j correspond to detector coordinates:
        x = x0 + i * sx
        y = y0 + j * sy

    deltax/deltay arrays are stored in pixels (delta in pixels) and
    sampled with bilinear interpolation.
    """
    def __init__(self, deltax: np.ndarray, deltay: np.ndarray, origin_x: float, scale_x: float, origin_y: float, scale_y: float):
        self.deltax = np.asarray(deltax, dtype=float)
        self.deltay = np.asarray(deltay, dtype=float)
        if self.deltax.shape != self.deltay.shape:
            raise ValueError('deltax/deltay shape mismatch')
        self.dimenx = self.deltax.shape[1]
        self.dimeny = self.deltax.shape[0]
        self.origin_x = float(origin_x)
        self.scale_x = float(scale_x)
        self.origin_y = float(origin_y)
        self.scale_y = float(scale_y)

    def sample_delta_at_pixel(self, x_pix: float, y_pix: float) -> tuple[float, float]:
        """Return (deltax_pix, deltay_pix) at detector pixel coordinates using bilinear interpolation."""
        # map pixel coordinates to fractional map indices
        fx = (x_pix - self.origin_x) / self.scale_x
        fy = (y_pix - self.origin_y) / self.scale_y
        # fx corresponds to column index, fy to row index
        if fx < 0 or fy < 0 or fx > (self.dimenx - 1) or fy > (self.dimeny - 1):
            # outside map: return zero delta
            return 0.0, 0.0
        i0 = int(np.floor(fx))
        j0 = int(np.floor(fy))
        i1 = min(i0 + 1, self.dimenx - 1)
        j1 = min(j0 + 1, self.dimeny - 1)
        tx = fx - i0
        ty = fy - j0
        # bilinear interp
        v00x = self.deltax[j0, i0]
        v10x = self.deltax[j0, i1]
        v01x = self.deltax[j1, i0]
        v11x = self.deltax[j1, i1]
        vx0 = v00x * (1 - tx) + v10x * tx
        vx1 = v01x * (1 - tx) + v11x * tx
        vx = vx0 * (1 - ty) + vx1 * ty

        v00y = self.deltay[j0, i0]
        v10y = self.deltay[j0, i1]
        v01y = self.deltay[j1, i0]
        v11y = self.deltay[j1, i1]
        vy0 = v00y * (1 - tx) + v10y * tx
        vy1 = v01y * (1 - tx) + v11y * tx
        vy = vy0 * (1 - ty) + vy1 * ty
        return float(vx), float(vy)

    def apply_xform2d(self, xform: XFORM2D, out_shape: tuple[int,int] | None = None) -> 'MapXform':
        """Return a new MapXform representing this map after applying `xform`.

        The implementation resamples the current deltas onto a grid that is
        transformed by `xform`. By default the output map has the same shape
        as the source (same number of cols/rows). For each output map cell
        (i,j) we compute its detector coordinates and map them back through
        the inverse of `xform` to sample the original deltas.

        This is a pragmatic regridding approach sufficient for many use
        cases; it does not attempt to analytically reproject the residuals.
        """
        inv = xform.inverse()
        # output shape
        out_nx = self.dimenx
        out_ny = self.dimeny
        if out_shape is not None:
            out_ny, out_nx = out_shape

        new_dx = np.zeros((out_ny, out_nx), dtype=float)
        new_dy = np.zeros((out_ny, out_nx), dtype=float)

        # determine new origin and scale by applying xform to original origin
        new_origin_x, new_origin_y = xform.apply(self.origin_x, self.origin_y)
        # compute effective scale by mapping one-step in x and y
        sx_x, sx_y = xform.apply(self.origin_x + self.scale_x, self.origin_y)
        sy_x, sy_y = xform.apply(self.origin_x, self.origin_y + self.scale_y)
        # approximate new scales as magnitudes along transformed axes
        new_scale_x = math.hypot(sx_x - new_origin_x, sx_y - new_origin_y)
        new_scale_y = math.hypot(sy_x - new_origin_x, sy_y - new_origin_y)

        # compute detector coords for each output grid cell using new origin/scale
        for j in range(out_ny):
            for i in range(out_nx):
                x_out = new_origin_x + i * new_scale_x
                y_out = new_origin_y + j * new_scale_y
                # map back to source detector coords
                src_x, src_y = inv.apply(x_out, y_out)
                # sample original deltas at that source detector coord
                dx, dy = self.sample_delta_at_pixel(src_x, src_y)
                new_dx[j, i] = dx
                new_dy[j, i] = dy

        return MapXform(new_dx, new_dy, new_origin_x, new_scale_x, new_origin_y, new_scale_y)


def radecroll_to_quat(ra_deg: float, dec_deg: float, roll_deg: float) -> np.ndarray:
    """Construct a quaternion from RA, Dec, Roll (degrees).

    This produces a quaternion that rotates the instrument frame so that
    the +Z axis points toward (RA,Dec) on the celestial sphere and the
    roll specifies rotation about that axis. The exact sign convention
    is the common aerospace convention: roll positive rotates the
    instrument x-axis toward the instrument y-axis.

    The returned quaternion is in (w,x,y,z) order and is normalized.
    """
    # Build rotation: first rotate by RA around Z, then by (90-dec) around X?
    # A robust approach: build target Z unit vector then compute rotation
    # from original z=(0,0,1) to target and apply roll about target axis.
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    roll = math.radians(roll_deg)

    # target unit vector in celestial cartesian coords
    tz = np.array([math.cos(dec)*math.cos(ra), math.cos(dec)*math.sin(ra), math.sin(dec)], dtype=float)

    # rotation to align +Z -> tz: axis = cross(z, tz)
    z = np.array([0.0, 0.0, 1.0], dtype=float)
    axis = np.cross(z, tz)
    norm_axis = np.linalg.norm(axis)
    if norm_axis < 1e-15:
        # tz is parallel to z (north or south pole)
        # rotation angle is 0 if same, pi if opposite
        dot = np.dot(z, tz)
        if dot > 0:
            q_align = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        else:
            # 180-degree rotation about X (or any orthogonal axis)
            q_align = quat_from_axis_angle([1.0, 0.0, 0.0], math.pi)
    else:
        axis_unit = axis / norm_axis
        angle = math.acos(np.clip(np.dot(z, tz), -1.0, 1.0))
        q_align = quat_from_axis_angle(axis_unit, angle)

    # roll about target axis tz (positive roll rotates x->y)
    q_roll = quat_from_axis_angle(tz, roll)

    # total rotation: first align, then roll in target frame -> q = q_roll * q_align
    q = quat_mul(q_roll, q_align)
    return quat_normalize(q)


def quat_to_radecroll(q: Sequence[float]) -> tuple[float, float, float]:
    """Convert quaternion (w,x,y,z) to RA (deg), Dec (deg), Roll (deg).

    This is the inverse of `radecroll_to_quat` under the same conventions.
    """
    qn = quat_normalize(q)
    R = quat_to_rotmatrix(qn)
    # rotated +Z axis
    tz = R @ np.array([0.0, 0.0, 1.0])
    # RA/Dec from tz
    dec = math.asin(np.clip(tz[2], -1.0, 1.0))
    ra = math.atan2(tz[1], tz[0])

    # Build quaternion that aligns +Z -> tz (same convention as radecroll_to_quat)
    z = np.array([0.0, 0.0, 1.0], dtype=float)
    axis = np.cross(z, tz)
    norm_axis = np.linalg.norm(axis)
    if norm_axis < 1e-15:
        dotz = np.dot(z, tz)
        if dotz > 0:
            q_align = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        else:
            q_align = quat_from_axis_angle([1.0, 0.0, 0.0], math.pi)
    else:
        axis_unit = axis / norm_axis
        angle = math.acos(np.clip(np.dot(z, tz), -1.0, 1.0))
        q_align = quat_from_axis_angle(axis_unit, angle)

    # q = q_roll * q_align  -> q_roll = q * inverse(q_align)
    q_inv_align = quat_conjugate(q_align)
    q_roll = quat_mul(q, q_inv_align)
    q_roll = quat_normalize(q_roll)

    # extract roll angle from q_roll: should be rotation about tz axis
    w = float(q_roll[0])
    w = max(min(w, 1.0), -1.0)
    phi = 2.0 * math.acos(w)
    s = math.sin(phi / 2.0)
    if abs(s) < 1e-15:
        roll = 0.0
    else:
        v = q_roll[1:] / s
        sign = 1.0 if float(np.dot(v, tz)) >= 0.0 else -1.0
        roll = sign * phi

    return (math.degrees(ra), math.degrees(dec), math.degrees(roll))


def apply_aberration_to_radec(ra_deg: float, dec_deg: float, v_kms: float, vhat: Sequence[float]) -> tuple[float, float]:
    """Apply an approximate relativistic aberration correction to (ra,dec).

    Parameters
    - ra_deg, dec_deg: input celestial direction in degrees
    - v_kms: spacecraft velocity magnitude in km/s
    - vhat: unit-vector direction of velocity (3-vector)

    Returns aberrated (ra_deg, dec_deg).

    Note: This implements a standard SR aberration formula using beta=v/c.
    It's an approximation adequate for small v<<c (e.g., <0.001 c) which is
    valid for spacecraft/earth motions. The sign convention applied returns
    the apparent direction seen in the moving frame.
    """
    # speed of light in km/s
    c = 299792.458
    beta = float(v_kms) / c
    if beta == 0.0:
        return ra_deg, dec_deg
    # direction unit vector for the source
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    s = np.array([math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)], dtype=float)
    bhat = np.asarray(vhat, dtype=float)
    bnorm = np.linalg.norm(bhat)
    if bnorm == 0:
        return ra_deg, dec_deg
    bdir = bhat / bnorm
    beta_vec = bdir * beta
    b2 = np.dot(beta_vec, beta_vec)
    gamma = 1.0 / math.sqrt(1.0 - b2)

    # aberration formula (source -> moving frame apparent direction)
    s_dot_b = float(np.dot(s, beta_vec))
    numerator = s + ((gamma - 1.0) * s_dot_b / b2) * beta_vec + gamma * beta_vec
    denom = gamma * (1.0 + s_dot_b)
    s_prime = numerator / denom
    # normalize
    s_prime = s_prime / np.linalg.norm(s_prime)

    dec_p = math.asin(np.clip(s_prime[2], -1.0, 1.0))
    ra_p = math.atan2(s_prime[1], s_prime[0])
    return math.degrees(ra_p), math.degrees(dec_p)
