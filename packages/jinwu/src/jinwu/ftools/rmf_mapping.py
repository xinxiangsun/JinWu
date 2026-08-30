"""RMF -> energy 映射工具模块

提供函数把观测到的道/PI 映射到能量（keV），支持：
- 基于 RMF 后验 P(E|C) 计算期望能量（expected）
- 基于 RMF 后验按能量格离散采样（sample）
- EBOUNDS 中点的回退策略

该实现尽量向量化：对大量事件先对唯一道号做一次后验计算/采样，然后将结果广播回事件数组。
使用 numba 加速核心循环以提升性能。
使用时请注意传入 RMF 矩阵的方向：本模块接受 `matrix` 形状为 (n_channels, n_energies) 或 (n_energies, n_channels)，
函数会自动检测并调整。
"""

from __future__ import annotations

from typing import Optional, Sequence, Any

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

__all__ = [
    'ebounds_midpoints',
    'map_channels_to_energy',
]


def ebounds_midpoints(e_min: Sequence[float], e_max: Sequence[float]) -> np.ndarray:
    """Compute midpoints from EBOUNDS arrays.

    Parameters
    - e_min/e_max: sequences of same length

    Returns
    - midpoints: ndarray of shape (N,)
    """
    e_min = np.asarray(e_min, dtype=float)
    e_max = np.asarray(e_max, dtype=float)
    if e_min.shape != e_max.shape:
        raise ValueError('e_min and e_max must have same shape')
    return 0.5 * (e_min + e_max)


def map_channels_to_energy(matrix: np.ndarray, e_centers: np.ndarray, channels: np.ndarray, *, method: str = 'expected', prior: Optional[np.ndarray] = None, seed: Optional[int] = None, prefer_sparse: bool = True, dtype: Any = np.float32) -> np.ndarray:
    """Map observed channels to energy values using RMF matrix.

    Parameters
    - matrix: 2D array either (n_channels, n_energies) or (n_energies, n_channels).
    - e_centers: 1D array of length n_energies with energy centers (keV).
    - channels: 1D integer array of observed channel indices (may be repeated).
    - method: 'expected' (return posterior mean) or 'sample' (draw one sample per event from posterior).
    - prior: optional prior over energies (length n_energies). If None, uniform prior is used.
    - seed: optional RNG seed for sampling mode.

    Returns
    - arr: 1D array of mapped energies (same length as `channels`).

    Notes
    - The function vectorizes work over unique channel values to avoid recomputing posteriors.
    - If a channel has zero likelihood across all energies, the mapped value is np.nan.
    """
    # Do not convert the input matrix to ndarray immediately because it
    # may already be a SciPy sparse matrix. Decide orientation first
    # using the matrix shape, then convert/prepare sparse or dense
    # representations as needed.
    e_centers = np.asarray(e_centers, dtype=dtype)
    channels = np.asarray(channels)

    # matrix may be ndarray or sparse; get its shape
    try:
        n0, n1 = matrix.shape
    except Exception:
        raise ValueError('matrix must be 2D')

    # Determine orientation: prefer (n_channels, n_energies)
    need_transpose = False
    if n1 == e_centers.size and n0 != e_centers.size:
        mat_obj = matrix  # (n_channels, n_energies)
    elif n0 == e_centers.size and n1 != e_centers.size:
        # Need to transpose the underlying object; handle sparse/dense later
        mat_obj = matrix
        need_transpose = True
    elif n0 == e_centers.size and n1 == e_centers.size:
        # square ambiguous: assume rows correspond to energies
        mat_obj = matrix
        need_transpose = True
    else:
        # Fallback: assume rows are channels
        mat_obj = matrix

    # If transpose required, create a transposed view (works for sparse and dense)
    if need_transpose:
        mat_obj = mat_obj.T

    # Now mat_obj is the matrix with shape (n_channels, n_energies)
    n_channels, n_energies = mat_obj.shape

    # prepare prior
    if prior is None:
        prior = np.ones(n_energies, dtype=dtype)
    prior = np.asarray(prior, dtype=dtype)
    if prior.size != n_energies:
        raise ValueError('prior length must match number of energy bins')
    # normalize prior
    if float(prior.sum()) <= 0:
        prior = np.ones_like(prior)
    prior = prior / float(prior.sum())

    # Decide on sparse path if requested and SciPy available
    use_sparse = False
    mat_sparse = None
    if prefer_sparse:
        try:
            from scipy.sparse import csr_matrix as _csr
            # if mat_obj exposes tocsr (is already sparse), use that
            if hasattr(mat_obj, 'tocsr'):
                mat_sparse = mat_obj.tocsr()
                use_sparse = True
            else:
                # attempt conversion (caller asked to prefer sparse)
                try:
                    mat_sparse = _csr(mat_obj)
                    use_sparse = True
                except Exception:
                    use_sparse = False
        except Exception:
            use_sparse = False

    # Work on unique channels to be efficient
    uniq, inv = np.unique(channels, return_inverse=True)
    uniq_i = np.asarray(uniq, dtype=int)
    out_vals = np.full(uniq_i.shape, np.nan, dtype=dtype)

    if use_sparse:
        # posterior_unnorm = P(C|E) * prior(E)  (sparse multiply by dense prior along columns)
        posterior_unnorm = mat_sparse.multiply(prior[np.newaxis, :])
        denom = np.asarray(posterior_unnorm.sum(axis=1)).ravel()

        if method == 'expected':
            numerator = np.asarray(posterior_unnorm.multiply(e_centers[np.newaxis, :]).sum(axis=1)).ravel()
            nz = denom > 0
            out_vals[nz] = (numerator[nz] / denom[nz]).astype(dtype)
        elif method == 'sample':
            rng = np.random.default_rng(seed)
            for i, c in enumerate(uniq_i):
                if (c < 0) or (c >= n_channels):
                    out_vals[i] = np.nan
                    continue
                if denom[c] <= 0:
                    out_vals[i] = np.nan
                    continue
                row = posterior_unnorm.getrow(c)
                cols = row.indices
                data = row.data.astype(dtype)
                probs = data / float(denom[c])
                if probs.size == 0:
                    out_vals[i] = np.nan
                    continue
                j = rng.choice(probs.size, p=probs)
                out_vals[i] = float(e_centers[cols[j]])
        else:
            raise ValueError('unknown method: ' + str(method))
    else:
        # Dense numpy path (fallback) — convert mat_obj to ndarray with dtype
        mat = np.asarray(mat_obj, dtype=dtype)
        posterior_unnorm = mat * prior[np.newaxis, :]
        denom = posterior_unnorm.sum(axis=1)
        posterior = np.zeros_like(posterior_unnorm)
        nz = denom > 0
        if np.any(nz):
            posterior[nz, :] = posterior_unnorm[nz, :] / denom[nz][:, None]

        for i, c in enumerate(uniq_i):
            if (c < 0) or (c >= n_channels):
                out_vals[i] = np.nan
                continue
            post = posterior[c]
            if post.sum() <= 0:
                out_vals[i] = np.nan
                continue
            if method == 'expected':
                out_vals[i] = float(np.dot(post, e_centers))
            elif method == 'sample':
                rng = np.random.default_rng(seed)
                idx = rng.choice(n_energies, p=post)
                out_vals[i] = float(e_centers[idx])
            else:
                raise ValueError('unknown method: ' + str(method))

    # Map back to original events
    out = out_vals[inv].astype(dtype)
    return out
