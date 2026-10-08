"""Counting-experiment significance with explicit background statistics.

The known-background, Poisson--Gaussian and systematic ON/OFF calculations
are derived from gv_significance, commit
946d761a41cc26bda21ead8b5ba7750dd5a389a4. Copyright (c) 2018 Giacomo
Vianello; BSD 3-Clause license, reproduced in ``GV_SIGNIFICANCE_LICENSE.txt``.
Reference: Vianello (2018), ApJS 236, 17, doi:10.3847/1538-4365/aab780,
https://arxiv.org/abs/1712.00118. Plain ON/OFF uses JinWu's existing Li--Ma
implementation, including its analytic zero-count limits.
"""

from __future__ import annotations

import math
from typing import Literal

import astropy.units as u
import numpy as np
from scipy import optimize, special

__all__ = ["snr", "li_ma_snr"]


def _values(value, name: str, *, counts: bool = False) -> np.ndarray:
    """在接口边界转为浮点数组 / Convert validated numeric inputs to float arrays.

    counts=True 时接受计数单位 Quantity 或无量纲值；否则仅接受无量纲。
    裸数直接转换，不自动乘曝光。所有值须有限，否则 ValueError；
    量纲不符抛 UnitConversionError。返回保留输入形状的 ndarray。
    With counts=True accept count-unit Quantity or dimensionless input; otherwise
    require dimensionless values. Bare numbers convert directly without exposure.
    Require finite values, raising ValueError or UnitConversionError as appropriate.
    Return an ndarray preserving shape; it may share input storage."""
    if isinstance(value, u.Quantity):
        if counts and value.unit.is_equivalent(u.ct):
            value = value.to_value(u.ct)
        else:
            try:
                value = value.to_value(u.dimensionless_unscaled)
            except u.UnitConversionError as exc:
                expected = "counts or dimensionless" if counts else "dimensionless"
                raise u.UnitConversionError(f"{name} must be {expected}") from exc
    result = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _xlogy(x, y):
    # Preserve the upstream zero-multiplier convention, including y=NaN.
    """保留上游零乘数规则 / Compute scalar x*log(y) with the upstream zero rule.

    x=0 返回 0，即使 y 为 NaN；否则 math.log 的域和类型异常原样上抛。
    x=0 returns 0 even for NaN y. Otherwise preserve math.log domain/type errors;
    this is a scalar helper, not a general broadcasting implementation."""
    return 0.0 if x == 0.0 else x * math.log(y)


def _xlogyv(x, y):
    """向量版上游 x*log(y) / Evaluate upstream masked x*log(y) arithmetic.

    x、y 至少一维化，x=0 元素不计算 log，包括 y=NaN 的零乘项。
    返回 squeeze 后结果；输入应为上游内核要求的相容形状，不提供完整
    NumPy 广播校验。使用 y 的 dtype 初始化输出，公共入口已保证浮点数。
    Promote x/y to at least 1-D, skipping logs at zero x including NaN y. Return
    squeezed results. Kernels require compatible shapes; this is not a general
    broadcast validator. Output dtype follows y; public inputs have float dtype."""
    x = np.array(x, ndmin=1)
    y = np.array(y, ndmin=1)
    results = np.zeros_like(y)
    idx = x != 0
    results[idx] = x[idx] * np.log(y[idx])
    return np.squeeze(results)


def _pg_profile(n, b, sigma):
    """PG 零源假设下的背景剖面 / Profile the PG background under the null source.

    n 为 ON 总计数，b 为高斯背景估计，sigma 为正的绝对背景标准误差，
    均以计数数值输入。返回 (B0, half_ts)：背景期望的零假设 MLE 与 TS/2。
    Public snr 已完成输入校验；这里保留冻结源码代数与计算顺序，不取根或符号。
    n is total ON counts, b a Gaussian background estimate, sigma a positive
    absolute error in count numbers. Return null background MLE B0 and TS/2.
    The public API validates inputs; preserve frozen source arithmetic here,
    without significance roots or signs.

    B0<=-0.01 时 ArithmeticError，允许范围内裁到非负。内部非有限/负
    half_ts 交由公共入口报错，而非偷偷改用其他统计量。
    Reject B0<=-0.01, otherwise clip to nonnegative. The public API rejects invalid
    half_ts instead of silently substituting a different statistical method."""
    B0 = 0.5 * (
        b - sigma**2
        + np.sqrt(b**2 - 2 * b * sigma**2 + 4 * n * sigma**2 + sigma**4)
    )
    if not np.all(B0 > -0.01):
        raise ArithmeticError("gv_significance PG null-background root is invalid")
    B0 = np.clip(B0, 0, None)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        half_ts = _xlogyv(n, n / B0) + (b - B0)**2 / (2 * sigma**2) + B0 - n
    return B0, np.atleast_1d(half_ts)


def _likelihood_with_sys(o, b, a, s, k, B, M):
    # Same barrier, likelihood and arithmetic as the frozen upstream source.
    """PP 相对系统误差的上游似然 / Evaluate the frozen PP systematic log likelihood.

    o/b 为 ON/OFF 计数，a 为名义比例，s 为相对高斯标准差，k 为偏差，
    B 为 OFF 背景期望，M 为 ON 纯源期望。保留源码的 -1000 有限屏障，
    不是 -inf；调用者须识别其导致的病态边界，不把屏障值当真实概率。
    o/b are ON/OFF counts, a nominal ratio, s relative Gaussian error, k offset,
    B OFF background mean and M source ON mean. Preserve the source's finite -1000
    barrier rather than -inf; callers guard its pathological boundary behavior.
    Return the source's likelihood value with constants omitted as in upstream."""
    if M + a * B <= 0 or k + 1 <= 0 or B <= 0:
        return -1000
    Ba = B * a
    Bak = B * a * k
    return -Bak - Ba - B - M + _xlogyv(b, B) - k**2 / (2 * s**2) + _xlogyv(o, Bak + Ba + M)


def _pp_gaussian_profile(n, b, alpha, sigma):
    """按冻结优化器剖面化 PP 系统误差 / Profile a PP Gaussian systematic with frozen settings.

    n/b 为 ON/OFF 计数，alpha 名义比例，sigma 为正的相对系统标准差。
    对单一 k 优化负的零源似然，初值 0、tol=1e-3，B 按源码解析表达式
    回代。返回 (TS, OptimizeResult)，公共入口决定有效性与显著性符号。
    Use ON/OFF counts, nominal alpha and positive relative sigma. Minimize the
    null negative likelihood over k with initial 0/tol=1e-3 and analytic B.
    Return TS and OptimizeResult; the public API checks TS and supplies the sign.

    双零计数因上游屏障伪显著性报 ValueError；优化失败或拟合背景/偏差
    无效报 ArithmeticError，不自动调换优化器或转为无系统误差方法。
    Reject both-zero counts with ValueError due to the upstream barrier artifact;
    failed/invalid optimization raises ArithmeticError without a backend/method change."""
    if n + b == 0:
        raise ValueError("PP Gaussian systematic does not support both counts zero (upstream barrier artifact)")

    def objective(kk):
        """零源负似然目标 / Evaluate the negative null likelihood for a candidate k.

        按源码 B=(b+n)/(alpha*k+alpha+1)，M=0，保留原有屏障与运算。
        Substitute the source's analytic B and M=0, retaining its barrier/arithmetic."""
        return -1 * _likelihood_with_sys(
            n, b, alpha, sigma, kk, B=(b + n) / (alpha * kk + alpha + 1), M=0,
        )

    fitted = optimize.minimize(objective, [0.0], tol=1e-3)
    kk = fitted.x[0]
    B0 = (b + n) / (alpha * kk + alpha + 1)
    if (not fitted.success or not np.isfinite(fitted.fun) or not np.isfinite(kk)
            or not np.isfinite(B0) or kk <= -1 or B0 <= 0):
        raise ArithmeticError(f"gv_significance PP systematic optimization failed: {fitted.message}")
    h1 = -(_xlogy(b, b) - b + _xlogy(n, n) - n)
    ts = 2 * (fitted.fun - h1)
    return ts, fitted


def snr(
    n_on,
    background,
    *,
    method: Literal["known", "pp", "pg"],
    alpha=None,
    background_error=None,
    systematic_fraction=0.0,
    systematic_sigma=0.0,
    signed: bool = True,
) -> float | np.ndarray:
    """计算五种计数统计情况下的显著性 / Compute counting significance.

    ON 指包含源和背景的测量区间或区域，OFF 指独立的背景测量。
    ON denotes the measurement interval/region containing source plus
    background; OFF denotes an independent background measurement.
    此处 SNR 是等效高斯显著性，不是净计数除以某个经验噪声估计。
    Here SNR means Gaussian-equivalent significance, rather than net counts
    divided by an empirical noise estimate.

    Parameters
    ----------
    n_on : float, array-like or Quantity
        ON 区的总观测计数，尚未扣除背景。裸数按计数解释；也接受计数单位
        或无量纲 Quantity。输入应为计数，而非计数率或已扣背景的净计数。
        Observed ON **total** counts, before background subtraction. Bare
        numbers mean counts; count or dimensionless Quantity is accepted.
        Supply counts, not count rates or background-subtracted net counts.
    background : float, array-like or Quantity
        ``pp`` 中为 OFF 区观测计数；``known`` 和 ``pg`` 中为 ON 区的
        预期背景计数。单位约定同 ``n_on``。高斯背景估计允许为负，
        其余计数必须非负；不要提前给 PP 的 OFF 计数乘以 ``alpha``。
        OFF observed counts for ``pp``; ON expected background counts for
        ``known`` and ``pg``. Same unit convention as ``n_on``. A Gaussian
        background estimate may be negative; other counts must be nonnegative.
        Do not pre-scale PP OFF counts by ``alpha``.
    method : {'known', 'pp', 'pg'}
        统计方法 / Statistical method:

        * ``known``：背景期望值精确已知，ON 计数服从 Poisson 分布；
          保留上游源码的尾概率 P(N > n_on)。
          Exactly known expected background with Poisson ON counts;
          preserves the upstream tail P(N > n_on).
        * ``pp``：ON/OFF 均为 Poisson 计数；系统误差为零时用 Li--Ma，
          也可指定固定相对偏差或高斯相对系统误差，共三种 PP 情况。
          Poisson ON/OFF counts: Li--Ma without systematics, a fixed relative
          adjustment, or a Gaussian relative systematic error (three cases).
        * ``pg``：ON 总计数服从 Poisson 分布，背景估计服从高斯分布，
          并提供其标准误差；
          适用于以高斯误差描述背景拟合结果的情况，例如 GBM 背景拟合。
          Poisson ON total counts with a Gaussian background estimate and
          its supplied standard error;
          suitable when background-fit uncertainty is represented as Gaussian,
          for example a GBM background fit.
    alpha : float, array-like or dimensionless Quantity, optional
        仅 ``pp`` 使用且必须为正：ON/OFF 的有效面积与活时间比例，
        ``alpha = (area_on / area_off) * (livetime_on / livetime_off)``。
        无系统偏差时，ON 区背景估计为 ``alpha * background``。
        Required positive ON/OFF efficiency ratio for ``pp``:
        (area_on / area_off) * (livetime_on / livetime_off).
        Without a systematic adjustment, the estimated ON background is
        ``alpha * background``. Only supported for ``pp``.
    background_error : float, array-like or Quantity, optional
        仅 ``pg`` 使用且必须为正：ON 区背景估计的绝对高斯标准误差，
        单位为计数；例如背景拟合得到的 1 sigma 误差。不能直接传入
        计数率误差，也不应默认用 ``sqrt(background)`` 代替拟合误差。
        Required positive absolute Gaussian standard error of the background
        estimate for ``pg``, in **counts**, e.g. the 1-sigma background-fit
        uncertainty. Do not supply a count-rate error or automatically replace
        the fit uncertainty with ``sqrt(background)``. Only supported for ``pg``.
    systematic_fraction : float, array-like or dimensionless Quantity
        ``pp`` 的非负固定相对背景调整量 k，默认 0：实际计算幅值时使用
        ``alpha * (1 + k)``。它是上游的固定调整，不是在有界偏差区间内
        对似然取极值，也不是绝对计数误差；其他方法仅允许为零。
        Nonnegative fixed relative background adjustment k for ``pp``:
        alpha becomes alpha * (1 + k). This is the upstream fixed adjustment,
        not a profile over a bounded interval of possible biases or an absolute
        count error. Defaults to 0; must be zero for other methods.
    systematic_sigma : float, array-like or dimensionless Quantity
        ``pp`` 的背景/效率相对系统误差的高斯标准差，非负且默认 0。
        正值沿用上游的一维似然优化（初始 k=0，tol=1e-3）。同一广播
        元素不能同时指定正的 ``systematic_fraction`` 和此参数；
        其他方法仅允许为零。它与 PG 的绝对 ``background_error`` 不同。
        Nonnegative Gaussian standard deviation of the relative background/
        efficiency error for ``pp``. Positive values use the upstream
        one-dimensional likelihood optimization (initial k=0, tol=1e-3).
        Cannot be positive together with ``systematic_fraction`` at an element.
        Defaults to 0; must be zero for other methods. Unlike the PG absolute
        ``background_error``, this parameter is a relative uncertainty.
    signed : bool
        默认 True，保留源码符号：PP 按 ``n_on - alpha * background``
        判断，即使存在固定调整仍使用名义 alpha；PG 按
        ``n_on - background`` 判断；known 直接返回单侧高斯分位数。
        False 则返回绝对值。
        True preserves upstream signs: pp follows n_on - alpha*background
        even with a fixed adjustment; pg follows n_on - background; known
        returns the one-sided normal quantile directly. False returns magnitude.

    Returns
    -------
    float or numpy.ndarray
        无量纲的等效高斯显著性。数值输入按 NumPy 规则广播；全部为
        标量时返回 float，否则保留广播后的数组形状（含长度为 1 的数组）。
        Dimensionless Gaussian-equivalent significance. Inputs broadcast;
        scalar inputs return float and arrays retain the broadcast shape.

    Raises
    ------
    ValueError, astropy.units.UnitConversionError
        参数无效、单位不兼容、缺少必要参数或统计情况相互冲突。
        本函数不隐式推断曝光时间，也不自动将计数率转换为计数。
        Invalid or conflicting inputs. No implicit rate/exposure conversion.
    ArithmeticError
        上游优化失败、似然比统计量 TS 无效，或尾概率/最终结果无法
        用浮点数表示；不会在失败后自动改用另一种统计方法。
        Failed upstream optimization, invalid TS, or unrepresentable p-value.
        A failure does not silently select a different statistical method.

    Notes
    -----
    已知背景使用源码的 ``pdtrc(n, b)``，计算 P(N > n)，而文章文字定义
    为 P(N >= n)。此处为保持迁移一致性保留源码端点。PG 不支持零背景
    误差，不会将其自动映射到不同的精确尾概率方法。带高斯系统误差的
    PP 在 ON/OFF 计数均为零时显式报错，因为上游似然屏障会产生伪显著性。
    The known-background tail follows the source's ``pdtrc(n, b)``, which
    computes P(N > n), whereas the article's text defines P(N >= n). This
    intentional compatibility choice is not corrected here. PG sigma=0 is
    unsupported and is not silently mapped to the distinct exact-tail method.
    PP with Gaussian systematics and both counts zero is rejected because the
    upstream likelihood barrier produces a spurious significance.

    这些结果是局部统计量，未校正跨时间、分箱、能段或探测器的搜索。
    负的似然根值沿用源码约定；低计数下不能将其解释为已校准的亏损
    显著性。已有 ``li_ma_snr`` 的历史输入处理保持不变。
    These are local statistics, without correction for searches across times,
    bins, energies or detectors. Negative likelihood-root values follow the
    source convention; they are not calibrated deficit significances at low
    counts. Existing ``li_ma_snr`` keeps its historical input handling.

    References
    ----------
    实现依据 / Implementation references:
    Vianello (2018), ApJS 236, 17, doi:10.3847/1538-4365/aab780.
    https://arxiv.org/abs/1712.00118
    Li & Ma (1983), ApJ 272, 317, Eq. 17.

    Examples
    --------
    PG：总计数 120，ON 区背景估计 80，背景拟合标准误差 5.3 计数。
    PG: 120 total ON counts, estimated ON background 80, fit error 5.3 counts.

    >>> snr(120, 80, method="pg", background_error=5.3)
    3.548732644109888

    PP：ON 总计数 20，OFF 观测计数 80，ON/OFF 比例 0.1。
    PP: 20 total ON counts, 80 observed OFF counts, ON/OFF ratio 0.1.

    >>> snr(20, 80, method="pp", alpha=0.1) > 3
    True
    """
    if method not in ("known", "pp", "pg"):
        raise ValueError("method must be 'known', 'pp' or 'pg'")
    # 在接口边界校验单位；不推断曝光时间或从计数率换算计数。
    # Validate units at the API boundary; do not infer exposure or convert rates.
    n = _values(n_on, "n_on", counts=True)
    b = _values(background, "background", counts=True)
    k = _values(systematic_fraction, "systematic_fraction")
    sys_sigma = _values(systematic_sigma, "systematic_sigma")
    if np.any(n < 0) or (method != "pg" and np.any(b < 0)):
        raise ValueError("observed counts and non-Gaussian background must be nonnegative")
    if np.any(k < 0) or np.any(sys_sigma < 0):
        raise ValueError("systematic uncertainties must be nonnegative")
    if method != "pp" and (np.any(k != 0) or np.any(sys_sigma != 0)):
        raise ValueError("systematic_fraction/systematic_sigma are only supported for pp")
    if method != "pp" and alpha is not None:
        raise ValueError("alpha is only supported for pp")
    if method != "pg" and background_error is not None:
        raise ValueError("background_error is only supported for pg")

    if method == "pp":
        if alpha is None:
            raise ValueError("alpha is required for pp")
        a = _values(alpha, "alpha")
        if np.any(a <= 0):
            raise ValueError("alpha must be positive")
        n, b, a, k, sys_sigma = np.broadcast_arrays(n, b, a, k, sys_sigma)
        if np.any((k > 0) & (sys_sigma > 0)):
            raise ValueError("fixed and Gaussian systematic uncertainties are mutually exclusive per element")
        # 保存广播形状，再逐元素调用上游的标量统计内核。
        # Preserve the broadcast shape before evaluating upstream scalar kernels.
        shape = n.shape
        n, b, a, k, sys_sigma = (v.ravel() for v in (n, b, a, k, sys_sigma))
        result = np.empty(n.size, dtype=float)
        for i in range(n.size):
            if sys_sigma[i] > 0:
                # 对高斯系统偏差进行似然优化，显著性幅值为 sqrt(TS)。
                # Profile the Gaussian systematic offset; magnitude is sqrt(TS).
                ts, _ = _pp_gaussian_profile(n[i], b[i], a[i], sys_sigma[i])
                if not np.isfinite(ts) or ts < 0:
                    raise ArithmeticError("gv_significance PP systematic TS is invalid")
                result[i] = np.sqrt(ts) * (1 if n[i] >= a[i] * b[i] else -1)
            elif k[i] > 0:
                # 幅值使用调整后的 alpha，符号仍按源码使用名义 alpha。
                # Adjust alpha for magnitude; retain the source's nominal-alpha sign.
                magnitude = li_ma_snr(n[i], b[i], a[i] * (1 + k[i]), signed=False)
                result[i] = magnitude * (1 if n[i] >= a[i] * b[i] else -1)
            else:
                result[i] = li_ma_snr(n[i], b[i], a[i])
    elif method == "pg":
        if background_error is None:
            raise ValueError("background_error is required for pg")
        sigma = _values(background_error, "background_error", counts=True)
        if np.any(sigma <= 0):
            raise ValueError("background_error must be positive for pg")
        n, b, sigma, k, sys_sigma = np.broadcast_arrays(n, b, sigma, k, sys_sigma)
        shape = n.shape
        n, b, sigma = (v.ravel() for v in (n, b, sigma))
        # 对真实背景期望值取剖面似然；half_ts 是 TS/2，保留源码计算次序。
        # Profile the true background mean; half_ts is TS/2, preserving source order.
        _, half_ts = _pg_profile(n, b, sigma)
        if np.any(~np.isfinite(half_ts)) or np.any(half_ts < 0):
            raise ArithmeticError("gv_significance PG TS is invalid")
        result = np.sqrt(2) * np.sqrt(half_ts) * np.where(n >= b, 1, -1)
    else:
        n, b, k, sys_sigma = np.broadcast_arrays(n, b, k, sys_sigma)
        shape = n.shape
        # 严格右尾 P(N > n) 保留上游端点，不能悄然换成 P(N >= n)。
        # Preserve the upstream strict tail P(N > n), rather than changing its endpoint.
        pvalue = special.pdtrc(n.ravel(), b.ravel())
        if np.any(pvalue <= np.finfo(float).tiny):
            raise ArithmeticError("gv_significance known-background p-value is too small")
        result = -special.ndtri(pvalue)

    if np.any(~np.isfinite(result)):
        raise ArithmeticError("gv_significance result is nonfinite")
    if not signed:
        result = np.abs(result)
    # 恢复广播形状；仅零维输入返回 Python float。
    # Restore broadcast shape; only zero-dimensional inputs return a Python float.
    result = result.reshape(shape)
    return float(result) if result.ndim == 0 else result


def li_ma_snr(n_on: float, n_off: float, alpha: float, *, signed: bool = True) -> float:
    """计算 Li-Ma 局部显著性 / Compute local Li & Ma significance (1983, Eq. 17).

    Parameters
    ----------
    n_on, n_off : float
        ON 区总观测计数与独立 OFF 背景计数；不可将净计数传为 n_on。
        Total observed ON counts and independent OFF counts, not net ON counts.
    alpha : float
        正的无量纲比例 (A_on/A_off)*(t_on/t_off)，背景 ON 期望为 alpha*n_off。
        Positive dimensionless ON/OFF exposure-area ratio; expected ON background
        estimate is alpha*n_off. No physical-unit/exposure conversion is performed.
    signed : bool
        默认 True，符号按 n_on-alpha*n_off；False 返回幅值。
        Default True follows the excess sign; False returns magnitude.

    Returns
    -------
    float
        无量纲显著性，零计数项采用解析极限。保持历史输入处理：两计数
        同时非正或 alpha<=0 返回 0，其余负计数各裁为 0；并非严格计数校验。
        Dimensionless significance with analytic zero-count limits. Historical
        handling returns 0 if both counts are nonpositive or alpha<=0, otherwise
        clips each negative count to 0. This is not strict count validation.

    Notes
    -----
    本函数不统一检查有限性、整数性或数组广播；需更严格接口时用 snr 的
    pp 模式。结果未校正多时间/能段搜索；低计数负号采用约定，不单独
    校准亏损事件尾概率。保留原计算体，当前任务仅补充说明。
    No unified finiteness/integrality/broadcast validation; use snr(method='pp')
    for the stricter API. No search-trials correction or separate low-count deficit
    calibration is supplied. The historical calculation body remains unchanged.

    References
    ----------
    Li & Ma (1983), ApJ 272, 317, Eq. 17."""
    if n_on <= 0 and n_off <= 0:
        return 0.0
    if alpha <= 0:
        return 0.0
    n_on = float(max(n_on, 0.0))
    n_off = float(max(n_off, 0.0))

    # Compute the unsigned Li & Ma magnitude S = sqrt(2 ln L)
    if n_on == 0.0 and n_off == 0.0:
        s = 0.0
    elif n_on == 0.0:
        # n_off > 0: term1 -> 0, term2 = n_off * ln(1+alpha)
        s = float(np.sqrt(2.0 * n_off * np.log(1.0 + alpha)))
    elif n_off == 0.0:
        # n_on > 0: term1 = n_on * ln((1+alpha)/alpha), term2 -> 0
        s = float(np.sqrt(2.0 * n_on * np.log((1.0 + alpha) / alpha)))
    else:
        term1 = n_on * np.log(((1.0 + alpha) / alpha) * (n_on / (n_on + n_off)))
        term2 = n_off * np.log((1.0 + alpha) * (n_off / (n_on + n_off)))
        val = 2.0 * (term1 + term2)
        s = float(np.sqrt(max(val, 0.0)))

    if signed:
        return math.copysign(s, n_on - alpha * n_off)
    return s
