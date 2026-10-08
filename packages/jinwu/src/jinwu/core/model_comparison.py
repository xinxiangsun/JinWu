"""Empirical model-comparison statistics shared by X-ray analyses.

The helpers deliberately avoid a Wilks/F-test shortcut.  They are useful when
the null is non-regular (for example an absorption column constrained at zero)
and a parametric bootstrap supplies the reference distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import beta
from scipy.special import gammaln, logsumexp


_NORMAL = NormalDist()


def _probability(value: float, *, name: str = "p") -> float:
    """校验有限概率 / Validate and return a finite probability in [0, 1].

    先转换 float；越界或非有限时抛 ValueError，name 仅用于报错。
    Convert to float; nonfinite/out-of-range input raises ValueError using name."""
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def p_to_sigma(p: float, *, sided: str = "one") -> float:
    """尾概率转高斯等效显著性 / Convert a tail probability to Gaussian-equivalent z.

    p 为 [0, 1] 无量纲概率；sided='one' 使用 Phi^-1(1-p)，'two' 使用
    Phi^-1(1-p/2)。单侧 p=0/1 返回 +inf/-inf；双侧 p=1 返回 0。
    单侧 p>0.5 的 z 为负；不会校准搜索次数或赋予 Bayes 因子 p 值语义。
    p is dimensionless in [0, 1]. 'one' uses Phi^-1(1-p), 'two' Phi^-1(1-p/2).
    One-sided endpoints give +inf/-inf; two-sided p=1 gives 0. Negative one-sided
    z is valid for p>0.5. No trials calibration or Bayes-factor reinterpretation.

    极小非零 p 的 1-p 可能舍入到 1，引发底层 StatisticsError；输入或
    sided 无效抛 ValueError。返回 float，不保证极端尾概率的数值精度。
    Very small nonzero p can round 1-p to 1 and raise StatisticsError. Invalid p
    or sided raises ValueError. Return float; extreme-tail precision is limited.
    参考 / Reference: https://docs.python.org/3/library/statistics.html#statistics.NormalDist.inv_cdf"""

    p = _probability(p)
    if sided not in {"one", "two"}:
        raise ValueError("sided must be 'one' or 'two'")
    if sided == "one":
        if p == 0.0:
            return float("inf")
        if p == 1.0:
            return float("-inf")
        return float(_NORMAL.inv_cdf(1.0 - p))
    if p == 0.0:
        return float("inf")
    if p == 1.0:
        return 0.0
    return float(_NORMAL.inv_cdf(1.0 - p / 2.0))


def sigma_to_p(sigma: float, *, sided: str = "one") -> float:
    """高斯 z 转尾概率 / Convert finite Gaussian z to a tail probability.

    单侧为 1-Phi(sigma)，双侧为 2*(1-Phi(abs(sigma)))；返回无量纲 float。
    非有限 sigma 或非法 sided 抛 ValueError；大 sigma 可能因相减消失为 0。
    Use 1-Phi(sigma) for one side, 2*(1-Phi(abs(sigma))) for two sides. Return
    float probability. Nonfinite sigma/unknown sided raises ValueError. CDF
    subtraction may round extreme tails to zero; no detection-trials correction."""

    sigma = float(sigma)
    if not np.isfinite(sigma):
        raise ValueError("sigma must be finite")
    if sided not in {"one", "two"}:
        raise ValueError("sided must be 'one' or 'two'")
    one_sided = 1.0 - _NORMAL.cdf(sigma)
    return float(one_sided if sided == "one" else 2.0 * (1.0 - _NORMAL.cdf(abs(sigma))))


@dataclass(frozen=True)
class EmpiricalTailResult:
    """Add-one empirical tail and exact Clopper--Pearson interval."""

    n_trials: int
    n_exceedances: int
    observed_statistic: float
    p_hat: float
    p_low: float
    p_high: float
    z: float
    z_low: float
    z_high: float
    confidence: float
    tail: str
    sided: str

    def as_dict(self) -> dict[str, float | int | str]:
        """导出经验尾概率结果 / Return a shallow dictionary of empirical-tail fields.

        保留样本数、点估计、区间、方向与置信度；不重新计算统计量。
        Preserve trial counts, estimates, intervals and conventions; no recalculation."""
        return self.__dict__.copy()


def empirical_tail_probability(
    statistics: np.ndarray | list[float],
    observed_statistic: float,
    *,
    tail: str = "greater",
    confidence: float = 0.95,
    sided: str = "one",
) -> EmpiricalTailResult:
    """用模拟统计量估计尾概率 / Estimate an empirical tail and binomial interval.

    Parameters
    ----------
    statistics : array-like
        同一统计量的参考模拟样本，展开为一维并去除非有限值。
        Reference samples of the same statistic; flattened, nonfinite values removed.
    observed_statistic : float
        有限的观测统计量，单位/定义须与样本一致。
        Finite observed statistic, matching the sample definition and unit.
    tail : {'greater', 'less'}
        greater 计 >= observed，less 计 <= observed；包含平局。
        Count >= or <= observed respectively, including ties.
    confidence : float
        二项区间置信度，要求 0 < confidence <= 1，默认 0.95。
        Binomial interval confidence, 0 < confidence <= 1; default 0.95.
    sided : {'one', 'two'}
        只决定 p_to_sigma 转换，不改变所选经验尾的计数。
        Gaussian conversion convention only; does not change the empirical tail.

    Returns
    -------
    EmpiricalTailResult
        B 为有限样本数，k 为越界数；p_hat=(k+1)/(B+1)。区间是基于原始
        (k, B) 的双侧 Clopper-Pearson 区间，并非给伪计数加一后的区间。
        z 区间由 p 区间反序换算；无足够有效输入时抛 ValueError。
        B finite trials and k exceedances give p_hat=(k+1)/(B+1). The two-sided
        Clopper-Pearson interval uses original (k, B), not added pseudocounts.
        Transform probability bounds in reverse order for z bounds. Invalid/empty
        samples raise ValueError; extreme Gaussian conversions can also fail.

    Notes
    -----
    参考分布的生成与搜索校准由调用者负责；不能仅凭此函数证明模拟代表
    真实零假设。此置信区间描述有限模拟对尾概率的抽样不确定度。
    The caller supplies a valid null/reference distribution and search calibration.
    This interval describes finite-simulation uncertainty in its tail probability.
    参考二项精确区间 / Exact binomial interval reference:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html"""

    values = np.asarray(statistics, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0 or not np.isfinite(observed_statistic):
        raise ValueError("statistics and observed_statistic must be finite")
    if tail not in {"greater", "less"} or sided not in {"one", "two"}:
        raise ValueError("invalid tail or sided argument")
    confidence = _probability(confidence, name="confidence")
    if confidence <= 0.0:
        raise ValueError("confidence must be greater than zero")
    observed = float(observed_statistic)
    k = int(np.count_nonzero(values >= observed) if tail == "greater" else np.count_nonzero(values <= observed))
    n = int(values.size)
    p_hat = float((k + 1) / (n + 1))
    alpha = 1.0 - confidence
    p_low = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    p_high = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return EmpiricalTailResult(
        n_trials=n,
        n_exceedances=k,
        observed_statistic=observed,
        p_hat=p_hat,
        p_low=p_low,
        p_high=p_high,
        z=p_to_sigma(p_hat, sided=sided),
        z_low=p_to_sigma(p_high, sided=sided),
        z_high=p_to_sigma(p_low, sided=sided),
        confidence=confidence,
        tail=tail,
        sided=sided,
    )


@dataclass(frozen=True)
class BayesFactorResult:
    r"""Numerically stable two-model evidence comparison.

    ``log_bayes_factor_10`` is :math:`\log Z_1-\log Z_0`, where model 1 is
    the alternative/evolution model and model 0 is the constant-absorption
    model.  Evidence errors are one-standard-deviation numerical estimates
    supplied by the nested sampler; the interval is therefore a numerical
    interval, not a posterior credible interval for the model itself.
    """

    log_evidence_h0: float
    log_evidence_h1: float
    log_bayes_factor_10: float
    bayes_factor_10: float
    log_bayes_factor_low: float | None
    log_bayes_factor_high: float | None
    confidence: float
    numerical_error: float | None

    def as_dict(self) -> dict[str, Any]:
        """导出证据比较结果 / Export evidence comparison and numerical-error fields.

        不把 numerical_error 或其区间改称模型后验可信区间。
        Preserve numerical-error terminology; these are not model posterior intervals."""
        return {
            "log_evidence_h0": self.log_evidence_h0,
            "log_evidence_h1": self.log_evidence_h1,
            "log_bayes_factor_10": self.log_bayes_factor_10,
            "bayes_factor_10": self.bayes_factor_10,
            "log_bayes_factor_low": self.log_bayes_factor_low,
            "log_bayes_factor_high": self.log_bayes_factor_high,
            "confidence": self.confidence,
            "numerical_error": self.numerical_error,
        }


def _confidence_z(confidence: float) -> float:
    """取对称高斯区间系数 / Return the symmetric Gaussian interval multiplier.

    使用 Phi^-1(0.5+confidence/2)，要求 confidence>0.5；confidence=1
    会由 NormalDist 的端点校验报错。返回无量纲 float。
    Use Phi^-1(0.5+confidence/2), requiring confidence>0.5. confidence=1 triggers
    the NormalDist endpoint error. Return dimensionless float."""
    confidence = _probability(confidence, name="confidence")
    if confidence <= 0.5:
        raise ValueError("confidence must be greater than 0.5")
    # NormalDist has no inverse survival function; the symmetric quantile is
    # equivalent and avoids adding a statsmodels dependency to jinwu core.
    return float(_NORMAL.inv_cdf(0.5 + confidence / 2.0))


def summarize_bayes_factor(
    log_evidence_h0: float,
    log_evidence_h1: float,
    *,
    log_evidence_error_h0: float | None = None,
    log_evidence_error_h1: float | None = None,
    confidence: float = 0.95,
) -> BayesFactorResult:
    """比较两个自然对数证据 / Compare two natural-log model evidences.

    log_evidence_h0/h1 为有限无量纲 ln Z。返回 ln B10=ln Z1-ln Z0
    及 B10，极端比值按当前浮点阈值置为 0 或 inf。
    Inputs are finite, dimensionless ln Z. Return ln B10=ln Z1-ln Z0 and B10;
    extreme factors are set to 0/inf according to the floating-point thresholds.

    若指定数值误差，须同时给出两项非负有限的 ln Z 标准误差。按独立
    误差平方和计算 ln B 的数值区间，不包含先验敏感性或模型不确定性；
    此时置信度须使对称高斯分位数有效。无误差时不生成区间。
    Supply both nonnegative finite ln Z standard errors or neither. Their independent
    quadrature gives a symmetric numerical interval in ln B, excluding prior/model
    uncertainty. Confidence must support a finite Gaussian quantile when errors
    are given. Without errors, no interval is produced.

    返回 BayesFactorResult；非法输入抛 ValueError 或底层分位数异常。
    Bayes 因子本身不是频率学检测 p 值。
    Return BayesFactorResult; invalid input raises ValueError or a quantile error.
    A Bayes factor is not a frequentist detection p-value."""

    logz0 = float(log_evidence_h0)
    logz1 = float(log_evidence_h1)
    if not np.isfinite(logz0) or not np.isfinite(logz1):
        raise ValueError("log evidences must be finite")
    errors = (log_evidence_error_h0, log_evidence_error_h1)
    if any(value is not None for value in errors):
        if any(value is None for value in errors):
            raise ValueError("provide both evidence errors or neither")
        err0 = float(log_evidence_error_h0)  # type: ignore[arg-type]
        err1 = float(log_evidence_error_h1)  # type: ignore[arg-type]
        if not np.isfinite(err0) or not np.isfinite(err1) or err0 < 0 or err1 < 0:
            raise ValueError("evidence errors must be finite and non-negative")
        error = float(np.hypot(err0, err1))
        z = _confidence_z(confidence)
        low = (logz1 - logz0) - z * error
        high = (logz1 - logz0) + z * error
    else:
        _probability(confidence, name="confidence")
        error = None
        low = high = None
    delta = logz1 - logz0
    if delta >= math.log(np.finfo(float).max):
        factor = float("inf")
    elif delta <= math.log(np.finfo(float).tiny):
        factor = 0.0
    else:
        factor = float(np.exp(delta))
    return BayesFactorResult(
        log_evidence_h0=logz0,
        log_evidence_h1=logz1,
        log_bayes_factor_10=float(delta),
        bayes_factor_10=factor,
        log_bayes_factor_low=None if low is None else float(low),
        log_bayes_factor_high=None if high is None else float(high),
        confidence=float(confidence),
        numerical_error=error,
    )


def model_posterior_probability(
    log_bayes_factor_10: float,
    *,
    prior_h1: float = 0.5,
) -> dict[str, float]:
    """用模型先验与 Bayes 因子更新模型概率 / Compute posterior model probabilities.

    log_bayes_factor_10 为有限 ln B10；prior_h1 严格介于 0 和 1，默认 0.5。
    使用 logsumexp 归一化，返回 {'p_h0', 'p_h1'} 无量纲概率；非法输入
    抛 ValueError。模型参数先验已属于证据计算，不在此重新积分。
    Require finite ln B10 and 0 < prior_h1 < 1 (default 0.5). Normalize with
    logsumexp and return dimensionless p_h0/p_h1. Invalid input raises ValueError.
    Parameter priors belong to the supplied evidences, not a new integration here."""

    logbf = float(log_bayes_factor_10)
    if not np.isfinite(logbf):
        raise ValueError("log_bayes_factor_10 must be finite")
    p1 = _probability(prior_h1, name="prior_h1")
    if p1 in (0.0, 1.0):
        raise ValueError("prior_h1 must be strictly between zero and one")
    scores = np.asarray([math.log1p(-p1), math.log(p1) + logbf], dtype=float)
    normalized = np.exp(scores - logsumexp(scores))
    return {"p_h0": float(normalized[0]), "p_h1": float(normalized[1])}


def model_averaged_direction_probabilities(
    log_bayes_factor_10: float,
    direction_given_h1: float,
    *,
    prior_h1: float = 0.5,
) -> dict[str, float]:
    """把条件方向概率纳入模型平均 / Average a conditional direction over H0/H1.

    direction_given_h1=q 是 H1 下预先定义方向的后验概率，范围 [0,1]。
    假设 H0 的严格方向事件概率为 0，返回 p_decrease=p_h1*q，
    p_increase=p_h1*(1-q)，并保留模型概率；H0 的质量未分配到两方向。
    q in [0,1] is the posterior of a predeclared direction conditional on H1.
    Assume a strict direction has zero probability under H0. Return p_decrease
    =p_h1*q and p_increase=p_h1*(1-q), plus model probabilities. H0 mass is
    not assigned to either direction. Invalid inputs propagate validation errors.

    两个方向的名称沿用接口；调用者须明确 q 对应的实际科学事件。
    Names follow the public API; callers define the actual scientific direction."""

    q = _probability(direction_given_h1, name="direction_given_h1")
    model = model_posterior_probability(log_bayes_factor_10, prior_h1=prior_h1)
    p_decrease = model["p_h1"] * q
    return {
        **model,
        "p_direction_given_h1": q,
        "p_decrease": float(p_decrease),
        "p_increase": float(model["p_h1"] * (1.0 - q)),
    }


def model_averaged_direction_probability_interval(
    log_bayes_factor_low: float,
    log_bayes_factor_high: float,
    direction_low: float,
    direction_high: float,
    *,
    prior_h1: float = 0.5,
) -> dict[str, float]:
    """传播数值范围到模型平均概率 / Propagate bounded numerical ranges to averaged probabilities.

    ln B 和 q 的上下界须有序，ln B 有限、q 在 [0,1]。由单调性使用
    两个边角给出 p_h1 和 p_decrease 范围；prior_h1 为固定模型先验。
    返回对应 *_low/*_high dict；非法输入抛 ValueError。
    Require ordered finite ln B bounds and ordered q bounds within [0,1]. Use
    monotonic corner values to return *_low/*_high for p_h1 and p_decrease with
    fixed prior_h1. Invalid inputs raise ValueError.

    输入是数值误差范围，输出不自动成为联合后验可信区间，也不估计
    两个输入的相关分布或覆盖率。
    Numerical ranges are not automatically joint posterior credible intervals;
    this operation does not estimate their correlation or coverage probability."""

    low_bf = float(log_bayes_factor_low)
    high_bf = float(log_bayes_factor_high)
    low_q = _probability(direction_low, name="direction_low")
    high_q = _probability(direction_high, name="direction_high")
    if not np.isfinite(low_bf) or not np.isfinite(high_bf) or low_bf > high_bf:
        raise ValueError("log Bayes-factor interval must be finite and ordered")
    if low_q > high_q:
        raise ValueError("direction interval must be ordered")
    p_h1_low = model_posterior_probability(low_bf, prior_h1=prior_h1)["p_h1"]
    p_h1_high = model_posterior_probability(high_bf, prior_h1=prior_h1)["p_h1"]
    return {
        "p_h1_low": float(p_h1_low),
        "p_h1_high": float(p_h1_high),
        "p_decrease_low": float(p_h1_low * low_q),
        "p_decrease_high": float(p_h1_high * high_q),
    }


def _as_count_array(value: Any, name: str) -> np.ndarray:
    """校验并展开原始整数计数 / Validate and flatten raw nonnegative integer counts.

    输入转 float 一维数组，须非空且有限；与最近整数偏差不超过 1e-9。
    返回 int64 数组；无效值抛 ValueError，不执行计数率或曝光换算。
    Flatten as float; require nonempty finite nonnegative values within 1e-9 of
    integers. Return int64 counts or raise ValueError. No rate/exposure conversion."""
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0 or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be a finite non-empty vector")
    if np.any(array < 0) or np.any(np.abs(array - np.rint(array)) > 1.0e-9):
        raise ValueError(f"{name} must contain non-negative integer counts")
    return np.rint(array).astype(np.int64)


def recover_raw_off_counts(
    source_scaled_background_rate: Any,
    source_exposure_s: Any,
    alpha: Any,
    *,
    tolerance: float = 1.0e-6,
) -> np.ndarray:
    """从源区缩放背景率恢复 OFF 计数 / Recover raw OFF counts from source-scaled rates.

    输入 source_scaled_background_rate 是已缩放到 ON 区的 counts/s，
    source_exposure_s 是 ON 曝光秒数，alpha 为 ON/OFF 曝光-面积比例。
    三者按 NumPy 广播，计算 raw=rate*exposure/alpha，再校验其整数性。
    Rates are counts/s already scaled to ON, exposure is ON seconds, and alpha
    is the positive dimensionless ON/OFF exposure-area ratio. Broadcast them,
    compute raw=rate*exposure/alpha, then require recoverable integer values.

    返回保留广播形状的非负 int64 ndarray。tolerance 为绝对计数残差
    容差（默认 1e-6），不是相对误差；超容差不静默四舍五入。非有限、
    非法曝光/比例、超 int64 或不可广播时抛 ValueError。
    Return nonnegative int64 ndarray with broadcast shape. tolerance is absolute
    count residual (default 1e-6), not relative error; reject nonintegral recovery
    beyond it. Invalid values, overflow or nonbroadcastable inputs raise ValueError.

    输入缩放语义由读取端确认，不能把原始 OFF 区 rate 当作源区缩放 rate。
    Readers must establish scaling semantics; raw OFF rates are not interchangeable
    with source-scaled background rates. This function does not read OGIP metadata."""

    rate = np.asarray(source_scaled_background_rate, dtype=float)
    exposure = np.asarray(source_exposure_s, dtype=float)
    scale = np.asarray(alpha, dtype=float)
    if not math.isfinite(float(tolerance)) or float(tolerance) < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    try:
        rate, exposure, scale = np.broadcast_arrays(rate, exposure, scale)
    except ValueError as exc:
        raise ValueError("background rate, source exposure and alpha are not broadcastable") from exc
    if rate.size == 0:
        raise ValueError("background rate must be non-empty")
    if np.any(~np.isfinite(rate)) or np.any(rate < 0.0):
        raise ValueError("source-scaled background rate must be finite and non-negative")
    if np.any(~np.isfinite(exposure)) or np.any(exposure <= 0.0):
        raise ValueError("source exposure must be finite and positive")
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("alpha must be finite and positive")
    raw = rate * exposure / scale
    if np.any(~np.isfinite(raw)) or np.any(raw < 0.0):
        raise ValueError("recovered OFF counts must be finite and non-negative")
    if np.any(raw > np.iinfo(np.int64).max):
        raise ValueError("recovered OFF counts exceed int64 range")
    counts = np.rint(raw)
    residual = np.abs(raw - counts)
    # The tolerance is an absolute count-contract tolerance.  A relative
    # allowance would grow with the count and could silently turn a metadata
    # scaling error into a different integer spectrum at large counts.
    allowed = float(tolerance)
    if np.any(residual > allowed):
        worst = float(np.max(residual))
        raise ValueError(
            "source-scaled background rate cannot recover integer raw OFF counts "
            f"within tolerance (max residual={worst:g})"
        )
    return counts.astype(np.int64)


def onoff_log_marginal_likelihood(
    source_counts: Any,
    background_counts: Any,
    source_model_counts: Any,
    alpha: Any,
    *,
    background_prior_shape: float = 0.5,
    background_prior_rate: float = 1.0e-6,
) -> float:
    """解析边际化 Poisson ON/OFF 背景 / Analytically integrate an ON/OFF background.

    source_counts=n 是 ON 总观测计数，background_counts=m 是独立 OFF
    原始计数；source_model_counts=s 是预期纯源计数（已按观测/响应折叠），
    alpha=a 是正的无量纲 ON/OFF 比例。模型为 n~Pois(s+a*b)，m~Pois(b)，
    b~Gamma(shape, rate)，这里 rate 是率参数而非 scale。
    n is total observed ON counts, m raw independent OFF counts, s expected
    source-only counts after observation/response folding, and a positive ON/OFF
    ratio. Use n~Pois(s+a*b), m~Pois(b), b~Gamma(shape, rate), with a rate rather
    than scale parameter. All inputs use count numbers, not rates or net counts.

    将输入展开为等长一维数组，alpha 可为正标量或等长数组；观测计数
    须为非负整数，源期望非负，Gamma 两参数有限且正。默认 shape=0.5、
    rate=1e-6，属于所选背景先验，证据解释须保留这两个设置。
    Flatten equal-length vectors; alpha may be scalar or matching vector. Observed
    counts are nonnegative integers, s nonnegative, Gamma parameters finite/positive.
    Defaults shape=0.5, rate=1e-6 define the background prior and affect evidence.

    返回含 Poisson 与归一化 Gamma 常数的标量自然对数似然，以逐通道
    有限多项展开与 logsumexp 积分。非法输入报错；不生成响应、不减背景，
    也不计算模型参数先验或完整模型证据。
    Return a scalar natural-log likelihood including normalized Poisson/Gamma
    constants, integrating a finite per-channel expansion with logsumexp. Invalid
    inputs raise; no response generation, background subtraction, parameter-prior
    integration or complete model evidence is performed."""

    on = _as_count_array(source_counts, "source_counts")
    off = _as_count_array(background_counts, "background_counts")
    source = np.asarray(source_model_counts, dtype=float).reshape(-1)
    scale = np.asarray(alpha, dtype=float)
    if source.shape != on.shape or np.any(~np.isfinite(source)) or np.any(source < 0):
        raise ValueError("source_model_counts must be finite, non-negative and match counts")
    if scale.ndim == 0:
        scale = np.full(on.shape, float(scale), dtype=float)
    else:
        scale = scale.reshape(-1)
    if scale.shape != on.shape or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("alpha must be a positive scalar or vector matching counts")
    shape = float(background_prior_shape)
    rate = float(background_prior_rate)
    if not np.isfinite(shape) or shape <= 0 or not np.isfinite(rate) or rate <= 0:
        raise ValueError("background Gamma shape and rate must be positive and finite")

    total = 0.0
    for n, m, s, a in zip(on, off, source, scale, strict=True):
        n_int = int(n)
        k = np.arange(n_int + 1, dtype=float)
        # Terms with s=0 and k<n are exactly zero.  Assign -inf explicitly
        # instead of evaluating log(0), which would create NaNs in 0-count
        # channels and hide a valid likelihood.
        source_power = np.zeros(k.shape, dtype=float)
        if s > 0:
            source_power = (n_int - k) * math.log(float(s))
        else:
            source_power[k < n_int] = -np.inf
        log_terms = (
            gammaln(n_int + 1.0)
            - gammaln(k + 1.0)
            - gammaln(n_int - k + 1.0)
            + source_power
            + k * math.log(float(a))
            + gammaln(float(m) + shape + k)
            - (float(m) + shape + k) * math.log1p(float(a) + rate)
        )
        log_integral = (
            -float(s)
            + shape * math.log(rate)
            - gammaln(shape)
            - gammaln(n_int + 1.0)
            - gammaln(float(m) + 1.0)
            + float(logsumexp(log_terms))
        )
        total += log_integral
    return float(total)


def onoff_log_profile_likelihood(
    source_counts: Any,
    background_counts: Any,
    source_model_counts: Any,
    alpha: Any,
) -> float:
    """对 ON/OFF 背景取剖面似然 / Profile a nonnegative Poisson ON/OFF background.

    输入契约同边际化函数的计数/alpha 部分：ON 总计数、原始 OFF 计数、
    预期纯源计数，以及正 ON/OFF 比例。逐通道求 b_hat>=0 的解析二次根，
    返回含 Poisson 归一化常数的标量自然对数似然。
    Use total ON counts, raw OFF counts, expected source-only counts and positive
    ON/OFF ratio. Solve a per-channel quadratic for b_hat>=0 and return the
    normalized scalar Poisson log likelihood. No background-prior integration.

    作为 XSPEC W-stat 的诊断桥；不能把剖面似然当作边际证据。含正计数
    而相应期望为零时返回 -inf，非法数据抛异常。实际 W-stat 常数约定须
    依据使用的 XSPEC 版本及同输入对照核验。
    Use as a diagnostic bridge to W-stat, not marginalized evidence. Positive counts
    with zero expectation return -inf; invalid inputs raise. Verify XSPEC constants
    against the actual build with identical inputs.
    参考 / Reference: https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/XSappendixStatistics.html"""

    on = _as_count_array(source_counts, "source_counts")
    off = _as_count_array(background_counts, "background_counts")
    source = np.asarray(source_model_counts, dtype=float).reshape(-1)
    scale = np.asarray(alpha, dtype=float)
    if source.shape != on.shape or np.any(~np.isfinite(source)) or np.any(source < 0):
        raise ValueError("source_model_counts must be finite, non-negative and match counts")
    if scale.ndim == 0:
        scale = np.full(on.shape, float(scale), dtype=float)
    else:
        scale = scale.reshape(-1)
    if scale.shape != on.shape or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("alpha must be a positive scalar or vector matching counts")

    total = 0.0
    for n, m, s, a in zip(on, off, source, scale, strict=True):
        n_int, m_int = int(n), int(m)
        # The derivative of the ON/OFF log likelihood is a quadratic in b:
        # a(a+1)b^2 + ((a+1)s-a(n+m))b - m*s = 0.
        aa = float(a) * (float(a) + 1.0)
        bb = (float(a) + 1.0) * float(s) - float(a) * (n_int + m_int)
        cc = -float(m_int) * float(s)
        discriminant = max(0.0, bb * bb - 4.0 * aa * cc)
        square_root = math.sqrt(discriminant)
        if bb >= 0.0:
            # Avoid cancellation when the source expectation is large and the
            # profiled background is close to zero.
            root = (2.0 * (-cc)) / (bb + square_root) if bb + square_root > 0 else 0.0
        else:
            root = (-bb + square_root) / (2.0 * aa)
        b_hat = max(0.0, float(root))
        on_mean = float(s) + float(a) * b_hat
        if on_mean <= 0.0 and n_int > 0:
            return float("-inf")
        if b_hat <= 0.0 and m_int > 0:
            return float("-inf")
        term = -on_mean - b_hat - gammaln(n_int + 1.0) - gammaln(m_int + 1.0)
        if n_int:
            term += n_int * math.log(on_mean)
        if m_int:
            term += m_int * math.log(b_hat)
        total += term
    return float(total)


class PreparedOnOffMarginalLikelihood:
    """Exact cached Gamma-Poisson ON/OFF marginal likelihood.

    Parameters are raw dimensionless ON/OFF counts and the positive ON/OFF
    exposure-area ratio ``alpha``. Background prior shape and rate use the
    same normalized convention as :func:`onoff_log_marginal_likelihood`.
    ``max_terms`` limits each temporary coefficient block; bins exceeding it
    use the reference implementation. Inputs are copied. Calling the object
    with expected source counts returns the normalized scalar log likelihood.

    Only data-dependent combinatorial terms are cached; no likelihood,
    response, parameter prior or model approximation is introduced.
    """
    def __init__(self, source_counts: Any, background_counts: Any, alpha: Any,
                 *, background_prior_shape: float = 0.5,
                 background_prior_rate: float = 1e-6, max_terms: int = 1_000_000):
        """缓存数据项以重复计算边际似然 / Prepare exact data-dependent ON/OFF coefficients.

        复制非负整数 ON/OFF 计数，检查等长，广播正 alpha；Gamma 参数约定
        同 onoff_log_marginal_likelihood。max_terms 为正整数，限制单个系数
        块的元素数，不是总体内存上限；更大的单 bin 保留为参考实现回退。
        Copy and validate matching integer counts and broadcast positive alpha. Gamma
        conventions match the reference marginal likelihood. max_terms is a positive
        integer limiting each coefficient block, not total memory. Oversized individual
        bins use the reference implementation when called.

        只缓存数据组合系数，尚未计算某个源模型的似然或证据；非法输入报错。
        Cache only combinatorial data terms, not any source-model likelihood/evidence.
        Invalid input raises; input arrays are copied before storage."""
        self.on = _as_count_array(source_counts, "source_counts").copy()
        self.off = _as_count_array(background_counts, "background_counts").copy()
        if self.on.shape != self.off.shape:
            raise ValueError("ON/OFF count shapes must match")
        scale = np.asarray(alpha, dtype=float)
        self.alpha = np.broadcast_to(scale, self.on.shape).copy()
        if np.any(~np.isfinite(self.alpha)) or np.any(self.alpha <= 0):
            raise ValueError("alpha must be positive and finite")
        self.shape = float(background_prior_shape)
        self.rate = float(background_prior_rate)
        if not np.isfinite(self.shape) or self.shape <= 0 or not np.isfinite(self.rate) or self.rate <= 0:
            raise ValueError("background Gamma parameters must be positive and finite")
        if isinstance(max_terms, bool) or int(max_terms) != max_terms or max_terms < 1:
            raise ValueError("max_terms must be a positive integer")
        self.blocks = []
        self.fallback = []
        start = 0
        while start < self.on.size:
            if self.on[start] + 1 > max_terms:
                self.fallback.append(start)
                start += 1
                continue
            end = start + 1
            width = int(self.on[start]) + 1
            while end < self.on.size:
                next_width = max(width, int(self.on[end]) + 1)
                if (end + 1 - start) * next_width > max_terms:
                    break
                width = next_width
                end += 1
            n = self.on[start:end, None]
            m = self.off[start:end, None]
            a = self.alpha[start:end, None]
            k = np.arange(width, dtype=float)[None, :]
            valid = k <= n
            power = np.maximum(n - k, 0)
            coefficients = (
                -gammaln(k + 1) - gammaln(power + 1)
                + k * np.log(a) + gammaln(m + self.shape + k)
                - (m + self.shape + k) * np.log1p(a + self.rate)
                + self.shape * math.log(self.rate) - gammaln(self.shape)
                - gammaln(m + 1)
            )
            coefficients[~valid] = -np.inf
            self.blocks.append((start, end, power, coefficients))
            start = end

    def __call__(self, source_model_counts: Any) -> float:
        """计算给定源期望的精确边际似然 / Evaluate the prepared marginal likelihood.

        source_model_counts 为有限非负纯源期望计数，与初始化的 ON 向量等长。
        缓存块使用 logsumexp/xlogy；超块限制的 bin 调用参考边际化函数。
        返回标量 ln L；不修改输入模型、不做拟合或模型参数先验积分。
        Supply finite nonnegative expected source-only counts matching prepared ON
        length. Use logsumexp/xlogy blocks and reference integration for fallback bins.
        Return scalar ln L without fitting or parameter-prior integration. Invalid
        input raises ValueError; cached arrays are used without revalidating their state."""
        from scipy.special import xlogy
        source = np.asarray(source_model_counts, dtype=float).reshape(-1)
        if source.shape != self.on.shape or np.any(~np.isfinite(source)) or np.any(source < 0):
            raise ValueError("source counts must be finite, non-negative and match data")
        total = 0.0
        for start, end, power, coefficients in self.blocks:
            s = source[start:end]
            total += float(np.sum(logsumexp(coefficients + xlogy(power, s[:, None]), axis=1) - s))
        for index in self.fallback:
            total += onoff_log_marginal_likelihood(
                self.on[index:index+1], self.off[index:index+1], source[index:index+1],
                self.alpha[index:index+1], background_prior_shape=self.shape,
                background_prior_rate=self.rate)
        return total


__all__ = [
    "PreparedOnOffMarginalLikelihood",
    "BayesFactorResult",
    "EmpiricalTailResult",
    "model_averaged_direction_probabilities",
    "model_averaged_direction_probability_interval",
    "model_posterior_probability",
    "onoff_log_marginal_likelihood",
    "onoff_log_profile_likelihood",
    "recover_raw_off_counts",
    "empirical_tail_probability",
    "p_to_sigma",
    "summarize_bayes_factor",
    "sigma_to_p",
]
