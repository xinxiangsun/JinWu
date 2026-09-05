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
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def p_to_sigma(p: float, *, sided: str = "one") -> float:
    """Convert a tail probability to a Gaussian-equivalent sigma."""

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
    """Convert a Gaussian-equivalent sigma to a tail probability."""

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
        return self.__dict__.copy()


def empirical_tail_probability(
    statistics: np.ndarray | list[float],
    observed_statistic: float,
    *,
    tail: str = "greater",
    confidence: float = 0.95,
    sided: str = "one",
) -> EmpiricalTailResult:
    """Estimate an empirical tail with ``(k+1)/(B+1)`` and exact CI."""

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
    """Compare two nested-sampling evidences in log space.

    Parameters are dimensionless natural logarithms.  The returned Bayes
    factor is ``Z1/Z0`` and is safely allowed to underflow/overflow to zero or
    infinity for very decisive comparisons.  If both evidence errors are
    provided, their quadrature is used for a symmetric numerical interval.
    """

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
    """Return posterior probabilities for H0/H1 from a Bayes factor.

    ``prior_h1`` is the prior probability of H1 before seeing the data.  The
    calculation uses a log-sum-exp form and remains stable for extreme Bayes
    factors.
    """

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
    """Combine H1's posterior direction probability with model averaging.

    ``direction_given_h1`` is the posterior probability of a decrease (or
    another predeclared direction) conditional on H1.  Under H0 the strict
    direction event has probability zero, so ``p_decrease = p_h1 * q``.
    """

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
    """Propagate conservative numerical intervals to model-averaged odds.

    The inputs are ordered numerical (not posterior-credible) intervals.  The
    posterior model probability is monotone in ``log B10`` and the directional
    probability is monotone in ``q``; consequently the corner values give a
    conservative interval for ``P(H1 | D)`` and ``P(decrease | D)``.  Keeping
    this operation in the reusable core prevents individual analyses from
    silently reporting a conditional ``q`` as a model-averaged probability.
    """

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
    r"""Recover raw OFF-region PHA counts from a source-scaled background rate.

    PyXspec exposes ``Spectrum.background.values`` in counts s\ :sup:`-1`
    *scaled to the ON/source extraction region*.  For the standard ON/OFF
    convention

    .. math:: n \sim \mathrm{Pois}(s + \alpha b),\qquad m \sim \mathrm{Pois}(b),

    the raw OFF count vector is therefore

    .. math:: m = \operatorname{round}(r_{\rm bkg,src} t_{\rm on}/\alpha).

    Parameters
    ----------
    source_scaled_background_rate
        Scalar or array of source-region-scaled background rates in counts/s.
    source_exposure_s
        Positive ON/source exposure in seconds.  It may be scalar or
        broadcastable to the rate array.
    alpha
        Positive ON/OFF scale ``BACKSCAL_on * EXPOSURE_on /
        (BACKSCAL_off * EXPOSURE_off)``.  It may be scalar or broadcastable.
    tolerance
        Absolute numerical tolerance (in counts) for the required integer
        recovery.  A non-integral result is a data-contract failure, not a
        quantity to round silently.  An absolute tolerance prevents a
        relative allowance from hiding metadata mistakes at large counts.

    Returns
    -------
    numpy.ndarray
        Raw non-negative OFF counts as ``int64``, with the broadcast shape of
        the inputs.
    """

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
    r"""Marginalize a Poisson ON/OFF background analytically.

    The source-region (ON) counts and background-region (OFF) counts obey

    .. math:: n_i\sim\operatorname{Pois}(s_i+\alpha_i b_i),\quad
              m_i\sim\operatorname{Pois}(b_i),

    while ``b_i ~ Gamma(shape, rate)``.  The expansion of
    ``(s + alpha*b)**n`` makes the integral exact and stable in log space.
    ``source_model_counts`` are expected *source* counts, not background
    subtracted counts.  This is a reusable likelihood primitive; instrument
    code remains responsible for deriving and validating ``alpha`` from OGIP
    metadata.
    """

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
    r"""Profile (rather than marginalize) the Poisson ON/OFF background.

    This is a normalized Poisson likelihood with the non-negative background
    expectation profiled independently in each channel.  It is provided as a
    diagnostic bridge to XSPEC W-stat: for a fixed source model, differences
    between this value and the W-stat log likelihood should be data-only
    constants (up to the convention used by the XSPEC build).  It is not the
    formal evidence likelihood; use :func:`onoff_log_marginal_likelihood` for
    that purpose.
    """

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


__all__ = [
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
