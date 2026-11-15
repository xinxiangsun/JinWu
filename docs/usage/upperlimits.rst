上限与检测灵敏度
================

观测上限来自实际数据的似然；检测灵敏度来自给定误报率与检测概率的校准实验。
选择符合 Poisson ON/OFF、带误差背景或 Gaussian 净谱的策略，并记录固定谱模板与响应。

Jinwu separates two quantities which are often both called an upper limit:

* an **observed-data upper bound**, obtained from the likelihood of the
  measured source and background spectra;
* a **detection sensitivity**, obtained from background-only and
  signal-injection trials for a specified false-alarm threshold and detection
  probability.

The statistical strategy is selected by ``InstrumentConfig.upper_limit``.
Callers do not pass a free ``method`` string.  WXT and FXT use spatial
Poisson ON/OFF likelihoods, GBM and GECAM use modeled-background count
spectra, and BAT uses its coded-mask spectrum policy.

BAT detection sensitivity is only enabled when an instrument-native
``DetectionSensitivityAdapter`` is supplied.  Jinwu never substitutes a
simple count-spectrum S/N for a coded-mask search statistic.

Response-folded count template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The likelihood scans one non-negative amplitude while keeping the spectral
shape fixed.  The example below uses a count template which has already been
folded through the WXT RMF and ARF.  Disable the Monte Carlo result in the
configuration when only the observed bound is needed.

.. code-block:: python

   from dataclasses import replace
   import numpy as np

   from jinwu.core.config import instrument
   from jinwu.core.upperlimit import (
       CountTemplateModel,
       UpperLimitObservation,
       estimate_upper_limit,
   )

   cfg = instrument("WXT")
   cfg.upper_limit = replace(
       cfg.upper_limit,
       result_modes=("observed_upper_bound",),
   )

   observation = UpperLimitObservation(
       name="CMOS40",
       source_counts=np.asarray(on_counts),
       background_counts=np.asarray(off_counts),
       alpha=alpha,
       unit_source_counts=np.asarray(expected_counts_for_unit_flux),
       exposure_s=exposure,
       response_path="wxt.rmf",
       arf_path="wxt.arf",
   )
   model = CountTemplateModel(
       name="powerlaw(Gamma=1.8)",
       amplitude_unit="1e-10 erg cm^-2 s^-1",
       flux_per_amplitude=1e-10,
   )
   result = estimate_upper_limit(
       observation,
       model=model,
       instrument_config=cfg,
       interval=(tstart, tstop),
       energy_band=(0.5, 4.0),
       output_dir="upper_limit",
   )
   print(result.observed_upper_bound.amplitude_upper)

The production bootstrap defaults (200,000 null trials and 20,000 trials per
signal level) are intentionally expensive.  Reduce them only for exploratory
checks by replacing the corresponding fields in ``cfg.upper_limit`` and record
that override with the result.

Use :class:`~jinwu.core.upperlimit.CountTemplateModel` with
:class:`~jinwu.core.upperlimit.XspecCountPredictor` for deterministic PyXspec
``fakeit(applyStats=False)`` folding, or with
:class:`~jinwu.core.upperlimit.CallablePhotonModelPredictor` for a callable
photon model and Jinwu's RMF/ARF readers.  Supply ``flux_per_amplitude`` or
``fluence_per_amplitude`` to report physical units in addition to model
normalization.

Gaussian net-rate profile
~~~~~~~~~~~~~~~~~~~~~~~~~

Instruments that publish background-subtracted rates with Gaussian errors
(such as the Swift/BAT survey products and GBM background spectra) use the
signed net-rate profile API.  A measurement is described by
:class:`~jinwu.core.upperlimit.GaussianNetRateObservation` — the
``net_rate`` and response-folded ``unit_source_rate`` are
:class:`~astropy.units.Quantity` vectors, with either ``rate_error`` for
independent channels or a full channel ``covariance``:

.. code-block:: python

   from jinwu.core.upperlimit import (
       GaussianNetRateObservation,
       profile_gaussian_upper_bound,
   )

   obs = GaussianNetRateObservation(
       name="BAT 14-195 keV",
       net_rate=net_rate,               # Quantity, count rate (may be negative)
       unit_source_rate=rate_per_norm,  # response-folded rate for amplitude 1
       rate_error=rate_error,           # or covariance=... in rate^2 units
       exposure=exposure,
       energy_band_keV=(14.0, 195.0),
   )
   result = profile_gaussian_upper_bound(obs, sigma=3.0)
   print(result.upper_bound_quantity)

The statistic :math:`q(A) = (r - A\,t)^T C^{-1} (r - A\,t)` is profiled
while the source amplitude :math:`A` is constrained to be non-negative.
Signed input rates are intentional: a downward background fluctuation is
data, not a zero-flux replacement, and the resulting crossing is an
*observed, conditional* profile bound.

Detection sensitivity for these instruments is never folded into the same
number.  An :class:`~jinwu.core.upperlimit.EmpiricalCalibrationAdapter`
supplies blank-sky control samples plus fixed-position signal injections,
and the pipelines report ``observed_upper_bound`` and
``detection_sensitivity`` as separate results; without adequate control
trials the sensitivity is reported as ``calibration_status=unavailable``.
The BAT survey (:doc:`bat_survey`) and GBM (:doc:`fermi_gbm`) pipelines
build on exactly this contract.

Confidence semantics
~~~~~~~~~~~~~~~~~~~~

The response-aware API uses :class:`~jinwu.core.upperlimit.OneSidedLevel`.
A one-sided ``+3 sigma`` bound has probability ``Phi(3)=0.9986501`` and an
asymptotic single-parameter profile threshold ``delta_stat=9``.  This is not
the two-sided coverage ``0.9973002``.

RapidXMM fixed-background Bayesian option
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For spatial aperture Poisson data (WXT/FXT), select
``observed_method="rapid_xmm"`` to obtain the fixed-background posterior
quantile described by `Ruiz et al. (2022), equations (7)--(10)
<https://doi.org/10.1093/mnras/stac272>`_. The default remains
``observed_method="profile_likelihood"``.

For a count upper limit alone:

.. code-block:: python

   from jinwu.core import rapid_xmm_upper_limit

   counts_upper = rapid_xmm_upper_limit(counts=5, background=3.2, confidence=0.9)
   rate_upper = counts_upper / (exposure_s * encircled_energy_fraction)

``counts`` is the total observed aperture count, and ``background`` is the
fixed expected background in that aperture, with the same time and energy
selection. The prior is uniform on non-negative source counts. Background
uncertainty is not propagated; this is a Bayesian credible upper bound,
not a calibrated frequentist confidence bound or a detection sensitivity.

The response-aware interface uses the same observations and model as above:

.. code-block:: python

   from jinwu.core.upperlimit import OneSidedLevel

   cfg.upper_limit = replace(
       cfg.upper_limit,
       observed_method="rapid_xmm",
       default_sigma=OneSidedLevel.from_confidence(0.9).sigma,
       result_modes=("observed_upper_bound",),
   )
   result = estimate_upper_limit(
       observation, model=model, instrument_config=cfg,
       interval=(tstart, tstop), energy_band=(0.5, 4.0),
       output_dir="rapid_xmm_limit",
   )

For ``poisson_onoff`` inputs the background expectation is fixed to
``sum(alpha * background_counts)``. Alternatively, set
``background_likelihood="known_background"`` and supply the expectation in
``observation.background_model``. Non-zero background systematics and explicit
background error arrays are rejected for this method.

Selected channels and modules are pooled: with total counts :math:`N`,
fixed background :math:`B`, and summed response-folded counts per unit
amplitude :math:`K`, the result is :math:`A_{\rm UL}=S_{\rm UL}(N,B)/K`.
This follows the pooled aperture construction, rather than multiplying
channel/module posteriors for a common amplitude. Each module retains its
own exposure, response, and background scaling before summation. A response
template must predict counts inside the source aperture; apply its aperture
correction only once. The flux band and absorbed/unabsorbed definition are
those used to build the fixed model template.

Results record ``construction="rapid_xmm"``,
``confidence_convention="posterior_quantile"``, the prior, and aperture
totals. The JSON profile threshold is ``null``. ``default_sigma=3`` selects
the posterior probability ``Phi(3)``; the example explicitly requests 90%.
The likelihood scan is diagnostic and its delta statistic does not define
the Bayesian bound. If detection sensitivity is also requested, it continues
to use the configured detection calibration separately.

``FXTPipeline`` accepts this option through its ``config`` argument, for
example with an ``instrument("FXT")`` configuration modified as above;
``upper_limit_sigma`` on the pipeline sets its reported upper-limit level.

Legacy XSPEC parameter limits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The original helper remains available for direct XSPEC chain quantiles and
``Fit.error`` endpoints:

.. code-block:: python

   from jinwu.core.upperlimit import UpperLimit

   chain_result = UpperLimit("lg10Flux__6").from_chain("chain.fits")
   error_result = UpperLimit(6).error()

For numerical compatibility, ``UpperLimit.from_chain()`` retains Jinwu's
historical sigma-labelled quantiles.  Those labels used central Gaussian
coverages directly as posterior quantiles: for example, legacy ``3sigma`` is
the 0.997300 quantile, not the one-sided ``+3 sigma`` quantile 0.998650.  The
result includes a warning.  Request explicit one-sided quantiles with
``DEFAULT_ONE_SIDED_CHAIN_LEVELS``:

.. code-block:: python

   from jinwu.core.upperlimit import DEFAULT_ONE_SIDED_CHAIN_LEVELS

   chain_result = UpperLimit("lg10Flux__6").from_chain(
       "chain.fits",
       levels=DEFAULT_ONE_SIDED_CHAIN_LEVELS,
   )

For compatibility, the legacy ``error_result.limits["90%"]`` entry still uses
XSPEC ``delta_stat=2.706``, the central 90% profile convention.  Each
``UpperLimitPoint`` separately records its central coverage and the one-sided
probability corresponding to its high endpoint; these conventions must not be
silently interchanged.

GECAM configuration
~~~~~~~~~~~~~~~~~~~

GECAM has no guessed detector or analysis band.  Both must identify a
calibrated response explicitly:

.. code-block:: python

   cfg = instrument(
       "GECAM",
       detector="GRD01",
       energy_range_keV=(20.0, 1000.0),
   )
