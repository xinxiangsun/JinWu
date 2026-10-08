计数实验显著性（SNR）
=============================================

``jinwu.core.snr`` provides five cases from Vianello (2018) and the matching
``gv_significance`` source. The method is selected explicitly; rates and net
counts must not be passed as ON total counts.

.. list-table:: Background assumptions
   :header-rows: 1

   * - Case
     - Arguments
     - Meaning of ``background``
   * - Exactly known background
     - ``method="known"``
     - Expected ON background counts
   * - Poisson ON/OFF, Li--Ma
     - ``method="pp", alpha=...``
     - Observed OFF counts
   * - Fixed relative background adjustment
     - PP with ``systematic_fraction=k``
     - OFF counts; effective alpha is alpha*(1+k)
   * - Gaussian relative systematic error
     - PP with ``systematic_sigma=...``
     - OFF counts; sigma is dimensionless
   * - Poisson measurement, Gaussian background estimate
     - ``method="pg", background_error=...``
     - Expected ON counts; error is absolute counts

Usage / 调用
------------

.. code-block:: python

   import astropy.units as u
   from jinwu.core import snr

   # ON/OFF: alpha = ON/OFF area ratio * ON/OFF livetime ratio.
   z_pp = snr(20, 80, method="pp", alpha=0.1)
   z_fixed = snr(20, 80, method="pp", alpha=0.1,
                 systematic_fraction=0.1)
   z_systematic = snr(20, 80, method="pp", alpha=0.1,
                      systematic_sigma=0.1)

   # Gaussian background estimate: observed TOTAL counts, predicted background,
   # and its propagated standard error in that same ON interval/energy band.
   z_pg = snr(120*u.ct, 80*u.ct, method="pg",
              background_error=5.3*u.ct)
   z_known = snr(5, 2, method="known")

Inputs broadcast with NumPy rules. Scalars return ``float``; a one-element
array remains an array. Counts accept bare numbers, count Quantity or
dimensionless Quantity. ``alpha`` and both relative systematic parameters
must be dimensionless. Invalid units, conflicting parameters and nonfinite
inputs raise errors. Positive fixed and Gaussian systematics are mutually
exclusive per element; arrays may mix the three PP cases.

GBM with a polynomial background fit usually uses PG: propagate the full
coefficient covariance into the predicted counts in the chosen ON window.
For a bin-averaged rate basis ``q``, coefficients ``c``, covariance ``C``
and livetime ``t``, background counts are ``t*(q @ c)`` and their variance is
``t**2 * (q @ C @ q)``. For independent detector fits, add expected counts
and variances separately. ``sqrt(background)`` is not the fitted background
standard error. The public SNR function performs no implicit rate, exposure,
energy-band, covariance or deadtime conversion.

Source compatibility / 源码一致性
---------------------------------

The source is pinned to ``946d761a41cc26bda21ead8b5ba7750dd5a389a4``.
``GV_SIGNIFICANCE_PROVENANCE.json`` records source file hashes and
adaptations; the complete BSD 3-Clause license is included in wheel/sdist.
The main source equations are preserved, including the default SciPy
one-dimensional systematic optimization, initial k=0 and tolerance 1e-3.

The existing scalar ``li_ma_snr`` implementation is unchanged and remains
available from ``jinwu.core.utils`` and the historical lightcurve wrapper.
Plain and fixed-adjustment PP reuse it. Its analytic zero-count limits
replace upstream's 1e-25 padding; the tiny numerical differences are recorded
in the source-equivalence evidence. The new ``snr`` API validates inputs more
strictly than the legacy function.

The known-background branch intentionally follows the source's
``-ndtri(pdtrc(n, b))``: **P(N > n)**. The article's text defines **P(N >= n)**;
these differ by one count at the lower summation endpoint. This migration
preserves source results. A correction requires a separate method/change.

``signed=True`` preserves source conventions. PP signs use nominal
``n_on-alpha*background`` even with the fixed adjustment, PG signs use
``n_on-background``, and the known-background branch returns the one-sided
normal quantile directly. ``signed=False`` gives absolute magnitude. The
fixed adjustment is not a profile over a bounded nuisance interval, and its
positive nominal sign does not establish an excess above the adjusted
background. Negative likelihood-root scores are not calibrated deficit
significances in the low-count regime.

PG requires positive ``background_error``. Its zero-error likelihood limit
is different from the known-background exact-tail test and is not silently
substituted. PG permits negative measured background estimates, as in the
paper; the profiled null background is nonnegative.

The original PP Gaussian-systematic optimizer returns a spurious score when
both counts are zero because of its likelihood barrier. That input is
rejected. Failed convergence, negative/nonfinite TS or unrepresentable tail
probabilities raise explicit errors. Extremely small positive systematic
sigma can also cause the frozen optimizer to lose precision; choosing sigma=0
is a distinct explicitly known input, not an automatic fallback.

All returned statistics are local. Searching times, bin widths, energy bands
or detectors requires its own trials calibration. Adding this API does not
change the frozen MVT detector selection, Haar statistic or empirical curve.

Validation / 验收
-----------------

.. code-block:: bash

   conda activate hea
   MPLCONFIGDIR=/tmp/jinwu-snr-mpl python -m scripts.validate_snr \
       --output reviews/evidence/snr/validation.json

The validator executes original function bodies with obsolete unused
``Z_Bi/ncephes`` imports omitted, compares final and intermediate values,
reads real WXT ON/OFF PHA, and refits two real GBM triggers plus an off-source
control from original TTE. GBM validation uses fixed complete 1-s bins,
native-channel deadtime, and full fitted coefficient covariance. No original
data or previous MVT result is overwritten. Analytical tolerances are
rtol=atol=1e-10; numerical systematics use 1e-7. See
``reviews/snr-gv-significance-migration.md`` for provenance and limitations.

References
------------

* `Vianello (2018), Significance of an excess in a counting experiment:
  assessing the impact of systematic uncertainties and the case with
  Gaussian background <https://arxiv.org/abs/1712.00118>`_
* `Frozen gv_significance source
  <https://github.com/giacomov/gv_significance/tree/946d761a41cc26bda21ead8b5ba7750dd5a389a4>`_
* `SciPy pdtrc definition
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pdtrc.html>`_
