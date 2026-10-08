事件时标：T50 / T90 / T100
=======================================================

本版本以有符号累计净计数定义时标，保留显式观测窗口、背景变换与 Koshut 误差状态。
先核查共同连续 GTI、时间零点与 alpha，再查看分块、累计曲线、窗口和未跨越阈值。

``timescale.compute(method="aanda")`` and ``jinwu.core.ops.txx`` retain their
public route and result shapes. The corrected implementation is identified by
``implementation_version="extras_timewarp_reference_edges_signed_koshut_onoff_v4"``. This is an
algorithm correction; it deliberately changes scientific results and error
semantics, rather than preserving the old algorithm as a migration.

调用入口
------------------------------

源/背景事件需具有兼容的时间参考与相同的连续 GTI。
``alpha`` 从本观测源/背景面积与相对曝光确定，下例中的变量应先独立核验：

.. code-block:: python

   from jinwu.core.ops import txx
   result = txx(
       'source.evt', background='background.evt', alpha=alpha,
       percent=(0.5, 0.9), nmc=0, seed=42,
       cumulative_mode='adaptive',
   )
   print(result['t90'], result['t100'])

``nmc=0`` 关闭 bootstrap 诊断，不关闭条件 Koshut 误差计算。
检查具体字段的状态与诊断；不能把这个无 bootstrap 的调用当作完整的不确定度校准。
分箱迭代背景的独立方法为 ``lc.timescale(...).compute(method='iterbkg')``，
其输入、误差与窗口语义以 :func:`jinwu.core.timescale.txx_iterbkg` 为准。

Method and data contract
------------------------

* An OFF-event BB rate model defines the time transform
  :math:`t'(t)=\int_0^t \alpha\lambda_{\rm OFF}(u)\,du`. Source BB edges are
  mapped back to seconds using the same fitted integral nodes. The physical
  edges then follow the event snapping and isolated-event additions of
  ``wuqinyu/EFXT_WXT_data_processing`` at commit
  ``dd5a3f55c7c60d56901a2de29860bd8eeceb2d0c``. Signed Li & Ma block
  significance **greater than or equal to** the threshold selects the activity
  window. Every candidate block is counted as ``[left, right)`` including the
  last candidate. This differs from the cumulative histogram endpoint policy
  below. Moving edges does not re-optimize BB fitness.
* By the user's explicit decision, the reference empirical OFF-CDF transform
  and source-event inverse interpolation are **deferred**. The retained fitted
  integral transform follows De Luca et al. (2021), Sect. 5.4. This intermediate
  combination is not a reproduction of the full reference WXT or EXTraS
  pipelines. ``raw_bb_edges_time/tprime`` record edges before refinement;
  ``bb_edges_time/tprime`` record the refined edges in the retained transform.
* ON/OFF inputs must have identical continuous GTIs and compatible FITS time
  references in seconds. GTI gaps, unmatched coverage, unsupported time units,
  and a background sample with no events are rejected. Explicitly restrict
  both inputs to a fully observed interval before calling the estimator.
  Constant relative exposure/area ``alpha`` is assumed; instrument response
  and fractional exposure changes are not inferred from event arrivals.
* The cumulative uses signed :math:`N_{\rm ON}-\alpha N_{\rm OFF}` counts.
  In ``adaptive`` mode, actual ON/OFF counts are measured again on the union
  of refined source and background BB boundaries, then accumulated with a
  constant net rate within each segment. This is not strict integration of
  the independently fitted ON and OFF models;
  ``fixed`` mode uses ``evt_binsize`` with the last bin strictly ending at the
  window boundary. Uniform bins outside it retain plateau fluctuations.
* ``plateau_intervals=((pre_start, pre_stop), (post_start, post_stop))`` gives
  two emission-free cumulative plateau intervals, in absolute input seconds.
  They must precede/follow the BB window (or explicit ``burst_tstart/stop``).
  By default the complete observed intervals outside that window are automatic
  candidates. Inspect them for weak emission, background drift and truncation.
* ``T100`` is the interval from the first significant BB start to the last
  significant BB stop, or the explicit ``burst_tstart/stop`` interval. Let
  :math:`C_s(t)` be signed net counts accumulated from this start, using only
  bins within T100. The nominal fluence is :math:`N=C_s(t_{100,\rm stop})>0`,
  and percentile thresholds :math:`fN` are searched **only inside T100**.
  Hence T50 is contained in T90, and T90 is contained in T100. This is enforced
  for nominal estimates and every reselected bootstrap window, without
  truncating a duration after computing it.
* Multiple crossings use the first/last midpoint convention of ``battblocks``.
  Linear interpolation replaces BATSE's bin-edge rounding. Full-observation
  ``cumulative_signed_counts`` and ``plateau_levels`` remain diagnostics;
  ``cumulative_signal_signed_counts`` starts at zero and ends at
  ``signal_total_net_counts``. ``cumulative_levels`` gives the full-curve values
  at the T100 boundaries. Plateau means do not define nominal thresholds.
  Internal histogram bins are left-closed/right-open; only the GTI final
  endpoint is inclusive. The nominal and diagnostic counts use one histogram.
* If a plateau is unobserved, the same window-normalized point estimate remains
  available, with ``plateau_status="missing_no_koshut_error"``. This cannot
  establish that a burst ending at the GTI boundary was fully observed.

Reference boundary settings
---------------------------

``reference_boundary_options`` accepts explicit overrides in seconds. The
resolved settings are recorded in every result. In reference order:

* Snap to the closest ON event if within ``dt0_threshold=1`` s or the next
  ON event is more than ``next_event_gap=100`` s away; otherwise use the next
  ON event.
* Snap toward internal ON events with an adjacent gap of at least
  ``short_gap_1=30`` s within ``short_diff_1=15`` s, then repeat using
  ``short_gap_2=100`` s and ``short_diff_2=25`` s.
* Add the closest ON event to each combined ON/OFF endpoint or internal event
  with an adjacent gap of at least ``all_gap=200`` s. Also add internal ON
  events with an adjacent gap of at least ``src_gap=400`` s.
* Snap existing edges toward those added edges within
  ``added_edge_distance=25`` s, then take the sorted unique union.

``additional_block_edges`` supplies absolute input seconds inside the GTI.
They undergo the same refinement. No implicit configuration or user-edge file
is read. A single unique refined edge cannot define an automatic interval;
the caller can instead provide an explicit positive-width signal window.

Error semantics and migration
-----------------------------

The error model adapts Koshut et al. (1996), Eqs. 8--16:

.. math::

   \sigma_f^2 = \sigma_{\rm cnt,f}^2 + (1-f)^2 V_z + f^2 V_t,
   \qquad \delta T_{90}=\sqrt{\Delta\tau_5^2+\Delta\tau_{95}^2}.

Here ``V_z`` and ``V_t`` are sample **scatter variances**, not variances of the
plateau means. The threshold levels are anchored to the same T100 boundaries
used for the nominal estimate. Using measured plateau scatter as their
fluctuation model is an approximation; this is **not** the paper's
plateau-mean nominal estimator. For independent measured OFF data the count variance is extended
to :math:`N_{\rm ON}+\alpha^2 N_{\rm OFF}`; the original paper assumes a known
background model. The count sum is restricted to T100. ``Delta tau`` is the **full** crossing span between
:math:`S_f-\sigma_f` and :math:`S_f+\sigma_f`.

``*_err`` and ``*_err_stat`` contain this conditional Koshut prescription as a
symmetric pair for compatibility. Endpoint error pairs contain the full span,
not half-width confidence limits. ``koshut`` stores the actual crossing times,
count thresholds, count variances, separate search ranges and status. Error
sigma crossings use the full observed curve and can be outside the nominal
window. This does not move the nominal endpoints. Uncrossed thresholds give NaN rather than
zero or an observation-boundary substitute. ``*_err_sys`` remains NaN because
no calibrated systematic uncertainty has been estimated. These outputs are not
claimed to be exact 68% confidence intervals, especially at low counts.

``nmc`` now requests a **full-observation parametric Poisson bootstrap**. Fitted
piecewise ON/OFF intensities generate new photon arrivals, and each sample
refits the background, time transform, BB window, plateaus and the same nominal
cumulative estimator, including the new edge refinement. An explicit manual
window stays fixed. The fitted ON generator preserves its previous rate
allocation inside the refined BB support, with zero expectation in the
photon-free regions outside it. ``bootstrap`` stores samples, raw 16/50/84 quantiles,
validity per percentile, failures, and observation-boundary hits. ``nmc=0``
skips this diagnostic, while Koshut errors still run. Bootstrap quantiles and
``binning_sensitivity`` are diagnostics under their stated assumptions, not
extra terms added in quadrature to Koshut errors.

Unsupported legacy options (edge-background fitting, peak selection and custom
``timebins``) now fail explicitly instead of being silently ignored. Use
``plateau_intervals`` and/or ``burst_tstart/stop`` for explicit intervals.

Reproducible EP260119a comparison
---------------------------------

``scripts/compare_ep260119a_duration.py`` loads a hash-pinned old source module
and runs it beside the corrected implementation on identical photons and
settings, including an independent PI 50--400 measurement. It writes JSON,
tables and cumulative/BB diagnostic plots outside the original EP260119a
project, and checks the hashes of its protected analysis/product files.
The existing EP260119a timing script and results are not edited.

The withdrawn v2 mixed a BB T100 with a full-observation plateau normalization
and full-observation nominal crossings. Its T90 could exceed T100. Relabeling
T100 as an activity window did not resolve the inconsistent definitions.
Its source and products remain archived as rejected evidence; the default
comparison historically wrote to
``reviews/evidence/duration-20261002/window_consistent_v3/ep260119a``. Preserve
these archived products by supplying a new ``--output-dir`` when rerunning it.

The v4 comparison is a separate entry point:
``reviews/evidence/duration-reference-edges-20261003/validate.py``. It loads
the hash-pinned v3 source, compares the edge rules against the original
reference statements on identical raw boundaries, and runs original
EP260119a, its independent PI 50--400 selection, and pointed EP260809a CMOS32
in adaptive/fixed modes. Its ``results.json`` contains full results,
parameters, source/input hashes, environment and preservation checks.
The background transform, inverse, cumulative, crossing and error choices
remain as in v3. Agreement of refinement given identical raw edges does not
establish agreement of the complete reference pipelines or uncertainty coverage.

References
------------

* `De Luca et al. (2021), The EXTraS project, A&A 650 A167, Sect. 5.4
  <https://doi.org/10.1051/0004-6361/202039783>`_
* `Koshut et al. (1996), Systematic Effects on Duration Measurements of
  Gamma-Ray Bursts, ApJ 463 570, Sects. 2.2--2.3
  <https://ntrs.nasa.gov/api/citations/19970025588/downloads/19970025588.pdf>`_
* `HEASoft battblocks crossing convention
  <https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/battblocks.html>`_
* `Pinned reference WXT boundary and T100 implementation
  <https://github.com/wuqinyu/EFXT_WXT_data_processing/blob/dd5a3f55c7c60d56901a2de29860bd8eeceb2d0c/wxt_pipeline/lc_analysis.py#L34-L155>`_
