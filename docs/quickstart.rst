Quick Start Guide
=================

.. admonition:: Prerequisites
   :class: important

   JinWu's spectral fitting relies on **HEASoft + XSPEC + model data**.
   The recommended way is via conda, but any working HEASoft installation works:

   .. code-block:: bash

      conda create -n hea -c https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/conda \\
          heasoft xspec xspec-data
      conda activate hea
      pip install jinwu

   If HEASoft is already installed outside conda, use:

   .. code-block:: python

      from jinwu.core.heasoft import HeasoftEnvManager
      HeasoftEnvManager().init_heasoft()

This guide walks through common JinWu workflows.

Reading OGIP FITS Files
-----------------------

JinWu provides a unified interface for reading standard OGIP FITS files:

.. code-block:: python

   import jinwu.core as jw

   # Read a PHA spectrum file
   pha = jw.read_pha("source.pha")
   print(f"Exposure: {pha.exposure} s")
   print(f"Channels: {len(pha.channels)}")

   # Read a lightcurve
   lc = jw.read_lc("lightcurve.fits")
   print(f"Time range: {lc.time.min():.1f} – {lc.time.max():.1f} s")

   # Read ARF and RMF
   arf = jw.read_arf("source.arf")
   rmf = jw.read_rmf("source.rmf")

   # Use the generic readfits to auto-detect type
   data = jw.readfits("mystery.fits")
   print(f"Detected: {data.kind}")

Working with Energy Bands
-------------------------

.. code-block:: python

   from jinwu.core import EnergyBand, ChannelBand, channel_mask_from_ebounds

   # Define an energy band (emin/emax each carry their own unit string)
   soft_band = EnergyBand(emin=0.3, emin_unit="keV", emax=2.0, emax_unit="keV")
   hard_band = EnergyBand(emin=2.0, emin_unit="keV", emax=10.0, emax_unit="keV")

   # Map an energy band onto detector channels via the EBOUNDS extension of
   # a PHA/RMF file (channels are defined by the response, not the ARF)
   pha = jw.read_pha("source.pha")
   mask_soft = channel_mask_from_ebounds(pha.ebounds, soft_band)
   mask_hard = channel_mask_from_ebounds(pha.ebounds, hard_band)

   # Restrict to a channel range on top of the energy selection
   ch_band = ChannelBand(ch_lo=0, ch_hi=100)
   mask_soft_ch = channel_mask_from_ebounds(pha.ebounds, soft_band, ch_band)

   # Reverse direction: the energy range covered by ARF bin indices
   band = jw.band_from_arf_bins("source.arf", bin_lo=81, bin_hi=780)

Computing Net Data
------------------

The :func:`jinwu.core.netdata` function computes the background-subtracted
net light curve with proper uncertainty propagation:

.. code-block:: python

   src = jw.read_lc("source.lc")
   bkg = jw.read_lc("background.lc")

   net = jw.netdata(src, bkg)            # scaling ratio auto-computed
   net = jw.netdata(src, bkg, ratio=0.1)  # manual source/background ratio
   net = src - bkg                        # equivalent shorthand

EP/WXT Pointing Pipeline
------------------------

For an EP-WXT pointing observation, the end-to-end pipeline goes from raw
L2/L3 data to lightcurves, duration, spectral fits, flux curve and a Chinese
quicklook report:

.. code-block:: python

   from jinwu.core.config import instrument
   from jinwu.ep.wxt import WXTPointingInput, WXTPointingPipeline

   inp = WXTPointingInput(
       target_id="EP260809a",
       root="/data/06800001692_32",        # 官方 L2/L3 数据目录
       output_root="/analysis/ep260809a_wxt", # 独立产物目录
       source_id="s1",
       ra_deg=..., dec_deg=...,
       obsid="06800001692",
       auto_approve_regions=False,
   )
   pipeline = WXTPointingPipeline(inp, config=instrument("WXT"))
   preview = pipeline.run(until="exposure_arm_qc", resume=False)
   print(preview.status, preview.workspace) # 预期 needs_review
   # 在图像上检查 regions/regions.json 指向的源区、背景区与 ARM，
   # 并阅读 regions/exposure_qc.json 中的覆盖率、alpha 与警告。
   pipeline.approve_regions(note="source and background reviewed")
   result = pipeline.run(resume=True)

   result.summary_text()                   # 中文快报文本
   result.display()                        # Jupyter 内嵌展示产物图
   print(result.workspace)                 # 所有产物都在这个工作区目录

Spectral fitting is configurable through the instrument config
(``FitConfig``: candidate models, ``comparison_intervals``, error
``error_delta_stat`` and so on); every fit result records per-parameter
profile-error status (``error_status``) instead of blindly trusting XSPEC
error output.

Next Steps
----------

* See :doc:`api` for the complete API reference.
* See :doc:`usage/spectral` and :doc:`usage/lightcurve` for fitting details,
  and :doc:`usage/bxa_fitting` for BXA/UltraNest Bayesian spectral fitting.
* For instrument pipelines, see :doc:`usage/swift_grb` (Swift BAT+XRT GRB),
  :doc:`usage/bat_survey` (Swift/BAT survey targets) and
  :doc:`usage/fermi_gbm` (Fermi/GBM continuous data), and
  :doc:`usage/gw_coverage` (GW localization and coverage plots).
* Check the `GitHub repository <https://github.com/xinxiangsun/jinwu>`_
  for examples and issue tracking.
