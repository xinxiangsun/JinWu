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

   import jinwu as jw

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

   from jinwu.core import EnergyBand, ChannelBand

   # Define an energy band
   soft_band = EnergyBand(0.3, 2.0, unit="keV")
   hard_band = EnergyBand(2.0, 10.0, unit="keV")

   # Convert to channel indices from an ARF
   arf = jw.read_arf("source.arf")
   ch_soft = ChannelBand.from_energy_band(soft_band, arf)
   ch_hard = ChannelBand.from_energy_band(hard_band, arf)

Computing Net Data
------------------

The :func:`jinwu.netdata` function computes net (background-subtracted)
count rates and errors with proper uncertainty propagation:

.. code-block:: python

   net, net_err = jw.netdata(
       src=src_counts,
       bkg=bkg_counts,
       exposure=1000.0,
       backscale=0.1,     # BKG / SRC area ratio
   )
   snr = net / net_err
   print(f"S/N = {snr:.1f}")

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
       source_id="s1",
       ra_deg=..., dec_deg=...,
       obsid="06800001692",
       auto_approve_regions=True,          # 跳过人工区域审批
   )
   pipeline = WXTPointingPipeline(inp, config=instrument("WXT"))
   result = pipeline.run(resume=False)     # 断点续跑用 resume=True

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
* See :doc:`usage/spectral` and :doc:`usage/lightcurve` for fitting details.
* Check the `GitHub repository <https://github.com/xinxiangsun/jinwu>`_
  for examples and issue tracking.
