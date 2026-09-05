
Light-curve Fitting
===================

.. warning::

   This page is a work in progress.  See :mod:`jinwu.core.fit` for the
   complete API.

JinWu's light-curve fitting module supports:

- Built-in models: power-law, broken power-law, smoothly broken power-law,
  exponential decay, Gaussian, constant
- Custom astropy ``Fittable1DModel`` subclasses
- Astropy-based fitting backends (Levenberg-Marquardt, TRF)
- Unified interface for ``LightcurveData`` and ``LightcurveDataset``

Example
~~~~~~~

.. code-block:: python

    import numpy as np
    from pathlib import Path
    from jinwu.core.data import LightcurveData
    from jinwu.core.fit import LightcurveFitter

    t = np.logspace(-1, 1, 20)
    lc = LightcurveData(
        time=t,
        value=100.0 * np.power(t, -1.5),
        error=np.full(20, 0.5),
        path=Path("dummy.lc"),   # OGIP dataclass 必需的元数据字段
        header={},
        meta=None,
        headers_dump=None,
    )

    fitter = LightcurveFitter(lc)
    result = fitter.fit(
        "smoothly_broken_powerlaw",
        bounds=([0.0, 0.0, 0.1, 0.5], [np.inf, 10.0, 10.0, 20.0]),
    )
    print(result.summary())

    # 参数按位置对应 result.param_names：
    for name, value, error in zip(result.param_names, result.params, result.errors):
        print(f"{name} = {value:.3g} ± {error:.2g}")

    # 绘图（返回 (fig, (主图, 残差图))）
    fig, (ax, ax_res) = fitter.plot_fit(result)
    fig.savefig("fit.png", dpi=300)

Notes
~~~~~

- ``LightcurveData`` 字段为 ``time`` / ``value`` / ``error``（另兼容
  ``counts`` / ``rate`` 等别名）。
- ``fitter.fit(model, ...)`` 接受注册表名称或 astropy 模型类；
  ``fitter_method`` 支持 ``"lm"``（默认）与 ``"trf"``。
- ``FitResult`` 提供 ``summary()``、``evaluate(t)`` 与 ``to_dict()``。

Multi-object plotting
~~~~~~~~~~~~~~~~~~~~~

:mod:`jinwu.core.plotpanel` adds lightcurve/spectrum overlay helpers that
accept any lightcurve- or spectrum-like object (``LightcurveData``,
``LightcurveDataset``, ``SpectrumDataset``, …).
:func:`jinwu.core.overlay` stacks several objects into one axes and
returns ``(fig, ax)``; :func:`jinwu.core.multi_panel` gives each object
its own vertically stacked, shared-x panel and returns ``(fig, axes)``,
optionally saving to ``out``:

.. code-block:: python

   from jinwu.core import multi_panel, overlay

   # One axes, several objects
   fig, ax = overlay([lc_a, lc_b], labels=["EP", "Swift/XRT"], xmode="relative")

   # One shared-x panel per object
   fig, axes = multi_panel([lc_a, lc_b], labels=["EP", "Swift/XRT"],
                           out="comparison", formats=("png",))

Panel appearance (height ratio, y-scale, colors, per-panel plot kwargs) is
configured through :class:`jinwu.core.PanelSpec` objects passed via
``multi_panel(..., panels=[...])``.
