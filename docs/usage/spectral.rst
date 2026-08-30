
Spectral Fitting
================

.. warning::

   This page is a work in progress.  See the module docstrings in
   :mod:`jinwu.core.fit` and :mod:`jinwu.core.upperlimit` for detailed
   API documentation.

JinWu provides a Pythonic wrapper around XSPEC/PyXspec for spectral fitting,
supporting:

- Standard models (power-law, absorbed power-law, blackbody, etc.)
- Multi-candidate model comparison ranked by AIC/AICc/BIC
- Per-parameter profile-error status in fit results
- Upper limit computation (Feldman-Cousins, Bayesian, inverse Li-Ma)

XSPEC/PyXspec must be importable in the current environment (HEASOFT
installation); these functions raise ``ImportError`` otherwise.

Prepare and Fit
~~~~~~~~~~~~~~~

The canonical flow is ``prepare_spectra`` → ``fit_prepared`` (single spectrum)
or ``fit_xray_models`` (multi-candidate comparison):

.. code-block:: python

    from jinwu.core.spectrum_prep import prepare_spectra
    from jinwu.core.fit import fit_prepared, fit_xray_models

    catalog = ...  # jinwu.core.instruments.Catalog built by scan() or by hand
    prepared_catalog = prepare_spectra(catalog, outdir="fit/prepared")
    prepared = prepared_catalog.spectra[0]

    # 单谱拟合：结果 dict 含 parameters / flux_abs / statistics /
    # error_status（每个自由参数的 profile 误差状态）
    result = fit_prepared(
        prepared,
        outdir="fit",
        model_name="tbabs*ztbabs*cflux*powerlaw",
        redshift=0.0,
        galactic_nh_1e22=0.05,
    )
    for name, parameter in result["parameters"].items():
        print(name, parameter["value"], parameter.get("error_status"))

    # 多候选模型比较（AICc 排序，自动采纳）
    comparison = fit_xray_models(
        prepared,
        outdir="fit",
        galactic_nh_1e22=0.05,
    )
    print(comparison.adopted_key, comparison.adopted_reason)
    print(comparison.to_dict()["ranking"])

Every free parameter in the result carries an ``error_status`` field
(``ok`` / ``uncomputed`` / ``boundary`` / ``failed``) plus the raw XSPEC
status string; ``error_lo`` / ``error_hi`` are only attached when the profile
interval is trustworthy.

Chain Analysis
~~~~~~~~~~~~~~

.. code-block:: python

    from jinwu.core.fit import run_xspec_chain

    # 在已加载光谱与模型的 XSPEC 会话上运行 MCMC 链
    # （先 xspec.AllData(...) 加载谱、xspec.Model(...) 定义模型并 fit）
    chain = run_xspec_chain(
        chain_path="chain.fits",
        chain_length=50000,
        chain_burn=10000,
    )
    payload = chain.to_dict()   # JSON 安全的结构化结果
