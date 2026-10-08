XSPEC 能谱拟合
==============

先核查源/背景 PHA、响应、曝光与区域缩放，再准备能谱并拟合候选模型。
记录统计量、NH / redshift、误差状态、模型选择与 flux 来源；运行需要真实 PyXspec。


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

An explicit missing ``BACKFILE``/``RESPFILE``/``ANCRFILE`` reference blocks
automatic pairing with a different file. Preparation also checks the ARF's
finite nonnegative effective areas and its incident-energy grid against the
RMF (relative tolerance 1e-6, absolute tolerance 1e-8 keV). Mismatched grids
require an explicit, validated rebin before fitting. A new prepared fit clears
previous XSPEC chains; joint data groups share the Galactic absorption column.
``freeze_galactic_nh`` applies even when no replacement value is supplied, in
which case it freezes the model's initial column rather than estimating one.

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
