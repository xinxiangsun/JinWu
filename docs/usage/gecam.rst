GECAMTools 接口与流程
========================================

``jinwu-gecam`` 提供 ``jinwu.gecam`` 命名空间，沿用共享 pipeline 的阶段、
输入/输出指纹与 ``run(until=..., resume=...)``。当前从源码安装；
GECAMTools ≥ 20240514 从官方发布页另装，不在导入插件时强制加载。

.. code-block:: python

   import astropy.units as u
   from jinwu.gecam import GECAMConfig, GECAMInput, GECAMPipeline

   config = GECAMConfig(
       detectors=(17,), gain="high",
       time_range=[74431550.6, 74431660.6] * u.s,
       time_bin=0.5 * u.s, energy_edges=[40, 100, 300] * u.keV,
   )
   job = GECAMPipeline(GECAMInput(
       "tn210511_112749_fb", "/data/gecam",
       output_root="/data/gecam_analysis",
       event_file="gbg_evt_tn210511_112749_fb_v00.fits",
   ), config=config)
   result = job.run(resume=True)

阶段为 ``inspect → lightcurve → spectra → analysis → report``。
源/本底区间均未指定时跳过谱生成；指定时调用上游原生 type-II 谱写出。
仅需提取时，使用 ``run(until="spectra")``。
``GECAMConfig`` 是工作流配置，与核心 ``instrument("GECAM", ...)`` 的
仪器/上限配置分开。

时间为带单位的绝对 GECAM MET，能量用 keV 或可转换单位。
EVT 的 UTC 换算读取 FITS TT 参考时刻和 ``TIMEZERO``，并将 MET 零点及其来源写入 NPZ。
每个探测器与增益单独保存；NPZ 保存计数及其误差，排列为能段×时间箱，
并记录上游实际时间、能量与通道边界。谱分箱与展示光变分箱独立。
``open_event``、``extract_lightcurve``、``extract_spectra`` 可独立调用，返回原生对象。

成品分析
--------

已有 GECAMTools full/background type-II 谱时，使用 ``GECAMInput.source_spectra``
与 ``background_spectra`` 传入以 GRD 编号为键的路径映射，并在 ``GECAMConfig.responses``
提供每个探测器及增益的响应。成品入口仅使用 JinWu 和 PyXspec。
每行 ``TIME`` 为区间左边界，``TIMEDEL`` 为持续时间，``EXPOSURE`` 为有效曝光。
GECAM-A/B 成品缺少 FITS 时钟时记录 2019-01-01T00:00:00 UTC 的假设。
GECAM-C/HEBS 成品缺时钟时必须用 ``met_reference_utc`` 明确提供 MET 零点。
full 与 background 都含 FITS 时钟时必须具有相同的 MET 零点；本底无时钟时，
诊断信息记录假设它与 full 共用零点。仅有相同的 ``TIME`` 数组不足以确认时间一致。

分析使用 Poisson full 谱和 Gaussian 模型本底，按显式 1-based 行号写出隔离的
Type-I PHA 副本，以 ``pgstat`` 拟合。默认 ``cflux*powerlaw``，用 ``energy_edges``
首末值作为观测者系拟合与通量能段，并链接同时段的谱参数。
``GECAMConfig.analysis`` 接收自定义的 ``SourceAnalysisConfig``。缺少响应时返回
``needs_review``，可用光变仍会写出。

``gecam_lightcurve_products`` 可把 NPZ 转换为保留实际时段、能段及死时间处理选择的
总计数率 ECSV。将其输出传入 ``GECAMInput.lightcurve_products``，或将它和
``prepare_gecam_spectra`` 的输出直接传入 :doc:`source_analysis` 做同源分析。

响应由调用方提供。校准有效性、残差和统计诊断需要审查，``fit_performed`` 记录
拟合是否执行，``science_status`` 为 ``requires_review``。响应索引、能道与能量边界
必须逐产品核对；官方样例的对照不能替代其他观测或校准版本的验收。
部分官方样例 RSP 有 ``NUMELT`` 求和溢出，并存在 ``F_CHAN=1``、缺失 ``TLMIN4``
与零起始 EBOUNDS/PHA 通道号冲突。准备阶段会拒绝索引不一致的输入。
仅在核对确切矩阵布局后对独立副本显式规范化并验证，保留原响应；
编码修正不能代替有效面积校准。

详细配置见源码包 README.md。官方接口与版本说明见
`GECAMTools-Public <https://github.com/zhangpeng-sci/GECAMTools-Public>`_。
