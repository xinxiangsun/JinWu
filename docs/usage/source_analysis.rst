同源多仪器分析
==============

``SourceAnalysisPipeline`` 接收已经完成归约的单源能谱和光变。每份谱保留自己的
PHA、本底、响应、拟合能段和统计量。只有显式指定相同 ``epoch``、时窗有交集、
且开始和结束时刻分别在 ``simultaneity_tolerance_s`` 内一致的能谱才进入同一次联合拟合。使用
同一标签前还应核对各仪器的 GTI 和曝光可代表同一物理阶段。不同历元独立拟合，
输出统一到同一个观测者 UTC 参考时刻。调用者负责关联天体及统一到达时间的修正
约定。可给每个产品填写 ``target_id``，流水线会核对其与输入目标一致。

同一目标的不同时段
------------------

下例使用已归约的 EP/FXT 余辉和 GECAM 提示辐射谱。为每份 EP 谱提供其实际提取
起止 UTC。GECAM 的转换入口从原生 type-II 谱读取提取时段，并对 Poisson 源谱
和 Gaussian 模型背景选择 ``pgstat``。EP 示例选择带 Poisson 背景的
``cstat``，XSPEC 对此使用 W-stat。不同历元可在同一观察者系能段比较模型通量。

.. code-block:: python

   from pathlib import Path

   import astropy.units as u

   from jinwu.core.config import FitConfig
   from jinwu.core.instruments import scan
   from jinwu.core.source_analysis import (
       SourceAnalysisConfig, SourceAnalysisInput, SourceAnalysisPipeline,
       SpectralObservation,
   )
   from jinwu.core.spectrum_prep import prepare_spectra
   from jinwu.gecam.analysis import prepare_gecam_spectra

   def analyze_prompt_and_afterglow(
       target, reference_utc, fxt_dir, fxt_bounds_utc, full, background,
       responses, output, flux_band_keV, simultaneity_tolerance_s,
       model_energy_grid=None,
   ):
       # full/background/responses 均为以 GRD 编号为键的文件字典。
       # fxt_dir 对应同一次余辉观测，fxt_bounds_utc 以模块名为键。
       output = Path(output).resolve()
       prepared = prepare_spectra(scan(fxt_dir, instrument="fxt"),
                                  outdir=output / "fxt-prepared")
       if not prepared.ready:
           raise ValueError(prepared.diagnostics)
       fxt = tuple(
           SpectralObservation(s, *fxt_bounds_utc[s.module], "cstat",
                               "fxt_afterglow", target_id=target)
           for s in prepared.spectra
       )
       gecam = prepare_gecam_spectra(
           full, background, responses, outdir=output / "gecam-prepared",
           energy_range=[40, 200] * u.keV, target_id=target, gain="high",
       )
       data = SourceAnalysisInput(
           target, output, output_root=output / "analysis",
           spectra=(*gecam, *fxt), reference_utc=reference_utc,
       )
       settings = SourceAnalysisConfig(
           fit=FitConfig(model_name="cflux*powerlaw"),
           flux_band_keV=flux_band_keV, joint_normalization="tied",
           simultaneity_tolerance_s=simultaneity_tolerance_s,
           fit_options={"model_energy_grid": model_energy_grid},
       )
       return SourceAnalysisPipeline(data, config=settings).run()

本例的幂律只示范接口。根据目标传入实际谱模型、吸收、红移和冻结参数。
``flux_band_keV`` 给出共同积分能段。若超出某份谱的原生模型能量网格，应
明确提供 ``model_energy_grid=(emin_keV, emax_keV, nbin)``，其范围须覆盖所用
原生响应和通量积分能段。网格扩展依赖谱模型外推，应检验结果对谱形和网格设置
的敏感性。通量表保留每份谱的实际纳入能量跨度和外推标记。

同一时段的联合谱
----------------

从两台仪器各取出确实代表同一时段的记录。例如将已验证的 ``ep_observation``
和 ``gecam_observation`` 设置为相同的 ``epoch``。保留各自原始的提取时刻及
统计量。以下设置共享源谱形状和归一化，独立拟合各仪器的相对校准系数。第一份
谱作为参考，其校准系数固定为 1。

.. code-block:: python

   from dataclasses import replace

   simultaneous = tuple(
       replace(obs, epoch="prompt_common")
       for obs in (ep_observation, gecam_observation)
   )
   data = SourceAnalysisInput(
       target, root, output_root=output,
       spectra=simultaneous, reference_utc=reference_utc,
   )
   config = SourceAnalysisConfig(
       fit=FitConfig(model_name="constant*cflux*powerlaw"),
       flux_band_keV=flux_band_keV,
       joint_normalization="cross_calibration",
       simultaneity_tolerance_s=simultaneity_tolerance_s,
       fit_options={"model_energy_grid": model_energy_grid},
   )
   result = SourceAnalysisPipeline(data, config=config).run()

``simultaneity_tolerance_s`` 默认是 0.001 秒。应按实际时钟精度与提取边界设置，
不能通过放大容差把不同时段合为同一历元；时窗即使符合容差，也必须有交集。
报告保留每份谱的实际起止时刻。

``tied`` 对同一历元内各谱共享归一化。``independent`` 保留底层
``fit_prepared`` 的仪器和拟合能段分组，每组独立归一化，默认共享谱形。
需要按谱共享吸收、独立拟合谱形时，使用 ``independent`` 并显式传入
``fit_options={"shared_absorption_only": True}``。

光变和产物
----------

``SourceAnalysisInput.lightcurves`` 接受多个 ``LightcurveProduct``。
每份 ECSV 记录一种仪器与能段或滤光片，包含四列：

- ``time_start``、``time_stop`` 使用可转换为秒的单位，相对于该产品的
  ``reference_utc``。时间箱依序排列且互不重叠。
- ``value``、``error`` 使用可相互转换的单位，误差为对称的测量误差。
  可在 ECSV 元数据中填写 ``kind``，如 ``total_rate``、``net_rate``
  或 ``flux``。如填写 ``target_id`` 或 ``reference_utc``，必须与声明一致。

流水线按产品绘制光变，将箱中心和边界转换到全局 ``reference_utc``。
各仪器的计数率与通量保持独立。默认生成光变图与摘要。显式设置
``lightcurve_model`` 可对每份产品单独用 ``LightcurveFitter`` 做 Gaussian
误差拟合，模型在时间箱中心求值。
仅在 Gaussian 误差近似适用时启用。宽时间箱或快速演化时，需评估箱中心值能否
代表箱内平均值。若测量误差中含零，仍会保留图表，拟合标为
``unavailable``。谱分析使用 XSPEC MLE。

含 ``cflux`` 的谱拟合生成共同能段的去吸收能量通量表与图。通量图用对数纵轴，
横误差条表示提取时窗。仅在双侧 profile 误差状态为 ``ok`` 时绘制有效的纵
误差条，其余点用空心符号标示。表中保留 ``error_delta_stat``、误差状态、所用
仪器、参考校准谱和模型能量网格，便于审查。每个历元另有 XSPEC 结果 JSON、文本
报告、日志、可回放 XCM 和谱残差图。总报告位于 ``source_analysis.json``。
``pipeline_status=completed`` 表示执行完成，``science_status=requires_review``
保留科学审查状态。恢复运行时会检查输入文件、配置、代码与运行环境的指纹。
运行完成和与直接调用结果一致仅验证接口与流程；谱模型、背景、响应校准及
Gaussian 光变误差假设仍需针对实际数据审查。
