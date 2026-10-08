Einstein Probe：WXT 与 FXT
==========================

选择处理入口
------------

WXT normal-pointing 提供从官方事件、响应和曝光产品到报告的可恢复流程。
FXT 提供目录扫描、仪器配置与共享 OGIP/能谱分析组件，没有独立端到端 FXT pipeline。
WXT slew 的运动曝光与响应需要专门验证，不能直接套用 pointing 假设。

WXT 输入与最小运行
------------------

先安装 ``jinwu-ep`` 并激活可运行 XSELECT / PyXspec 的环境。
把下面的路径、观测号和坐标替换为同一次真实观测的信息：

.. code-block:: python

   from jinwu.core.config import instrument
   from jinwu.ep.wxt import WXTPointingInput, WXTPointingPipeline

   inp = WXTPointingInput(
       target_id='my_transient', root='/data/wxt_observation',
       output_root='/analysis/my_transient_wxt',
       ra_deg=159.386, dec_deg=56.171,
       obsid='your_obsid', source_id='s1', auto_approve_regions=False,
   )
   job = WXTPointingPipeline(inp, config=instrument('WXT'))
   result = job.run(until='exposure_arm_qc', resume=False)
   print(result.status, result.workspace)

``root`` 应包含可匹配的 cleaned event、RMF、ARF 与曝光图；
具体布局由 ``discover_wxt_files`` / scanner 判断。多 detector 或多个候选源时显式选择。
可提供 ``source_region``、``background_region``、``trigger_time_utc`` 和 ``redshift``。
示例坐标是格式演示，不能与任意文件组合进行科学分析。

区域与曝光检查
--------------

默认在 ``exposure_arm_qc`` 保留人工检查点。检查源/背景位置、ARM、视野边缘、
曝光覆盖率、背景缩放 alpha 及其来源。确认实际产物后再运行：

.. code-block:: python

   job.approve_regions(note='source, background and exposure reviewed')
   result = job.run(resume=True)
   print(result.summary_text())
   result.display()

``auto_approve_regions=True`` 会跳过人工区域批准，使用者仍承担区域和曝光校验。
不应通过自动批准隐藏缺失曝光、错误源区或异常背景。

阶段与产物
------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 阶段组
     - 主要内容
   * - discover / galactic_absorption / pipeline_spectrum
     - 匹配文件、Galactic NH 来源、官方能谱基线
   * - regions / provisional_events / exposure_arm_qc / final_events
     - 区域、临时事件、曝光与 ARM 诊断、最终源/背景事件
   * - duration / lightcurves
     - 时标、净光变、背景缩放、时间参考
   * - t100_spectra / t90_spectra / ogip_finalize
     - 对应时间段的源/背景谱和 OGIP 完整性
   * - bayesian_block_spectra / fit
     - 合并后的分段谱、候选模型、误差与选择理由
   * - fluxcurve / report
     - 通量曲线、图、JSON、摘要与产物索引

在 result 的 ``workspace`` 下检查每个 stage manifest 与诊断产品。
fit 参数 ``error_status``、背景/曝光来源、flux 来源与未完成阶段都应保留。
结果文件存在不能替代响应、时间和区域验收。

FXT 扫描与能谱
--------------

.. code-block:: python

   from jinwu.core.instruments import scan
   from jinwu.core.spectrum_prep import prepare_spectra
   catalog = scan('/data/fxt_observation', instrument='FXT')
   prepared_catalog = prepare_spectra(catalog, outdir='analysis/fxt/prepared')

检查每个 module 的源/背景 PHA、RMF 和 ARF 是否匹配，再按 :doc:`spectral` 拟合。
FXT 时间仍使用 EP 格式，通道能量优先从 RMF 读取。
API：:mod:`jinwu.ep.wxt.pipeline`、:mod:`jinwu.core.instruments`。
