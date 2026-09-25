GBM 亚临界定向搜索
==================

``jinwu.fermi.gbm.subthreshold`` 面向给定外部触发时刻的连续 TTE 搜索。
它是独立的可恢复流水线，不要求 GBM 星上触发；不是全天连续盲搜。
已有 ``fermi_gbm`` 的谱提取、上限和几何覆盖功能仍各自使用原接口。

安装与输入
----------

在仓库根安装搜索依赖（天文分析使用 ``hea`` 环境）：

.. code-block:: bash

   conda run -n hea python -m pip install -e 'packages/jinwu-fermi[search]'
   # 需要概率天图时，另安装本地 jinwu-gw。
   conda run -n hea python -m pip install -e packages/jinwu-gw

响应模板从 `FSSC 官方模板归档 <https://fermi.gsfc.nasa.gov/ssc/data/analysis/gbm/templates.tar.gz>`_
取得并解包；传入含 ``direct/``、``atmo_nai/``、``atmo_bgo/`` 的 ``templates/GBM`` 目录。
预检记录每个文件的 SHA256、形状和清除的负数舍入残差数目。
不会隐式下载模板；``--download`` 只下载所需连续 TTE 和实际 POSHIST。
也可以在 ``--root`` 下提供本地归档，递归发现对应日期/小时的文件。

.. code-block:: bash

   conda run -n hea python -m jinwu.fermi.gbm.subthreshold search GW170817 \
     --time 2017-08-17T12:41:04.429126 --root /data/gbm \
     --templates /data/templates/GBM --output /data/results/GW170817 --download

默认使用 14 个探测器、5 度原生天区网格、hard/normal/soft 三种模板，
在触发前后 30 秒扫描 0.064--8.192 秒的二倍时间尺度。
每尺度最多 8 个相位，步长不小于 0.064 秒；背景窗口 125 秒，外围上下文 500 秒。
``--until background`` 可只准备数据；默认恢复已完成且输入、配置与代码哈希一致的阶段。

Python 接口要求时间偏移带单位：

.. code-block:: python

   import astropy.units as u
   from jinwu.fermi.gbm.subthreshold import (
       GBMTargetedSearchInput, GBMTargetedSearchConfig, run_targeted_search,
   )

   job = GBMTargetedSearchInput(
       target_id="external_event", trigger_time="2017-08-17T12:41:04.429126",
       root="/data/gbm", template_root="/data/templates/GBM",
       output_root="/data/results/external_event",
   )
   config = GBMTargetedSearchConfig(search_interval=[-5, 5] * u.s)
   result = run_targeted_search(job, config=config)
   print(result.science_status, result.products["report"])

位置与结果
----------

``position=SkyCoord(...)`` / ``--position RA DEC`` 或本地 ``skymap`` / ``--skymap``
二选一。空间先验在每个窗口评分时参与计算，然后按该评分筛选和去重。
始终另存 GBM 独立评分和候选，防止外部先验改变后的结果被误作独立探测。
点位置采用最近原生网格，并报告角偏移；天图概率投影至原生网格。
``visible_prior_probability`` 保留地球遮挡后的先验质量，评分使用可见区域内条件先验。
零可见质量不产生定向候选。

输出包括 ``full_search.ecsv``、两种评分的候选表、最终 ``candidates.ecsv``、
``waterfall.png``、``lightcurves.png``、候选统计 HEALPix FITS、背景诊断和 ``report.json``。
``loglr`` 与 ``prior_loglr`` 是未校准的排序统计量，不是 sigma。
``ra/dec/template/photon_flux`` 属于独立 GBM 最优解；``prior_`` 前缀字段属于先验加权解。
光子通量单位为 photon cm^-2 s^-1，模板归一化能段 50--300 keV。
定位图为 GBM 独立统计图，不含系统误差模型，也不等同于联合定位后验。

缺失数据、GTI/SAA/姿态间隙、背景不足均不会被解释成未探测。
数据阶段保留同一时刻不同通道的事件以及文件内多重事件；跨文件重叠去重。
背景用 GDT NaivePoisson，按局部活时间比例将率转换为每活秒，再乘搜索窗口活时间。
背景方差为完整的 ``(rate_uncertainty * exposure)**2``，不使用上游额外的 0.5 因子。
背景控制块没有独立留出，诊断会明确标识这一限制。
磷光事件筛除复用上游判据。超出大气响应摇摆角范围时保留直接响应的条件候选，
整体标为 ``needs_review``；不能据此宣称通过正式响应验证。
候选窗口端点响应与中点响应也会进行稳定性检查。

离源 FAR
--------

将经过审查、远离目标与已知瞬变的 UTC 时刻逐行写入 ``off_times.txt``，运行：

.. code-block:: bash

   conda run -n hea python -m jinwu.fermi.gbm.subthreshold calibrate GW170817 \
     --time 2017-08-17T12:41:04.429126 --root /data/gbm \
     --templates /data/templates/GBM --output /data/calibration/GW170817 \
     --off-times off_times.txt --download

搜索时添加 ``--calibration /data/calibration/GW170817/calibration.json``。
校准使用同一窗口网格、模板、探测器、筛选与空间先验；配置、版本和代码哈希必须匹配。
离源时刻的背景上下文不得污染目标上下文；重复时间去重，重叠搜索时段拒绝重复计曝光。
只有质量检查通过的离源结果才能进入校准。
零个超过阈值的事件返回 FAR 的 95% 上限，绝不返回零 FAR。
FAP 使用 Poisson 到达假设；离源时段的代表性和长期平稳性仍需数据验证。
本功能不附带已经校准的通用背景分布。软件测试和公开事件复现都不能替代 FAR 校准。

来源与改动
----------

似然核及相关算法来自 `USRA-STI gamma-ray-targeted-search
<https://github.com/USRA-STI/gamma-ray-targeted-search/tree/1bc1e913f97fd7195a7e297f8d6032a5c7758894>`_，
固定提交 ``1bc1e913f97fd7195a7e297f8d6032a5c7758894``。
``_vendor/license.txt`` 保留 Apache-2.0 许可，``UPSTREAM.json`` 记录原文件哈希和修改说明。
JinWu 适配层增加单位/活时间、连续文件、质量状态、周期方位角插值、空间先验、校准与流水线输出。
候选重叠聚类和完整背景方差等适配使结果不应逐数值等同旧版 GTS 发布结果。

方法参考：`Goldstein et al. (2019), arXiv:1903.12597 <https://arxiv.org/abs/1903.12597>`_；
`Kocevski et al. (2018), arXiv:1806.02378 <https://arxiv.org/abs/1806.02378>`_。
时间定义遵循 `FSSC MET 文档
<https://fermi.gsfc.nasa.gov/ssc/data/p7v6/analysis/documentation/Cicerone/Cicerone_Data/Time_in_ScienceTools.html>`_：
从 2001-01-01 UTC 起连续计 SI 秒，包含闰秒，以 TT 历元进行运算，避免导入 GDT 前后定义不同。

本地验证记录（2026-09-12）
--------------------------

复现脚本：``scripts/validate_gbm_subthreshold.py``，接受 ``--root``、``--templates``、
``--output`` 和可选 ``--download``；``--short`` 选择前后 5 秒的快速检查。
以下载的 FSSC 连续 TTE、实测 POSHIST 和官方模板运行默认配置，两事件各评估
4,581 个窗口，无 GTI/姿态排除和窗口计算失败。最终产物位于本地
``.runtime/gbm-search-validation/default/``，不随安装包分发。

.. list-table:: 默认配置的最高独立 GBM 评分候选（不是显著性）
   :header-rows: 1

   * - 参考事件
     - 参考 MET / s
     - 窗口起点 / 相对秒
     - 窗口长度 / s
     - loglr
     - 科学状态
   * - GW170817
     - 524666469.429126
     - 1.728
     - 0.512
     - 70.0211
     - uncalibrated_candidates
   * - GRB140606A
     - 423745096.496
     - -0.064
     - 0.256
     - 78.7048
     - needs_review

GW170817 的本次默认窗口背景和响应检查通过；没有输入离源校准，FAR 保持空值。
GRB140606A 的 n4/n7 背景检查未通过，且大气响应不适用，故该候选只供复核。
较短搜索窗口会改变背景控制块与诊断，不能把默认窗口的通过状态转用到其他配置。
两个事件的全部 11 张统计定位图通过有限值、非负和归一化检查；未验证系统定位误差覆盖率。

离线全套回归为 1,378 passed、43 skipped、6 deselected；随后新增可选依赖隔离测试，
搜索测试共 17 项，源码和安装到独立目录的 wheel 均通过。
wheel 检查还确认实际导入来自安装目录、保留 Apache 许可与来源记录，且配置导入不强制加载搜索依赖。
本次没有生成可用于科学结论的真实离源 FAR 校准库。
