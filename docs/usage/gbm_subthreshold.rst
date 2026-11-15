GBM 亚临界定向搜索
==================

``jinwu.fermi.gbm.subthreshold`` 面向给定外部触发时刻的连续 TTE 搜索。
它是独立的可恢复流水线，不要求 GBM 星上触发；不是全天连续盲搜。
已有 :doc:`fermi_gbm` 的谱提取、上限和几何覆盖功能仍各自使用原接口。

安装与输入
------------

在仓库根安装搜索依赖（天文分析使用 ``hea`` 环境）：

.. code-block:: bash

   conda run -n hea python -m pip install -e 'packages/jinwu-fermi[search]'
   # 搜索天图读取使用核心 skymap，不要求完整 GW 包。
   conda run -n hea python -m pip install -e 'packages/jinwu-fermi[search-skymap]'

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
------------

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
------------

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
------------

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

使用前验收
------------

以实际 TTE、POSHIST 和官方模板核查完整流程，检查窗口覆盖、背景残差、
响应适用性、候选表与定位图的有限值和归一化。改变搜索时窗也会改变背景控制块
与诊断，不能沿用其他配置的通过状态。统计定位图不包含系统误差覆盖率保证。
FAR 需要同配置、具代表性的独立离源数据；软件测试或公开事件复现不能提供通用校准库。
