任务时间、GTI 与曝光
====================

任务时间
------------

使用注册的任务格式转换 MET；EP 的入口如下：

.. code-block:: python

   from jinwu.core.time import Time, mission_time_format, time_from_mission_seconds
   t = Time(100.0, format='ep')
   print(t.utc.isot)
   print(mission_time_format('EP'))
   assert time_from_mission_seconds('unknown', 100.0) is None

支持的具体 epoch 和 time scale 以 :mod:`jinwu.core.time` 中各格式定义为准。
Swift 使用 ``swiftmet``；旧拼写 ``swift`` 仅在接受该兼容名的 helper 内解析。
FITS 时轴同时涉及 TIMEZERO、MJDREF、TIMESYS、TIMEUNIT、TIMEPIXR。
``extract_time_interval``、``compare_time_intervals`` 与 ``plot_time_intervals``
可读取或展示区间；未知任务时钟保留相对秒，不能捏造绝对时间。

GTI 与部分曝光
--------------

``merge_gti`` / ``union_gti`` 合并有效区间；``exposure_per_bins``
计算每个 bin 与 GTI 的交叠时长，单位为秒：

.. code-block:: python

   import numpy as np
   from jinwu.core.gti import merge_gti, exposure_per_bins
   start, stop = merge_gti(np.array([0., 1., 4.]), np.array([1., 2., 5.]))
   edges = np.array([0., 1., 2., 3., 4., 5.])
   exposure = exposure_per_bins(start, stop, edges)
   assert np.allclose(exposure, [1., 1., 0., 0., 1.])

无曝光的 bin 不构成零计数观测。源与背景的有效区间不同，需要共同网格和明确的
缩放策略。瞬变时标应保留边界、曝光与时间零点的来源；见 :doc:`duration`。

API：:mod:`jinwu.core.time`、:mod:`jinwu.core.gti`。
