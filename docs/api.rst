API 参考
============

API 按对象的定义模块生成，避免重导出造成重复。jinwu.core 是便捷导入入口；
仪器包共享 jinwu 命名空间。签名与说明来自当前源码，文档构建不运行科学分析。

.. toctree::
   :maxdepth: 2

   api/modules
   compatibility

常用对象在哪里
--------------

.. list-table::
   :header-rows: 1

   * - 对象/方法
     - 定义模块
   * - readfits、read_pha、read_lc、read_evt
     - jinwu.core.io
   * - Time、TimeDelta
     - jinwu.core.time
   * - LightcurveData、PhaData、EventData
     - jinwu.core.data
   * - netdata、数据集合
     - jinwu.core.datasets
   * - txx、txx_iterbkg
     - jinwu.core.timescale
   * - snr
     - jinwu.core.significance
   * - fit_spectral、LightcurveFitter
     - jinwu.core.fit
   * - BXA 先验和结果
     - jinwu.core.bxa_fit
   * - 仪器预设、拟合配置
     - jinwu.core.config
   * - 条件上限、灵敏度适配器
     - jinwu.core.upperlimit
   * - absorption_budget
     - jinwu.physics.absorption
