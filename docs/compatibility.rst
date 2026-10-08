版本兼容与迁移
==============

规范导入路径
------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 历史路径或用法
     - 当前建议
   * - ``jinwu.lightcurve`` / ``jinwu.spectrum`` 模拟转发
     - ``jinwu.lf.lcfake`` / ``jinwu.lf.specfake``
   * - ``jinwu.core.redshift`` / ``jinwu.core.lf``
     - ``jinwu.lf`` 中的定义模块
   * - 大写文件名 ``BATObservation`` / ``GBMObservation``
     - 包导出的类或小写定义模块
   * - ``from jinwu.core import timescale`` 作为分析器类
     - 该名称现在是模块；使用 ``lc.timescale(...)`` 或 ``jinwu.core.data.timescale``
   * - ``fig, (ax, ax_res) = fitter.plot_fit(...)``
     - 返回 ``ax, ax_res``；figure 为 ``ax.figure``
   * - ``fit_prepared(plot_dpi=...)``
     - ``plot_density`` 表示模型采样密度；图片 DPI 在绘图/保存处设置
   * - ``jinwu.response``
     - GBM 响应在 ``jinwu.fermi.gbm.response``
   * - ``jinwu.ftools.grppha_hsp``
     - 优先 ``ftgrouppha``；核查具体分组语义

``fit_prepared`` 使用显式默认值；统一分发与临时设置上下文见
:doc:`usage/bxa_fitting`。升级时请核对配置指纹及阶段缓存是否需要重跑。
本工作区安装元数据可能落后于源码声明版本，见 :doc:`installation`。

历史材料
------------

下列旧教程仅用于追溯，含遗留公式和接口；先阅读 :doc:`usage/simulation` 与已知限制。

.. toctree::
   :maxdepth: 1

   RedshiftExtrapolator
