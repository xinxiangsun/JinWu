JinWu 使用手册
==============

**从观测产品到可追溯的光变、能谱与统计推断。**

JinWu（金乌）为高能瞬变分析提供统一的数据对象、时间处理、统计接口和仪器流程。
本手册覆盖当前 |release| 版本，包括 WXT、Swift BAT/XRT、Fermi GBM、GW 覆盖，
以及有符号净计数时标、计数实验 SNR 和 Haar MVT。工作区功能不等同于已发布 wheel；安装时请核对版本。

.. container:: jinwu-start-grid

   .. container:: jinwu-start-card

      **第一次使用**

      :doc:`installation` → :doc:`quickstart`：选择依赖，理解数据与单位。

   .. container:: jinwu-start-card

      **已有观测数据**

      :doc:`pipelines`：按仪器选择输入、运行入口、阶段与产物。

   .. container:: jinwu-start-card

      **需要一个分析方法**

      :doc:`user_guide`：时间、背景、时标、拟合、上限、模拟与绘图。

   .. container:: jinwu-start-card

      **查询具体接口**

      :doc:`api`：按定义模块查看参数、返回值、单位和源码。

选择适合的起点
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - 你要做什么
     - 起点
     - 主要依赖
   * - 读 FITS、处理时间或绘制光变
     - :doc:`usage/ogip`、:doc:`usage/time_gti`、:doc:`usage/visualization`
     - 核心 jinwu
   * - T50/T90、计数显著性或光变拟合
     - :doc:`usage/duration`、:doc:`usage/significance`、:doc:`usage/lightcurve`
     - 按方法准备 ON/OFF 与曝光
   * - XSPEC 或 BXA 能谱分析
     - :doc:`usage/spectral`、:doc:`usage/bxa_fitting`
     - HEASoft / PyXspec；BXA 可选
   * - WXT、Swift、GBM 或 GW 数据处理
     - :doc:`pipelines`
     - 对应仪器包与任务依赖
   * - 元素吸收、模拟或红移转换
     - :doc:`usage/absorption_budget`、:doc:`usage/simulation`
     - 按后端选择 PyXspec 与响应

本版本的能力边界
----------------

读取成功、拟合结束或文件存在都不足以验证科学结论。流程保留
needs_review、误差失败、背景校准不足与数据缺失等状态。
:doc:`known_issues` 说明影响使用的已知限制；:doc:`overview` 区分完整流程、公共接口和实验功能。

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started
   user_guide
   pipelines
   api
   development
   release_notes

.. rubric:: 索引

:ref:`genindex` · :ref:`modindex` · :doc:`科学约定 <conventions>`
