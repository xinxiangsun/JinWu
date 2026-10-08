功能全览与能力边界
==================

这张表是本版本的功能地图。使用指南说明选择方法和准备输入，API 页说明具体签名。
“实验”表示存在接口但不应据此宣称已完成通用科学验证。

.. list-table:: 功能地图
   :header-rows: 1
   :widths: 23 47 30

   * - 功能
     - 当前能力
     - 文档入口
   * - OGIP FITS
     - PHA、ARF、RMF/RSP、光变与事件读写；切片、分箱、响应兼容性检查
     - :doc:`usage/ogip`
   * - 时间与曝光
     - 任务 MET、UTC、区间比较、GTI 合并和逐 bin 曝光
     - :doc:`usage/time_gti`
   * - 光变与时标
     - Astropy 模型拟合、Bayesian Blocks、Txx、有符号累计净计数
     - :doc:`usage/lightcurve`、:doc:`usage/duration`
   * - 计数统计
     - ON/OFF、Poisson 背景、带误差背景的 SNR / Li–Ma
     - :doc:`usage/significance`
   * - 能谱推断
     - OGIP 准备、XSPEC 候选模型、profile error、MCMC、BXA / UltraNest
     - :doc:`usage/spectral`、:doc:`usage/bxa_fitting`
   * - 非探测与上限
     - 计数上限、Gaussian 净计数率 profile；仪器响应折叠与单独灵敏度校准
     - :doc:`usage/upperlimits`
   * - 吸收
     - Galactic NH 查询、tbabs/ztbabs 元素不透明度分解与闭合检查
     - :doc:`usage/nhtot`、:doc:`usage/absorption_budget`
   * - 观测流程
     - WXT pointing、Swift GRB、BAT survey、GBM continuous、targeted search、Haar MVT、GW coverage
     - :doc:`pipelines`
   * - 绘图与产物
     - 多仪器叠加、panel、光变/谱图、统一样式、JSON/NPZ 与报告辅助
     - :doc:`usage/visualization`
   * - 模拟与红移
     - XSPEC fakeit、ON/OFF 模拟、红移触发与 detectability；模型结构和背景限制需要审查
     - :doc:`usage/simulation`
   * - FITS 工具
     - Python FTOOLS 对应接口、区域、TELDEF 与通道映射；可选 Rust 扩展
     - :doc:`usage/ftools_rust`
   * - 星表与聚类
     - 宿主星系查询、交互展示与机器学习聚类，需要额外依赖
     - :doc:`usage/catalogs`
   * - 模型扩展
     - 源模型基类、背景先验与特殊相对论辅助；通用辐射模型来自 naima
     - :doc:`usage/models_background`

FXT 有扫描器、配置与通用能谱处理入口，目前没有独立 FXT 端到端 pipeline。
WXT pointing 的审批和曝光处理不自动适用于 slew 或所有数据布局。
GW 覆盖积分衡量几何/概率覆盖，不直接给出源检测效率。
MVT 的测量、上限和未分辨状态不能混用；targeted search 的原始统计量需要 FAR 校准。

当前模块清单由构建时的源码 AST 生成，见 :doc:`api/modules`；
兼容路径见 :doc:`compatibility`，会影响使用的限制见 :doc:`known_issues`。
