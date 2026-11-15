当前版本说明
============

本手册对应当前源码声明的 |release|。安装方式见 :doc:`installation`；
各发行包的实际发布文件以 PyPI 为准。
仓库历史 changelog 的发布日期与修复条目属于原有记录，不代表本次文档任务重新执行了所有验收。

0.3.0 更新
----------

新增独立 :doc:`usage/gecam` 插件，使用可选 GECAMTools 按提取、成品准备、
能谱和光变拟合分阶段执行并恢复。:doc:`usage/source_analysis` 汇总共享分析入口，
显式传入 epoch/UTC，并支持逐谱统计量与联合谱交叉校准。
六个 Python 包与可选 Rust 包统一为 0.3.0，已有内部依赖版本范围同步为
``>=0.3,<0.4``。Swift 保持显式选择，``jinwu[all]`` 不包含 Swift。
源码更新与 PyPI 发布分别进行；使用限制见 :doc:`usage/gecam` 和 :doc:`known_issues`。

0.2.4 更新
----------

本版修复 GBM 源/背景 GTI 覆盖与缓存失效、CSPEC 文件发现和光变兼容性，
源谱及背景分箱保留同时间事件，曝光按全能道事件死时间计算。
WXT 抽谱在保存缓存前完成关键字与响应准备，恢复运行可复用完整谱产物。

GBM 背景残差门禁的统计标定及时间趋势仍待修正；本版保留失败时
``needs_review`` 的状态。背景末箱/BAK 曝光近似、XSELECT 退出异常、
严格文档构建警告以及历史开放项见 :doc:`known_issues`。
本轮软件和产物检查不构成响应、通量、上限或 FAR 的完整科学验收；
T90 与红移模拟仍按原决定暂缓。本次版本更新及源码推送未执行 PyPI 上传。

本版使用入口
------------

* :doc:`overview` 提供完整功能地图；:doc:`pipelines` 按仪器选择流程。
* 有符号净计数时标见 :doc:`usage/duration`；计数实验 SNR 见 :doc:`usage/significance`。
* GBM TTE Haar MVT 的算法来源、状态与 bin-width 诊断见 :doc:`usage/gbm_mvt`。
* 吸收预算和闭合检查见 :doc:`usage/absorption_budget`。
* 当前审查仍未修复的使用问题见 :doc:`known_issues`；升级导入/返回值见 :doc:`compatibility`。

本次文档更新
------------

补充安装、数据/时间约定、WXT/FXT、绘图、模型/背景、模拟、星表、FTOOLS/Rust 与开发说明；
重建导航和规范模块 API，修正光变绘图返回值、NH 单位和 PHA EBOUNDS 示例。
文档版本从源码配置读取，Read the Docs 使用固定工具版本与严格构建。

.. toctree::
   :maxdepth: 1

   changelog
