当前版本说明
============

本手册对应当前源码声明的 |release|。安装本版的命令见 :doc:`installation`；
各发行包的实际发布文件以 PyPI 为准。
仓库历史 changelog 的发布日期与修复条目属于原有记录，不代表本次文档任务重新执行了所有验收。

0.2.3 更新
----------

本版包含 0.2.2 的工作区变更，并补充 GBM 检测器编号与输入校验、谱文件引用及
ARF/RMF 网格检查、XSPEC 会话状态、BAT 时间单位、区域筛选和背景后验边界修正。
0.2.2 未单独发布到 PyPI；从 0.2.1 升级时也请阅读下方 changelog 中的 0.2.2 条目。
已有 T90 与红移模拟的算法限制仍见 :doc:`usage/duration` 和 :doc:`known_issues`，
本次发布不扩大其科学验收结论。

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
