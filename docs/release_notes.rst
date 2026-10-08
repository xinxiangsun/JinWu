0.2.2 工作区版本说明
====================

本手册对应当前源码声明的 0.2.2。它描述工作区的实际能力，并不确认这些变更已发布到 PyPI。
仓库历史 changelog 的发布日期与修复条目属于原有记录，不代表本次文档任务重新执行了所有验收。

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
