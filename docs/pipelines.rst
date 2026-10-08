仪器与可恢复流程
================

选择与输入产品匹配的流程。原始观测与本次产物分目录保存；输出包含阶段 manifest。
网络下载、在线产品请求与区域确认有各自的显式入口。

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - 仪器/任务
     - 入口
     - 结果与边界
   * - EP WXT 指向观测
     - WXTPointingPipeline
     - 区域/曝光、事件与光变、时标、谱和报告
   * - EP FXT
     - scan → prepare_spectra → 拟合
     - 已有产品分析；无与 WXT 等价的原始数据端到端流程
   * - Swift BAT + XRT GRB
     - SwiftGRBPipeline
     - 单 GRB 目录、光变、分段、谱和报告
   * - Swift BAT survey
     - BATSurveyPipeline
     - 编码掩模净率、谱、条件上限；灵敏度另需校准
   * - Fermi GBM 连续数据
     - GBMPipeline
     - 覆盖、探测器、背景、PHA/BAK/RSP、条件上限
   * - GBM 外部触发搜索
     - GBMTargetedSearchPipeline
     - 候选与排序；显著性需匹配的 off-source 校准
   * - GBM Haar MVT
     - run_gbm_mvt / compute_mvt
     - TTE 或均匀计数光变、重采样、分类与诊断
   * - GW 定位与覆盖
     - jinwu-gw plot
     - 天区概率与几何覆盖；不输出已验证的能谱/曝光

.. toctree::
   :maxdepth: 1

   usage/pipeline_framework
   usage/ep_wxt_fxt
   usage/swift_grb
   usage/bat_survey
   usage/fermi_gbm
   usage/gbm_subthreshold
   usage/gbm_mvt
   usage/gw_coverage
