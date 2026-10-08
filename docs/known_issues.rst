已知限制与使用检查
==================

本页记录当前工作区审查中仍会影响使用的问题；文档修正不会自动修复底层算法。
完整的本地审查记录保存在仓库 ``reviews/``，该目录按工作区规则被 gitignore。

光变拟合状态
------------

0.2.2 已修正 ``LightcurveFitter.success``：后端未收敛或模型/统计量非有限时返回 False，
且不提供有效参数误差。零值、负值或非有限 measurement error 会抛出 ValueError；
调用方须提供有统计依据的误差或显式筛选数据。收敛仍不能替代残差、边界与模型检验。

PHA 通道能量
------------

PHA 不一定包含 EBOUNDS。请从匹配的 RMF/RSP 获取通道定义，
核查通道编号一致；见 :doc:`quickstart`。ARF 的能量 bin 不能代替 EBOUNDS。

Swift Burst Analyser ECF 表
---------------------------

0.2.2 已修正 ``read_burst_analyser_dat`` 的 14 列 ``XRTBand`` 表：
ECF 读取第 12 列，计数率由 flux/ECF 得到；计数率统计误差使用未包含 ECF
不确定度的 flux 误差列。遗留独立脚本的列定义仍须对照原始表头核验。
原始格式见 `UKSSDC GRB 数据说明 <https://www.swift.ac.uk/API/ukssdc/data/GRB.md>`_。

红移与模型语义
--------------

遗留 ``RedshiftExtrapolator`` 示例不适合作为通用红移算法依据。
``powerlaw`` 和 ``zpowerlw`` 的归一化变换不同；吸收红移、cosmology 和能段
必须随模型结构处理。通用变换的设计说明见 :doc:`lf_redshift_transform`。
当前模拟接口的计算结果仍需逐模型验证，尤其不能把未验证的高红移触发边界当测量。

计数统计与未探测
----------------

SNR 的背景已知分支保留移植实现的严格尾概率语义；小计数时与其他尾概率约定可能不同，
见 :doc:`usage/significance`。Gaussian 净计数率上限需要相应误差/协方差，
GBM/BAT 固定模板 limit 需要响应、背景与校准质量。
搜索 ``loglr``、程序返回成功或文件存在均不足以构成显著检测。

数据对齐与环境
--------------

0.2.2 的 ``netdata`` 要求相同 TIMEZERO 和严格相合的 bin；细背景 bin 可在完整覆盖
且不跨源 bin 边界时聚合。错位或不完整覆盖会抛出 ValueError。调用前仍须核对时间系统与 GTI。
网络星表、UKSSDC 与 GraceDB 服务可能不可用；配置本地缓存并保留查询来源。
部分 API 页仅展示可选后端的签名，运行前仍需安装真实依赖。
