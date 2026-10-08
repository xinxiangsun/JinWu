宿主星系、星表与聚类
====================

宿主候选查询
------------

``jinwu.core.host.HostGalaxyFinder`` 提供候选星系查询、匹配与展示。
需要 ``jinwu[crossmatch]``：pandas、astroquery、ipyaladin、plotly、regions 与 requests。
网络查询应保存服务、查询范围、时间和原始表；候选空间重合不能独自确定物理宿主关联。
字段、坐标单位与各服务可用性见 API，使用前核对当前星表版本。

聚类
------------

``jinwu.cluster.cluster.ClusterAnalyzer`` 提供聚类与可视化辅助，
需要 ``jinwu[cluster]`` 的 pandas、seaborn 和 scikit-learn。
明确特征、缺失值、缩放、算法参数与随机状态。
聚类分组是探索结果，需独立验证选择效应与物理含义。

这些工具是分析辅助，当前不提供可直接采纳的宿主概率或完整物理分类 pipeline。
API：:mod:`jinwu.core.host`、:mod:`jinwu.cluster.cluster`。
