科学与数据约定
==============

单位、参考系与置信状态
----------------------

公共接口中的物理量优先使用 ``astropy.units.Quantity``。
遗留数值参数的单位以 API 文档为准，例如时间秒、能量 keV、坐标度、
XSPEC 吸收参数以 :math:`10^{22}\,\mathrm{cm}^{-2}` 为单位。

记录 flux、fluence 与 luminosity 时写明单位、能段、observed/rest frame、
吸收状态、模型与不确定度。固定模板上限与后验预测回答不同问题，不能交换解释。
profile 区间未跨越阈值、参数触及边界、后验受先验主导或校准不足时保留这些状态。

时间与背景
------------

EP MET 使用 ``jinwu.core.time.Time(format='ep')``；不要手动拼接 MJDREF。
数组上的相对秒不能在缺乏参考零点时标成 UTC。
源区/背景区应在共同时间网格和可用曝光上比较；计数、计数率与曝光修正值必须区分。
保留有符号净计数，Poisson 计数观测与 Gaussian 背景扣减观测使用相应似然。

可复现产物
------------

为每次运行使用独立输出目录，保留原始数据。记录观测标识、输入哈希、
软件环境、配置、区域与响应、随机种子、诊断图和最终状态。
``completed`` 表示流程阶段完成，科学判断仍需查看产品质量和模型检验。
缓存指纹可能包含源码文件哈希，文档字符串修改也可能使相应阶段重新运行。

方法来源
------------

* Bayesian Blocks：Scargle et al. (2013)，`arXiv:1207.5578 <https://arxiv.org/abs/1207.5578>`_。
* ON/OFF 显著性与不同背景实验：见 :doc:`usage/significance` 的原始文献与实现来源。
* XSPEC 模型与单位：`XSPEC 官方手册 <https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/XspecManual.html>`_。
* 红移转换：见 :doc:`lf_redshift_transform`，其中区分已实现接口与待实现的通用变换。
