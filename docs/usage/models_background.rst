源模型、背景与物理工具
======================

源模型接口
------------

能谱优先使用验证过的 XSPEC 模型，入口见 :doc:`spectral`。
光变模型注册表支持 powerlaw、broken/double-broken/smoothly-broken powerlaw、
exponential、gaussian、constant 和 linear，见 :doc:`lightcurve`。

``jinwu.model.modelbase`` 提供 ``ModelBase``、``AdditiveModel``、
``MultiplicativeModel``、``ConvolutionModel`` 抽象接口，用于扩展模型协议；
这些基类不等于完整的独立能谱拟合后端。仪器特有的响应或校准模型应与可复用源模型分开。

背景先验
------------

``jinwu.background.backprior`` 提供 ``BackgroundPrior``、
``BackgroundCountsPosterior``、``BackgroundSpectralPrior``。
使用前核查观测模型、先验参数化、源/背景面积与曝光，不能把背景后验直接当已知常数。
Poisson ON/OFF 和带 Gaussian 不确定度的背景对应不同实验；见 :doc:`significance`。
BAT survey 的净谱协方差与 GBM TTE 多项式背景各有自己的流程诊断。

物理工具与边界
--------------

:mod:`jinwu.physics.absorption` 将 XSPEC absorption 的光学深度按元素等成分分解，
强调单位、吸收模型数据与 closure 检查，见 :doc:`absorption_budget`。
``jinwu.physics.radiation`` 导入 naima 模型，需另装 naima，未定义 JinWu 自有模型。
``GeneralRelativity`` 的当前能力主要是特殊相对论运动学辅助，
不能据类名推断已经实现广义相对论时空度规或完整宇宙学求解器。

API：:mod:`jinwu.model.modelbase`、:mod:`jinwu.background.backprior`、
:mod:`jinwu.physics.gr`、:mod:`jinwu.physics.absorption`。
