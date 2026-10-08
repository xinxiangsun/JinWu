光变拟合
============

选择模型与后端
--------------

``LightcurveFitter`` 接受 ``LightcurveData``、``LightcurveDataset`` 或
``(time, value, error)`` 数组，支持 Astropy ``Fittable1DModel`` 子类。
注册表提供 powerlaw、broken/double-broken/smoothly-broken powerlaw、
exponential、gaussian、constant 和 linear。
默认 ``lm`` 与 ``trf`` 适用性不同；有参数边界时显式选择合适后端并核查诊断。

可运行示例
------------

这里的下降幂律为合成输入，内置模型使用 ``norm * (t/t0)**index``，
所以下降对应负 index。

.. code-block:: python

   import numpy as np
   from jinwu.core.fit import LightcurveFitter

   t = np.logspace(-1, 1, 30)
   y = 100.0 * t**(-1.5)
   error = np.full(t.size, 0.5)
   fitter = LightcurveFitter((t, y, error))
   result = fitter.fit(
       'powerlaw', p0=[90.0, -1.4, 1.0],
       bounds=([0.0, -5.0, 0.1], [np.inf, 0.0, 10.0]),
       fitter_method='trf',
   )
   assert np.all(np.isfinite(result.params))
   assert np.isfinite(result.chisq)
   print(result.summary())
   ax, ax_res = fitter.plot_fit(result)
   ax.figure.savefig('fit.png', dpi=150, bbox_inches='tight')

``powerlaw`` 中 norm 与 t0 有尺度退化，上例验证程序调用，不能用于独立约束三者。
观测拟合可使用固定 t0 的自定义 Astropy 模型，或依据物理先验选取可辨识参数化。

.. warning::

   0.2.2 起 ``result.success`` 同时核验后端收敛与有限模型/统计量；非法误差会抛出异常。
   检查警告、有限数值、残差、边界、参数退化和模型域；需要科学区间时不能仅依赖
   局部 covariance。净计数的负值应保留，幂律的有效时间域需单独选择。

结果与绘图
------------

``param_names`` / ``params`` 给出对应参数，``errors`` / ``covariance`` 可能不可用。
``summary()``、``evaluate(t)`` 和 ``to_dict()`` 用于查看、预测和保存。
``plot_fit`` 返回 ``(ax, ax_res)``，figure 为 ``ax.figure``。
多对象叠加与 panel 见 :doc:`visualization`。

平滑双幂律有五个参数 ``norm, index1, index2, t_break, smoothness``，
bounds 和 p0 均须按该顺序给出五项；不能照搬普通 broken powerlaw 的四项边界。
API：:mod:`jinwu.core.fit`。
