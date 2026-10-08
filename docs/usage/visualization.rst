绘图、panel 与分析产物
======================

单对象与多对象
--------------

``LightcurveData`` / dataset 的绘图接口、:mod:`jinwu.core.plot`
和 :mod:`jinwu.core.plotpanel` 用于光变、谱图、残差、多仪器叠加。

.. code-block:: python

   from jinwu.core import overlay, multi_panel
   # lc_a / lc_b 为已经正确标定的光变对象
   fig, ax = overlay([lc_a, lc_b], labels=['WXT', 'XRT'], xmode='relative')
   fig, axes = multi_panel([lc_a, lc_b], labels=['WXT', 'XRT'])
   fig.savefig('comparison.png', dpi=150, bbox_inches='tight')

``PanelSpec`` 配置单 panel 的比例、坐标轴和绘图参数。
叠加前核查时间参考、能段、物理单位与背景处理，不能因曲线画在同一张图就直接比较。
``LightcurveFitter.plot_fit`` 返回主图/残差图两个 Axes，见 :doc:`lightcurve`。

样式与保存
------------

:mod:`jinwu.core.plotstyle` 提供 ``apply_style``、``save_figure`` 与对数轴格式化。
论文图需在实际栏宽核查字号、图例、误差棒和裁切；透明度或对数坐标不应隐藏负净计数的处理。
使用中文时环境需要可用的 CJK 字体，工具会按实际可用字体选择。

产物辅助
------------

:mod:`jinwu.core.products` 提供净光变/flux curve 的序列化、环境记录、
artifact index、quicklook 图与观察摘要。区分谱拟合 flux、固定 conversion 的 quicklook
与外推值，并保留 ``flux_origin`` 等实际结果元数据。
JSON 保存工具处理非有限值和原子写入；科学无效值仍需要状态解释。
