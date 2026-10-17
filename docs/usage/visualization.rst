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

论文中的最终尺寸
~~~~~~~~~~~~~~~~

``paper_style`` 是临时上下文：按目标宽高（英寸）创建图，轴标题使用
``font_size`` pt，刻度为其 90%，图例为 85%；点、描边与主次刻度同步缩放。
默认无网格，保留原配色；退出时恢复之前的样式。单栏或跨栏宽度应从稿件
取得，不把某个项目的尺寸当作通用标准。

.. code-block:: python

   import matplotlib.pyplot as plt
   from jinwu.core.plotstyle import paper_style, save_figure

   # 宽高仅为示例；替换为稿件实际插入尺寸。
   with paper_style(width=3.5, height=2.8, font_size=10):
       fig, ax = plt.subplots()
       ax.errorbar(time, flux, yerr=flux_error, fmt='o', linestyle='none')
       ax.set(xlabel='Time (s)', ylabel=r'Flux (erg cm$^{-2}$ s$^{-1}$)')
       save_figure(fig, 'lightcurve', formats=('png', 'pdf'), bbox_inches=None)
       plt.close(fig)

默认文字字体为 Times New Roman，缺失时发出警告并给出实际替代字体；
数学字符使用 STIX，PDF 使用 TrueType 嵌入。可通过 ``font_family`` 指定其他字体；
额外的本地字体需先用 Matplotlib ``font_manager.fontManager.addfont`` 注册。
创建和保存均应放在上下文中；它作用于进程全局设置，不适合线程间并发切换。
上下文只设置新建对象的默认值：绘图函数显式传入的 ``figsize``、字号、颜色等
仍优先；已创建的图不会被改造。核心绘图入口在未收到显式 ``figsize``/``ax`` 时
会采用当前论文画布：``plot_spectrum``、``plot_lightcurve``、
``plotpanel.overlay``/``multi_panel``、``LightcurveFitter.plot_fit``、
``plot_net_lightcurve``、``save_flux_curve``、``plot_time_intervals`` 和
``plotfit``。
仪器流水线或其他专用绘图入口若固定画布，需要显式传入 ``figsize``/``ax``，
或先单独适配，不能假定 ``paper_style`` 会覆盖显式尺寸。

``save_figure`` 在 ``paper_style`` 外默认裁边；上下文内默认保留画布物理尺寸，
显式传入 ``bbox_inches='tight'`` 仍可裁边，``bbox_inches=None`` 始终保留尺寸。
保留画布时应预先安排边距，检查文字和误差棒有无裁切。
不要将 ``tight`` 导出的实际宽度等同于 ``figsize``。
若论文再次缩放，使用相同长度单位计算 ``q = 插入宽度 / 导出宽度``：
最终字号、点直径、线宽与箭头大小均乘以 q；提高 DPI 只增加栅格清晰度。
在实际栏宽及邻近正文中检查 PNG 和 PDF，保留原始数值、误差、上限方向及模型参数。

产物辅助
------------

:mod:`jinwu.core.products` 提供净光变/flux curve 的序列化、环境记录、
artifact index、quicklook 图与观察摘要。区分谱拟合 flux、固定 conversion 的 quicklook
与外推值，并保留 ``flux_origin`` 等实际结果元数据。
JSON 保存工具处理非有限值和原子写入；科学无效值仍需要状态解释。
