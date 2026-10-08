快速开始
============

先跑一个无外部数据的例子
------------------------

下面在内存中建立一条合成光变并拟合。时间与测量值是演示数值，
输出不代表观测结果。更多模型与拟合诊断见 :doc:`usage/lightcurve`。

.. code-block:: python

   from pathlib import Path
   import numpy as np
   from jinwu.core.data import LightcurveData
   from jinwu.core.fit import LightcurveFitter

   t = np.linspace(1.0, 10.0, 30)
   lc = LightcurveData(
       time=t, value=2.0 * t + 3.0,
       error=np.full(t.size, 0.2), path=Path('synthetic.lc'),
       header={}, meta=None, headers_dump=None,
   )
   fitter = LightcurveFitter(lc)
   result = fitter.fit('linear', p0=[1.0, 1.0], fitter_method='trf')
   assert np.all(np.isfinite(result.params))
   assert np.isfinite(result.chisq)
   print(result.summary())
   ax, ax_res = fitter.plot_fit(result)
   ax.figure.savefig('synthetic_fit.png', dpi=150)

.. warning::

   当前 ``FitResult.success`` 不能独自判定收敛。还应核查后端警告、有限数值、残差、
   参数边界与模型适用性；见 :doc:`known_issues`。

读取真实产品
------------

把下面的文件名替换成观测产品路径。源、背景、RMF 与 ARF 应来自同一观测和正确提取区域：

.. code-block:: python

   import numpy as np
   import jinwu.core as jw

   pha = jw.read_pha('source.pha')
   rmf = jw.read_rmf('source.rmf')
   arf = jw.read_arf('source.arf')
   src = jw.read_lc('source.lc')
   bkg = jw.read_lc('background.lc')
   print(pha.exposure, len(pha.channels))

PHA 常常没有 EBOUNDS，应从对应 RMF 读取通道能量定义：

.. code-block:: python

   band = jw.EnergyBand(emin=0.3, emin_unit='keV', emax=2.0, emax_unit='keV')
   ebounds = pha.ebounds
   if ebounds is None:
       if any(x is None for x in (rmf.channel, rmf.e_min, rmf.e_max)):
           raise ValueError('需要带 EBOUNDS 的匹配响应')
       ebounds = (rmf.channel, rmf.e_min, rmf.e_max)
   if not np.array_equal(ebounds[0], pha.channels):
       raise ValueError('PHA 与 EBOUNDS 通道顺序不一致，应先按编号映射')
   mask = jw.channel_mask_from_ebounds(ebounds, band)

掩膜按 EBOUNDS 通道顺序返回，应用到 PHA 前核对通道编号与长度。
ARF 的能量网格描述有效面积，不能替代探测器通道定义。

源减背景
------------

.. code-block:: python

   # 先确认 time、bin width、GTI 与计数/计数率约定一致
   net = jw.netdata(src, bkg)
   # 有经验证的源/背景缩放时，可显式给 ratio
   # net = jw.netdata(src, bkg, ratio=0.1)

不要把负净计数裁成零。自动缩放依赖产品元数据；曝光不均匀、GTI 不一致或已扣背景的
数据需要单独检查，具体见 :doc:`usage/ogip` 和 :doc:`usage/time_gti`。

进入仪器流程
------------

WXT 的可恢复流程示例、区域审批、阶段与产物见 :doc:`usage/ep_wxt_fxt`。
Swift、GBM 与 GW 的输入要求见 :doc:`pipelines`。
能谱从 :doc:`usage/spectral` 开始；需要 BXA 时使用 :doc:`usage/bxa_fitting`。
所有接口定义可从 :doc:`api` 查询。
