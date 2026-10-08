Galactic NH 查询与 XSPEC 单位
=============================

``nhtot`` 查询 Swift UKSSDC 服务，返回 Willingale et al. (2013) 方法的
Galactic 氢柱密度，包括 HI 和估计的分子成分。
返回列密度的单位为 **cm⁻²**；总氢为 :math:`N_{H,tot}=N_{HI}+2N_{H_2}`。
服务依赖网络，应保存查询坐标、日期、原始响应与结果。

.. code-block:: python

   from jinwu.core.utils import nhtot
   result = nhtot(ra=159.386, dec=56.171)  # 十进制度
   print(result['nhi_weighted'])
   print(result['nh2_weighted'])
   print(result['nhtot_weighted'])
   # XSPEC tbabs 参数单位为 10^22 cm^-2
   nh_gal_1e22 = result['nhtot_weighted'] / 1.0e22

也可使用性角坐标字符串：``nhtot('10:37:32.6', '+56:10:15.6')``。
``fit_prepared(galactic_nh_1e22=...)`` 与 ``tbabs.nH`` 接受的是转换后的数值。
不要把 cm⁻² 原始值直接写入 XSPEC，也不要把本例坐标的查询结果用到其他视线。

选择 Galactic NH 估计需核查所用地图、分子气体假设和吸收模型，
并在科学报告中说明来源；不同估计不能只因名称相近而直接替换。

来源：Willingale et al. (2013)，MNRAS 431, 394，
`arXiv:1303.0843 <https://arxiv.org/abs/1303.0843>`_；
`UKSSDC nhtot <https://www.swift.ac.uk/analysis/nhtot/>`_；
`XSPEC tbabs 模型单位 <https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/XSmodelTbabs.html>`_。
API：:func:`jinwu.core.utils.nhtot`。
