OGIP 数据、读写与变换
=====================

选择对象
------------

``read_pha``、``read_arf``、``read_rmf``、``read_lc``、``read_evt`` 返回对应 dataclass；
``readfits`` 根据 FITS 内容自动判断类型。对象保留文件路径、header 与 OGIP 元数据。
``LightcurveDataset``、``SpectrumDataset`` 与 ``JointDataset`` 用于组合源、背景与分析信息。

.. code-block:: python

   from jinwu.core import readfits, read_pha, read_rmf
   obj = readfits('source.pha')
   pha = read_pha('source.pha')
   rmf = read_rmf('source.rmf')
   print(obj.kind, pha.exposure, pha.channels.shape)

响应与区域缩放
--------------

RMF 的 EBOUNDS 给出探测器通道的能量边界，矩阵行能量网格描述入射光子。
ARF 提供有效面积。RSP/DRM 可包含有效面积；具体能否另乘 ARF 应核对响应语义，避免重复折叠。
工具见 :func:`jinwu.core.ogip.check_response_compatibility` 与
:func:`jinwu.core.io.channel_mask_from_ebounds`。

源/背景关系依赖 EXPOSURE、BACKSCAL、AREASCAL 与区域定义。
``netdata`` 提供源减背景与误差传播，不能代替不同时间网格、GTI 或曝光图的科学核验。
曝光不均匀的 WXT 区域处理见 :doc:`ep_wxt_fxt`。

变换与写出
------------

``jinwu.core.ops`` 提供 lightcurve/PHA/event 切片与重分箱、事件到光变转换。
``jinwu.core.io`` 提供对应 ``write_*`` 和 ``writefits``。
选择重分箱规则时核查 counts/rate、误差、分数曝光、GTI 与 QUALITY/GROUPING 的含义。
不要把净谱的 Gaussian 误差问题转成 Poisson 源计数输入。

建议步骤：读取与核查元数据 → 确定时间/通道选择 → 同步处理源与背景 →
检查响应兼容性 → 写入独立目录 → 重新读回并比较关键数组与缩放。
FTOOLS 对应实现及通道映射见 :doc:`ftools_rust`。

API：:mod:`jinwu.core.base`、:mod:`jinwu.core.data`、:mod:`jinwu.core.datasets`、
:mod:`jinwu.core.io`、:mod:`jinwu.core.ogip`、:mod:`jinwu.core.ops`。
