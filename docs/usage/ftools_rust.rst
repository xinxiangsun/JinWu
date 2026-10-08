FITS / FTOOLS 与 Rust 扩展
==========================

Python 工具
------------

``jinwu.ftools`` 提供 FITS extension 提取、表筛选、PHA 分组/重分箱、RMF 重分箱、
DS9 区域处理、TELDEF 变换、XSELECT mission database 与 PHA/RMF 通道映射。
具体参数与覆盖的 HEASoft 语义见各定义模块 API。

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 模块
     - 职责
   * - fextract / ftselect
     - FITS extension 与行筛选
   * - ftgrouppha / grppha
     - QUALITY/GROUPING；不完整尾组应核查标记
   * - ftrbnpha / ftrbnrmf
     - 通道/响应网格重分箱，核查整除与 REDIST 归一化
   * - rmf_mapping
     - 通道编号与响应能量映射
   * - region / teldef / teldef_helpers
     - 区域、坐标与姿态变换
   * - xselect_mdb
     - mission database 读取与缓存

Python 实现不表示覆盖了原任务全部功能。
迁移科学流程时用相同真实输入比较原 HEASoft 输出，记录版本、参数和容差，
尤其检查矩阵、缩放、分组、边缘通道与元数据。
``jinwu.core.xselect`` 的提取仍依赖真实 XSELECT；其产物需做 OGIP 核查。

Rust 是可选后端
---------------

``jinwurs`` 当前提供重分箱底层加速接口，Python 分发和构建源码位于 ``packages/jinwurs``。
核心适配见 :mod:`jinwu.core.rebin_rs`，缺少扩展时该入口抛出 ImportError，不自动回退；应核对两种实现的语义一致性，
不要把性能优化当成算法校准。

.. code-block:: bash

   python -m pip install jinwurs
   # 从源码构建需按 packages/jinwurs/README.md 准备 Rust / maturin

Rust 的原生函数不是 Sphinx Python autodoc 的完整对象清单；其使用入口、源代码与
当前可用符号以实际安装的扩展为准。

当前原生符号
------------------------------

* ``rebin_counts_core(orig_counts, orig_errors, orig_left, orig_right, orig_width, orig_exposure, new_edges)``：
  按区间交叠分配计数，返回 counts、variance 与 exposure 三个数组。
* ``rebin_finalize(counts, var, exposure, method, empty_nan)``：
  将累计量转换为计数或计数率，返回 value / error 数组。

底层输入要求连续 float64 一维数组与一致长度；可空误差/曝光参数传 None。
优先使用 Python 适配入口的单位和数据校验，不直接向原生 kernel 传入未验证数组。
