安装与环境
============

先选择要运行的分析
------------------

核心包负责 FITS、时间、统计与绘图；仪器包共用 ``jinwu`` namespace。
各包当前源码版本为 **0.2.2**，声明 Python ≥ 3.11，本手册的构建环境为 Python 3.12。
HEASoft、PyXspec、任务校准库与外部任务程序按流程配置，不由普通 pip 安装代替。

.. list-table:: 依赖选择
   :header-rows: 1
   :widths: 25 35 40

   * - 用途
     - 包 / extra
     - 额外条件
   * - 核心读写、统计、绘图
     - ``jinwu``
     - NumPy、SciPy、Astropy、Matplotlib
   * - WXT pointing
     - ``jinwu-ep``
     - HEASoft / XSELECT、有效事件和响应、曝光图
   * - Swift GRB
     - ``jinwu-swift[ukssdc]``
     - UKSSDC 网络或本地缓存；能谱阶段用 PyXspec
   * - BAT survey
     - ``jinwu-swift[survey]``
     - BatAnalysis、HEASoft、CALDB 与 survey 产品
   * - GBM continuous / TTE
     - ``jinwu-fermi``
     - Astro-GDT；生成响应另选官方程序或 ``[rsp]``
   * - GBM targeted search
     - ``jinwu-fermi[search]``
     - 搜索输入、响应数据库、背景和 FAR 校准样本
   * - GW coverage
     - ``jinwu-gw``
     - HEALPix / MOC；GBM 几何另需 POSHIST
   * - BXA / UltraNest
     - ``jinwu[bxa]``
     - 可导入的 PyXspec；源码要求 BXA ≥ 5
   * - 宿主星表 / 聚类
     - ``jinwu[crossmatch]`` / ``jinwu[cluster]``
     - 星表服务可能需要网络
   * - Rust 加速
     - ``jinwurs`` 或 ``jinwu[rust]``
     - 可选；源码构建需 Rust / maturin

安装当前工作区
--------------

仓库根目录是多包管理入口，不能直接 ``pip install -e .``。
下面从仓库根目录安装核心与实际需要的仪器包：

.. code-block:: bash

   conda activate hea
   python -m pip install -e ./packages/jinwu
   python -m pip install -e ./packages/jinwu-ep
   # 按需选择，extras 必须加引号
   python -m pip install -e './packages/jinwu-swift[ukssdc,survey]'
   python -m pip install -e './packages/jinwu-fermi[search,rsp]'
   python -m pip install -e ./packages/jinwu-gw
   python -m pip install -e './packages/jinwu[bxa]'

普通发布包可以通过 ``python -m pip install jinwu`` 安装，再加所需插件。
本手册包含尚未证明已发布的工作区功能；请用源码安装复现本版本，不能用 PyPI 的包名推断功能相同。

核对运行环境
------------

.. code-block:: python

   from importlib.metadata import version
   import inspect
   from jinwu.core import read_pha

   print(version('jinwu'))
   print(inspect.getfile(read_pha))

editable 源码修改立即生效，但分发元数据可能仍保留旧版本号，需重新安装更新。
核查模型后端应在真正用于分析的环境中执行：

.. code-block:: bash

   conda activate hea
   python -c 'import xspec; print(xspec.Xset.version)'
   command -v xselect

本机已有 ``hea`` 时先激活环境。其他机器请按
`HEASoft 官方安装说明 <https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/install.html>`_
准备 HEASoft、PyXspec 和任务校准文件。
只有现有安装无法正常初始化时，才检查 :mod:`jinwu.core.heasoft` 的环境管理工具。

.. note::

   文档构建会模拟少数可选模块的导入，以展示接口；这不会提供 XSPEC、CALDB 或响应生成能力。
   能运行文档站点与能运行天文分析是两个不同的环境验收。
