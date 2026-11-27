Einstein Probe：WXT 与 FXT
==========================

选择处理入口
------------

WXT normal-pointing 提供从官方事件、响应和曝光产品到报告的可恢复流程。
FXT 提供目录扫描和 L2/L3 后随分析流程，使用已配对的谱和响应完成逐观测分析。
WXT slew 的运动曝光与响应需要专门验证，不能直接套用 pointing 假设。

WXT 输入与最小运行
------------------

先安装 ``jinwu-ep`` 并激活可运行 XSELECT / PyXspec 的环境。
把下面的路径、观测号和坐标替换为同一次真实观测的信息：

.. code-block:: python

   from jinwu.core.config import instrument
   from jinwu.ep.wxt import WXTPointingInput, WXTPointingPipeline

   inp = WXTPointingInput(
       target_id='my_transient', root='/data/wxt_observation',
       output_root='/analysis/my_transient_wxt',
       ra_deg=159.386, dec_deg=56.171,
       obsid='your_obsid', source_id='s1', auto_approve_regions=False,
   )
   job = WXTPointingPipeline(inp, config=instrument('WXT'))
   result = job.run(until='exposure_arm_qc', resume=False)
   print(result.status, result.workspace)

``root`` 应包含可匹配的 cleaned event、RMF、ARF 与曝光图；
具体布局由 ``discover_wxt_files`` / scanner 判断。多 detector 或多个候选源时显式选择。
可提供 ``source_region``、``background_region``、``trigger_time_utc`` 和 ``redshift``。
示例坐标是格式演示，不能与任意文件组合进行科学分析。

区域与曝光检查
--------------

默认在 ``exposure_arm_qc`` 保留人工检查点。检查源/背景位置、ARM、视野边缘、
曝光覆盖率、背景缩放 alpha 及其来源。确认实际产物后再运行：

.. code-block:: python

   job.approve_regions(note='source, background and exposure reviewed')
   result = job.run(resume=True)
   print(result.summary_text())
   result.display()

``auto_approve_regions=True`` 会跳过人工区域批准，使用者仍承担区域和曝光校验。
不应通过自动批准隐藏缺失曝光、错误源区或异常背景。

阶段与产物
------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 阶段组
     - 主要内容
   * - discover / galactic_absorption / pipeline_spectrum
     - 匹配文件、Galactic NH 来源、官方能谱基线
   * - regions / provisional_events / exposure_arm_qc / final_events
     - 区域、临时事件、曝光与 ARM 诊断、最终源/背景事件
   * - duration / lightcurves
     - 时标、净光变、背景缩放、时间参考
   * - t100_spectra / t90_spectra / ogip_finalize
     - 抽谱阶段补齐背景缩放、响应和背景引用；ogip_finalize 只读校验计数与 OGIP 完整性
   * - bayesian_block_spectra / fit
     - 合并后的分段谱、候选模型、误差与选择理由
   * - fluxcurve / report
     - 通量曲线、图、JSON、摘要与产物索引

在 result 的 ``workspace`` 下检查每个 stage manifest 与诊断产品。
fit 参数 ``error_status``、背景/曝光来源、flux 来源与未完成阶段都应保留。
结果文件存在不能替代响应、时间和区域验收。

``t100_spectra`` / ``t90_spectra`` 完成时，PHA 的 ``BACKSCAL``、``BACKFILE``、
``RESPFILE``、``ANCRFILE`` 已设置，响应已复制到同目录，再记录缓存哈希。
后续校验不会回写这些文件；未改动的产物可直接恢复运行。修改 PHA 或响应内容
仍会使对应缓存失效。升级前由旧实现生成的缓存会因代码变化重建一次。

WXT 报告选项
------------

``config.reporting.enabled=False`` 跳过报告生成和摘要打印，分析结果仍然可用；
返回的 ``result.report`` 与 ``result.summary`` 为 ``None``。已有报告文件保留，
但不会作为本次结果返回。仅关闭控制台打印时，使用 ``print_summary=False``。

启用报告时，目前仅支持 ``language="zh"``。其他语言为未来实现；
例如 ``language="en"`` 会在运行入口抛出 ``NotImplementedError``，
不会继续生成与配置不符的中文报告。关闭报告时不检查语言。

FXT 扫描与能谱
--------------

.. code-block:: python

   from jinwu.core.instruments import scan
   from jinwu.core.spectrum_prep import prepare_spectra
   catalog = scan('/data/fxt_observation', instrument='FXT')
   prepared_catalog = prepare_spectra(catalog, outdir='analysis/fxt/prepared')

检查每个 module 的源/背景 PHA、RMF 和 ARF 是否匹配，再按 :doc:`spectral` 拟合。
FXT 时间仍使用 EP 格式，通道能量优先从 RMF 读取。
``FXT().energy_range_keV`` 是 0.3–10 keV 仪器名义能段。``FXT().spectrum.fit_energy_range_keV``
默认采用较保守的 0.5–10 keV。使用已确认适用的低能标定时，可通过谱配置明确启用 0.3–10 keV。

.. code-block:: python

   from dataclasses import replace
   from jinwu.core.config import FXT
   from jinwu.ep.fxt import FXTInput, FXTPipeline

   config = FXT()
   config.spectrum = replace(config.spectrum, fit_energy_range_keV=(0.3, 10.0))
   inputs = FXTInput(target_id="source", root="/data/fxt_observation",
                     ra_deg=158.08, dec_deg=65.50)
   pipeline = FXTPipeline(inputs, config=config)

若传入 ``fit_energy_range``，该次运行使用显式参数。若谱配置的拟合能段为 ``None``，
使用仪器名义能段。有效能段传给谱准备、计数、XSPEC 和通量上限。
通量的积分能段由 ``config.flux_curve.energy_range_keV`` 或 ``flux_band_keV`` 指定。

API：:mod:`jinwu.ep.fxt.pipeline`、:mod:`jinwu.core.instruments`。

Docker 中运行官方 Level-1 处理
------------------------------

``DockerRunner`` 通过本机 Docker CLI 执行参数列表，不依赖 Docker Python SDK。
``WXTDockerRunner`` / ``FXTDockerRunner`` 只负责官方软件初始化、挂载与命令执行，
不替代上面的科学分析流程，也不选择 GTI、观测模式、源区或筛选表达式。
要求 Docker 可用、镜像已载入、CALDB 完整，且工作目录已被 Docker Desktop 共享。

.. code-block:: python

   from pathlib import Path
   from shlex import join
   from jinwu.ep import FXTDockerRunner, WXTDockerRunner

   # workspace 取输入目录与默认输出目录的共同母目录。
   root = Path('/data/ep_work').resolve()
   fxt = FXTDockerRunner(
       image='fxtsoft:v2.0', caldb='/data/FXT_CALDB', workspace=root,
       context='desktop-linux',  # Linux Engine 等环境可省略，使用 Docker 当前 context
   )
   command = fxt.chain_command(
       root / '06800001128',  # 默认输出 root / '06800001128_fxtchain'
       exitstage=2, module='a', expr='DEFAULT',
       srcffregion=Path('regions/source.reg'),
       bkgffregion=Path('regions/background.reg'),
   )
   print(join(fxt.build_command(command)))  # 仅预览，不启动容器
   result = fxt.run(command, log_path=root / 'fxtchain.log')
   # 等价快捷入口；params 中透传官方任务参数
   # result = fxt.run_chain(indir, params={'exitstage': 2}, log_path=log)

   wxt = WXTDockerRunner(
       image='wxtdas:R1', caldb='/data/WXT_CALDB', workspace=root,
       context='desktop-linux',
   )
   # steminputs 与观测号匹配；其他参数依照相应镜像中的官方任务帮助设置。
   result = wxt.run_pipeline(
       root / '01709321360',  # 默认输出 root / '01709321360_wxtpipeline'
       params={'steminputs': 'ep01709321360', 'stemcmosid': '3', 'clobber': True},
       log_path=root / 'wxtpipeline.log',
   )

``outdir`` 可省略，默认在输入目录同一母目录下创建 ``<input.name>_fxtchain`` 或
``<input.name>_wxtpipeline``。例如 ``fxt.run_chain(root / '06800001128')`` 的输出为
``root / '06800001128_fxtchain'``。需要其他位置时显式传入第二个参数或
``outdir=root / 'custom_output'``。输入路径先解析相对路径与符号链接，再计算默认输出。

输入目录必须存在；输入输出必须位于 workspace 内且不能互相包含。
若输入就是 workspace 本身，默认兄弟目录将位于挂载范围外，此时应使用包含二者的
更大 workspace，才能同时挂载输入与独立输出目录。``workspace`` 整体以读写方式挂载到 ``/work``，CALDB 只读挂载到
``/caldb``。请将 CALDB 放在 workspace 之外，避免其文件还可通过可写的 ``/work`` 访问。建议在独立工作目录中使用原始数据的副本。输入输出路径自动映射；
其他任务参数中，``Path`` 对象映射为容器路径，相对 ``Path`` 以 workspace 为基准，
字符串保持原值。因此 ``'CALDB'``、筛选表达式和 ``'NONE'`` 等哨兵可直接传字符串，
源区、背景区及其他宿主机文件应传 ``Path``。FXT 的 ``srcffregion``、
``srcpwregion``、``srctmregion`` 等仍为不同的官方参数，不自动混用。
参数列表安全传入 ``exec "$@"``；表达式不需要额外 shell 引号。
官方任务启动前会创建并切换到对应输出目录，使 XSPEC 等任务能找到相对响应文件，
并让不同输出目录的 XSELECT 固定临时文件彼此隔离；字符串中的相对文件路径
以输出目录为基准。一般的 ``run([其他命令, ...])`` 或不含 ``outdir`` 的帮助/版本命令
仍在容器私有 ``/tmp`` 执行。宿主机文件路径应使用 ``Path``，其相对路径仍以
workspace 为基准。并行任务应选择不同输出目录。

WXT 入口显式默认 ``clobber=False``，覆盖官方默认值。
实际任务需要覆盖时在 ``params`` 或 ``pipeline_command`` 中显式设置 ``clobber=True``。WXTDAS 某些观测的多模式处理中会重复生成共享中间文件，
即使输出目录最初为空也可能需要此选项；应先选择独立输出目录。
FXT 的 ``fxtchain`` 不支持 ``clobber`` 参数，默认拒绝非空输出目录；
确需复用时在 ``chain_command`` 或 ``run_chain`` 中传 ``allow_existing=True``，
此时覆盖行为由官方任务决定。程序退出零仍需检查任务日志和必要产品，
部分官方任务的内部错误不一定传播为非零退出码。

环境变量与镜像归档
~~~~~~~~~~~~~~~~~~

``from_env(workspace, ...)`` 在调用时读取当前进程中的 ``WXTCALDB`` / ``WXTDASIMG``
或 ``FXTCALDB`` / ``FXTDASIMG``；可通过 ``environ={...}`` 提供显式映射，
``image=`` / ``caldb=`` 优先于环境变量。模块导入不读取这些变量，也不会执行
``~/.bashrc``。请在启动 Python 前导出变量，或直接传参。

``DASIMG`` 若为镜像归档路径，先显式载入，再选择其返回的镜像引用：

.. code-block:: python

   import os
   from jinwu.core import DockerRunner

   images = DockerRunner.load_image(os.environ['WXTDASIMG'], context='desktop-linux')
   print(images)  # 归档可以包含多个镜像；明确选择所需的 tag 或 ID
   wxt = WXTDockerRunner.from_env(
       root, image='wxtdas:R1', context='desktop-linux',
   )

载入镜像会改变所选 Docker daemon 的镜像存储；构造 runner、``from_env`` 和命令
预览均不会隐式载入、下载或启动镜像。已载入的镜像直接传 tag/ID 即可。

初始化会 source 镜像的 ``$HEADAS/headas-init.sh``，然后设置容器内 ``CALDB``、
``CALDBCONFIG``、``CALDBALIAS``，为每次运行新建容器内独立的 ``PFILES`` 用户目录，
并关闭交互提示。WXT 默认 CALDBCONFIG 为 ``/caldb/software/tools/caldb.config``，
FXT 为 ``/caldb/caldb.config``；可通过 ``env={'JINWU_CALDB_CONFIG': ...}`` 修改。
两者 CALDBALIAS 均为 ``/caldb/software/tools/alias_config.fits``。
其他需要的容器变量可通过 ``env=`` 传入；上述初始化变量最终由初始化脚本设置。

日志、错误与通用 Docker 命令
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from jinwu.core import DockerMount, DockerRunner

   docker = DockerRunner(
       'ubuntu:24.04', mounts=[DockerMount(root, '/work')], workdir='/work',
       context='desktop-linux', env={'LANG': 'C.UTF-8'},
   )
   result = docker.run(['printf', '%s\n', 'an argument with spaces'])
   print(result.stdout)
   process = fxt.start(command, log_path=root / 'fxtchain.log')
   returncode = process.wait()  # start 不自动检查退出状态

``run`` 默认捕获 stdout/stderr，并在非零退出时抛 ``subprocess.CalledProcessError``；
``log_path`` 改为将合并输出写入该文件，覆盖旧日志。``capture_output=False`` 将输出
直接显示在终端。``start`` 返回 ``Popen``，日志追加写入，调用者负责等待与检查退出码。
日志目录必须事先存在。``timeout`` 或终止 ``Popen`` 只保证结束 Docker CLI，不能保证
容器停止；需要时通过 Docker 查看并停止仍在运行的容器。

通用 runner 支持 ``container='name'`` 配合 ``existing=True`` 执行 ``docker exec``；
此时使用既有容器的挂载，不会套用 runner 的镜像、挂载或 entrypoint。EP runner
要求每次使用新容器，以保证初始化与参数隔离，不支持 ``existing=True``。

官方参数与任务定义以所用镜像的 ``$HEADAS/help/wxtpipeline.html`` 和
``$HEADAS/help/fxtchain.html`` 为准。
