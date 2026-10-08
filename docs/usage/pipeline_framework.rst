可恢复 pipeline 与配置
======================

共同机制
------------

``PipelineInput`` 标识目标、输入与输出目录；``InstrumentPipeline``
按 ``PipelineStage`` 的依赖运行，写入阶段 manifest，记录状态与产物。
``run(until=..., resume=...)`` 控制截止阶段和缓存复用；
``run_stage`` 运行指定阶段及其依赖。

.. code-block:: python

   from jinwu.core.config import instrument
   from jinwu.core.pipeline import pipeline
   # inp 必须是与配置匹配的仪器 Input 对象
   job = pipeline(instrument('WXT'), inp)
   result = job.run(until='exposure_arm_qc', resume=True)

检查实际结果与阶段状态，包括 ``needs_review``、失败和完成。
改变输入文件、配置、代码依赖或已缓存输出会影响指纹。
删除 stage JSON 来强制复用无效产物会破坏可追溯性，应使用 ``resume=False``
或独立输出目录，并保留旧运行证据。

配置选择
------------

``jinwu.core.config`` 中 ``instrument`` 返回预设；WXT、FXT、Swift 与 GBM
按不同仪器能段、响应和执行要求配置。预设不是所有观测的校准保证。
``FitConfig``、``BXAConfig``、``ExecutionConfig`` 等 dataclass 的具体字段见 API。
显式记录采用的模型、NH、红移、统计量、profile 阈值和执行开关。

扩展接口
------------

自定义 pipeline 使用 ``register_pipeline`` 注册；实现输入校验、stage 执行和结果构建。
可覆盖配置/输入/代码依赖指纹接口缩小重跑范围。
保留临时 PFILES 的隔离与环境清理，不能污染并行任务。
API：:mod:`jinwu.core.pipeline`、:mod:`jinwu.core.config`。
