# WXT 指向观测值班演示 / Pointing observation demonstration

使用 `hea` 环境打开 [Notebook](pointing_pipeline.ipynb)，依次运行输入检查、区域提案、曝光诊断、人工批准和结果检查。配套 [脚本](pointing_pipeline.py) 默认在区域检查后停止；加 `--approve-regions` 才会显示交互式批准提示。

Open the Notebook in the `hea` environment. Run its input, region, exposure QC, approval and result cells in order. The paired script stops after region QC unless `--approve-regions` is supplied; it still asks for interactive confirmation.

本机默认案例为 `test/EP260809adata/EP260809a/06800001692_32`，源坐标 RA=328.573°、Dec=17.607°。数据不随仓库分发。学生可设置 `JINWU_WXT_OBS_ROOT` 指向自己的官方 WXT L2/L3 单观测目录，并按实际源修改 `target_id`、坐标及 `source_id`。

The local default is EP260809a observation `06800001692_32`. Data are not shipped with the repository. Set `JINWU_WXT_OBS_ROOT` to another official single-observation WXT L2/L3 directory and update the source position and `source_id` for that target.

输出根目录由 `JINWU_EXAMPLE_OUTPUT_ROOT` 控制，默认 `examples/_outputs`；每次运行建独立工作区。Notebook 中须核查源区、背景区、ARM 排除区与曝光诊断后才能执行批准单元。`needs_review` 是预期的暂停状态，不是失败。最终检查 `status`、`alpha`、时标、PHA 背景与响应关联、拟合诊断、光变和报告；模型结果不可仅凭 `completed` 判为科学可信。

`JINWU_EXAMPLE_OUTPUT_ROOT` controls the output base (default `examples/_outputs`); each run gets a distinct workspace. Review the source, background and ARM exclusion regions plus exposure diagnostics before executing the approval cell. `needs_review` is the expected pause. After completion, inspect the status, alpha, duration, PHA background/response links, fit diagnostics, light curves and report. A completed software run alone does not establish a scientific result.
