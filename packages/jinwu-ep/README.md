# jinwu-ep

Einstein Probe (WXT/FXT) instrument support for the
[jinwu](https://pypi.org/project/jinwu/) analysis toolkit: observation
discovery, WXT pointing-mode data reduction orchestration, event/light-curve
products and spectral extraction helpers.

```bash
pip install jinwu-ep
```

Import the normal-pointing workflow as
`from jinwu.ep.wxt import WXTPointingInput, WXTPointingPipeline`.
Run through `exposure_arm_qc`, inspect the proposed source/background regions
and exposure diagnostics, call `approve_regions()`, then resume. Supply an
independent `output_root` so products do not land beside the input observation.

## TA: WXT + FXT

在 `hea` 环境运行，输入已下载的 cleaned events / L3 产品。WXT 每项明确一个
obsid、CMOS和源；FXT可包含多次后随。输出目录必须独立于输入目录。

```python
from jinwu.ep import TAInput, TAPipeline
from jinwu.ep.wxt import WXTPointingInput
from jinwu.ep.fxt import FXTInput

pipeline = TAPipeline(TAInput(
    target_id="EP_source", ra_deg=123.4, dec_deg=56.7,
    output_root="/path/to/new_analysis",
    trigger_time_utc="2026-01-19T00:00:00", redshift=None,
    wxt=(WXTPointingInput(target_id="EP_source", root="/path/to/wxt",
                         ra_deg=123.4, dec_deg=56.7, source_id="s1"),),
    fxt=FXTInput(target_id="EP_source", root="/path/to/fxt",
                 ra_deg=123.4, dec_deg=56.7),
))
result = pipeline.run()
print(result.status, result.products, result.warnings)
```

WXT首先生成源/背景区域及曝光诊断。检查后调用
`pipeline.wxt_pipelines[0].approve_regions(note="已检查区域与曝光")`，再
`pipeline.run(overwrite=True)`继续。宇宙线只自动筛查并画帧诊断；确认后在相应
`WXTPointingInput.confirmed_cosmic_ray_frames`填写真实帧号，再运行。
单帧现象和APEC偏好均为候选证据，不自动完成物理分类。

- WXT新谱bin严格要求背景缩放后的净计数 >20且Li–Ma SNR >3；合并邻近不达标bin，
  不跨GTI或仪器。各bin共享吸收，Gamma/温度/flux独立；TA也链接多个WXT观测。
- WXT默认PL、BB+PL、Band、bknpower的零/自由额外吸收方案，另加裸APEC和双APEC，
  共10候选。APEC用于近邻恒星，丰度固定为所选XSPEC solar表的1倍，发射及吸收红移均固定0，
  不沿用目标的非热模型红移；吸收APEC是显式选项。
- 默认保留`cflux`拟合通量，单分量`norm=1`固定。BB+PL/双APEC只固定一个分量的
  `norm=1`，其他norm从1起步并拟合相对强度，避免锁死混合比例。不推荐省略`cflux`。
  普通联合拟合将同一分量的norm跨数据组链接到第一组，不把不同分量的norm相互链接；
  WXT时变bin仅共享吸收，形状及分量比例仍逐bin独立。
  通量拟合仅接受`吸收*cflux*(源分量之和)`：所有吸收放左边，完整源谱放右边。
  自定义`bbodyrad`、`gaussian`等带norm的加性模型同样固定一个基准norm。
  吸收位于cflux内部或只对部分源分量使用cflux时明确报错，不误标为总去吸收通量。
- FXT默认每次观测一bin、A/B联合PL，各观测独立。更小的早期bin通过
  `FXTInput.intervals`指定（相对T0秒；无T0时为EP MET秒），需事件、区域和相容响应。
  其他谱型通过`fxt_options={"candidate_keys": (...)}`明确指定。
  A/B时间交集需覆盖各自时段的至少99%；否则本epoch报错，需重抽一致时段或分别分析。
- 默认按AIC比较，保留BIC、统计量、参数数目和有效拟合bins数。带Poisson背景的
  XSPEC `cstat`实际为profile W-stat；这些信息准则分数不是Bayesian evidence。
- 主图统一observer-frame 0.5–4 keV去吸收通量；未知z固定0的额外NH不解释为宿主NH。
  汇总JSON保留完整来源，ECSV/CSV带单位；不重复加入T90/T100聚合点和已覆盖的子bin。
  表中保留参数profile置信水平、误差状态及上限构造；默认Δstat=1对应约68.27%，
  不是90%。固定上限谱形的NH必须与本次输入的红移、丰度和吸收模型一致。
- 当前L2/L3流程将pile-up保留为代办（`pileup_status="deferred"`），不因此阻断或降级运行，
  也不把它记为已核查。非探测观测与其固定谱形必须显式提供：
  `FXTInput.nondetection_obsids`及`fxt_options["upper_limit_shape"]`中的
  `photon_index`、`intrinsic_nh_1e22`，不静默假定谱形或检测阈值。

APEC表缺失时可以在`WXT(fitting=replace(WXT().fitting, apec_root="/path/apec_vX"))`
中显式指定完整AtomDB文件前缀（对应`_line.fits`、`_coco.fits`），传给
`TAPipeline(..., wxt_config=config)`；不改变用户的XSPEC环境默认设置。
只有L3谱时输出其拟合/通量，并明确时标、事件与宇宙线核查缺少输入。
`overwrite=True`只允许重写选定分析目录，原始文件和既有分析结果保留。

当前自动流程针对已有L2/L3定点产品；FXT pile-up处理和slew时变响应保留为代办，
本入口不自动生成校准后的挖孔响应或重建slew响应。
