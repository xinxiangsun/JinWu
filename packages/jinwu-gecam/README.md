# jinwu-gecam

`jinwu.gecam` 是官方 GECAMTools 的接口包，沿用 JinWu 的
`PipelineInput → InstrumentPipeline → StageResult` 与阶段指纹/恢复机制。
不复制上游分析算法，也不把多探测器或高低增益直接叠加。

## 安装

当前为源码新增包，尚未发布：

```bash
python -m pip install -e packages/jinwu-gecam
# 从官方发布页下载并解压源码包，然后安装其中的 zip：
python -m pip install /path/to/gecamTools-v20240514.zip
```

GECAMTools 的 Python 导入名是 `gecam`，与 `jinwu.gecam` 分开。
官方后端不随本包自动安装；仅导入/发现插件不需要后端。
后续发行时核心 extra 为 `jinwu[gecam]`，并纳入 `jinwu[all]`。

官方来源：[任务网站](https://gecam.ihep.ac.cn/grbDataAnalysisSoftware.jhtml)、
[源码与版本说明](https://github.com/zhangpeng-sci/GECAMTools-Public)。
20240514 的发布记录取消了旧版依赖上限；本接口要求该版本或更新版本，
已验证范围见 [VALIDATION.md](VALIDATION.md)。

## 使用

```python
import astropy.units as u
from jinwu.gecam import GECAMInput, GECAMConfig, GECAMPipeline

# 官方公开样例 tn210511_112749_fb；MET 是绝对秒，不是相对触发的秒数。
config = GECAMConfig(
    detectors=(17, 18), gain="high",
    time_range=[74431550.6, 74431660.6] * u.s,
    time_bin=0.5 * u.s, energy_edges=[40, 100, 300] * u.keV,
    # 不需要谱时省略下列三项：
    source_intervals=[[74431599.6, 74431609.6]] * u.s,
    background_intervals=[[74431560.6, 74431580.6],
                          [74431625.6, 74431660.6]] * u.s,
    channel_bin=4,
    # responses={17: "/data/grd17_high.rsp", 18: "/data/grd18_high.rsp"},
)
job = GECAMPipeline(GECAMInput(
    target_id="tn210511_112749_fb", root="/data/gecam",
    event_file="gbg_evt_tn210511_112749_fb_v00.fits",
    output_root="/data/gecam_analysis",
), config=config)
result = job.run(until="lightcurve")
result = job.run(resume=True)
```

也可用 `jinwu.core.pipeline.pipeline(config, input_data)` 自动发现插件。
这里 `GECAMConfig` 配置 GECAMTools 工作流；核心 `instrument("GECAM", ...)`
是已有的仪器/上限配置，不与该工作流配置互换。

- 阶段为 `inspect → lightcurve → spectra → analysis → report`。仅提取时可停在 `spectra`。
  不提供源/本底区间时，谱阶段记录 `enabled=False`。
- `open_event(path)` 返回原生 `Evt`，可继续使用官方低层接口。
- `extract_lightcurve(...)` 返回原生 `LightCurve`；`extract_spectra(...)` 返回官方结果字典。
- NPZ 保存 MET 分箱边界、原始/修正计数及其误差、死时间和**实际**能道/能量边界。
  计数不是计数率；数组排列为 `(energy_bin, time_bin)`。能量边界由上游映射至通道，
  时间边界沿用上游对齐策略，可能超出请求窗口；需检查实际边界。
- 所有时间参数要求带单位。EVT 按 FITS `MJDREF` 或 `MJDREFI/MJDREFF`、`TIMESYS=TT`
  及 `TIMEZERO` 换算 UTC。输出 NPZ 同时保留换算后的 MET 零点和来源。
- 谱阶段直接调用官方多探测器谱接口，保留其分箱、死时间/背景处理、误差和
  full/background/net **type-II** 格式。光变能段与 `correct_dead_time` 不控制谱提取。
  谱默认时间箱为 1 s，通道分组和背景多项式阶数可显式设置。
- 输入文件、响应、配置、上游 Python 源码和数值依赖版本纳入缓存指纹。
  强制重跑创建新产物子目录；旧产物保留，当前报告更新。

## 成品能谱与光变

可用 `source_spectra`、`background_spectra` 和 `lightcurve_products` 从成品开始。
这一入口使用 JinWu 和 PyXspec，无需安装 GECAMTools。每个探测器使用对应增益的响应，
谱文件与光变文件路径可相对于输入 `root`。以下复用上例的配置和观测标识。

```python
from dataclasses import replace

finished_config = replace(
    config, source_intervals=None, background_intervals=None,
    responses={17: "/data/bg17H.rsp", 18: "/data/bg18H.rsp"},
)
finished = GECAMPipeline(GECAMInput(
    target_id="tn210511_112749_fb", root="/data/gecam_products",
    output_root="/data/gecam_finished_fit",
    source_spectra={17: "grd17_full.fits", 18: "grd18_full.fits"},
    background_spectra={17: "grd17_background.fits", 18: "grd18_background.fits"},
), config=finished_config)
result = finished.run(resume=True)
```

- full type-II 谱按显式的 1-based 行号配对。`TIME` 是区间左边界，`TIMEDEL` 为持续时间，
  `EXPOSURE` 为有效曝光。无时钟的 GECAM-A/B 成品会记录采用 `2019-01-01T00:00:00 UTC`
  的假设。GECAM-C/HEBS 成品若缺时钟，须在 `GECAMInput.met_reference_utc` 指定 MET 零点。
  full 与 background 均含 FITS 时钟时，必须给出相同的 MET 零点；本底无时钟时，诊断信息
  明确记录假设它与 full 共用零点。相同的 `TIME` 数组不能替代时钟核对。
- 原始计数与高斯模型本底在隔离目录写为明确的 Type-I PHA 及响应副本，保留 `GROUPING`、
  `QUALITY` 和曝光。拟合用 `pgstat`，默认 `cflux*powerlaw`，取 `energy_edges` 首末值作为
  观测者系能段，并链接同一时段的谱参数。通过 `GECAMConfig.analysis` 指定 `SourceAnalysisConfig`
  可更换模型、共同通量能段或参数链接方式。启用拟合但缺少响应时，分析返回 `needs_review`，
  可用的光变仍会写出。补齐响应后可恢复运行。
- `gecam_lightcurve_products(...)` 把 NPZ 转为按实际箱宽计算的总计数率 ECSV，
  保留能段、死时间校正选择和相对 UTC 参考的实际箱边界。可把其返回值传给
  `lightcurve_products`。可用 `prepare_gecam_spectra(...)` 独立准备谱，并与其他仪器的
  同一源产品一起交给 `jinwu.core.source_analysis.SourceAnalysisPipeline`。

## 边界

首版针对 A/B/C GRD EVT；真实数据验收限 GECAM-B 官方样例。
下载、姿态选探头、CALDB 响应生成、T90、CPD 及并道数据由独立的原生工具处理。
`science_status="requires_review"` 与 `fit_performed` 分别记录科学状态和拟合执行状态。
使用响应前需检查校准版本、生成警告及响应关联。[样例响应编码问题、显式副本和验证记录](VALIDATION.md)。
