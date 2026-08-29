# REUSABLE_FUNCTIONS — jinwu 公共 API 索引

> jinwu 是跨项目**可复用库**（`~/research/jinwu/src`，editable 安装，改源码即时生效）。
> 本索引列出高频公共入口，供所有 AI 代理（Codex / Claude / Hermes）在写任何分析
> 代码前快速定位："库里已有 X，别重复造"。
>
> 规则：
> 1. 先查本表 + 模块 docstring + 测试，再写新代码。
> 2. 新增公共函数 → 在文末「登记区」追加；已有函数不改签名时无需条目。
> 3. 提升新函数：遵守 `ARCHITECTURE_ROADMAP.md` 分层（`core` 不导入任务包/HEASoft/XSPEC 等
>   外部运行时；任务相关放 `jinwu.<mission>`；适配器放 `integrations/`）。参考现有
>   `jinwu.core.fit`（XSPEC 调用封装为函数）与 `jinwu.core.heasoft` 的边界方式。

## core 层高频入口

| 模块 | 入口 | 用途 |
|---|---|---|
| `jinwu.core.utils` | `nhtot(ra, dec, equinox=2000, timeout)` | 银河 NH / E(B-V) 查询（EP 的 `resolve_galactic_nh` 包装自它） |
| | `li_ma_snr(n_on, n_off, alpha, signed)` | Li-Ma (1983) 显著性 |
| | `generate_xspec_result(model, spectrum)` | XSPEC 拟合结果结构化 |
| | `get_asym_err` / `flux_err_from_log10` | 不对称误差提取/换算 |
| | `generate_download_url` / `extract_all_gz_recursive` / `gunzip` | 数据获取与解压 |
| `jinwu.core.time` | `TimeEP` / `TimeSwift` / `TimeFermi` / `TimeGECAM` / `TimeHXMT` / `TimeMAXI` / `TimeLIGO` 等 | 任务时间系统（EP 秒、MET、卫星时间转换），禁止手工 MJDREF |
| | `extract_time_interval` / `check_time_overlap` / `get_overlap_duration` / `compare_time_intervals` | 时间区间运算 |
| `jinwu.core.io` | `readfits(path, kind)` / `read_arf/rmf/pha/lc/evt` | OGIP 读取（自动识别类型） |
| | `writefits` / `write_pha/arf/rmf/lc/evt` | OGIP 写入 |
| | `guess_ogip_kind` / `band_from_arf_bins` / `channel_mask_from_ebounds` | 类型判定、能带工具 |
| `jinwu.core.xselect` | `extract_products_with_xselect` / `extract_spectrum_with_xselect` / `XSelectSession` | xselect 产品提取 |
| | `select_events` / `filter_time` / `filter_energy` / `filter_region` / `merge_gti` / `trim_events_to_gti` | 事件筛选与 GTI |
| | `extract_spectrum` / `extract_curve` / `extract_image` / `accumulate_spectrum_from_events` | 从事件提取产品 |
| | `write_curve` / `write_image` / `write_pha` | 产品写入 |
| `jinwu.core.spectrum_prep` | `prepare_spectra` / `PreparedSpectrum` / `PreparedJointSpectrum` / `PreparedCatalog` | 谱准备、链接、联合 |
| `jinwu.core.fit` | `fit_spectrum` / `fit_spectrum_from_files` / `fit_prepared` / `fit_xray_models` / `fit` / `run_xspec_chain` | XSPEC 拟合入口（EP 项目的拟合均可用） |
| | `PowerLawModel` / `BrokenPowerLawModel` / `SmoothlyBrokenPowerLawModel` / `DoubleBrokenPowerLawModel` / `ExponentialModel` / `GaussianModel` / `ConstantModel` | 模型类 |
| | `ModelRegistry` / `LightcurveFitter` / `resolve_xray_model_specs` / `calculate_model_fit_metrics` | 模型注册、拟合器、指标 |
| | `FitResult` / `XspecChainResult` / `XRayModelSpec` | 结果与规格类型 |
| `jinwu.core.plot` | `plot_spectrum` / `plot_lightcurve` / `plot_event_txx` / `plot_ogip` / `plot_xspec_origin` / `plotfit` | X 射线绘图 |
| `jinwu.core.galactic` | `resolve_galactic_absorption` / `GalacticAbsorptionResult` | 银河吸收 |
| `jinwu.core.datasets` | `netdata` / `LightcurveDataset` / `SpectrumDataset` / `JointDataset` | 数据集契约 |
| `jinwu.core.config` | `instrument(name, **kwargs)` / `register_instrument` / `FXT` / `WXT` / `BAT` / `GBM` / `GECAM` / `UVOT` + 各 `Config` | 仪器与配置 |
| `jinwu.core.units` | `magnitude_to_flux` / `Magnitude` / `FilterInfo` / `InstrumentFilterLibrary` | 光学 mag/mJy 转换。⚠️ 2026-08-22 实测有缺陷待重写：默认 Vega（AB 需显式传 system）、强制 filter_info（纯 AB→fν 传 None 报错）、`u.ABmag` 不自动切系统。重写前勿推荐复用，详见 EP `REUSABLE_FUNCTIONS.md` 已知问题节 |
| `jinwu.core.base` | `EnergyBand` / `RegionArea` / `OgipMeta` / `PhaBase` / `RmfBase` / `ArfBase` / `EventDataBase` / `LightcurveDataBase` | 基类与数据原语 |
| `jinwu.core.ogip` | `ValidationReport` / `OgipFitsBase` | OGIP 校验 |
| `jinwu.core.pipeline` | `register_pipeline` / `pipeline` / `InstrumentPipeline` / `PipelineStatus` | 流水线 |
| `jinwu.core.host` | `HostGalaxyFinder` | 宿主星系查找/分类 |
| `jinwu.core.heasoft` | `HeasoftEnvManager` | HEASoft 环境管理（生产标准路径） |
| `jinwu.core.rebin_rs` | `rebin_lightcurve_rs` | 光变重分箱 |
| `jinwu.core.products` | `FitProductSet` / `ExternalRunArtifacts` | 产物集 |

## 使用示例（规范写法）

```python
from jinwu.core.utils import nhtot, li_ma_snr
from jinwu.core.time import TimeEP, extract_time_interval
from jinwu.core.io import readfits
from jinwu.core.xselect import extract_spectrum_with_xselect
from jinwu.core.spectrum_prep import prepare_spectra
from jinwu.core.fit import fit_spectrum_from_files
```

## 登记区（新增/提升公共函数时，在此追加）

格式：`| 函数或类 | 模块 | 一句话用途 | 日期 |`

| 函数或类 | 模块 | 用途 | 日期 |
|---|---|---|---|
| `GBMFlareInterval` / `check_gbm_coverage` / `select_gbm_detectors` | `jinwu.fermi.gbm.pipeline` | 连续 GBM 覆盖、遮挡/SAA/GTI 判定与几何选探测器 | 2026-08-27 |
| `fetch_gbm_continuous_products` / `extract_gbm_spectral_products` | `jinwu.fermi.gbm.pipeline` | 可恢复连续数据下载及 TTE 局部多项式 PHA/BAK 提取 | 2026-08-27 |
| `build_gbm_response_command` / `generate_gbm_response` | `jinwu.response.gbm` | 官方 GBM 响应生成器的无 shell 安全封装 | 2026-08-27 |
| `profile_source_amplitude` | `jinwu.core.upperlimit` | 与 response-aware 上限同契约的非负幅度 profile 显著性 | 2026-08-27 |
