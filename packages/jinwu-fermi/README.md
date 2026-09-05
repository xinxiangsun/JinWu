# jinwu-fermi

Fermi/GBM instrument support for the
[jinwu](https://pypi.org/project/jinwu/) analysis toolkit: continuous GBM
coverage checks, detector selection, TTE spectral product extraction and
response generation (built on [astro-gdt](https://astro-gdt.readthedocs.io/)).

```bash
pip install jinwu-fermi
pip install "jinwu-fermi[rsp]"   # optional: pure-Python response stack (gbm_drm_gen)
```

Import as `jinwu.fermi`, e.g. `from jinwu.fermi.gbm import GBMObservation`.

## GBM 连续数据单目标管线 (`jinwu.fermi.gbm.pipeline`)

可恢复的单目标 Fermi/GBM 连续数据（continuous/TTE）分析管线，注册为 `"fermi.gbm"`
，构建于 `jinwu.core.pipeline`，阶段图：

```
preflight → coverage → detectors → download → windows
                                              ↙        ↘
                              lightcurve     spectra → response → fit → report
```

- **preflight** — 检查 `gdt-data`、两套响应生成器（官方 `SA_GBM_RSP_Gen.pl` +
  Perl `Astro::FITS::CFITSIO`；纯 Python `gbm_drm_gen`/`responsum`/`gbmgeometry`
  + `BALROG_DB`）以及 PyXspec，写 `preflight.json`；缺工具不失败，由后续阶段门控。
- **coverage** — 本地定位（可选下载）position-history 文件并执行可见性/航天器状态/
  GTI 覆盖检查；`none`/`data_missing` 记为 `needs_review`，绝不当作“无探测”。
- **detectors** — 按几何角度上限选择最优 NaI（默认 ≤60°，最多 3 台）与 BGO
  （≤90°，最多 2 台）探测器。
- **download** — 仅为选中探测器拉取 TTE/CSPEC（HTTP Range 断点续传），下载范围自动
  覆盖最大背景窗；本地优先，`--download` 显式开启网络。
- **windows** — 源段（覆盖段或 `--window` 显式指定）+ 耀发前后紧贴的多项式背景窗
  （guard 10 s，每侧 300–1800 s，按数据覆盖截断）；无窗或不满足两侧包夹即门控。
- **lightcurve** — 各探测器在分析能段的分箱光变（NPZ + PNG），产物失败仅降级为警告。
- **spectra** — 逐探测器从 TTE 提取 OGIP PHA/BAK，AICc 自动选择 0–2 阶多项式背景。
- **response** — 双后端二选一（`--response-backend auto/official/gbm_drm_gen`）：
  官方 `SA_GBM_RSP_Gen.pl`（CSPEC + poshist，RSP2 按实际 GTI 曝光加权）或纯 Python
  `gbm_drm_gen.DRMGenTTE`（段中点单矩阵）。当中点响应在所选能段的折叠率与 GTI
  加权响应相差不超过 1% 时才保留中点近似，否则写出加权 DRM。两分支产物统一做
  1-based CHANNEL 重编号与 `MATRIX` EXTNAME 后处理，并通过 `read_ogip_products`
  一致性校验。
- **fit** — 逐探测器组（nai/bgo）× 光子指数网格（2.0 主值 + 1.5/2.5 敏感性）用
  `jinwu.core.upperlimit.estimate_upper_limit` 做固定谱形幂律剖面。源区计数仍按
  Poisson 处理，BAK 的 `STAT_ERR`（以及显式协方差扩展）作为 Gaussian 背景 nuisance
  进入 profile；不会默认加入未经校准的 5% 系统误差，也不会把同一误差再转换成
  分数系统学。结果同时写出 `observed_upper_bound` 和独立的
  `detection_sensitivity`；没有空白控制样本时后者为 `unavailable`。sqrt(null_statistic)
  仅是固定位置的模型显著性诊断，低于阈值才给出条件上限。
- **report** — `report/gbm_summary.json` + `summary_row.csv`。

科学门槛：没有通过验证的 PHA/BAK/RSP 三元组，任何阶段都不会发布流量或上限；缺失的
门槛记为 `needs_review` 并在报告中写明下一步要求，而不是伪造非探测。

运行（需 `hea` 环境 + HEASoft/PyXspec；官方响应后端先 `source` gbmrsp 环境，或安装
`jinwu-fermi[rsp]` 并设 `BALROG_DB`）：

```bash
python -m jinwu.fermi.gbm.pipeline Mrk421 \
    --ra 166.1138 --dec 38.2088 \
    --start 2024-07-16T12:00:00 --stop 2024-07-16T12:00:30 \
    --root ~/data/gbm-cache --output /tmp/mrk421_gbm --download
# --until windows 只跑到某阶段；--no-resume 忽略缓存；--response-backend
# 覆盖配置后端；退出码 0=完成 2=需人工复核 1=失败
```

Python API：

```python
from jinwu.core.config import GBMContinuous
from jinwu.core.pipeline import pipeline
from jinwu.fermi.gbm import GBMPipelineInput

result = pipeline(
    GBMContinuous(),
    GBMPipelineInput(target_id="Mrk421", root="~/data/gbm-cache",
                     output_root="/tmp/mrk421_gbm",
                     source_name="Mrk421", ra_deg=166.1138, dec_deg=38.2088,
                     start_utc="2024-07-16T12:00:00",
                     stop_utc="2024-07-16T12:00:30",
                     download=True),
).run()
print(result.science_status, result.science_result, result.products["report"])
```

`GBMContinuous` 是 `jinwu.core.config` 中的公开预设，分析参数（探测器角度上限、背景窗、
光子指数网格、能段、显著性阈值等）集中在 `GBMAnalysisConfig`，可通过关键字直接覆盖，
如 `GBMContinuous(lc_bin_s=0.5, response_backend="gbm_drm_gen")`。中断后重跑同一命令即从
工作区 `.pipeline/` manifest 断点恢复。

背景模型在连续留出时间块上用 Poisson deviance 检查；残差均值、趋势以及 68/95% 预测覆盖
均按有限样本二项区间核验。若任一项不合格，报告保留条件 profile 数值但将
`analysis_status` 标为 `needs_review`，不会把它写成已校准的有限上限。`BAK.EXPOSURE` 必须与源 PHA 的实际 GTI 曝光一致（允许有限的
浮点舍入），短窗口只改变 GTI 交叠记录，不按墙钟时长缩放完整源谱。

GBM/GECAM 的统计语义在核心层统一为 `poisson_gaussian_profile`：可以传入带单位的
背景协方差并保留固定 Γ=2 的条件结果。当前仓库没有完整 GECAM 产品提取器，因此 GECAM
配置只提供公共统计契约；真实 GECAM 结果在完成产品和背景校准前不得标记为已验证。
