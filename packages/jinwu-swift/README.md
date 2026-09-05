# jinwu-swift

Swift/BAT instrument support for the
[jinwu](https://pypi.org/project/jinwu/) analysis toolkit: BAT observation
handling, attitude reconstruction and survey/posted data products.

```bash
pip install jinwu-swift
```

Import as `jinwu.swift`, e.g. `from jinwu.swift.bat import BATObservation`.

## Optional extras

```bash
pip install "jinwu-swift[gdt]"     # gdt-swift: SAO/poshist 指向文件读取与天图
pip install "jinwu-swift[ukssdc]"  # swifttools: UKSSDC 目录查询与 Burst Analyser 下载
pip install "jinwu-swift[survey]"   # BatAnalysis + HEASARC query for BAT survey
```

Without the `gdt` extra the core BAT workflows work; SAO/poshist-based
features degrade gracefully (`jinwu.swift.bat.bat_observation.HAS_GDT_SWIFT`
is `False`).

## BAT+XRT GRB pipeline (`jinwu.swift.grb`)

Resumable per-GRB pipeline ported from the `swift_highz_grb` research
workflow, registered as `"swift.grb"` and built on `jinwu.core.pipeline`:

```
catalog → download → {lightcurve, prompt, galactic_absorption, bblocks}
        → spectra → fit → report
```

- **catalog** — resolve the GRB in `catalog/swift_grb_catalog_merged.csv`
  (research merge of SDC/UK_XRT/BAT_GRB) with redshift priority selection and
  the NASA BAT duration windows (`bat_t90_start_s/stop_s`).
- **download** — verify Burst Analyser `parsed/` products; optionally fetch
  via `SwiftBurstAnalyserFetcher` (3-tier strategy, BadBin preservation,
  requires the `ukssdc` extra).  Local XRT product directories are preferred.
  `--request-xrt` is the only switch that can submit an UKSSDC job; its saved
  state is resumed by `--poll-xrt` and an ambiguous submission is never sent a
  second time automatically.
- **lightcurve** — BadBin-filtered BAT + XRT WT/PC flux series → NPZ +
  three-panel figure (flux / N_H / Γ) in the research styling.
- **prompt** — strict prompt-stage judgment (XRT start inside the absolute
  BAT T90 window); missing inputs yield `UNDECIDED`.
- **galactic_absorption** — weighted Galactic N_H via
  `jinwu.core.galactic.resolve_galactic_absorption` (cached per workspace).
- **bblocks** — Bayesian blocks on XRT source/background events (WT per-GTI,
  PC per observation, GTI fallback), Li-Ma significance with SRCAREA/BGAREA
  alpha, and low-SNR merging.  An absent area scale stops that segment rather
  than assuming alpha = 1.
- **spectra** — XSELECT source/background PHA per segment (right edge +ε),
  `grppha_hsp` grouping with interval0 RMF/ARF links.
- **fit** — `jinwu.core.fit.fit_prepared` per segment with the research
  model `tbabs*ztbabs*cflux*powerlaw`, frozen Galactic N_H, catalog redshift
  (requires a HEASoft/PyXspec environment).
- **report** — `grb_summary.json`, `summary.txt`, and
  `spectral_parameters.csv`; the optional overlay is written as a separate
  report NPZ/PNG and never rewrites the lightcurve-stage NPZ.

`SwiftGRB` in `jinwu.core.config` is the public preset.  Its BAT Burst
Analyser product is `SNR5_sinceT0:BATBand` and **BATBand is 15--50 keV**;
that product band is distinct from Swift/BAT's 15--150 keV instrument range.
XRT fitting uses 0.3--10 keV, C-stat (reported as W-stat when a Poisson
background is loaded), group 20, and Δstat=1 profile errors.

Run from the command line against a research-style workspace:

```bash
python -m jinwu.swift.grb \
    --root ~/research/swift_highz_grb/output \
    --grb 050904 --nh 0.09 --output-dir /tmp/grb050904_jinwu
# --until bblocks 只跑到某阶段；--no-resume 忽略缓存；缺失网络产品时该阶段
# 记为 needs_review，补齐数据后重跑即从断点继续

# Only with a registered UKSSDC XRT-product account:
python -m jinwu.swift.grb --root ~/research/swift_highz_grb/output --grb 050904 \
    --request-xrt --xrt-user "$SWIFT_XRT_USER"
```

Python API:

```python
from jinwu.core.config import SwiftGRB
from jinwu.core.pipeline import pipeline
from jinwu.swift.grb import SwiftGRBInput

result = pipeline(
    SwiftGRB(),
    SwiftGRBInput(target_id="050904", root=workspace,
                  output_root="/tmp/grb050904_jinwu",
                  xrt_products_dir=workspace / "xrt_product_requests_maybe/downloads/GRB_050904"),
).run()
```

## BAT survey 单目标管线 (`jinwu.swift.bat.survey`)

该入口处理一个已知位置的 Swift/BAT survey 目标，流程为：

```
预检 → 观测发现 → 可选下载 → survey → 光变 → 可选 mosaic → PHA/响应 → 拟合或上限 → 报告
```

```python
from jinwu.core.config import BATSurvey
from jinwu.swift.bat.survey import BATSurveyInput, BATSurveyPipeline

job = BATSurveyInput(
    target_id="NGC4253",
    root="/data/swift/batdata",
    output_root="/tmp/ngc4253_bat_survey",
    source_name="NGC4253",
    coord=(183.5625, 29.8125),       # ICRS degrees
    obsids=("00098092002",),
    time_windows=(("2024-07-16T12:00:00", "2024-07-16T13:00:00"),),
    download=False,                  # local-first; explicit opt-in
    mosaic=False,
)
result = BATSurveyPipeline(job, config=BATSurvey()).run()
print(result.science_status, result.products["report"])
```

命令行等价用法：

```bash
python -m jinwu.swift.bat.survey NGC4253 \
  --root /data/swift/batdata --output /tmp/ngc4253_bat_survey \
  --source-name NGC4253 --ra 183.5625 --dec 29.8125 \
  --obsid 00098092002 \
  --window 2024-07-16T12:00:00 2024-07-16T13:00:00
```

同样的参数也可直接运行带注释的
[`examples/bat_survey_pipeline.py`](examples/bat_survey_pipeline.py)；它支持
`--until`、`--no-resume`、本地 survey/mosaic 路径以及与命令行一致的查询、下载、
并行和超时选项。

已有结果可用 `--survey-products` 或 `--mosaic-products` 直接接入；原始单观测目录
可用 `--raw-products` 指定。`--until` 和 `--no-resume` 分别控制阶段停止和缓存恢复。

`--query` 和 `--download` 分别显式启用在线发现和下载，`--mosaic` 启用时间窗
mosaic，`--profile lmjagn` 使用参考项目的 8000/0.01 筛选配置，默认配置仍为
10000/0.05。survey 率保留 `CENT_RATE` 的正负号，优先使用 `RATE_ERR`，并保留
`BKG_VAR` 供产品定义的 SNR 和缺失误差回退；它们是 Gaussian count/s/fully
illuminated detector 测量，不是能流。请求窗口
和实际完整指向曝光分开写入报告，短窗口不会把完整指向 PHA 按比例缩放。
每个 survey CAT 的八个原生能道会保留，并在没有附带总能段时按误差平方和推导一个
第九个总能段；已有第九列不会再次累加。`RATE_ERR` 优先作为源率误差，只有缺失时
才回退到 `BKG_VAR`，而 `Flux/ECF` 推导的结果会标记为计数率来源。仅有
`batsurvey.pickle` 或完成标记的目录不会被当作可用科学产品；PHA、响应和处理状态
会在光谱阶段重新核验。兼容的 QDP 光变会标记为 `native_qdp_rate`，但因其只给出
单点时间误差，不会被当作完整指向曝光参与窗口谱选择。
若结果树含有逐指向 `*.gti`，光变选择会记录真实 GTI 重叠；该重叠只用于窗口覆盖
判定，完整 survey 曝光和 PHA 不会按重叠比例缩放。

BAT survey 光谱的上限使用带符号 `RATE` 的 Gaussian profile：`STAT_ERR` 是统计误差，
OGIP `SYS_ERR` 只转换并加入一次，`BKG_VAR` 只用于原生 TOTSNR 的灵敏度校准，不会
混入光谱拟合误差。响应模板用同一 PHA/RSP 通过 `fakeit(applyStats=False)` 生成，默认
固定 Γ=2、`Δχ²=9`（单侧高斯等效置信度 `0.9986501019683699`）。每个结果分别保存
`observed_upper_bound`（本次观测的条件 profile）和 `detection_sensitivity`（空白控制
样本加固定位置注入的 90% 检出灵敏度）；没有足够控制尾部时后者明确为
`calibration_status=unavailable`，不会用人工 `nσ×BKG_VAR` PHA 代替。

```bash
# 明确提供空白 survey 控制表后才会估计固定位置灵敏度
python -m jinwu.swift.bat.survey AT20G_J182338-345412 \
  --root /data/swift/batdata --survey-products /data/swift/00097302084_surveyresult \
  --sensitivity-controls /data/swift/blank_sky_sources.fits \
  --output /tmp/at20g_bat --obsid 00097302084 \
  --window 2024-07-16T12:06:10 2024-07-16T12:19:58
```

`--sensitivity-controls` 必须来自源不活跃、质量筛选一致的八能道 survey 产品；其
`BKG_VAR` 是局部天空噪声，不与 PHA `STAT_ERR` 相加。`--profile lmjagn` 只改变参考项目
确认过的 8000/0.01 筛选和 0.01 mosaic 部分编码门槛，不改变能段或上限统计契约。

需要控制资源时，可在命令行追加 `--processes N`、`--internal-threads 1` 和
`--task-timeout SECONDS`。网络阶段可分别用 `--network-timeout`、`--retries`、
`--retry-wait` 和 `--query-margin` 调整；每个阶段仍会在工作区下使用独立的
PFILES 和日志位置。

HEASoft/XSPEC 测试应在 `hea` 环境中运行。`conda run` 不会自动 source
HEASoft 的初始化脚本，因此管线在可选的 Swift 后端边界补齐
`heainit.sh` 所需的 `HEADAS`、Perl/PGPLOT 运行变量，并为每个阶段建立可写的
`HOME`、PFILES 和临时目录；核心配置导入仍不会加载 BatAnalysis 或发起网络请求。

```bash
# 纯离线契约测试
PYTHONPATH=$PWD/packages/jinwu/src:$PWD/packages/jinwu-swift/src \
  conda run --no-capture-output -n hea python -m pytest \
  packages/jinwu-swift/tests/test_bat_survey.py -q -p no:cacheprovider

# 本地 AT20G PHA/RSP 的 XSPEC 上限验收（显式 opt-in）
JINWU_RUN_REAL_DATA=1 \
  PYTHONPATH=$PWD/packages/jinwu/src:$PWD/packages/jinwu-swift/src \
  conda run --no-capture-output -n hea python -m pytest \
  packages/jinwu-swift/tests/test_bat_survey_heasoft.py -q -p no:cacheprovider
```

真实观测查询、下载和在线 mosaic 均默认关闭；只有显式加入 `--query`、
`--download` 或 `--mosaic` 才会启用相应阶段。未设置注册的 Swift/UKSSDC 账号时，
不要提交在线产品请求。
