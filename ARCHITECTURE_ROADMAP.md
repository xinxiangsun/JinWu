# Jinwu 2030 时域多信使与 AI 原生架构蓝图

> 文档状态：架构方向与实施路线图<br>
> 当前基线：Jinwu 0.2.0（monorepo：`jinwu` 核心 + `jinwu-ep/-fermi/-swift` 仪器发行包），Alpha<br>
> 最后更新：2026-09-05（源码审查、定向回归与实施顺序修订，见 §19）<br>
> 复核周期：至少每 6 个月一次<br>
> 状态标记约定：【✅】已实现 ·【🔶】部分实现/有前身 ·【⬜】未开始。未注明更新日期的状态保留 2026-08-30 历史复核口径；2026-09-05 的复核结论与执行优先级见 §19，冲突时以较新的明确记录为准。规划登记不代表实现或验收完成。

## 1. 执行摘要

Jinwu 的长期目标，是成为面向 2030 年代时域天文学和多信使天文学的通用 Python
科学平台。它不应只是一组 X 射线任务脚本，也不应通过重新实现所有成熟软件来追求
表面上的“独立”。Jinwu 应提供稳定的科学语义、任务无关的数据契约、可组合的模型与
似然、可追踪的工作流，以及能够被研究者和 AI 系统一致调用的公共接口。

2026-09-05 定位补充：近期聚焦“连接多任务观测数据与可复现物理推断的公共分析层”，
以 EP、Swift、GBM 的完整研究流程作为外部采用的起点。优先提升科学结果可信度、
接口稳定性和外部研究者独立使用能力，再逐步扩大波段与信使范围。

长期能力范围包括：

- 光学、X 射线和伽马射线的事件、光变、能谱、成像与联合拟合。
- 引力波和中微子事件的 alert、定位、关联、外部 likelihood 接入与联合推断。
- 数据获取、任务标定、产品生成、背景分析、物理建模、统计推断和可重复发布。
- 面向人类和机器的稳定 API、结构化错误、机器可读 provenance 和分析规范。

实施策略必须保持务实：

1. 近期优先完成并稳定 Einstein Probe 数据处理流程。【🔶 WXT 已有流水线与 fake 后端 e2e 测试，但仍有缓存依赖缺陷；FXT 待做，见 §19】
2. 生产级 EP、Swift 等 reduction 继续依赖 HEASoft、任务软件和 CALDB。【✅ 维持该策略】
3. 从已有统计与响应折叠组件提炼公共契约，原生拟合与外部采用并行推进。
   【🔶 已有上限分析中的原生组件；完整原生谱拟合仍待建设，不作为告警接入和多任务产品互操作的前置条件】
4. 脱离 HEASoft 只作为远期、选择性的能力，不是近期成功标准。【✅ 维持该策略】
5. 新架构在同一个 `jinwu` 发行包中从 `jinwu.kernel` 开始建设。
   【🔶 0.2.0 起修订：`kernel` 仍待建，但发行结构已改为 monorepo——任务插件拆出
   `jinwu` 主包（见 §5、§14.3），与"避免主包承担所有任务发布周期"的方向一致】
6. NumPy/SciPy 是可信参考后端，JAX 是可选高性能后端。【✅ NumPy/SciPy 维持；JAX ⬜】
7. 参考其他项目不等于把它们全部加入基础依赖；外部生态通过显式适配器接入。
   【✅ 已执行：核心依赖收窄至 numpy/scipy/astropy/matplotlib/pillow】

第一条端到端的新内核能力仍以已有 OGIP 产品的响应折叠、统计计算和参数拟合为目标，
并使用 XSPEC 作为数值校验基准；优先迁移现有真实调用路径，避免新旧内核长期维护两套
统计实现。文档、发布门禁、插件规范与社区维护从当前阶段开始建设。

## 2. 当前基线与主要差距

Jinwu 已经具有可继续发展的基础，而不是从零开始（2026-08-30 状态）：

- 【✅ 属实】`core` 中已有 OGIP、事件、光变、PHA、RMF、ARF、GTI、时间系统和数据校验代码。
- 【✅ 属实，且已升级】`ep`、`swift`、`fermi` 已承载任务相关能力（0.2.0 起为独立发行包）。
- 【🔶】`background`、`lightcurve`、`spectrum`、`physics`、`lf` 已有领域模块；
  模块存在不等于计算能力可用，`physics.GeneralRelativity` 的公开接口缺陷见 §19。
- 【✅ 属实】已存在纯 Python FTOOLS 替代实现和可选 Rust 加速路径（`jinwurs` 独立发行）。
- 【🔶 需更新表述】已有 HEASoft 环境管理、XSPEC 拟合、上限和红移外推等工作代码。
  （2026-09-05：BXA 已作为可选 extra，并有 `core.bxa_fit` 与拟合方法分发；
  逐参数 profile 误差状态、模型比较、flux curve 与微信快报产物已有实现，
  统一结果契约及各路径科学验收仍需推进）
- 【✅ 属实且已扩充】已有覆盖 OGIP、时间、GTI、数据集、拟合、任务 I/O 和显著性的
  测试资产（历史记录为 949 项通过，含 fake 后端管线测试；不能代表当前全套测试状态。
  2026-09-05 定向回归及明确失败项见 §19.1）。

当前结构也存在需要在 Phase 0 处理的债务（状态按 2026-08-30 复核标注）：

- 【✅ 已解决】`bxa`、`pymc`、`swiftbat`、`batanalysis` 等仍是基础安装的强制依赖。
  （0.2.0：核心依赖仅 numpy/scipy/astropy/matplotlib/pillow；任务能力拆为独立发行包，
  swiftbat 经 `jinwu-swift` 提供；2026-09-05：BXA 与 BatAnalysis 已有按需启用的
  extra/适配路径，不能再表述为从项目彻底移除）
- 【🔶 部分解决】XSPEC 调用和 HEASoft 初始化分散在 `core`、`lf` 和绘图代码中。
  （拟合入口已集中到 `core.fit` + `_require_xspec` 单点；`plotfit`/`upperlimit` 仍直接
  import xspec；`HeasoftEnvManager` 存在但无 CapabilityProbe 抽象）
- 【⬜ 未处理】当前 `ModelBase` 和数据集容器不足以表示单位、先验、参数链接和多后端编译。
- 【⬜ 未处理】响应、数据集、统计量和推断器之间尚未形成稳定的公共契约。
- 【🔶 部分解决】任务发现和产品扫描中仍混有较多 EP 特定逻辑。
  （Catalog/Manifest/DataFile 抽象已任务无关，但 `core/instruments.py` 仍内置
  WXT/FXT 扫描器与 FXTA/FXTB 合束逻辑）
- 【🔶 部分解决】发布 CI 负责构建和上传，但当前没有测试、导入和依赖矩阵门禁。
  （publish workflow 已按 tag lockstep 构建 4 个发行包 + Rust wheel 矩阵，但仍不运行 pytest）
- 【🔶 部分解决】`make sync` 与部分运行记录使用统一版本信息，但文档仍有漂移。
  （2026-09-05：Quick Start 接口不匹配，Sphinx 仍引用旧 `src` 目录并从缺失的
  `jinwu.__version__` 回退至旧版本；见 §19.2）
- 【✅ 基本解决】部分模块导入会隐式要求交互环境或外部软件，不利于最小安装和自动化。
  （库代码 plt.show() 已清除；swiftbat/astroquery/plotly 等改为惰性导入 + 守卫报错；
  仪器 pipeline 经 entry point 懒发现）

这些问题应作为迁移输入，而不是通过一次性重写全部代码解决。

## 3. 架构原则

以下原则是新内核和新公共 API 的强制约束。

### 3.1 科学语义优先于计算后端

模型、参数、响应和统计量的定义属于 Jinwu。SciPy、lmfit、JAX、XSPEC、BXA、Bilby
或 SkyLLH 只是执行后端或外部能力，不能反向决定 Jinwu 的公共语义。

### 3.2 数据产品是 HEASoft 与内核之间的边界

生产流程可以继续使用 HEASoft 生成事件、PHA、RMF、ARF 和图像。内核处理这些标准化
产品，而不关心它们来自 HEASoft、任务软件还是未来经过验证的原生 reducer。

### 3.3 无隐藏全局状态

模型求值、似然计算和数据变换必须是显式输入、显式输出。XSPEC 全局模型、随机数种子、
CALDB、宇宙学参数、丰度表和截面表都必须被记录，不得依赖不可见的进程状态。

### 3.4 单位、时间和坐标是类型语义

公共边界使用 `astropy.units.Quantity`、`astropy.time.Time` 和
`astropy.coordinates`。性能循环内部可以转为规范单位的数值数组，但转换必须由对象
记录，不能依赖调用方猜测单位。

### 3.5 组合优先于深层继承

使用小型数据类、`Protocol` 和组合构建数据集、响应和插件。任务包负责 I/O 与校准差异，
不通过深层继承树重新定义通用科学行为。

### 3.6 可追踪性是科学结果的一部分

任何结果都必须能够追溯到输入校验和、选择条件、软件版本、外部命令、CALDB、响应、
模型、先验、统计量、随机种子和运行环境。

### 3.7 AI 与人类使用同一科学接口

AI 层只能调用公共 Python 服务层，不得拥有另一个更宽松的分析实现，也不得绕过单位、
数据质量、统计量和外部工具能力检查。

### 3.8 兼容通过适配器实现

新内核不承诺复刻所有旧对象。旧 API 在迁移期通过明确的转换器或兼容包装器连接，避免
让历史接口永久限制内核设计。

## 4. 非目标

以下事项明确不属于近期目标：

- 不立即重写 EP、Swift 或 Fermi 的全部 HEASoft/任务软件流程。
- 不重新实现 GWpy、Bilby、PyCBC、SkyLLH 或 IceCube 的成熟探测器 likelihood。
- 不把所有参考包都加入基础依赖。
- 不用一个无语义的通用 `ndarray` 代替事件、光谱、天空图和响应对象。
- 不把显著性、交叉匹配概率和物理参数 likelihood 混为一个统计接口。
- 不允许 AI 自动化以牺牲科学校验和可重复性为代价。
- 不在没有真实数据和数值对照的情况下默认启用原生 reduction 替代品。

## 5. 分层架构与依赖方向

依赖只能从外层指向内层：

```text
AI / CLI / notebooks
          |
          v
workflows and domain services
          |
          +----------> external integrations
          |                |
          v                v
dataset and inference contracts
          |
          v
data / model / response / stats / provenance
```

禁止 `jinwu.kernel` 导入任务包、HEASoft、XSPEC、BXA、GWpy、Bilby、SkyLLH 或 alert
客户端。外部适配器可以导入内核并把外部对象转换为 Jinwu 类型。

目标包结构如下：

```text
jinwu/
  kernel/
    data/
    model/
    response/
    stats/
    dataset/
    inference/
    multimessenger/
    provenance/
    schema/
  integrations/
    heasoft/
    xspec/
    bxa/
    lmfit/
    alerts/
    crossmatch/
    gw/
    neutrino/
  workflows/
  ai/
  ep/
  swift/
  fermi/
  background/
  lightcurve/
  spectrum/
  timing/
  physics/
  lf/
```

现有领域包不要求立即搬迁。它们先通过新契约接入，只有在职责清晰且测试充分时才逐步
调整目录。

> 【🔶 2026-08-30 状态】`kernel/` 与 `integrations/` 层【⬜】未开始建设；但任务插件
> 独立发行已提前落地——0.2.0 起仓库为 monorepo，`ep`/`swift`/`fermi` 拆为独立发行包
> （`packages/jinwu-ep|-fermi|-swift`），经 PEP 420 命名空间（`jinwu` 无 `__init__.py`）
> 与 `jinwu.instruments` entry points 接入核心（见 §14.3）。`ep`/`swift`/`fermi`/
> `background`/`lightcurve`/`spectrum`/`timing`/`physics`/`lf` 目录结构与上图一致；
> `workflows/`、`ai/` 尚未建立。

## 6. 内核契约

### 6.1 数据层

`jinwu.kernel.data` 提供任务无关的数据原语：

- `Axis`：坐标值、边界、单位、尺度和排序信息。
- `IntervalSet`：GTI、坏时间区间、选择区间及集合运算。
- `EventTable`：时间、通道或能量、质量标记和任务扩展列。
- `BinnedCounts`：计数、曝光、区间、mask 和统计语义。
- `CountsSpectrum`：探测器通道空间中的计数数据。
- `FluxPoints`：已经经过响应推断的 flux 或 luminosity 点及协方差。
- `Photometry`：滤光片、flux/magnitude 系统、零点和 bandpass 身份。
- `ImageData`：像素坐标、WCS、曝光、mask 和 PSF 引用。
- `Observation`：数据产品、GTI、指向、探测器、响应引用和 provenance 的组合对象。

公共数据对象必须满足：

- 单位、时间尺度、参考时刻和坐标框架不可省略或含糊。
- 原始值、质量 mask、选择 mask 和派生值相互分离。
- slicing、grouping 和 rebinning 返回新对象，并记录操作历史。
- I/O 元数据与科学语义分离，但原始 header 可以完整保留。
- 数据对象不持有 XSPEC、HEASoft 或任务 pipeline 会话。

### 6.2 模型层

`jinwu.kernel.model` 将声明式模型和执行模型分离：

- `Parameter`：稳定 ID、名称、单位、初值、边界、固定状态、变换和先验。
- `ParameterLink`：等值、比例、函数关系和跨数据集共享关系。
- `ModelComponent`：加性、乘性、卷积和变换组件。
- `ModelGraph`：不可变的声明式组件图和参数图。
- `CompiledModel`：针对 NumPy、JAX 或外部后端生成的执行对象。
- `DerivedQuantity`：flux、luminosity、能量、时标等从参数和模型推导的量。

关键不变量：

- `ModelGraph` 不存放拟合过程中变化的全局值。
- `evaluate()` 对相同输入产生相同输出，不修改模型或参数。
- 参数向量的顺序由稳定 ID 映射产生，不能依赖 XSPEC 式扁平 tuple 索引。
- 模型必须声明输入坐标和输出物理量，例如 `ph / (cm2 s keV)`。
- 红移、宇宙学、吸收丰度和截面是显式配置或参数。

### 6.3 响应层

`jinwu.kernel.response.ResponseOperator` 把潜在物理信号映射到观测空间。首批实现包括：

- `RedistributionMatrix`：光子能量到探测器通道的概率或有效响应。
- `EffectiveArea`：能量、时间、方向或探测器相关的有效面积。
- `Bandpass`：光学/红外通量到观测 photometry 的映射。
- `PSFOperator`：天空模型到图像像素或区域的映射。
- `ExposureOperator`：时间、姿态、死时间和有效曝光映射。
- `CompositeResponse`：按明确顺序组合多个响应算子。
- `ResponseSeries`：具有有效时间区间的一组响应，支持选择、插值和有依据的加权。

响应身份至少包含：

- 输入和输出轴。
- 探测器、任务和校准版本。
- 有效时间、指向和源方向。
- 文件校验和与创建工具。
- 插值、重采样和加权历史。

响应矩阵不能被作为普通可逆矩阵使用。所有推断均采用 forward folding。

### 6.4 统计层

`jinwu.kernel.stats` 定义统计语义，优化器只消费其输出：

- `CashStatistic`：Poisson 源计数，背景已知或包含在模型中。
- `WStatStatistic`：Poisson ON/OFF 或源/背景谱。
- `PGStatStatistic`：Poisson 源计数与 Gaussian 背景约束。
- `GaussianStatistic`：具有方差或协方差的近似 Gaussian 数据。
- `EventLikelihood`：未分箱事件 likelihood 的通用协议。
- `LikelihoodTerm`：任意数据集对参数的标量 log-likelihood 贡献。

统计量必须定义零计数、零曝光、空通道、负背景估计和 nuisance parameter 的行为。
Li & Ma 显著性、检测阈值和 goodness-of-fit 属于诊断层，不伪装成参数 likelihood。

### 6.5 数据集与联合推断

`Dataset` 组合：

- 一个或多个观测数据对象。
- 响应算子。
- 模型绑定。
- 统计量。
- 选择 mask 和质量信息。
- 数据集级 nuisance parameters。

`JointDataset` 组合多个 `LikelihoodTerm`，通过稳定参数 ID 共享源参数。独立数据集的
log-likelihood 可以相加；相关系统误差必须通过显式协方差或共享 nuisance model 表示，
不能默认独立。

`FitProblem` 包含数据集、模型图、参数状态和后端配置。`FitResult` 至少保存：

- 最优参数、协方差、profile interval 或后验样本。
- 每个数据集的统计贡献和有效数据点。
- 优化或采样诊断。
- posterior predictive checks、bootstrap 或 goodness-of-fit 结果。
- 派生物理量及其不确定度。
- 完整 `ResultManifest`。

### 6.6 最小公共协议草图

下面的协议表达职责边界，不规定首个版本必须采用继承还是具体实现类：

```python
class Model(Protocol):
    @property
    def parameters(self) -> ParameterSet: ...

    def evaluate(
        self,
        coordinates: ModelCoordinates,
        parameters: ParameterValues,
    ) -> PhysicalPrediction: ...


class ResponseOperator(Protocol):
    def apply(
        self,
        prediction: PhysicalPrediction,
        context: ObservationContext,
    ) -> ExpectedData: ...


class LikelihoodTerm(Protocol):
    @property
    def parameter_ids(self) -> tuple[str, ...]: ...

    def log_likelihood(self, parameters: ParameterValues) -> float: ...


class InstrumentPlugin(Protocol):
    def capabilities(self) -> CapabilityManifest: ...
    def read_observation(self, source: DataSource) -> Observation: ...
```

后端适配器接收这些 Jinwu 对象并返回 Jinwu 结果。任何公共签名都不应要求用户传入
XSPEC model、lmfit `Parameters`、Bilby likelihood 或 SkyLLH dataset。

## 7. HEASoft 与任务软件边界

### 7.1 当前生产路径

```text
Raw mission data
    -> HEASoft / mission pipeline / CALDB
    -> calibrated event, image, PHA, RMF, ARF
    -> Jinwu readers and validators
    -> Jinwu kernel analysis
```

该路径在可预见阶段内仍是 EP 和 Swift 的生产标准。Jinwu 必须把 HEASoft 视为正式的
外部运行时，而不是临时导入技巧。

### 7.2 规划中的集成接口

【🔶 2026-08-30 状态】`jinwu.core.heasoft.HeasoftEnvManager` 已存在（HEADAS 探测、
`init_heasoft()`/`reset_environment()`，quickstart 在用），是本节设计的部分前身；
`HeasoftEnvironment` 子进程隔离、`CapabilityProbe/Report`、`ToolSpec/Runner/Invocation`
provenance 链【⬜】未实现。

`jinwu.integrations.heasoft` 应提供：

- `HeasoftEnvironment`：构造隔离的子进程环境，不修改宿主 Python 进程。
- `CapabilityProbe`：检查 `HEADAS`、CALDB、XSPEC、任务软件和关键命令。
- `CapabilityReport`：机器可读地返回版本、路径、可用性和修复建议。
- `ToolSpec`：命令、输入、输出、所需能力和副作用声明。
- `ToolRunner`：执行、超时、日志、退出码、临时目录和失败分类。
- `ToolInvocation`：写入 provenance 的完整调用记录。

工作流必须在开始前声明并验证能力，例如：

```text
ep.reduce_wxt:
  requires:
    - heasoft.ftools
    - ep.wxtpipeline
    - caldb.ep

swift.extract_xrt:
  requires:
    - heasoft.xselect
    - heasoft.xrtpipeline
    - caldb.swift
```

缺少能力时必须 fail fast，不能静默使用科学含义不同的近似实现。

### 7.3 原生替代策略

未来原生 reducer 必须输出同一个标准产品契约，并在成为默认路径前通过：

1. 文件格式和元数据对照。
2. 曝光、GTI、区域、能道和响应对照。
3. 计数、背景和统计量对照。
4. 至少两个真实观测和一组合成数据的科学结果对照。
5. 任务专家审阅和可重复环境记录。

完整脱离 HEASoft 不是 1.0 的发布条件。

## 8. 多信使预留接口

多信使支持首先统一事件语义、时间、定位、alert 和 likelihood 边界，而不是立即实现每种
探测器的底层算法。

### 8.1 通用事件对象

`TransientEvent` 包含：

- 全局或来源内稳定的事件 ID。
- 开放的 messenger vocabulary，例如 photon、gravitational-wave、neutrino。
- 事件时间或时间概率分布，以及原始时间尺度。
- `Localization`。
- 分类概率和来源。
- 距离或红移分布。
- alert revision、状态、撤回信息和父版本。
- 原始 payload 引用及 provenance。

messenger 类型使用开放字符串或 URI vocabulary，不使用不可扩展的封闭枚举。

### 8.2 定位对象

- `PointLocalization`：位置和二维误差模型。
- `RegionLocalization`：置信区域或几何区域。
- `HealpixLocalization`：归一化天空概率图、RING/NESTED 和坐标框架。
- `VolumeLocalization`：天空概率与每个方向上的距离分布。

定位对象必须区分 likelihood map、posterior map 和 credible region，不能只保存一组轮廓。

### 8.3 引力波接口

- `StrainSeries`：探测器、通道、采样率、GPS/UTC 参考和质量区间。
- `FrequencySeries`：频率轴和复数或实数频域数据。
- `NoisePSD`：有效时间、估计方法和频率范围。
- `ExternalLikelihoodAdapter`：把 Bilby 或其他框架的 likelihood 暴露为 `LikelihoodTerm`。

Jinwu 应优先与 GWpy、Bilby 和 `ligo.skymap` 互操作，不复制其底层信号处理。

### 8.4 中微子接口

`DirectionalEvent` 保存：

- 到达时间和不确定度。
- 重建方向及每事件方向概率。
- 能量 proxy 及其校准语义。
- 事件质量和 detector configuration。
- 信号/背景 PDF 所需的附加字段。

SkyLLH 或 Flarestack 可以通过 adapter 提供时间相关、方向相关和能量相关的 unbinned
likelihood。Jinwu 负责联合参数和 provenance，不复制其成熟 detector model。

### 8.5 Alert 与关联

`AlertPacket` 保存规范化字段和未经修改的原始 payload。`EventStream` 负责：

- 持续消费和可重放存档。
- 去重和顺序处理。
- revision 和 retraction。
- schema 版本迁移。
- 网络中断后的恢复。

`AssociationProblem` 与 `JointDataset` 分离。前者计算多目录对应体、时间符合、空间符合、
距离一致性和偶然符合概率；后者进行物理模型联合 likelihood。NWAY 可作为交叉匹配适配器。

## 9. 技术雷达

### 9.1 状态定义

- **Adopt**：允许用于生产接口，并有维护者、适配器和持续测试。
- **Trial**：在受控垂直切片中试用，不承诺长期公共接口。
- **Assess**：研究设计和互操作，不进入默认安装。
- **Watch**：持续观察成熟度、维护状态或许可证。
- **Reject**：已记录不采用的原因，避免反复讨论。
- **Reference**：学习架构或算法，但不计划成为运行时依赖。

### 9.2 初始雷达

| 技术 | 状态 | Jinwu 中的角色 |
|---|---|---|
| NumPy / SciPy / Astropy | Adopt | 最小科学运行时和参考实现 |
| [lmfit](https://lmfit.github.io/lmfit-py/) | Trial | 参数约束与确定性优化 adapter |
| JAX | Trial | 可选编译、自动微分和批量计算后端 |
| [ELISA](https://astro-elisa.readthedocs.io/) | Reference | `Model`/`CompiledModel`、JAX 和结果设计 |
| [GDT](https://astro-gdt.readthedocs.io/en/latest/) | Reference | 任务无关数据原语和时间依赖响应 |
| [HEApy](https://github.com/jyangch/heapy) | Reference | 瞬变端到端工作流和任务工具桥接 |
| [nDspec](https://ndspec.readthedocs.io/en/latest/) | Assess | 多维响应、功率谱和 cross-spectrum 互操作 |
| Stingray | Assess | 通用谱时产品和时间序列分析 |
| [NWAY](https://johannesbuchner.github.io/nway/) | Assess | 多目录贝叶斯交叉匹配服务 |
| [BXA](https://johannesbuchner.github.io/BXA/) | Adopt, optional → Assess* | XSPEC/Sherpa 与 UltraNest 的兼容桥 |
| UltraNest / dynesty / iminuit | Assess | 采样、evidence 和优化后端 |
| ArviZ / xarray | Assess | 后验诊断和带标签结果表示 |
| 3ML / Sherpa / Gammapy | Reference | 插件、拟合职责和 Dataset 设计 |
| GWpy / Bilby / `ligo.skymap` | Assess | 引力波数据、likelihood 和定位 adapter |
| SkyLLH / Flarestack | Assess | 中微子 unbinned likelihood adapter |
| [GCN Kafka](https://gcn.nasa.gov/docs/notices) | Adopt as standard | 实时 alert 交换 |
| [IVOA VOEvent](https://www.ivoa.net/documents/VOEvent/) | Adopt as standard | 兼容历史和 VO 生态的事件格式 |
| ASDF | Assess | Jinwu 原生模型、结果和 provenance 序列化 |
| Rust extension | Trial → 独立发行* | 经过 Python 参考实现验证的热点加速 |

\* 2026-08-30 状态变化：BXA 已从全部依赖（含 extras）中移除——当前 XSPEC 误差采用
逐参数 profile interval 状态记录，无 Bayesian 桥接需求；BXA 产生的 MCMC chain 文件
仍可经 `UpperLimit.from_chain` 消费。重引入须按 §9.3 走 ADR。
Rust 加速已落地为独立发行包 `jinwurs`（`jinwu[rust]` extra 引用），由 CI 构建多平台
abi3 wheel。

技术雷达至少每 6 个月复核，并在发布说明中记录状态变化。

### 9.3 依赖准入

新增依赖必须通过 ADR，并回答：

1. 它解决的是科学语义、执行后端还是用户体验问题？
2. 能否通过 `Protocol` 或文件格式适配，而不暴露第三方对象？
3. 维护活跃度、许可证、Python 支持和二进制兼容性如何？
4. 是否能在 CI 中安装和验证？
5. 是否会使无 HEASoft、无网络或无 GPU 的基础安装失效？
6. 替换或移除它时，公共数据和结果能否继续读取？
7. 它的科学定义能否由独立测试验证？

只有具备适配器、错误隔离、文档和集成测试的包才能进入 optional extras。

## 10. 依赖与发布策略

目标基础安装只保留通用内核依赖，例如 NumPy、SciPy、Astropy 和必要的轻量基础设施。
2026-09-05 核心依赖为 NumPy、SciPy、Astropy、Matplotlib 和 Pillow；任务工具已拆入
插件，BXA 等按需安装。后续以最小安装测试维持此边界，不重新引入可选重依赖。

规划中的 extras 按能力而不是按流行包命名：

```text
jinwu[optim]           # lmfit, iminuit adapters
jinwu[bayes]           # generic posterior diagnostics and samplers
jinwu[xspec]           # XSPEC integration glue; HEASoft remains external
jinwu[xspec-bayes]     # BXA / UltraNest compatibility
jinwu[ep]
jinwu[swift]
jinwu[fermi]
jinwu[spectral-timing]
jinwu[crossmatch]
jinwu[alerts]
jinwu[gw]
jinwu[neutrino]
jinwu[viz]
jinwu[jax]
jinwu[rust]
```

注意：`jinwu[xspec]` 不能声称通过 pip 安装 HEASoft。它只安装 Python 适配代码，并通过
capability probe 检查外部环境。大型 `[all]` extra 在兼容矩阵成熟前不提供。

> 【🔶 2026-08-30 状态】已提供：`jinwu[ep]`【✅】、`jinwu[swift]`【✅】、`jinwu[fermi]`
> （含 `gbm` 别名）【✅】、`jinwu[crossmatch]`【✅】、`jinwu[rust]`【✅】，另有计划外的
> `cluster`/`docs`；仪器依赖进一步独立为 `jinwu-ep/-swift/-fermi` 发行包（swift 包内
> 另有 `[gdt]` extra）。未提供：`optim`/`bayes`/`xspec`/`xspec-bayes`/
> `spectral-timing`/`alerts`/`gw`/`neutrino`/`viz`/`jax`【⬜】。注意 BXA 已不在任何
> extra 中（见 §9.2 脚注）。

FITS、OGIP、VOTable、GCN JSON 和 VOEvent 继续作为交换格式。Jinwu 原生分析规范和
结果使用版本化 schema；具体采用 ASDF 或其他容器必须经过 ADR，但 JSON Schema 是机器
接口的稳定契约。

## 11. AI 原生能力

“AI 原生”表示机器可以发现、验证、规划、执行和解释公共科学操作，不表示在内核中加入
不可审计的自动决策。

### 11.1 机器契约

- `AnalysisSpec`：输入、数据选择、产品、模型、统计量、后端、随机种子和输出。
- `CapabilityManifest`：可用任务、仪器、模型、统计量、后端、HEASoft 和 CALDB。
- `ResultManifest`：结果、诊断、假设、单位、版本、校验和和操作历史。
- `StructuredError`：稳定错误码、阶段、上下文、缺失能力和修复建议。
- `SideEffectSpec`：网络、文件写入、外部命令、凭据和预估计算成本。

JSON Schema 是跨语言、CLI 和 agent 的稳定表示。Python 内部可以用 dataclass 或
Pydantic 实现，但实现库不能成为 wire format。

### 11.2 CLI 与服务层

规划接口：

```text
jinwu capabilities --json
jinwu schema AnalysisSpec
jinwu validate analysis.json
jinwu run analysis.json --dry-run
jinwu run analysis.json
jinwu inspect result.asdf --json
jinwu doctor
jinwu doctor heasoft
```

`--dry-run` 必须解析输入、验证能力、列出外部命令和输出位置，但不执行科学计算或写入
最终产品。

### 11.3 MCP 与 agent

MCP/agent 层只能封装同一个服务层。它必须：

- 暴露机器可读 schema 和 capability。
- 为有副作用操作提供明确确认边界。
- 返回结构化结果和错误，而不是只返回终端文本。
- 保存用户提供、AI 生成和最终执行的规范版本。
- 不自动改变模型、先验、统计量或数据质量筛选而不记录理由。

MCP 进入公开 API 的前提是 `AnalysisSpec` 和服务层已稳定，不能先于它们设计。

### 11.4 AI benchmark

AI benchmark 至少覆盖：

1. 从机器可读 capability 选择可用工作流。
2. 为 golden observation 生成有效 `AnalysisSpec`。
3. 识别缺失 HEASoft/CALDB 并停止，而不是伪造结果。
4. 正确区分 Cash、WStat、PGStat、Li & Ma 和 association probability。
5. 解释单位、响应、背景和参数链接。
6. 执行 dry-run，再在授权后运行。
7. 从 `ResultManifest` 重建结果来源。
8. 对 revision/retraction alert 更新结果而不覆盖历史。

## 12. 路线图与退出门槛

路线图按能力门槛推进，不按日期强行切换生产路径。

2026-09-05 执行顺序修订：以下 Phase 保留为能力分组，版本号不表示对应能力已经交付。
实际排期采用 §19.5 的验收顺序。Phase 6 中的文档、治理、引用和插件规范前移至
当前阶段；Phase 2 原生拟合可并行研发，不阻塞基于现有后端的外部采用与告警适配。

### Phase 0：EP 稳定期，0.0.x

目标：保证当前科学生产流程可靠，并为新内核建立可信基线。

产物：

- 【🔶】完成 EP WXT/FXT 的核心 reduction 和分析流程。
  （WXT 🔶：已有含拟合/通量曲线/中文快报的流水线及 fake 后端 e2e 测试，
  仍需修复 §19 的缓存问题并形式化真实数据验收；FXT ⬜：仅 config 与扫描器，无 pipeline）
- 【🔶】选择弱源、强源、复杂 GTI/响应各至少一个 golden observation。
  （有真实数据回归工作区如 EP260703a，但未形式化 golden 基准与重放记录）
- 【🔶】记录 HEASoft、任务软件、CALDB、输入和输出校验和。
  （pipeline stage manifest 记录输入/输出 sha256 指纹；拟合产物含可回放 .xcm/.log 与
  `collect_runtime_environment`；2026-09-01 BAT 验证记录已有 CALDB 配置校验和，
  尚需将实际使用的校准文件身份与环境记录推广为各任务统一契约）
- 【🔶】集中 HEASoft 环境检查和外部命令 provenance。
  （HeasoftEnvManager 见 §7.2；xselect 调用记录 .xco/.log）
- 【✅】把 BXA、PyMC 和任务工具从基础依赖迁移到 extras。
  （任务工具升级为独立发行包；BXA、BatAnalysis 按需接入，保持核心安装轻量）
- 【🔶】增加最小安装、完整安装、无 XSPEC 和 HEASoft 集成测试矩阵。
  （已有测试与 network/heasoft/real_data markers；CI 尚无安装矩阵，当前定向结果见 §19.1）
- 【⬜】发布前 CI 必须运行测试、导入检查和 wheel smoke tests。
  （publish workflow 目前只构建与发布）

退出门槛：

- 【🔶】Golden workflows 可以从干净环境重放。
  （e2e 测试以 fake 外部后端可重放；真实数据 golden 未形式化）
- 【✅】无 HEASoft 环境能够导入通用模块。
- 【✅】生产 pipeline 缺少能力时给出明确错误。
  （XSPEC ImportError 带 pip 提示；entry-point 缺仪器包时提示 `pip install jinwu-ep`）
- 【✅】基础安装不再拉取任务和 Bayesian 重依赖。

### Phase 1：Kernel 契约期，0.1

【⬜ 2026-08-30 状态】未开始。仅有雏形：兼容 shim（`jinwu.core.lf`/`jinwu.core.redshift`
弃用包装、lightcurve/spectrum 再导出）可视为 adapter 思路的雏形；`FitResult.to_dict`/
拟合结果 JSON/逐参数误差状态是 ResultManifest 的素材，但均非冻结契约。

目标：冻结最小数据、模型、响应、统计、provenance 和 schema 契约。

产物：

- `jinwu.kernel` 初始包和依赖方向测试。
- `Parameter`、`ModelGraph`、`ResponseOperator`、`LikelihoodTerm` 和 `Dataset`。
- `AnalysisSpec`、`CapabilityManifest`、`ResultManifest`。
- 多信使基础类型和 alert revision 模型。
- 旧数据对象到新对象的显式 adapter。

退出门槛：

- 内核可在最小依赖环境独立导入。
- 所有公共类型可序列化并通过 round-trip tests。
- 模型求值无副作用，参数链接不依赖位置索引。
- 人类 API 与机器 schema 表达同一操作。

### Phase 2：Native OGIP 期，0.2

【🔶 2026-09-05 状态】已有 `CallablePhotonModelPredictor`、Gaussian 净率对象和
ON/OFF likelihood 等可复用组件，完整的通用原生谱拟合路径与公共契约尚未完成。
0.2.0 版本号对应 monorepo 打包重构，不表示本阶段已验收。

目标：完成第一条脱离 XSPEC 的端到端拟合路径。

产物：

- PHA、背景、RMF、ARF 到内核对象的读取。
- NumPy forward folding。
- PowerLaw 和经过版本化数据支持的吸收模型。
- Cash、WStat 和 PGStat。
- SciPy 参考优化器和 lmfit adapter。
- XSPEC parity suite 和可选 BXA 路径。

退出门槛：

- 无 XSPEC 环境可完成标准 OGIP 拟合。
- 简单模型的预测计数和统计量通过预先声明的 parity tolerance。
- XSPEC 丰度、截面和宇宙学设置被完整记录。
- `FitResult` 可以在没有原拟合后端时读取和检查。

### Phase 3：多波段与谱时期，0.3

【⬜ 2026-08-30 状态】未开始。

目标：从 X 射线内核扩展到真正的多仪器联合分析。

产物：

- 光学 `Photometry` 与 `Bandpass`。
- 伽马事件、时间依赖响应和多探测器数据集。
- 功率谱、cross-spectrum、lag 等谱时对象。
- nDspec/Stingray 互操作实验。
- NWAY 交叉匹配 adapter。
- 共享源参数和独立仪器 calibration nuisance parameters。

退出门槛：

- 一个光学 + X 射线 + 伽马合成联合拟合通过端到端测试。
- 联合 log-likelihood 和参数链接可独立验证。
- 时间依赖响应保留有效区间和加权 provenance。

### Phase 4：Alert 与关联期，0.4

【⬜ 2026-08-30 状态】未开始。

目标：接入实时事件生态，并建立可重放的关联分析。

产物：

- GCN Kafka 和 VOEvent adapters。
- `EventStream`、revision、retraction 和 schema migration。
- HEALPix/3D localization。
- `AssociationProblem` 和偶然符合概率模型。

退出门槛：

- 网络中断后可恢复且不重复处理事件。
- revision/retraction 不覆盖历史。
- alert 存档可离线完整重放。
- association 与 physical likelihood 在 API 和结果中明确区分。

### Phase 5：GW 与中微子适配期，0.5+

【⬜ 2026-08-30 状态】未开始。

目标：让 Jinwu 能够参与多信使联合推断，而不复制成熟生态。

产物：

- GWpy 数据对象转换和 `ligo.skymap` 定位转换。
- Bilby likelihood/result adapter。
- SkyLLH 或 Flarestack likelihood adapter。
- photon + GW + neutrino 合成联合案例。

退出门槛：

- 外部对象不会泄漏为 Jinwu 永久公共格式。
- 每个外部 likelihood 的参数、归一化和独立性假设均有记录。
- 合成联合案例可以共享时间、天空位置、距离和源参数。

### Phase 6：AI 与社区成熟期，1.0

2026-09-05 修订：本阶段表示成熟度验收，不表示届时才启动社区建设。
文档、贡献指南、科学审查、弃用政策、引用信息及插件规范从 Phase 0/1 开始交付。

【⬜ 2026-08-30 状态】未开始。唯一相关的雏形是面向 AI 代理的
`REUSABLE_FUNCTIONS.md` 公共 API 索引（非机器契约）。

目标：稳定公共契约、插件生态和长期治理。

产物：

- MCP/agent 适配层。
- 任务插件 SDK 和 conformance suite。
- 公开 golden datasets 与 AI benchmark。
- 英文规范文档和中文翻译。
- ADR/RFC、弃用、发布、引用和安全政策。
- Zenodo DOI 和可引用数据/软件版本。

1.0 不要求完全脱离 HEASoft，但要求所有 HEASoft 依赖均被显式隔离、探测和记录。

## 13. 验证战略

### 13.1 测试层级

- **Unit tests**：公式、单位、边界条件和纯函数。
- **Contract tests**：插件、响应、模型、likelihood 和序列化协议。
- **Property tests**：rebin、slice、坐标转换、概率归一化和 round trip。
- **Numerical parity tests**：NumPy/JAX、Python/Rust、Jinwu/XSPEC。
- **Integration tests**：HEASoft、任务 pipeline、alert broker 和外部 adapters。
- **Golden tests**：真实观测的科学关键输出，而不是只比较文件存在。
- **Architecture tests**：禁止内核导入任务包和可选重依赖。
- **AI benchmark**：规范生成、校验、执行、解释和 provenance。

### 13.2 数值门槛

- 解析模型和纯矩阵 folding 的参考测试目标为 `rtol <= 1e-10`。
- NumPy/JAX float64 结果目标为 `rtol <= 1e-7`。
- 简单 PowerLaw 的 XSPEC folding 初始目标为 `rtol <= 1e-4`，使用完全一致的能格、
  通道 mask 和响应。
- 吸收模型 parity 必须固定 abund、xsect、红移和积分设置，阈值按模型建立，不能使用
  一个掩盖物理差异的全局容差。
- 拟合参数应在统计不确定度的 `0.1 sigma` 内一致，或通过预先说明的数值误差预算。
- 原生 reduction 比较必须同时检查文件、计数、曝光、背景、响应和最终科学结论。

### 13.3 CI 矩阵

至少包含：

- 最小依赖、无网络、无 XSPEC。
- 支持的 Python 版本。
- 常规 extras 的组合安装。
- NumPy 参考路径。
- 可选 JAX 和 Rust parity。
- 专用 HEASoft runner 上的 integration tests。
- wheel 安装和导入 smoke tests。
- 文档链接、示例和 schema 校验。

## 14. 治理与社区影响力

成为有影响力的科学软件依赖治理，而不只是功能数量。

### 14.1 决策机制

- 架构和依赖变化使用 ADR。
- 大型功能使用公开 RFC。
- 公共 API 遵循语义化版本和明确弃用周期。
- 技术雷达、支持矩阵和路线图每 6 个月复核。
- 统计公式、标定处理和物理模型需要科学审阅者。

### 14.2 文档与教育

- 英文文档作为国际协作的规范版本，中文文档提供完整翻译。
- 每个公共对象包含单位、统计假设、失败模式和最小示例。
- 教程基于可下载的小型真实数据，并在 CI 中执行。
- 同时提供研究者教程、插件开发指南和 AI 工具指南。
- 结果页面说明如何引用 Jinwu、任务数据、外部软件和校准文件。

### 14.3 插件生态

任务插件必须通过 conformance suite，验证：

- 时间、坐标、单位和 GTI。
- 数据质量和 mask。
- 响应和校准身份。
- capability 与失败报告。
- provenance 和序列化。
- 无可选依赖时的隔离行为。

长期允许任务插件在独立发行包中通过 entry points 注册，避免 Jinwu 主包承担所有任务的
发布周期。

> 【✅ 2026-08-30 状态】该长期项已提前实现：`jinwu-ep`/`jinwu-swift`/`jinwu-fermi`
> 独立发行，经 `jinwu.instruments` entry points 注册 pipeline；核心注册表懒发现，
> 缺包时给出 `pip install jinwu-<name>` 提示。conformance suite 部分【⬜】未建。

## 15. 成功指标

成功不能只用代码行数或支持任务数量衡量。

### 科学可靠性

- Golden workflows 的可重放率。
- 统计量和响应 parity 覆盖率。
- 已发表研究中的复现成功率。
- 科学缺陷发现到修复和发布的时间。

### 生态采用

- 外部维护的任务插件数量。
- 外部贡献者和机构数量。
- 使用 Jinwu 生成的同行评议论文和公开数据产品。
- 与 Astropy、GDT、GCN 等标准生态的互操作案例。

### API 与运维

- 最小安装成功率和导入时间。
- 公共 API 破坏性变更数量。
- 外部工具调用的 provenance 完整率。
- CI 支持平台、Python 和 extras 的覆盖率。

### AI 可用性

- AI benchmark 通过率。
- 生成 `AnalysisSpec` 的首次校验成功率。
- 对单位、统计量和能力缺失的正确识别率。
- AI 执行结果的可重放率和 provenance 完整率。

## 16. 风险与应对

| 风险 | 应对 |
|---|---|
| 目标过大导致长期没有可用成果 | 每个阶段必须交付一条端到端 vertical slice |
| 为兼容旧代码污染新内核 | 使用显式 adapter，不在内核复制旧状态模型 |
| 依赖爆炸和安装冲突 | 最小基础安装、capability extras、持续安装矩阵 |
| 过早替代 HEASoft 产生科学偏差 | 生产路径不变，原生替代必须经过任务级 golden validation |
| 多信使对象过度抽象 | 先实现 alert、时间、定位和 likelihood 协议，再扩展探测器细节 |
| AI 自动化掩盖分析假设 | 版本化 AnalysisSpec、dry-run、结构化 provenance 和强制校验 |
| 单一维护者成为瓶颈 | RFC、插件维护者、英文文档、公开 benchmark 和发布治理 |
| 后端成为公共 API | Jinwu 类型作为边界，第三方对象只存在于 integration 层 |

## 17. 首批 ADR 清单

实施前应依次完成以下 ADR：

1. `ADR-001`：内核包边界和禁止依赖规则。
2. `ADR-002`：公共单位、时间尺度和内部规范单位。
3. `ADR-003`：`Parameter`、参数链接和变换语义。
4. `ADR-004`：`ModelGraph` 与 `CompiledModel`。
5. `ADR-005`：统计量、likelihood 和显著性的职责边界。
6. `ADR-006`：HEASoft capability、子进程和 provenance。
7. `ADR-007`：`AnalysisSpec`、JSON Schema 和版本迁移。
8. `ADR-008`：原生结果序列化容器，评估 ASDF。
9. `ADR-009`：插件 entry points 和 conformance suite。
10. `ADR-010`：alert revision/retraction 与事件身份。
11. `ADR-011`：HEALPix、MOC 和三维定位表示。
12. `ADR-012`：AI/MCP 副作用和授权模型。

## 18. 参考项目与标准

这些项目用于学习和互操作，不代表自动成为 Jinwu 依赖：

- [Astropy](https://www.astropy.org/)：单位、时间、坐标、FITS 和社区治理。
- [ELISA](https://astro-elisa.readthedocs.io/)：高能能谱模型编译和现代推断。
- [GDT](https://astro-gdt.readthedocs.io/en/latest/)：任务无关的高能数据原语和响应。
- [HEApy](https://github.com/jyangch/heapy)：多任务瞬变数据处理工作流。
- [Gammapy](https://docs.gammapy.org/)：Dataset、IRF 和 likelihood 组合。
- [3ML](https://threeml.readthedocs.io/)：多任务插件和联合 likelihood。
- [Sherpa](https://sherpa.readthedocs.io/)：数据、模型、统计和优化职责。
- [lmfit](https://lmfit.github.io/lmfit-py/)：参数约束和优化接口。
- [nDspec](https://ndspec.readthedocs.io/en/latest/)：多维谱时建模。
- [NWAY](https://johannesbuchner.github.io/nway/)：多目录贝叶斯交叉匹配。
- [BXA](https://johannesbuchner.github.io/BXA/)：X 射线 Bayesian 分析桥接。
- [GWpy](https://gwpy.readthedocs.io/)：引力波时间序列和频域数据。
- [Bilby](https://bilby-dev.github.io/bilby/)：通用与引力波 Bayesian inference。
- [ligo.skymap](https://lscsoft.docs.ligo.org/ligo.skymap/)：引力波定位和 HEALPix。
- [SkyLLH](https://icecube.github.io/skyllh/)：中微子和多信使 likelihood。
- [GCN](https://gcn.nasa.gov/docs/notices)：实时天文 alert 和 JSON schema。
- [IVOA VOEvent](https://www.ivoa.net/documents/VOEvent/)：瞬变事件交换标准。

Jinwu 当前采用 GPL-3.0-or-later，并已注明从 HEApy 修改代码的来源。未来复用任何第三方
代码时，必须继续保存版权、许可证、修改记录和引用信息；只借鉴设计时也应在架构文档或
ADR 中注明来源。

## 19. 2026-09-05 基础设施建设审查与行动计划

### 19.1 审查范围与验证证据

本节来自对 Jinwu 0.2.0 当前工作区（包含未提交修改）的源码、公共接口、仪器插件、
文档、发布流程和已有路线图的只读审查。审查未修改科学代码，也未重跑全部任务的
HEASoft reduction、标定和真实数据流水线。本次只登记问题和计划，不将其标记为已修复。

在 `hea` 环境中执行的核心定向回归：

```bash
conda run -n hea pytest \
  test/test_base_time.py test/test_ogip.py test/test_writefits_roundtrip.py \
  test/test_datasets.py test/test_config.py test/test_pipeline.py \
  test/test_fit_settings.py test/test_model_comparison.py test/test_bxa_fit.py \
  test/test_xray_model_comparison.py test/test_pipeline_products.py test/test_plotpanel.py \
  -q -p no:cacheprovider -m 'not network and not heasoft and not real_data'
```

实际结果：**528 passed, 6 skipped, 1 deselected in 7.99s**。
执行时使用 Agg 绘图后端、临时 matplotlib 缓存、禁写字节码，并将数学库线程数限制为 1。

追加运行：

```bash
conda run -n hea pytest test/test_txx_iterbkg_validation.py -q -rs -p no:cacheprovider
```

实际结果：**1 failed, 1 passed in 0.91s**。有符号分位数测试通过；真实事件文件路径与
对象等价性测试失败，错误为 `ValueError: Lightcurve HDU lacks RATE/COUNTS column`。
该测试模块曾在 [Swift 验证记录](packages/jinwu-swift/VALIDATION.md) 的受影响回归中
被排除；历史通过总数不得用作当前该路径已经通过验证的证据。

上述结果是定向软件回归证据，不是全套测试通过或全部任务科学标定完成的声明。

### 19.2 已确认问题：先修复并建立发布门禁

审查时点所有条目均为**待修复/待验收**；六项已于 2026-09-05 实施修复
（见下方"修复记录"），正式验收仍按各自退出门槛执行。源码位置使用文件与符号定位，避免行号随修改失效。

| 编号 / 优先级 | 源码与本次证据 | 影响 | 修复方向与退出门槛 |
|---|---|---|---|
| AUD-01 / 高 | [WXT `stage_code_dependencies`](packages/jinwu-ep/src/jinwu/ep/wxt/pipeline.py) 使用 `Path(__file__).resolve().parents[2] / "core"`；拟合阶段列出的 `fit.py`、`products.py`、`plot.py` 全部位于不存在的 `packages/jinwu-ep/src/jinwu/core/` 下 | 核心算法变化可能未触发缓存失效，复用旧科学结果 | 按实际导入模块定位源码，复用 Swift GRB 的模块定位思路；检查依赖文件存在与声明完整性；验证核心辅助模块改变后仅相应依赖阶段重算，并覆盖 editable 与 wheel 安装 |
| AUD-02 / 高 | [`txx_iterbkg._to_binned`](packages/jinwu/src/jinwu/core/timescale.py) 对路径直接使用 `read_lc`；[现有事件回归](test/test_txx_iterbkg_validation.py) 中对象调用完成、路径调用失败 | 相同数据通过不同入口行为不一致，影响批处理和复用 | 复用通用 FITS 类型发现/读取，统一事件与光变输入归一；路径/对象须在绝对时间、共同网格、曝光、BACKSCAL 和 Txx 上一致；修复后将该回归纳入适当的真实数据验收门禁 |
| AUD-03 / 高 | [`GeneralRelativity`](packages/jinwu/src/jinwu/physics/gr.py) 已从 `jinwu.physics` 导出；`g.v = 1.0` 实测触发 `NameError: name 'u' is not defined`；源码也未导入 `np`，`beta` 固定返回占位值 `0.0` | 公开计算能力与实际实现不符 | 盘点公开导出；实验功能显式声明未支持，禁止返回占位科学结果；完成单位、解析解和适用边界测试后再声明可用 |
| AUD-04 / 高 | [publish workflow](.github/workflows/publish.yml) 仅构建 Python/Rust 产物并发布，未运行 pytest；publish 只依赖构建任务 | 构建成功无法阻止接口或科学回归进入发布版本 | PR 与发布接入核心离线测试、插件契约、实际 wheel 安装/发现、支持版本与 extras 矩阵、示例执行；真实数据/HEASoft 验收使用具备相应能力的独立环境 |
| AUD-05 / 中 | [Quick Start](docs/quickstart.rst) 的 `EnergyBand(..., unit="keV")` 实测 TypeError，`ChannelBand.from_energy_band` 不存在 | 新用户照文档操作即失败 | 统一公开能段接口和教程；带固定小样例执行文档，不能仅检查语法或用 mock 代替关键调用 |
| AUD-06 / 中 | [Sphinx 配置](docs/conf.py) 仍指向旧 `src`；实查命名空间包无 `jinwu.__version__`，配置因此回退 `0.0.27`；[RTD 配置](.readthedocs.yaml) 仍从根目录安装，而发行项目已移入 `packages/` | 文档版本与安装路径漂移；RTD 干净环境安装尚需验证 | 使用发行元数据读取版本，按 monorepo 安装文档所需包；在干净环境完成文档构建和关键示例测试，发布版本与文档版本必须一致 |

AUD-01 的修复不能止于路径拼接：还需审查如时标模块等实际算法依赖是否完整登记。
AUD-02 的本次失败发生在读取阶段，后续时间与缩放断言尚未执行，不能视为已通过。
AUD-06 的 RTD 安装失败未在本次远程构建中复现，应保留为明确配置风险和待验收项。

#### 19.2.1 修复记录（2026-09-05）

- **AUD-01 已修复**：WXT `stage_code_dependencies` 改为"导入真实模块 +
  `inspect.getsourcefile`"定位（与 Swift GRB/BAT 一致，editable 与 wheel 均
  正确）；补登 fit 阶段实际执行链 `spectrum_prep`/`bxa_fit`/`config`，以及
  duration 阶段的 `timescale`（ops 仅为重导出垫片，正是本条补充告诫所指）。
  基类 `_stage_code_fingerprint` 对声明了但不存在的文件显式报错（不再静默
  `sha256=None`）；Swift 两处 `getsourcefile(x) or ""` 退化同样改为显式报错。
  新增 `test/test_stage_code_deps.py`：三管线依赖存在性、声明完整性、指纹
  内容敏感性与缺失报错。wheel 布局下的同一断言由发布门禁执行。
- **AUD-02 已修复**：`_to_binned` 路径输入经 `guess_ogip_kind` 按内容判型
  （evt→`read_evt`，lc→`read_lc`），与对象输入汇合到同一处理路径；
  `rebin_events_to_lightcurve` 透传 `timezero`/`timezero_obj`；`_to_binned`
  把 bin 几何平移回绝对时间框架，使 src/bkg（`TIMEZERO` 相差 2.6 s）在绝对
  网格上投影对齐；返回 dict 补 `alpha` 与 `time_reference` 键。
  `test_txx_iterbkg_validation.py` 标记 `real_data` 并实测通过（两分支
  t100/t90/t50/peak 1e-9 一致、alpha=0.18796801885192388、绝对 MET）。
- **AUD-03 已修复**：`gr.py` 补 `numpy`/`astropy.units`/光速导入，`beta`
  改为解析 `v/c`，`v>=c` 报错；`physics/__init__` docstring 移除未实现的
  redshift 宣称；IPython 改为可选（缺失时 show_* 纯文本回退），修正 show_*
  raw-string 双反斜杠转义。新增 `test/test_gr.py`（单位、解析解、边界）。
- **AUD-04 已修复**：新增 `.github/workflows/ci.yml`（PR/push，Python
  3.11–3.13 矩阵，editable 安装四包后运行离线套件 `not network and not
  heasoft and not real_data`，`--strict-markers`）；`publish.yml` 新增
  `wheel-gate`（干净 venv 安装全部 Python wheel → 导入与
  `jinwu.instruments` entry point 发现 → 离线门禁测试子集），publish 依赖
  该门禁。真实数据/HEASoft 验收仍留在具备能力的独立环境。
- **AUD-05 已修复**：quickstart/index 能段示例改为真实 API
  （`EnergyBand` 四字段构造、`channel_mask_from_ebounds`、
  `band_from_arf_bins`，并修正"ARF 转通道"的说法）；新增
  `test/test_quickstart_examples.py` 以固定合成 OGIP 样例执行教程调用。
  顺带发现并修复 `WXTPointingResult.display()` 无 IPython 分支输出契约
  不一致（会中断 CI 环境的教程回归）。
- **AUD-06 已修复**：`docs/conf.py` 删除指向不存在 `src/` 的死路径，版本改
  由 `importlib.metadata.version("jinwu")` 读取（缺失时显式报错，不再回退
  0.0.27），标题显示发布版本；`.readthedocs.yaml` 改为安装 monorepo 四个
  发行包（根目录无 `[project]` 表不可安装）；docs extra 钉
  `sphinx-automodapi>=0.21`（Sphinx 8.2+/9 兼容）。已在干净 venv 按新配置
  完成文档构建（版本 v0.2.0 可见）；RTD 远程构建保留为待验收项。

### 19.3 架构、基本功能与易用性调整

以下是设计与开发计划，已有组件仅作为复用起点，不代表目标契约已经实现。

| 方向 | 当前可复用起点与差距 | 计划与验收要求 |
|---|---|---|
| 数据的科学语义 | [OGIP 数据结构](packages/jinwu/src/jinwu/core/base.py)、时间、GTI 已存在；`SpectrumDataset`/`JointDataset` 主要为容器 | 明确原始计数、净率、通量及其转换来源，显式保存背景模型、曝光、协方差、能段和参考系；公共物理量使用 Quantity，EP 时间复用 `jinwu.core.time.Time(format="ep")`；统计方法根据数据契约验证兼容性 |
| 最小推断内核 | [upperlimit](packages/jinwu/src/jinwu/core/upperlimit.py) 已有 `CountPredictor`、`CallablePhotonModelPredictor`、`GaussianNetRateObservation`；[model_comparison](packages/jinwu/src/jinwu/core/model_comparison.py) 已有 ON/OFF profile/marginal likelihood | 从真实调用路径提炼参数、响应、似然和结果协议，先迁移一个完整案例；保持 profile 与 marginal 的语义区别，避免新旧层复制统计实现 |
| 联合推断 | 已有 XSPEC 分组与 prepared spectra，但通用模型/数据集契约未冻结 | 显式表达共享物理参数、标定 nuisance、共享背景/协方差和观测重叠；每项似然记录独立性假设，防止同一事件生成的谱和光变被重复计入 |
| 一致结果接口 | [fit_spectral](packages/jinwu/src/jinwu/core/fit.py) 按方法返回 dict 或 `BXAFitResult` | 统一参数、区间类型、诊断、来源及序列化接口；分别记录执行、产品、收敛、边界和科学验证状态；保留后端细节但不要求消费者安装后端才能读结果 |
| 插件边界 | 已拆分发行包并使用 entry points；[InstrumentConfig](packages/jinwu/src/jinwu/core/config.py) 仍承载 BAT/GBM 专用配置 | 通用协议留在核心，任务预设、发现、校准和特定配置归插件；按模型/统计/外部执行/诊断/报告拆分大模块；外部团队添加仪器无需修改核心源码，并通过 conformance suite |
| 可复现分析包 | 已有阶段 manifest、schema version、文件指纹和部分环境记录 | 保存生效配置、选择条件、输入与实际校准文件身份、模型/先验、软件版本、随机种子、日志和结果；支持跨目录迁移、schema 升级、历史结果只读和独立复核，逐步向 IVOA Provenance 映射 |
| 批处理与隔离 | 已有阶段恢复和外部命令超时；仍有进程级拟合设置、XSPEC 全局状态及 GBM 上限超时待办 | 将生效配置固化为每次执行上下文；以独立进程运行 PyXspec 拟合，统一中止、超时、CPU/内存预算、PFILES 与工作目录锁；中断不污染其他任务，恢复不误用部分产物 |
| 用户入口 | 已有任务级 CLI、preflight、Notebook 展示与产物报告 | 建议新增统一 `jinwu doctor`、`jinwu run analysis.yaml`、`jinwu explain result`，这些命令当前是规划；全部复用同一 Python 服务层，明确可用能力、缺失条件及结果为何需要复查 |
| 文档与交互 | 已有教程和绘图资产，但入门接口漂移 | 分别提供首次分析、批量分析、模型/插件开发路径；提供固定校验和的小样例；区域、时间窗、背景、残差和质量状态可检查，交互修改能保存为可重放配置；英文规范与中文教程同步维护 |
| 科学验证 | 已有软件回归、部分真实数据与 sensitivity 验证资产 | 建立公开的弱源、强源、无源、复杂背景、间断 GTI、变化响应及仪器边缘基准；检查计数、曝光、背景缩放、参数偏差、区间覆盖率和误报率；为每项能力公布适用范围与验证等级 |

### 19.4 优先开发的科学成果

| 编号 | 方向与已有起点 | 下一步交付与科学边界 |
|---|---|---|
| DEV-01 | EP 发现—后随分析链：已有 WXT 流程与 FXT 配置/扫描 | 补齐 FXT pipeline，统一 WXT、FXT 与 Swift 后随产品组织，以公开真实观测形成完整可重放案例；参考任务软件完成产品及科学对照 |
| DEV-02 | 按坐标和时间查询多任务观测与非检测：复用 BAT survey、GBM 连续数据及覆盖判断 | 返回可用数据、有效曝光、测量、上限及质量状态；明确区分无覆盖、无数据、质量不足和观测后未检测；几何覆盖不能代替有效曝光或通量约束 |
| DEV-03 | 时间与能谱联合建模：复用分段能谱、响应和背景组件 | 逐步实现共享 `F(E,t)`，处理响应变化、GTI、曝光和背景不确定度；以共享参数、预测计数及模拟恢复验证联合模型，记录分段近似与数据依赖 |
| DEV-04 | 注入恢复与选择函数：扩展现有 sensitivity、模拟和高红移可探测性工具 | 输出依赖物理参数、位置、时间和任务选择条件的检测概率及其不确定度；验证误报与恢复率，服务发生率、光度函数和高红移群体研究；单个事件的最大可探测红移不能替代调查选择函数 |
| DEV-05 | 告警与定位适配：接入 GCN 与选定 broker 的相关子流 | 支持版本化 schema、更新、撤回、去重、关联历史和离线重放；区分定位概率图与时空覆盖，MOC 用于覆盖交换；优先复用现有事件分发生态 |

DEV-02 与 DEV-04 是重点差异化方向：将弱信号、非检测及选择过程作为可复用科学产品。
本节为建设判断，不承诺尚未测量的精度、吞吐量或调查完备性。

### 19.5 实施顺序、维护与成功标准

以下顺序以验收结果推进，不构成固定工期承诺：

| 阶段 | 主要交付 | 退出门槛 |
|---|---|---|
| 下一次稳定发布 | 修复 AUD-01～06，建立发布门禁、可执行教程与能力验证清单 | 明确失败回归修复；缓存依赖可验证；实际 wheel 和文档在干净环境可用；每个公开功能标明支持范围，科学验证缺口不被“测试通过”掩盖 |
| 外部可用闭环 | 公开真实观测基准，推进 EP 后随流程与可移植分析包 | 外部研究者无需作者手工调整即可完成声明支持的流程，独立复核输入、结果与质量状态 |
| 公共接口与生态扩展 | 从已有组件提炼内核、交付插件规范，推进 DEV-02/03/05 | 一个真实工作流迁移完成；外部插件通过契约测试；同一分析规范可以由 Python、CLI 和后续 AI 适配层一致执行 |
| 群体分析与性能扩展 | 推进 DEV-04，按实际瓶颈评估 JAX/GPU 和更大规模任务处理 | 检测概率及其误差经过注入恢复验证；优化具有性能测量与数值一致性证据；维护成本和适用范围明确 |

原生拟合可与以上阶段并行研发；更大范围的任务处理、独立 broker 和高性能后端，
以明确合作需求、有效基准和实际瓶颈作为启动条件。

社区建设从现在开始：维护贡献指南、科学审查责任、弃用政策、引用信息与版本 DOI；
逐步为关键模块培养第二位维护者，组织外部团队完成一次独立分析和一次插件开发。
新可复用接口仍按项目约定登记到 `REUSABLE_FUNCTIONS.md` 并配套测试。

长期追踪外部团队独立完成分析的比例、旧结果跨版本复现率、第三方插件数量、科学缺陷
修复周期，以及能够独立维护和发布的贡献者数量。每项指标需定义样本、统计窗口和
验证方式；本次未测量其基线值，不填入推测数值。

### 19.6 本次规划的文献与标准依据

以下来源在本次审查中查阅；用于方法学和互操作设计，不代表 Jinwu 已达到相同能力：

- Vianello et al. (2015), *The Multi-Mission Maximum Likelihood framework (3ML)*,
  [arXiv:1507.08343](https://arxiv.org/abs/1507.08343)：任务插件、响应折叠与联合推断。
- Donath et al. (2023), *Gammapy: A Python package for gamma-ray astronomy*,
  [arXiv:2308.13584](https://arxiv.org/abs/2308.13584)：科学分析体系与长期支持版本实践。
- HEASARC, *XSPEC — Parameter Estimation*,
  [官方文档](https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/node393.html)：Gaussian、Poisson 与背景统计语义。
- Servillat et al. (2020), *IVOA Provenance Data Model 1.0*,
  [IVOA Recommendation](https://www.ivoa.net/documents/ProvenanceDM/)：数据来源与处理活动交换。
- Astropy Collaboration (2022), *The Astropy Project: Sustaining and Growing a Community-oriented Open-source Project and the Latest Major Release (v5.0) of the Core Package*,
  [arXiv:2206.14220](https://arxiv.org/abs/2206.14220)：互操作生态与可持续维护。
- Thrane & Talbot (2019), *An introduction to Bayesian inference in gravitational-wave astronomy: parameter estimation, model selection, and hierarchical models*,
  [arXiv:1809.02293](https://arxiv.org/abs/1809.02293)：层次推断与选择效应；高能仪器仍需自身的检测与注入验证。
- NASA GCN, [Schema Browser](https://gcn.nasa.gov/docs/schema)；Rubin Observatory,
  [Alerts and brokers](https://rubinobservatory.org/for-scientists/data-products/alerts-and-brokers)：现有告警协议与分发生态。
- Fernique et al. (2022), *MOC: Multi-Order Coverage map 2.0*,
  [IVOA Recommendation](https://www.ivoa.net/documents/MOC/)：天空和时间覆盖的标准表示。
