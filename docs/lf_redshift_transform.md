# LF 红移转换：推导核验与实现规格

核验日期：2026-09-10。本文依据当前工作区源码、独立推导及下列原始资料编写；历史聊天只用于确定待核验的问题，不作为物理结论的证据。

**结论：同一本征源的光子谱变换为 $N_2(E,t)=dr^2N_1(rE,t/r)$；普通幂律在固定观测能段中的归一化因子为 $dr^{2-\Gamma}$。原讨论这两项正确。** 实际计数与可探测性还取决于吸收、响应、有效曝光、背景和统计估计，不能只缩放一条带噪声的光变。

本文是 `jinwu.lf` 后续实现的科学与接口规格，**不是新增 API 已实现的声明**。本次仅新增本文并在旧说明中增加链接；没有修改分析代码、模拟产物或校准文件。

## 1. 定义、假设与单位

把同一个本征瞬变源从 $z_1$ 放到 $z_2$：保持源静止系光子发射历史、视向和本征物理参数不变。采用透明传播的 FLRW 宇宙学，忽略透镜和额外传播损失；吸收在第 4 节单独施加。各向异性源可将下述光度理解为固定视向的各向同性等效光度。本征谱演化允许存在。

| 符号 | 定义 | 单位 |
|---|---|---|
| $Q(E_{\rm em},\tau)$ | 源静止系 $dN_\gamma/(d\tau\,dE_{\rm em})$ | photon s$^{-1}$ keV$^{-1}$ |
| $N_z(E,t)$ | 未吸收的观测者系光子通量密度 | photon cm$^{-2}$ s$^{-1}$ keV$^{-1}$ |
| $D_L,D_M,D_C$ | 光度距离、横向共动距离、径向共动距离 | 计算绝对通量时统一为 cm |
| $t,\tau$ | 相对于同一物理参考事件的观测者时间、源时间 | s |
| $K_{\rm pl},K_{\rm zpl}$ | 默认 XSPEC 幂律模型的归一化系数 | photon cm$^{-2}$ s$^{-1}$ keV$^{-1}$ |
| $C,\mu_s$ | 探测器源计数率、源期望计数 | ct s$^{-1}$、ct |
| $\kappa=C/F$ | 指定谱、响应和能段的计数率／能流转换系数 | ct cm$^2$ erg$^{-1}$ |

全篇采用

$$
r=\frac{1+z_2}{1+z_1},\qquad d=\left[\frac{D_L(z_1)}{D_L(z_2)}\right]^2.
\tag{1}
$$

$D_L=(1+z)D_M$；仅空间平直时 $D_M=D_C$。不能将任意宇宙学中的 `comoving_distance` 都当作横向距离，也不将共动距离泛称为“真实物理距离”。这一区分及谱能流关系见 [Hogg 1999，2000 修订版，§§5–7、式 (21)–(22)](https://arxiv.org/pdf/astro-ph/9905116)。

绝对时间另行保存：若参考时刻为 $T_{0,1},T_{0,2}$，应转换
$T_2=T_{0,2}+r(T_1-T_{0,1})$，不能把 MJD、MET 或 EP 绝对时标直接乘 $r$。EP 时间通过 `jinwu.core.time.Time(format="ep")` 解释。

## 2. 从光子数守恒重新推导

源在微元内发出 $dN_\gamma=Q(E_{\rm em},\tau)d\tau\,dE_{\rm em}$。到达观测者时，几何稀释面积为 $4\pi D_M^2$，且

$$
E_{\rm em}=(1+z)E,\quad dt=(1+z)d\tau,\quad dE_{\rm em}=(1+z)dE.
$$

因此光子谱的时间和能宽 Jacobian 抵消：

$$
N_z(E,t)=\frac{Q((1+z)E,t/(1+z))}{4\pi D_M^2}
\frac{d\tau}{dt}\frac{dE_{\rm em}}{dE}
=\frac{(1+z)^2}{4\pi D_L^2}Q((1+z)E,t/(1+z)).
\tag{2}
$$

在两个红移处消去同一个 $Q$，得到任意时变谱的转换算子：

$$
\boxed{N_2(E,t)=dr^2N_1(rE,t/r).}
\tag{3}
$$

这是本节的独立推导，而非对 XSPEC norm 意义的预设。源吸收若固定在静止系，也可以预先并入 $Q$；观测者前景吸收不可以。

### 2.1 能流与积分交叉检查

设 $F_E(E,t)=E N(E,t)$、$L_E(E_{\rm em},\tau)=E_{\rm em}Q(E_{\rm em},\tau)$；能量需转换为 erg，光子数按计数处理。则

$$
F_E(E,t)=\frac{1+z}{4\pi D_L^2}L_E((1+z)E,t/(1+z)),
\quad F_{{\rm bol},z}(t)=\frac{L_{\rm bol}(t/(1+z))}{4\pi D_L^2}.
\tag{4}
$$

第二式要求积分收敛；无限能域纯幂律不能用作有限总光度的检验源。式 (4) 与 [Hogg 的谱能流定义](https://arxiv.org/pdf/astro-ph/9905116)一致。

对任意未吸收谱，换元 $x=rE$ 给出

$$
F_{2,[a,b]}(t)=\int_a^b E\,dr^2N_1(rE,t/r)dE
=dF_{1,[ra,rb]}(t/r).
\tag{5}
$$

对应静止系能段 $[a_{\rm rest},b_{\rm rest}]$ 的光度应由观测者能段
$[a_{\rm rest}/(1+z),b_{\rm rest}/(1+z)]$ 的未吸收能流乘 $4\pi D_L^2$ 得到。固定观测能段与固定静止系能段是不同约束。

## 3. 公式转换表与 XSPEC 参数

令 $A=dr^{2-\Gamma}$。下表中的源率不含背景；计数指期望值，而非某次 Poisson 实现。

| 对象 | 转换 | 必要条件 |
|---|---|---|
| 对应相位与能量 | $t_2=rt_1,\ E_2=E_1/r$ | 参考物理事件一致 |
| 对应 bin | $[t^-_2,t^+_2]=[rt^-_1,rt^+_1]$ | 尚未施加目标观测窗口 |
| 一般光子谱 | $N_2(E,t)=dr^2N_1(rE,t/r)$ | 前景吸收已分离 |
| 普通幂律 norm | $K_{{\rm pl},2}(t)=A K_{{\rm pl},1}(t/r)$ | 默认 norm、固定 $\Gamma$ |
| 红移幂律 norm | $K_{{\rm zpl},2}(t)=dr^2K_{{\rm zpl},1}(t/r)$ | 默认 norm、同步设置模型红移 |
| 能段能流 | $F_{2,[a,b]}(t)=dF_{1,[ra,rb]}(t/r)$ | 未吸收谱；目标所需能域已覆盖 |
| 源计数率 | $C_2(t)=A C_1(t/r)$ | 纯幂律；对应时刻的响应、通道选择及观测者谱衰减相同 |
| 对应 bin 源期望计数 | $\mu_{s,2}=Ar\mu_{s,1}$ | 上一行成立，且完整曝光或曝光比例按对应相位相同 |
| 相同观测相对时刻的源率 | $C_2(t)/C_1(t)=Ar^\alpha$ | $C_1(t)\propto t^{-\alpha}$，同一幂律时间段且 $t>0$ |
| 时间积分能流 | $\mathcal S_{2,[a,b]}=dr\mathcal S_{1,[ra,rb]}$ | 对应完整时间区间、未吸收谱 |

### 3.1 `powerlaw` 与 `zpowerlw` 不能混用 norm 因子

用固定参考能量 $E_0=1$ keV 消除幂函数的量纲。普通幂律为

$$
N_1(E,t)=K_{{\rm pl},1}(t)(E/E_0)^{-\Gamma},\quad
N_2(E,t)=dr^{2-\Gamma}K_{{\rm pl},1}(t/r)(E/E_0)^{-\Gamma}.
\tag{6}
$$

而 `zpowerlw` 定义为 $N(E,t)=K_{\rm zpl}(t)[(1+z)E/E_0]^{-\Gamma}$。
同一红移、同一物理谱满足 $K_{\rm pl}=K_{\rm zpl}(1+z)^{-\Gamma}$，代入式 (6) 即得 $K_{{\rm zpl},2}=dr^2K_{{\rm zpl},1}$（对应相位）。只设置 `Redshift` 不会自动加入宇宙学距离衰减；再对其 norm 使用普通幂律因子则重复引入谱红移。

上述模型定义来自 [XSPEC：powerlaw, zpowerlw](https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/node251.html)。规范实现必须检查 `POW_EMIN/POW_EMAX`：这些设置可令 norm 表示能段能流或微 Jy 谱密度。首版默认 norm 快捷路径遇到这些设置应明确拒绝，不能静默沿用式 (6)。适配器记录设置并管理 XSPEC 会话状态。

若 $\Gamma$ 随相位演化，则逐时段用 $\Gamma_1(t/r)$ 计算因子，不能对整条光变乘同一个 $A$。`cutoffpl`、Band 等曲谱直接按式 (3) 转换；峰值／截止能量随 $1/r$ 移动，但归一化必须依据各模型定义推导。

### 3.2 `cflux` 与 `clumin` 是积分约束

`cflux` 的值是其包裹组件在指定观测能段内的 $\log_{10}F$。先构造物理上正确的目标谱，再积分该组件，并设

$$
\mathrm{lg10Flux}_2=\log_{10}\!\left(\frac{F_{\rm wrapped,2}}{1\ {\rm erg\,cm^{-2}\,s^{-1}}}\right).
\tag{7}
$$

对于固定 $\Gamma$、包裹未吸收纯幂律、固定观测能段，可简化为
$\mathrm{lg10Flux}_2=\mathrm{lg10Flux}_1+\log_{10}d+(2-\Gamma)\log_{10}r$。
若包裹吸收组件则按实际包裹范围重算；`tbabs*ztbabs*cflux*powerlaw` 与 `cflux*tbabs*ztbabs*powerlaw` 的积分含义不同。内层加性归一化固定为非零，避免同时施加第二次物理振幅缩放。依据：[XSPEC cflux](https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/node338.html)。

`clumin` 官方参数顺序为 **Emin、Emax、Redshift、lg10Lum**，能段在源静止系。保持同一本征源、对应相位和同一静止系积分定义时，保持其光度并更新红移；内层谱形中观测者能量尺度仍须正确变换，`clumin` 本身不是任意曲谱的能量红移算子。银河吸收在包裹范围内时不能简单称为本征光度不变。依据：[XSPEC clumin](https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/node340.html)。

两种卷积模型均须覆盖积分所需能域；参数按实际组件名称定位，不硬编码旧文档中的位置。XSPEC 距离计算与传入的 Astropy 宇宙学须验证一致；配置字符串名字相同不能代替数值交叉验证。

## 4. 从谱到实际观测

### 4.1 分离吸收

将未吸收本征谱转换后，目标到达仪器的谱为

$$
N_{{\rm arr},2}(E,t)=T_{{\rm Gal},2}(E)\,
T_{\rm host}((1+z_2)E,t/(1+z_2))\,N_{{\rm intr},2}(E,t).
\tag{8}
$$

式 (8) 是基于各吸收屏所在参考系的建模约定：宿主柱密度及其源相位演化保持不变，银河吸收在观测者能量求值。同一视线可保持银河柱密度；模拟不同视线必须明确传入新值。不要对强吸收的数据点直接除以接近零的透射率来反演本征谱，应使用拟合的谱模型及不确定性。

若仅有固定银河吸收、纯幂律和相同响应，式 (6) 的率因子仍成立，因为相同 $T_{\rm Gal}(E)$ 在线性折叠中抵消。但固定宿主柱密度通常不代表观测者吸收谱形不变。

### 4.2 响应必须按通道折叠

令 $P(i\mid E,t)$ 为入射光子落到通道 $i$ 的重分配概率，$A_{\rm eff}$ 为有效面积，定义 $R_i=A_{\rm eff}P(i\mid E,t)$。则

$$
C_i(t)=\int_{\mathcal E_{\rm response}}R_i(E,t)N_{\rm arr}(E,t)dE,
\qquad C_{\mathcal I}(t)=\sum_{i\in\mathcal I}C_i(t).
\tag{9}
$$

例如“0.5–4 keV 计数率”指选定通道集合 $\mathcal I$；响应能域外观测能段的入射光子仍可重分配到这些通道。只有理想对角响应才可直接用 $\int_{0.5}^{4}A_{\rm eff}N\,dE$ 替代。使用 EBOUNDS 和明确的通道选择策略，不把任意仪器的 PI 编号视为固定能量。

离散响应网格中先计算 $n_k(t)=\int_{E_k^-}^{E_k^+}N_{\rm arr}(E,t)dE$，再按响应定义求 $C_i\simeq\sum_k R_{ik}n_k$。RMF+ARF 的因子只用一次；完整 RSP 已含有效面积，不能再乘同一 ARF。见 [George 等，OGIP CAL/GEN/92-002，§§2–3](https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html)。

原讨论“不能提出因子因为响应不恒定”需要修正：$dr^2$ 本来就能提出积分；不能将积分化为原计数率乘一个普适常数的根本原因，是 $N_1(rE,t/r)$ 通常不与 $N_1(E,t/r)$ 成比例。即使有效面积恒定，曲谱在固定能段内也需要能段移动。反过来，纯幂律的比例关系允许任意能量依赖的相同线性响应。

### 4.3 时间 bin、曝光与重分箱

定义目标 bin $J_j=[t_j^-,t_j^+)$ 和有效观测比例 $0\leq\ell(t)\leq1$（GTI 外为零），则

$$
\mu_{s,ij}=\int_{J_j}\ell(t)C_i(t)dt,\qquad
e_j=\int_{J_j}\ell(t)dt.
\tag{10}
$$

有效曝光平均率为 $\mu_{s,j}/e_j$；若另外报告按墙钟时间平均的率，应单独标注。零曝光保留零期望计数，曝光平均率标为无效。有效面积、曝光或光变的已有校正不能重复使用。

理想对应 bin 且 $\ell_2(rt)=\ell_1(t)$ 时，式 (10) 给出 $\mu_{s,2}=Ar\mu_{s,1}$。真实目标观测的 GTI、死时间、姿态和响应必须单独提供，不能因源红移自动拉长实际望远镜的观测机会。

默认的离散模型：参考 bin 内谱和源期望率分段常数。先伸长源 bin 边界，再与目标 bin 和 GTI 求交，累加“率 × 有效重叠时长”。单 bin 直接使用显式边界，末 bin 也有独立右边界；不靠相邻时间中心的差补齐。没有观测的参考时间区间应标为未约束；只有模型显式规定零源流时才填零，不跨缺口自动插值。

### 4.4 背景和随机观测

背景属于目标观测条件。设 OFF 区域背景率 $b$，ON/OFF 的背景率比例为 $\eta$，有效曝光分别为 $e_{{\rm on},j},e_{{\rm off},j}$。对恒定背景率，条件采样为

$$
n_{{\rm on},j}\mid b\sim\mathrm{Poisson}(\mu_{s,j}+\eta b e_{{\rm on},j}),\qquad
n_{{\rm off},j}\mid b\sim\mathrm{Poisson}(b e_{{\rm off},j}).
\tag{11}
$$

二者给定 $b$ 独立。有背景后验时，每条曲线抽取一次 $b$，再执行式 (11)；不能先抽未来 OFF 计数、除以曝光当成潜在率、然后再对同一观测抽 Poisson。面积／BACKSCAL 的率比例 $\eta$ 与缩放实测 OFF 计数的 $\alpha_j=\eta e_{{\rm on},j}/e_{{\rm off},j}$ 分开，后者仅在 OFF 曝光正时定义。非均匀背景须分别积分，不能默认只按面积缩放。

以下均为式 (11) 的概率推导。若 $b\sim\mathrm{Gamma}(a,\mathrm{rate}=\beta)$，则 OFF 计数

$$
\mathbb E[n]=e\,a/\beta,\qquad
\mathrm{Var}(n)=e\,a/\beta+e^2a/\beta^2.
\tag{12}
$$

若错误地先抽 $n'\mid b\sim\mathrm{Poisson}(bT)$ 再令 $\hat b=n'/T$，就多出 $\mathrm{Var}(\hat b)-\mathrm{Var}(b)=\mathbb E[b]/T$；随后采样长度 $e$ 的 bin，会额外增加 $e^2\mathbb E[b]/T$ 方差。这不是背景率后验本身的不确定性。

实测净计数 $n_{\rm on}-\alpha n_{\rm off}$ 可以为负，应保留其符号及误差。物理源强度需由显式背景与非负源模型推断，不能逐 bin 裁零来代替推断。有效面积校正后的计数或 Gaussian 净率也不是原始 Poisson 计数，不能直接使用式 (11) 的输入口径。

背景主导、背景条件相同、曝光窗口随对应相位伸长，且背景估计误差的相对结构也一致时，$S/N\propto C_s\sqrt{\Delta t}$ 给出 $S/N$ 比值约为 $A\sqrt r$。这是近似尺度估计，不是低计数显著性、固定 OFF 曝光或多窗口搜索的通用公式。固定触发窗口下必须重新积分及评估检测统计量。

无噪声、固定谱形、完整对应观测下，源累计计数分位定义的持续时间伸长 $r$；实际阈值、背景、谱演化及观测缺口可改变测得 T90。不能据“更慢”推定更容易触发，也不保证固定能段观测亮度对红移严格单调。

## 5. 可落地的模块接口规格（待实现）

实现顺序：**本征谱变换 → 吸收 → 响应折叠 → 曝光积分 → 随机观测 → 既有检测算法**。纯物理计算放 `jinwu.lf` 内，仪器特有校准仍由对应仪器包提供。下面是能力边界，不是现有导入示例。

| 能力 | 显式输入 | 显式输出及约束 |
|---|---|---|
| 光子谱红移转换 | 参考未吸收谱求值器、$z_1,z_2$、`cosmology`、目标能量与相对时间 | 式 (3) 的 `Quantity` 光子谱；记录参考／目标能域与时间支持范围 |
| 幂律快捷转换 | norm、photon index、两个红移、`cosmology`、归一化约定 | 新 norm 和无量纲率因子；默认 norm 以外拒绝走此路径 |
| 仪器响应预测 | 目标谱、吸收配置、RMF+ARF 或完整 RSP、通道选择 | 源模型期望率、明确能段／吸收口径的能流与 $\kappa$；无背景随机数混入 |
| 光变积分 | 源模型及显式 bin 边界、参考时刻、目标 bin、GTI／曝光 | 浮点期望计数、有效曝光及有效性标记；不得提前取整 |
| ON/OFF 采样 | 非负源期望、背景率或后验、两区曝光、$\eta$、`rng` | 整数 ON/OFF 观测及保留的原始期望；每条曲线共用其潜在背景率样本 |

### 5.1 复用与最小接入

- 优先复用 [XspecKFactory](../packages/jinwu/src/jinwu/lf/specfake.py) 的 `get_K_with_values()`；其旧返回值 `(K, rate, flux)` 在适配边界命名为 `(kappa, source_rate, energy_flux)`。当 $N_2$ 已有绝对振幅时，直接折叠返回源率，不再额外乘 $d$、$r^2$ 或 $A$。
- 对只有参考率轮廓的固定谱形输入，明确 $p(t)=C_{{\rm source},1}(t)/C_{{\rm model},1}$，并采用 $C_2(t)=p(t/r)C_{{\rm model},2}$。两个模型率必须以一致的物理参数配对；此路径依赖谱形假设，不能从一条宽能段计数光变唯一恢复任意时变谱。
- 若继续走 $F_1=C_1/\kappa_1$ 路径，必须显式计算目标 $F_2$，再求 $C_2=\kappa_2F_2$。不能把 $\kappa_2/\kappa_1$ 当作宇宙学振幅比。
- [core.ops](../packages/jinwu/src/jinwu/core/ops.py) 已有 `rebin_lightcurve` 及按区间重叠投影的代码。未来实现先核对其 counts/rate、曝光和方差契约；满足时复用，不满足时提升共享的重叠积分函数，避免在 `lf` 再复制一份算法。
- [BackgroundCountsPosterior](../packages/jinwu/src/jinwu/background/backprior.py) 已有内部 Gamma 率采样 `_sample_lambda_off`。未来需要公共率采样接口时，在此模块封装该能力，不复制后验推导或调用 `sample_off` 代替率采样。

### 5.2 单位、状态和失败边界

新物理接口必须使用 `astropy.units.Quantity` 表达能量、时间、距离、率及能流；红移和比值是无量纲量。旧裸数接口仅在适配层按其约定显式转换。`cosmology` 必须传入，示例选 Planck18；不读取隐藏的模块全局宇宙学。

距离搬移接口要求有限 $z_1,z_2>0$ 和有限正 $D_L$，不静默截断为很小正数。局部源或 $z=0$ 的有限距离情形应另设计显式距离接口，本规格不隐式支持。

参考谱必须覆盖目标响应所需的 $rE$ 能域，以及 $t/r$ 时间域。超出时返回明确的未约束状态或报错；采用拟合模型外推必须显式授权并记录范围与模型，不能自动补零。绝对时间、参考时刻、输入是否为净率／ON 计数／校正计数，以及吸收位置均是必需元数据。

XSPEC 适配器须显式配置并记录模型参数、丰度、截面、宇宙学、特殊 norm 设置、积分能域、响应和背景标识，避免共享进程全局状态串扰。输入模型及数组不原地修改；随机接口使用传入的 `numpy.random.Generator`。新增可复用函数时才追加 `REUSABLE_FUNCTIONS.md` 并配套测试；本文没有新增函数，故不登记尚不存在的 API。

## 6. 当前实现审计与修正方向

以下结论来自 2026-09-10 工作区源码，不只比较分支提交；已有未提交修改保留。表中“符合”只针对指定步骤，不表示整个模拟管线通过校准。

| 源码位置／符号 | 当前证据与影响 | 后续实现要求 |
|---|---|---|
| [redshift.py](../packages/jinwu/src/jinwu/lf/redshift.py)：`_adjust_params_for_redshift` | 普通幂律使用 $dr^{2-\Gamma}$，符合式 (6)；`zpowerlw/cflux` 抛出 `NotImplementedError`；`clumin` 保持光度 | 保留正确幂律因子，按第 3 节补齐模型语义和包裹范围检验 |
| 同文件：红移输入与宇宙学 | `max(float(z), 1e-6)` 静默改变目标红移，距离使用模块级 Planck18 | 显式校验红移，显式传宇宙学 |
| 同文件：`_normalize_lightcurve_profile` 与 `_generate_lightcurve_at_redshift` | 归一化代数上是 `net_rate/rate_z0`，再乘 `rate_z`；没有额外距离因子重复相乘 | 固定谱形且输入源率／曝光正确时可复用；不是谱演化支持的证明 |
| [lcfake.py](../packages/jinwu/src/jinwu/lf/lcfake.py)：两个 `build_fake_*_from_npz` | 用 $C_1/\kappa_1$ 得到能流，时间伸长后能流振幅保持原值，再乘 $\kappa_2$；纯归一化同时约去 rate 与 flux，因此不能携带距离衰减 | 明确为响应转换步骤，补目标能流转换，或采用成对的模型绝对率比 |
| 同文件：`load_counts_npz` → 两个构建入口 | 优先读 `corrected_counts_src`，或由 net+background 重建 ON 总计数；调用者随后将其当源信号，配置背景时再次加入 | 对这些 ON 输入分支会重复计算背景；分开 ON/OFF 数据、源期望和校正率的数据类型 |
| 同文件：时间与无噪声分支 | `_infer_dt` 取中位数；部分路径对时间点插值；`add_poisson=False` 仍取整计数 | 显式边界及有效曝光重叠积分；期望计数始终为浮点 |
| 同文件：`generate_redshift_lightcurves` | 用相邻时间差作宽度、末 bin 重复上一宽度；单 bin 使用未伸长 `dt`，且总时长为零；多 bin 的 `last-first` 也未包含末 bin 宽度 | 单 bin 与最后一个 bin 均采用明确左右边界；背景曝光与实际积分窗口一致 |
| 同函数与 [backprior.py](../packages/jinwu/src/jinwu/background/backprior.py) | `sample_off(T)/T` 将后验预测计数变成率后再采样，额外增加方差；源 Poisson 均值使用 `clip(...,0,None)` | 直接抽潜在背景率；负净计数另行建模，不能静默裁零 |
| [detectability.py](../packages/jinwu/src/jinwu/lf/detectability.py)：`simulate_snr_at_z` | 调用 `build_fake_from_npz(add_poisson=False)` 后再 Poisson，继承振幅、输入背景口径及提前取整的问题 | 用未取整的物理源期望与目标背景驱动检测模拟 |
| [test_lf_model_structure.py](../test/test_lf_model_structure.py) | 当前 14 项通过，但主要是解析／mock 测试；`clumin` fixture 把 `lg10Lum` 放在 `Redshift` 前，与官方顺序相反 | 修正真实顺序的 fixture 并做真实 PyXspec 参数与谱积分测试；不能把通过数当作响应验证 |
| [旧 RedshiftExtrapolator 文档](RedshiftExtrapolator.md) | 入口仍指向旧 `core.utils`，使用径向共动距离及含糊距离术语 | 标记历史说明；平直宇宙学下 $[D_C(z_1)/D_C(z_2)]^2r^{-\Gamma}=dr^{2-\Gamma}$，不能误报此代数关系错误；不将它推广到任意模型 |

## 7. 验证记录与未来验收

### 7.1 本次已完成的验证

- 独立查阅 Hogg、XSPEC 模型定义与 OGIP 响应文档，得到第 2–4 节推导及边界。
- 在 `hea` 环境运行 `conda run -n hea pytest test/test_lf_model_structure.py -q`：**14 passed**。这是现有软件测试结果，未新增或修正这些测试。
- 下列可复现的独立数值检查使用合成谱和合成响应；所有断言通过。它验证公式代数与积分，不调用生产转换函数，不代表生产 API 回归或真实仪器校准。
- 本次没有执行真实 PyXspec 响应、WXT/FXT 实测产品或注入回收实验。这些验证保持待完成；普通 shell 中未初始化 PyXspec 的导入状态不作为“未安装”的证据。

在仓库根目录执行下述完整命令即可复核数值检查；示例参数均为人工构造，不是科学观测结果。绝对谱振幅取任意正值，所有比较均用比值消去其影响。

```bash
conda run --no-capture-output -n hea python - <<'PY'
import numpy as np
from astropy.cosmology import Planck18
from scipy.integrate import quad

z1, z2, z3, gamma = 1.0, 3.0, 5.0, 1.7
dl = lambda z: Planck18.luminosity_distance(z).to_value("cm")
r = (1 + z2) / (1 + z1)
d = (dl(z1) / dl(z2)) ** 2
A = d * r ** (2 - gamma)
q = lambda e, t: e * np.exp(-e / 3) * np.exp(-t / 30)
n = lambda z, e, t: (1 + z)**2 / (4*np.pi*dl(z)**2) * q((1+z)*e, t/(1+z))
integral = lambda f, lo, hi: quad(f, lo, hi, epsabs=0, epsrel=1e-10)[0]
close = lambda x, y: np.testing.assert_allclose(x, y, rtol=1e-8, atol=0)
move = lambda f, za, zb, e, t: (dl(za)/dl(zb))**2 * ((1+zb)/(1+za))**2 * f(((1+zb)/(1+za))*e, t/((1+zb)/(1+za)))
e, t = np.geomspace(0.1, 20, 101), 17.0
f1 = lambda e, t: n(z1, e, t)
f2 = lambda e, t: move(f1, z1, z2, e, t)
close(f2(e, t), n(z2, e, t))
close(move(f1, z1, z1, e, t), f1(e, t))
close(move(f2, z2, z1, e, t), f1(e, t))
close(move(f2, z2, z3, e, t), move(f1, z1, z3, e, t))
close(integral(lambda x: x*n(z2, x, t), 0, np.inf),
      integral(lambda x: x*q(x, t/(1+z2)), 0, np.inf)/(4*np.pi*dl(z2)**2))
close(integral(lambda x: x*f2(x, t), 0.5, 4),
      d*integral(lambda x: x*f1(x, t/r), 0.5*r, 4*r))
# Non-constant synthetic response; no real telescope calibration.
response = lambda x: 20*(1+0.3*np.sin(x))
p1 = lambda x, t: x**(-gamma)*np.exp(-t/30)
p2 = lambda x, t: move(p1, z1, z2, x, t)
rate1 = lambda t: integral(lambda x: response(x)*p1(x, t), 0.3, 10)
rate2 = lambda t: integral(lambda x: response(x)*p2(x, t), 0.3, 10)
close(rate2(t), A*rate1(t/r))
close(integral(rate2, 10*r, 20*r), A*r*integral(rate1, 10, 20))
# Norm versus count-to-flux conversion: amplitude cancels from kappa.
flux1 = integral(lambda x: x*p1(x, t/r), 0.3, 10)
flux2 = integral(lambda x: x*p2(x, t), 0.3, 10)
close(rate2(t)/flux2, rate1(t/r)/flux1)
close(d*r**2*((1+z2)*e)**(-gamma), A*((1+z1)*e)**(-gamma))
print("PASS: spectrum, identity, inverse, composition, bolometric, band flux, response, bin counts, kappa, zpowerlw")
PY
```

### 7.2 后续实现的验收矩阵

| 层次 | 必须覆盖的情景 | 判断标准 |
|---|---|---|
| 纯谱变换 | 恒等、往返、$z_1\to z_2\to z_3$ 与直接变换 | 同一支持域上相符；代数相对误差目标 $10^{-12}$ |
| 积分物理 | $\Gamma=2$、曲谱峰值移动、式 (4)–(5) | $\Gamma=2$ 时普通幂律率因子为 $d$；平滑合成谱积分相对误差 $<10^{-8}$ |
| XSPEC 参数 | 默认 powerlaw/zpowerlw 等价、特殊 norm 设置、cflux 包裹范围、真实 clumin 参数顺序 | 根据公式生成同一目标谱；能域收敛后比对积分与 norm；特殊 norm 不静默走默认路径 |
| 吸收与响应 | 固定银河屏、随源移动宿主屏、非对角响应与能段外漏入、FULL RSP | 手工合成矩阵和折叠结果一致；不重复吸收或有效面积因子 |
| 时间与曝光 | 非零参考时刻、非均匀 bin、单 bin、末 bin、固定目标宽度、GTI 缺口和部分曝光 | 完整对应 bin 得到 $Ar$；重分箱前后总期望按有效曝光守恒；零曝光无虚假率 |
| 数据语义 | 原始 ON/OFF、带负净计数、已校正率、仅期望值输出 | 不重复加背景；不把校正／净数据强行当 Poisson 样本；期望不取整 |
| 背景统计 | 固定率及 Gamma 后验、ON/OFF 不同曝光、每曲线共用背景率 | 均值／方差满足式 (11)–(12)；随机检查偏差在预先计算的 Monte Carlo 标准误差范围内 |
| 状态与失效 | 非正或非有限红移、能时域外求值、单位错配、复用 rng、不同 XSPEC 配置 | 明确报错／未约束状态；重复种子可复现且会话配置不串扰 |
| 仪器与检测 | 一个明确绑定 PHA/BAK/RMF/ARF 的目标、空场控制、注入回收及多时间窗搜索 | 记录校准版本、统计量、假警与恢复率；通过后才评估特定配置的可探测红移 |

真实 XSPEC 的数值容差应根据模型精度和响应能量网格收敛实验确定并在测试中固定，不能机械沿用纯代数容差。运行时使用 `hea` 并初始化 HEASoft；隔离其可写运行目录。软件回归、物理数值检验与仪器校准三类结果分别记录。

本文为谱—计数—光变的基础契约，不实现光度函数拟合、宇宙事件率或完备性校正。最大可探测红移仍需先验证检测概率随红移的行为，不能从宇宙学衰减直觉直接假定二分搜索总成立。

## 8. 参考资料与定位

1. David W. Hogg (1999; v4, 2000), *Distance measures in cosmology*, [astro-ph/9905116](https://arxiv.org/abs/astro-ph/9905116)。使用距离及谱能流关系，不采用其早期示例宇宙学参数作为默认观测结论。
2. HEASARC XSPEC Manual, [powerlaw, zpowerlw: power law photon spectrum](https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/node251.html)：默认归一化、红移模型与特殊 norm 设置。
3. HEASARC XSPEC Manual, [cflux: calculate flux](https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/node338.html)：观测能段能流与卷积包裹范围。
4. HEASARC XSPEC Manual, [clumin: calculate luminosity](https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/node340.html)：源能段、参数顺序与光度定义。
5. I. M. George et al., [OGIP Calibration Memo CAL/GEN/92-002: The Calibration Requirements for Spectral Analysis](https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html)：响应重分配、有效面积、RMF/ARF 及完整响应格式。

以上链接于核验日查阅。XSPEC 网页编号可能随版本变化；复核时同时检查页面标题和本地软件版本，不只检查链接是否返回成功。
