# JinWu：Joint Inference for high energy transient light‑curve & spectral analysis With Unifying physical modeling

## 项目简介 / Project Introduction

金乌（JinWu）是中国古代神话中的太阳神鸟，象征着光明、能量与希望。传说中，金乌为三足乌，栖于扶桑，驾驭太阳穿行于天际，赋予万物生机。以“金乌”为名，寓意本项目致力于高能瞬变天体（如伽马暴、超新星等）的联合光变曲线与光谱物理建模与推断，探索宇宙中最明亮、最剧烈的能量释放过程。

本项目旨在为高能天体物理领域的研究者，提供统一、灵活且易于扩展的分析工具，支持多种物理模型、数据拟合与推断方法，促进科学交流与创新。尤其是关于EP WXT, FXT的数据处理和产品pipeline.

JinWu, the Golden Crow, is a legendary solar bird in ancient Chinese mythology, symbolizing light, energy, and hope. According to legend, JinWu is a three-legged bird dwelling in Fusang, driving the sun across the sky and bringing vitality to all things. Naming this project "JinWu" reflects our dedication to joint inference and physical modeling of high-energy transients (such as gamma-ray bursts and supernovae), aiming to explore the brightest and most energetic phenomena in the universe.

This project provides a unified, flexible, and extensible toolkit for researchers and enthusiasts in high-energy astrophysics, supporting various physical models, data fitting, and inference methods, and fostering scientific communication and innovation.

Especially, this repo devoting on EP/WXT&FXT data products process.

## 安装前置条件

| 使用场景 | 需要准备 |
|---|---|
| FITS / OGIP 读写、光变分析与绘图 | Python ≥ 3.11（推荐 3.12）和 `pip`。NumPy、SciPy、Astropy、Matplotlib 和 Pillow 由 pip 安装。 |
| 在 Python 中用 XSPEC 拟合，或在本机调用 XSELECT | 在运行 JinWu 的同一个 Conda 环境安装并激活 HEASoft。XSPEC 的 Python 模块名为 `xspec`。 |
| 从 EP Level-1 数据运行 `wxtpipeline` 或 `fxtchain` | `jinwu[ep]`、可用的 Docker Engine 或 Docker Desktop、对应仪器的处理软件镜像、匹配的 CALDB 和 Level-1 数据。只运行容器内的处理任务时，可以省去本机 HEASoft。 |

推荐用 **Conda 管理本机 HEASoft**，用 **Docker 运行 EP 官方处理软件**。如果处理完还要在本机用 PyXspec 拟合，两套环境都要准备。

### 通过 PyPI 安装 JinWu

在打算运行 JinWu 的 Python 环境执行：

```bash
python -m pip install jinwu
# 处理 Einstein Probe WXT / FXT 数据时安装 EP 插件
python -m pip install "jinwu[ep]"
```

核心与仪器包共享 `jinwu` 命名空间。其他仪器插件和可选功能见下方的可选依赖。

### 通过 Conda 安装 HEASoft

先安装 Conda（例如 [Miniforge](https://github.com/conda-forge/miniforge)），再从 HEASARC 的官方频道创建独立环境。做 EP 谱分析时，可以在同一环境安装 EP 插件。

```bash
conda create -n hea python=3.12 pip heasoft \
  -c https://heasarc.gsfc.nasa.gov/FTP/software/conda/ \
  -c conda-forge
conda activate hea
command -v xselect
python -c "import xspec; print(xspec.Xset.version)"
python -m pip install "jinwu[ep]"
```

每次打开新终端后先执行 `conda activate hea`。使用 Jupyter 时，选择该环境对应的 Python 内核。如果激活后找不到 HEASoft 命令，按官方说明检查初始化（`source heainit.sh`）。部分 XSPEC 模型还需要额外的数据文件。HEASoft 6.37 起使用 `ftgetmodeldata` 获取，6.35 和 6.36 使用 `xspec-data`。参见 [HEASARC Conda 安装说明](https://heasarc.gsfc.nasa.gov/docs/software/conda.html)。

### 通过 Docker 安装并配置 EP 处理软件

1. 安装并启动 Docker Engine 或 Docker Desktop，运行 `docker info` 确认当前用户能连接 Docker，运行 `docker context show` 确认正在使用的 context。下文的加载和运行使用同一个 context。
2. 获取所需仪器的镜像归档和匹配版本的 CALDB，并解压 CALDB。FXT 可从 [FXTDAS 官方下载页](https://epfxt.ihep.ac.cn/ftl/das/latest)和 [FXT CALDB 下载页](https://epfxt.ihep.ac.cn/ftl/caldb/index)获取。WXT 镜像及标定库按 [EP 任务门户](https://ep.bao.ac.cn/ep/cms/helpdesk)提供的分发途径获取。
3. 将所需仪器的配置加入 `~/.bashrc`（Zsh 用 `~/.zshrc`），将以下示例路径替换为本机实际路径。CALDB 变量指向解压后的根目录，`DASIMG` 变量在加载阶段指向镜像归档。

```bash
export WXTCALDB="$HOME/ep/caldb/wxt"
export WXTDASIMG="$HOME/ep/images/wxtdas.tar"
export FXTCALDB="$HOME/ep/caldb/fxt"
export FXTDASIMG="$HOME/ep/images/fxtsoftv2.0.img.tar.gz"
```

保存后让当前终端读取配置，然后按需加载镜像。镜像只需加载一次，升级镜像时再加载新归档。

```bash
source ~/.bashrc                     # Zsh 用户改用 source ~/.zshrc
docker load -i "$WXTDASIMG"           # 需要 WXT 时执行
docker load -i "$FXTDASIMG"           # 需要 FXT 时执行
docker image ls
```

记录 `docker load` 输出的镜像名和标签。Python 调用时通过 `image=` 传入该引用，`from_env()` 会读取对应的 CALDB 变量。也可以在载入后把 `WXTDASIMG` / `FXTDASIMG` 改成镜像引用，省略 `image=`。从 IDE 或 Jupyter 启动 Python 时，请让它继承这些环境变量，或在构造 runner 时显式传入 `image=` 和 `caldb=`。

```python
from pathlib import Path
from jinwu.ep import FXTDockerRunner, WXTDockerRunner

# 替换为实际工作目录，其中包含 Level-1 观测目录。
workspace = Path("/data/ep_work")
# 标签以 docker load 的实际输出为准，只需构造所用仪器的 runner。
fxt = FXTDockerRunner.from_env(workspace, image="fxtsoft:v2.0")
fxt.run_chain("06800001128", params={"exitstage": 2})

# WXT 示例。换成实际的观测目录、文件名前缀和 CMOS 编号。
# 多模式观测如需重写共享中间文件，可在独立输出目录下设置 clobber=True。
wxt = WXTDockerRunner.from_env(workspace, image="wxtdas:R1")
wxt.run_pipeline(
    "01709321360",
    params={"steminputs": "ep01709321360", "stemcmosid": "3"},
)
```

传给 runner 的相对路径以 `workspace` 为基准。默认输出到输入目录旁的 `<观测目录>_fxtchain` 或 `<观测目录>_wxtpipeline`，也可手动指定 `outdir=`。请将 CALDB 放在 `workspace` 之外，runner 会将 CALDB 单独挂载为只读。使用 Docker Desktop 时，确认数据和 CALDB 路径可供 Docker 共享。如果加载镜像时显式指定了 `docker --context NAME`，构造 runner 时也传入 `context="NAME"`。WXT 运行示例及参数见 [EP WXT / FXT 使用指南](docs/usage/ep_wxt_fxt.rst)。

### 源码安装

```bash
git clone https://github.com/xinxiangsun/jinwu.git
cd jinwu
pip install -e packages/jinwu            # 核心（可编辑安装）
pip install -e packages/jinwu-ep         # 按需安装仪器包
pip install -e packages/jinwu-swift      # Swift/BAT（survey 适配器可选依赖）
pip install -e packages/jinwu-fermi      # Fermi/GBM（响应生成可选依赖）
pip install -e packages/jinwu-gecam       # GECAMTools 接口，后端另装官方源码
pip install -e packages/jinwu-gw          # GW 天区与覆盖绘图
```

Swift/BAT 单目标 survey 管线位于 `jinwu.swift.bat.survey`，按“本地发现 →
可选查询/下载 → survey 光变 → 可选 mosaic → PHA/响应与 Gaussian-χ²
拟合/上限 → 报告”运行。BatAnalysis 和 HEASARC 查询只在明确启用相应选项时
延迟加载或访问网络；参见
[`packages/jinwu-swift/README.md`](packages/jinwu-swift/README.md) 的 Python
示例和命令行用法。

### 可选依赖 / Optional Extras

当前源码的 `all` 包含除 Swift 外的全部可选依赖。Swift 相关开发暂缓，
需显式选择 `[swift]`（或 `[all,swift]`）。

```bash
pip install -e './packages/jinwu[all]' -e ./packages/jinwu-gecam  # 当前源码：全部可选依赖，不含 Swift
pip install "jinwu[ep]"          # Einstein Probe 仪器包 (jinwu-ep)
pip install "jinwu[swift]"       # Swift 仪器包 (jinwu-swift, 含 swiftbat/batanalysis)
pip install "jinwu[fermi]"       # Fermi 仪器包 (jinwu-fermi, 含 astro-gdt)
pip install "jinwu[crossmatch]"  # 星表交叉证认 (astroquery/plotly/ipyaladin/regions)
pip install "jinwu[cluster]"     # 聚类分析 (pandas/seaborn/scikit-learn)
pip install "jinwu[rust]"        # Rust 加速重采样 (jinwurs)
pip install "jinwu[docs]"        # 文档构建
```

## License / 许可证

This project is licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later). See the `LICENSE` file for the full text.

本项目采用 GNU 通用公共许可证第 3 版或更高版本（GPL-3.0-or-later）授权。详见根目录的 `LICENSE` 文件。

SPDX identifier (optional for source headers): `SPDX-License-Identifier: GPL-3.0-or-later`.

### Copyright

推荐在可交互程序启动时或源文件头部加入如下声明：

```
JinWu — Copyright (C) 2025 Xinxiang Sun

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
```

或使用 SPDX 简洁标识：

```
# SPDX-License-Identifier: GPL-3.0-or-later
```

## Acknowledgments

This project incorporates code modified from [heapy](https://github.com/jyangch/heapy) (https://github.com/jyangch/heapy)

by Jun Yang, licensed under GPLv3.
