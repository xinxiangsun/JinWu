# 安装方式 / Installation

金乌现已拆分为一个核心包和多个仪器包，它们共享 `jinwu` 导入命名空间：

| 发行包 | 导入路径 | 内容 |
|---|---|---|
| `jinwu` | `jinwu.core`、`jinwu.lightcurve` 等 | 核心分析层（OGIP I/O、时间、拟合、背景、绘图） |
| `jinwu-ep` | `jinwu.ep` | Einstein Probe (WXT) |
| `jinwu-swift` | `jinwu.swift` | Swift/BAT |
| `jinwu-fermi` | `jinwu.fermi` | Fermi/GBM |

### 通过 PyPI 安装

```bash
pip install jinwu              # 仅核心
pip install "jinwu[ep]"        # 核心 + EP 支持
pip install "jinwu[swift]"     # 核心 + Swift 支持
pip install "jinwu[fermi]"     # 核心 + Fermi/GBM 支持
```

# JinWu：Joint Inference for high energy transient light‑curve & spectral analysis With Unifying physical modeling

## 项目简介 / Project Introduction

金乌（JinWu）是中国古代神话中的太阳神鸟，象征着光明、能量与希望。传说中，金乌为三足乌，栖于扶桑，驾驭太阳穿行于天际，赋予万物生机。以“金乌”为名，寓意本项目致力于高能瞬变天体（如伽马暴、超新星等）的联合光变曲线与光谱物理建模与推断，探索宇宙中最明亮、最剧烈的能量释放过程。

本项目旨在为高能天体物理领域的研究者，提供统一、灵活且易于扩展的分析工具，支持多种物理模型、数据拟合与推断方法，促进科学交流与创新。尤其是关于EP WXT, FXT的数据处理和产品pipeline.

JinWu, the Golden Crow, is a legendary solar bird in ancient Chinese mythology, symbolizing light, energy, and hope. According to legend, JinWu is a three-legged bird dwelling in Fusang, driving the sun across the sky and bringing vitality to all things. Naming this project "JinWu" reflects our dedication to joint inference and physical modeling of high-energy transients (such as gamma-ray bursts and supernovae), aiming to explore the brightest and most energetic phenomena in the universe.

This project provides a unified, flexible, and extensible toolkit for researchers and enthusiasts in high-energy astrophysics, supporting various physical models, data fitting, and inference methods, and fostering scientific communication and innovation.

Especially, this repo devoting on EP/WXT&FXT data products process.

### 源码安装

```bash
git clone https://github.com/Charon0922/jinwu.git
cd jinwu
pip install -e packages/jinwu            # 核心（可编辑安装）
pip install -e packages/jinwu-ep         # 按需安装仪器包
pip install -e packages/jinwu-swift      # Swift/BAT（survey 适配器可选依赖）
```

Swift/BAT 单目标 survey 管线位于 `jinwu.swift.bat.survey`，按“本地发现 →
可选查询/下载 → survey 光变 → 可选 mosaic → PHA/响应与 Gaussian-χ²
拟合/上限 → 报告”运行。BatAnalysis 和 HEASARC 查询只在明确启用相应选项时
延迟加载或访问网络；参见
[`packages/jinwu-swift/README.md`](packages/jinwu-swift/README.md) 的 Python
示例和命令行用法。

### 必要依赖 / Required Dependencies

核心依赖随 `pip install jinwu` 自动安装：

- numpy
- scipy
- astropy
- matplotlib
- pillow

谱拟合需要 HEASOFT 的 XSPEC（PyXspec）运行环境，不随 pip 安装。

### 可选依赖 / Optional Extras

```bash
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
