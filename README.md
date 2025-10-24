# 安装方式 / Installation

### 通过 PyPI 安装

```bash
pip install jinwu
```

# JinWu：Joint Inference for high energy transient light‑curve & spectral analysis With Unifying physical modeling

## 项目简介 / Project Introduction

金乌（JinWu）是中国古代神话中的太阳神鸟，象征着光明、能量与希望。传说中，金乌为三足乌，栖于扶桑，驾驭太阳穿行于天际，赋予万物生机。以“金乌”为名，寓意本项目致力于高能瞬变天体（如伽马暴、超新星等）的联合光变曲线与光谱物理建模与推断，探索宇宙中最明亮、最剧烈的能量释放过程。

本项目旨在为高能天体物理领域的研究者，提供统一、灵活且易于扩展的分析工具，支持多种物理模型、数据拟合与推断方法，促进科学交流与创新。

JinWu, the Golden Crow, is a legendary solar bird in ancient Chinese mythology, symbolizing light, energy, and hope. According to legend, JinWu is a three-legged bird dwelling in Fusang, driving the sun across the sky and bringing vitality to all things. Naming this project "JinWu" reflects our dedication to joint inference and physical modeling of high-energy transients (such as gamma-ray bursts and supernovae), aiming to explore the brightest and most energetic phenomena in the universe.

This project provides a unified, flexible, and extensible toolkit for researchers and enthusiasts in high-energy astrophysics, supporting various physical models, data fitting, and inference methods, and fostering scientific communication and innovation.

### 源码安装

```bash
git clone https://github.com/Charon0922/jinwu.git
cd jinwu
pip install .
```

### 必要依赖 / Required Dependencies

本包依赖以下 Python 库，请确保已安装：

- numpy
- scipy
- astropy
- emcee/pymc
- astro-gdt

### Fermi GBM Response Generator


> git clone [https://github.com/xinxiangsun/responsum.git](https://github.com/xinxiangsun/responsum.git)

> pip install ./responsum

> git clone [https://github.com/xinxiangsun/gbmgeometry.git](https://github.com/xinxiangsun/gbmgeometry.git)

> pip install ./gbmgeometry
>
> git clone [https://github.com/xinxiangsun/gbm_drm_gen.git](https://github.com/xinxiangsun/gbm_drm_gen.git)

> pip install ./gbm_drm_gen

> git clone https://github.com/xinxiangsun/hea.git

> pip install ./hea

## License / 许可证

This project is licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later). See the `LICENSE` file for the full text.

本项目采用 GNU 通用公共许可证第 3 版或更高版本（GPL-3.0-or-later）授权。详见根目录的 `LICENSE` 文件。

SPDX identifier (optional for source headers): `SPDX-License-Identifier: GPL-3.0-or-later`.

### Copyright

推荐在可交互程序启动时或源文件头部加入如下声明（请按需替换年份/程序名）：

```
JinWu Copyright (C) 2025  Xinxiang Sun
This program comes with ABSOLUTELY NO WARRANTY; for details type `show w`.
This is free software, and you are welcome to redistribute it
under certain conditions; type `show c` for details.
```

或使用 SPDX 简洁标识：

```
# SPDX-License-Identifier: GPL-3.0-or-later
```
