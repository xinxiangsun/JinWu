"""狭义相对论运动学与辐射变换工具类（实验性）。

当前已实现并有测试覆盖（test/test_gr.py）：
- 速度 v 的单位化存储：接受 astropy Quantity（自动换算到 m/s），
  裸数值按 m/s 解释；必须满足 0 <= v < c。
- beta = v/c、洛伦兹因子 gamma = 1/sqrt(1 - beta^2) 的解析实现。
- 时间膨胀 time_dilation 与长度收缩 length_contraction。

尚未实现、不要用于科学计算的功能：
- 宇宙学红移与 GR 度规效应。show_* 系列方法仅做 LaTeX 公式展示，
  不参与数值计算。
"""

from astropy.constants import c as _c

import astropy.units as u
import numpy as np

# IPython 是可选依赖，仅 show_* 公式展示需要；缺失时回退为纯文本打印，
# 保证 import jinwu.physics 不依赖 Jupyter 环境（如 RTD 文档构建）。
try:
    from IPython.display import Math, display
except ImportError:  # pragma: no cover - 仅在无 IPython 的精简环境触发
    Math = None
    display = None

#: 真空光速 (m/s)，用于 beta = v/c 与适用边界检查
C_M_S = float(_c.to_value(u.meter / u.second))


def _display_math(latex: str) -> None:
    """优先用 IPython 富显示，缺 IPython 时打印 LaTeX 源。"""
    if display is not None and Math is not None:
        display(Math(latex))
    else:
        print(latex)


class GeneralRelativity:
    """基础相对论/多普勒与辐射变换工具类（实验性）。

    - ``v``: astropy Quantity（任意速度单位，换算到 m/s）或裸数值（m/s）；
    - ``beta``: v/c（解析值，非占位）；
    - ``lorentz_factor``: 1/sqrt(1 - beta^2)；
    - ``show_*``: 仅公式展示，不做计算。
    宇宙学红移等 GR 效应尚未实现。
    """

    def __init__(self):
        self._v = None

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, value):
        if hasattr(value, 'unit'):
            velocity = value.to(u.meter / u.second)
        else:
            velocity = float(value) * u.meter / u.second
        if velocity.value < 0:
            raise ValueError("速度必须大于等于0")
        if velocity.value >= C_M_S:
            raise ValueError("速度必须小于光速 c")
        self._v = velocity

    # 方法：狭义相对论运动学：时间膨胀 Δt = γ·Δt_rest（固有时与坐标时之比），
    #       长度收缩 L = L_rest/γ（沿运动方向）；
    #       关键式：γ = 1/sqrt(1 - beta^2)，beta = v/c
    # 参考：MTW, Misner, Thorne & Wheeler, 1973, "Gravitation" (W.H. Freeman) §2；
    #       Rindler, 2006, "Relativity: Special, General, and Cosmological" (Oxford) Ch.3
    def time_dilation(self, t_rest, frame_from="静止系", frame_to="运动系"):
        result = self.lorentz_factor * t_rest
        print(f"时间膨胀: 从 {frame_from} 到 {frame_to}，输入 {t_rest}，输出 {result}")
        return result

    def length_contraction(self, l_rest, frame_from="静止系", frame_to="运动系"):
        result = l_rest / self.lorentz_factor
        print(f"长度收缩: 从 {frame_from} 到 {frame_to}，输入 {l_rest}，输出 {result}")
        return result

    @property
    def beta(self):
        if self._v is None:
            raise ValueError("速度未设置")
        return float(self._v.to_value(u.meter / u.second)) / C_M_S

    @property
    def lorentz_factor(self):
        beta = self.beta
        return 1.0 / np.sqrt(1.0 - beta ** 2)

    @classmethod
    def show_formula(cls, formula_type="all"):
        # 方法：相对论多普勒/像变换公式表（展示用，不参与计算）。
        #       多普勒因子 D = 1/[gamma*(1 - beta*cos(theta))] = gamma*(1 + beta*cos(theta'))；
        #       光行差 cos(theta) = (cos(theta') + beta)/(1 + beta*cos(theta'))，
        #       dcos(theta) = D^-2 dcos(theta')，d(Omega) = D^-2 d(Omega')；
        #       不变量 I_nu/nu^3 给出 I_nu = D^3 I'_nu'；辐射转移自洽要求 ds = D ds'。
        #       各式已由光行差公式与 I_nu/nu^3 不变性逐条数值核验。
        # 参考：Rybicki & Lightman, 1979, "Radiative Processes in Astrophysics"
        #       (Wiley-Interscience) §4（I_nu/nu^3 不变性与相对论变换）；
        #       Urry & Padovani, 1995, PASP 107, 803 (doi:10.1086/133758)（多普勒 boosting）；
        #       内激波时标比 1 : 2*gamma^2 : 2*gamma : 1 见 Piran, 2004, Rev. Mod. Phys. 76, 1143
        #       (arXiv:astro-ph/0405503)。
        formulas = {
            "lorentz": r"\text{洛伦兹因子:}\quad \gamma = \frac{1}{\sqrt{1-\beta^2}}",
            "doppler": (r"\text{Doppler因子:}\quad "
                        r"\mathcal{D} = \frac{1}{\gamma (1 - \beta \cos\theta)}"
                        r"= \gamma (1 + \beta \cos\theta')"),
            "volume": r"\text{体积变换:}\quad ds = \mathcal{D}\,ds',\quad dV = D\,dV'",
            "length": r"\text{长度变换:}\quad ds = \mathcal{D}\,ds'",
            "time": r"\text{时间变换:}\quad dt = \mathcal{D}^{-1} \,dt'",
            "energry": r"\text{能量变换:}\quad E = \mathcal{D}E'",
            "dcos_theta": (r"\text{微分余弦变换:}" r"\quad d\cos\theta = \frac{d\cos\theta'}{\gamma^2(1 + \beta\cos\theta')^2} = D^{-2} d\cos\theta'"),
            "cos_theta": r"\text{余弦变换:}\quad \cos\theta = \frac{\cos\theta' + \beta}{1 + \beta\cos\theta'}",
            "sin_theta": r"\text{正弦变换:}\quad \sin\theta = \frac{\sin\theta'}{\gamma(1 + \beta\cos\theta')}",
            "tan_theta": r"\text{正切变换:}\quad \tan\theta = \frac{\sin\theta'}{\gamma(\cos\theta' + \beta)}",
            "solid_angle": r"\text{立体角变换:}\quad d\Omega = \mathcal{D}^{-2} d\Omega'",
            "time_ratio_simple": (r"\Delta t_{\text{eng}} : \Delta t_e : \Delta t_e' : \Delta t_{\text{obs}} \simeq 1 : 2\gamma^2 : 2\gamma : 1."),
            "time_ratio_full": (r"\Delta t_{\text{eng}} : \Delta t_e : \Delta t_e' : \Delta t_{\text{obs}} = "
                                 r"\frac{1-\beta}{1-\beta\cos\theta} : \frac{1}{1-\beta\cos\theta} : \frac{1}{\gamma(1-\beta\cos\theta)} : 1."),
            "tobs_teng": (r"\Delta t_{\text{obs}} = \frac{1-\beta\cos\theta}{1-\beta} \Delta t_{\text{eng}}."),
            "intensity": (r"\text{辐射强度变换:}\quad I_\nu(\nu) = \mathcal{D}^3 I'_{\nu'}(\nu')"),
        }
        header = r"\text{带'}\text{的是共动系，不带的是近邻观测者系}\\"
        note = r"\text{尤其需要特别注意的事情是: 近邻观测者系仍然需要经过宇宙学的变换才能得到观测的结果}"
        note2 = r"\text{另外由于视超光速效应,引擎系下两束光的间隔在辐射过程中会导致间隔观测到的信号间隔变短,这完全不涉及相对论}"
        if formula_type == "all":
            _display_math(header)
            _display_math(note)
            for key in formulas:
                _display_math(formulas[key])
        else:
            _display_math(header)
            _display_math(note)
            _display_math(formulas.get(formula_type, r"\text{未知公式类型}"))

    @classmethod
    def show_radiation_transform(cls, formula_type="all"):
        """展示常用辐射变换公式"""
        # 方法：共动系 ↔ 观测者系的辐射量变换（展示用）：不变量 I_nu/nu^3 与
        #       j_nu/nu^2、alpha_nu*nu 的变换给出 I_nu = D^3 I'_nu'、j_nu = D^2 j'_nu'、
        #       alpha_nu = D^-1 alpha'_nu'；真实（束流内）光度 L_nu = D L'_nu'、
        #       L = D^2 L'；各向同性等效光度 L_nu,iso = D^3 L'_nu'、L_iso = D^4 L'
        #       （因 nu*F_nu 不变量与 d(Omega) = D^-2 d(Omega')）；观测能流
        #       F_nu = (1+z)*D^3*j'_nu'*V'/D_L^2（含 Hogg 宇宙学 (1+z) 因子）。
        # 参考：Rybicki & Lightman, 1979, "Radiative Processes in Astrophysics"
        #       (Wiley-Interscience) §4.3-4.4；
        #       Urry & Padovani, 1995, PASP 107, 803 (doi:10.1086/133758)；
        #       Hogg, 1999, MNRAS 310, 940 (arXiv:astro-ph/9905116)（f_nu 与 L_nu 的宇宙学变换）；
        #       Sari, Piran & Narayan, 1998, ApJ 497, L17 (doi:10.1086/311269)（GRB 余辉中的用法）。
        formulas = {
            "flux1": (r"F_\nu(\nu_{\text{obs}}) = \frac{(1+z)\mathcal{D}^3 j'_{\nu'}(\nu')V'}{D_L^2}."),
            "flux2": (r"F_\nu(\nu_{\text{obs}}) = \frac{(1+z)L_{\nu,\text{iso}}(\nu)}{4\pi D_L^2},"),
            "l_iso": (r"L_{\text{iso}}(\nu) = \nu L_{\nu,\text{iso}}(\nu) = \mathcal{D}^4 (\nu' L'_{\nu'}(\nu'))."),
            "l_nu_iso": (r"L_{\nu,\text{iso}}(\nu) = \mathcal{D}^3 L'_{\nu'}(\nu')."),
            "l_nu": (r"L_\nu(\nu) = \mathcal{D} L'_{\nu'}(\nu')."),
            "l": (r"L(\nu) = \mathcal{D}^2 L'_{\nu'}(\nu')."),
            "intensity": (r"I_\nu(\nu) = \mathcal{D}^3 I'_{\nu'}(\nu'),"),
            "emissivity": (r"j_\nu(\nu) = \mathcal{D}^2 j'_{\nu'}(\nu'),"),
            "absorption": (r"\alpha_\nu(\nu) = \mathcal{D}^{-1} \alpha'_{\nu'}(\nu')."),
        }
        header = r"\text{带'}\text{的是共动系，不带的是近邻观测者系}\\"
        if formula_type == "all":
            _display_math(header)
            for key in formulas:
                _display_math(formulas[key])
        else:
            _display_math(header)
            _display_math(formulas.get(formula_type, r"\text{未知公式类型}"))

    @classmethod
    def show_grmhd_equations(cls):
        """显示理想磁流体的GRMHD方程组（MHD守恒形式）"""
        eqs = [
            r"\frac{\partial (\gamma \rho)}{\partial t} + \nabla \cdot (\gamma \rho \mathbf{v}) = 0",
            r"\frac{\partial}{\partial t} \left( \frac{\gamma^2 h}{c^2} \mathbf{v} + \frac{\mathbf{E} \times \mathbf{B}}{4\pi c} \right)"\
            r"+ \nabla \cdot \left[ \frac{\gamma^2 h}{c^2} \mathbf{v} \otimes \mathbf{v} + \left( p + \frac{E^2 + B^2}{8\pi} \right) \mathbf{I} - \frac{\mathbf{E} \otimes \mathbf{E} + \mathbf{B} \otimes \mathbf{B}}{4\pi} \right] = 0",
            r"\frac{\partial}{\partial t} \left( \gamma^2 h - p - \gamma \rho c^2 + \frac{B^2 + E^2}{8\pi} \right)"\
            r"+ \nabla \cdot \left[ (\gamma^2 h - \gamma \rho c^2) \mathbf{v} + \frac{c}{4\pi} \mathbf{E} \times \mathbf{B} \right] = 0",
            r"\frac{\partial \mathbf{B}}{\partial t} + c \nabla \times \mathbf{E} = 0"
        ]
        _display_math(r"注意方程组中\otimes表示张量积,通过假设E=B=0, GRMHD方程可以演化为一般的广义相对论流体力学方程")
        for eq in eqs:
            _display_math(eq)
