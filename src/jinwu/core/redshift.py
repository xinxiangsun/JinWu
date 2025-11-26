"""
红移触发外推器：基于光变曲线和能谱参数，外推最大可触发红移。

物理流程：
1. 对T100范围内能谱拟合得到平均谱参数
2. 计算转化因子 K = rate/flux（使用XspecKFactory）
3. 归一化光变轮廓 → (T/T100, 计数/平均期望计数率)
4. 对每个红移z：
   a. 计算K(z)，得到源信号期望计数率
   b. 从背景先验推断背景后验
   c. 分别生成Poisson随机数：
      - 源区域：源信号 + 背景
      - 背景区域：背景
   d. 生成N条模拟光变曲线（默认10000条）
   e. 评估SNR是否≥7
5. 二分搜索找到最大可触发红移

数据结构说明（npz文件）：
- T0_T100: [T0, T100] 事件的起止时间（MJD秒）
- time_series: 时间序列（MJD秒）
- corrected_counts_src: 源区域计数（已ARF修正）
- corrected_counts_back: 背景区域计数（已ARF修正）
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Dict, Any, Union
from pathlib import Path
import os

from jinwu.spectrum.specfake import XspecKFactory
from jinwu.lightcurve.trigger import TriggerDecider, BackgroundSimple
from jinwu.lightcurve.duration import LightcurveSNREvaluator
from jinwu.background.backprior import BackgroundPrior, BackgroundCountsPosterior
from astropy.cosmology import Planck18 as cosmo


class RedshiftTriggerExtrapolator:
    """
    红移触发外推器：通过模拟大量光变曲线，找到能触发SNR≥7的最高红移。

    核心逻辑：
    - 归一化原始光变曲线为轮廓（T/T100, 计数/平均计数率）
    - 对每个红移z，计算转化因子K(z)，缩放轮廓生成模拟曲线
    - 添加Poisson噪声和背景，评估是否能触发
    - 二分搜索找到最大z
    """

    def __init__(
        self,
        # 原始光变曲线
        time: np.ndarray,
        counts_src: np.ndarray,
        counts_back: np.ndarray,
        # 基本参数
        z0: float,
        t0: float,
        t100: float,
        area_ratio: float,
        # 谱参数
        model: str,
        params: Tuple[float, ...],
        arf: str | Path,
        rmf: str | Path,
        exposure: float ,
        back_exposure: float,
        # 背景先验
        background_prior: BackgroundPrior,
        background: Optional[str | Path] = None,
        band: Tuple[float, float] = (0.5, 4.0),
        # 触发窗口方法
        trigger_method: str = 'sliding_window',
        trigger_window: float = 1200.0,
        trigger_fraction: float = 0.9,
        # 曝光时间

    ):
        """
        初始化红移触发外推器
        
        参数：
        - time: 时间序列（秒，从某参考时刻开始）
        - counts_src: 源区域计数（已ARF修正）
        - counts_back: 背景区域计数（已ARF修正）
        - z0: 观测红移
        - t0: 事件开始时间（用于计算相对时间）
        - t100: T100持续时间（秒）
        - area_ratio: 源/背景区域面积比
        - model: XSPEC模型名称
        - params: 模型参数
        - arf, rmf: 响应文件路径
        - background_prior: 背景先验
        - background: 背景文件路径（可选）
        - band: 能段范围（keV）
        - trigger_method: 触发判断方法 ('sliding_window', 'head_window', 'cumulative_from_t0')
        - trigger_window: 触发窗口时长（秒，默认1200.0）
        - trigger_fraction: 触发所需的曲线比例（0-1之间，默认0.9即90%）
        - exposure: 源区域曝光时间（秒，默认1.0）
        - back_exposure: 背景区域曝光时间（秒，默认1.0）
        """
        # 光变曲线数据
        self.time = np.asarray(time, dtype=float)
        self.counts_src = np.asarray(counts_src, dtype=float)
        self.counts_back = np.asarray(counts_back, dtype=float)
        
        # 计算实际时间间隔（不假设均匀采样）
        if len(self.time) > 1:
            self.dt_array = np.diff(self.time)  # 实际时间间隔数组
            self.dt = float(np.median(self.dt_array))  # 中位数，用于参考
        else:
            self.dt_array = np.array([0.5])  # 默认0.5秒
            self.dt = 0.5
        
        # 基本参数
        self.z0 = float(z0)
        self.t0 = float(t0)
        self.t100 = float(t100)
        self.area_ratio = float(area_ratio)

        # 谱配置
        self.model = model
        self.params = params
        self.arf = str(arf)
        self.rmf = str(rmf)
        self.background = str(background) if background else None
        self.band = band
        self.exposure = float(exposure)
        self.back_exposure = float(back_exposure)

        # 触发参数
        if trigger_method not in ['sliding_window', 'head_window', 'cumulative_from_t0']:
            raise ValueError(f"trigger_method必须是 'sliding_window', 'head_window' 或 'cumulative_from_t0'，得到: {trigger_method}")
        self.trigger_method = trigger_method
        self.trigger_window = float(trigger_window)
        if not 0.0 < trigger_fraction <= 1.0:
            raise ValueError(f"trigger_fraction必须在(0, 1]范围内，得到: {trigger_fraction}")
        self.trigger_fraction = float(trigger_fraction)

        # 背景
        self.background_prior = background_prior
        
        # 从背景先验推断后验（用于生成背景Poisson随机数）
        # 这里使用先验的n_off和t_off作为后验的Gamma参数
        self._background_posterior = BackgroundCountsPosterior(
            a_total=self.background_prior.n_off_prior,
            b=self.background_prior.t_off,
            area_ratio=self.area_ratio
        )

        # K工厂
        self._k_factory = XspecKFactory()

        # 归一化轮廓（延迟计算）
        self._normalized_profile: Optional[Tuple[np.ndarray, np.ndarray]] = None
        
        # 解析模型结构以找到norm和alpha参数的位置
        self._parse_model_structure()
        
    @classmethod
    def from_npz(
        cls,
        npz_path: str | Path,
        # 谱参数
        model: str,
        params: Tuple[float, ...],
        arf: str | Path,
        rmf: str | Path,
        # 基本参数
        z0: float,
        exposure: float ,
        back_exposure: float,
        area_ratio: float = 1/12,
        # 背景先验
        background_prior: Optional[BackgroundPrior] = None,
        background: Optional[str | Path] = None,
        band: Tuple[float, float] = (0.5, 4.0),
        # 触发窗口方法
        trigger_method: str = 'sliding_window',
        trigger_window: float = 1200.0,
        trigger_fraction: float = 0.9,
        # 曝光时间
        
    ) -> 'RedshiftTriggerExtrapolator':
        """
        从npz文件创建RedshiftTriggerExtrapolator实例
        
        参数：
        - npz_path: 光变曲线npz文件路径
        - model: XSPEC模型名称
        - params: 模型参数
        - arf, rmf: 响应文件路径
        - z0: 观测红移
        - area_ratio: 源/背景区域面积比（默认1/12）
        - background_prior: 背景先验（如果None，使用默认）
        - background: 背景文件路径（可选）
        - band: 能段范围（keV）
        - trigger_method: 触发判断方法 ('sliding_window', 'head_window', 'cumulative_from_t0')
        - trigger_window: 触发窗口时长（秒，默认1200.0）
        - trigger_fraction: 触发所需的曲线比例（0-1之间，默认0.9即90%）
        - exposure: 源区域曝光时间（秒，默认1.0）
        - back_exposure: 背景区域曝光时间（秒，默认1.0）
        
        返回：
        - RedshiftTriggerExtrapolator实例
        """
        # 读取npz文件
        data = np.load(npz_path)
        
        # 提取光变曲线数据
        time_series = data['time_series']
        counts_src = data['corrected_counts_src']
        counts_back = data['corrected_counts_back']
        
        # 提取T0和T100
        t0_t100 = data['T0_T100']
        t0 = float(t0_t100[0])
        t100_time = float(t0_t100[1])
        t100_duration = t100_time - t0
        
        # 如果没有提供背景先验，使用默认
        if background_prior is None:
            from jinwu.background.backprior import BackgroundPrior
            background_prior = BackgroundPrior.from_epwxt_background_default()
        
        # 创建实例
        return cls(
            time=time_series - t0,  # 转换为相对时间
            counts_src=counts_src,
            counts_back=counts_back,
            z0=z0,
            t0=t0,
            t100=t100_duration,
            area_ratio=area_ratio,
            model=model,
            params=params,
            arf=arf,
            rmf=rmf,
            background_prior=background_prior,
            background=background,
            band=band,
            trigger_method=trigger_method,
            trigger_window=trigger_window,
            trigger_fraction=trigger_fraction,
            exposure=exposure,
            back_exposure=back_exposure,
        )

    def _normalize_lightcurve_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        归一化光变曲线为轮廓：T/T100, 计数率/平均计数率
        
        物理流程（更新后）：
        1. 计算净光变：源区域 - 背景区域×area_ratio
        2. 通过 K(z0) 将净计数率转为 flux 光变
        3. 归一化并插值：得到平滑的 flux 轮廓（T/T100, flux/平均flux）
        4. 重建计数率轮廓：flux轮廓 × 平均flux × K(z0) → 归一化计数率
        
        返回：
        - (t_norm, counts_norm): T/T100, 计数率/平均计数率
        """
        if self._normalized_profile is not None:
            return self._normalized_profile

        # 找到T100范围内的数据点
        mask = self.time <= self.t100
        if not np.any(mask):
            mask = np.ones(len(self.time), dtype=bool)
        
        time_t100 = self.time[mask]
        counts_src_t100 = self.counts_src[mask]
        counts_back_t100 = self.counts_back[mask]
        
        # 1. 计算净光变（源 - 背景×area_ratio）
        net_counts = counts_src_t100 - counts_back_t100 * self.area_ratio
        
        # 计算对应的时间间隔
        # dt_array[i] 是 time[i+1] - time[i]，对应第 i 个 bin 的宽度
        # 对于 T100 范围内的数据，我们需要相应的 dt
        if len(time_t100) > 1:
            dt_t100 = np.diff(time_t100)
            # 为了匹配 counts 的长度，最后一个 bin 使用倒数第二个 dt
            dt_t100 = np.append(dt_t100, dt_t100[-1])
        else:
            dt_t100 = np.array([self.dt])
        
        # 净计数率：使用实际的时间间隔
        net_rate = net_counts / dt_t100
        
        # 2. 计算 K(z0) 转化因子（使用z0对应的参数）
        from jinwu.lightcurve.lcfake import XspecConfig
        
        # 为z0调整参数（虽然z0时norm不变，但保持一致性）
        params_z0 = self._adjust_params_for_redshift(self.z0)
        
        cfg = XspecConfig(
            arf=self.arf,
            rmf=self.rmf,
            background=self.background,
            model=self.model,
            params=params_z0,
            band=self.band,
            exposure=self.exposure,
            back_exposure=self.back_exposure,
            xspec_abund='wilm',
            xspec_xsect='vern',
            xspec_cosmo='67.66 0 0.6888463055445441',
            allow_prompting=False
        )
        # 使用 get_K_with_values 获取 K(z0), rate(z0), flux(z0)
        K_z0, rate_z0, flux_z0 = self._k_factory.get_K_with_values(
            arf=cfg.arf,
            rmf=cfg.rmf,
            background=cfg.background,
            model=cfg.model,
            params=cfg.params,
            band=cfg.band,
            exposure=cfg.exposure,
            back_exposure=cfg.back_exposure,
            xspec_abund=cfg.xspec_abund,
            xspec_xsect=cfg.xspec_xsect,
            xspec_cosmo=cfg.xspec_cosmo,
            allow_prompting=cfg.allow_prompting,
        )
        
        # 3. 将净计数率转为 flux 光变
        # 注意：XSPEC 给出的 flux_z0 是平均 flux，而 net_rate 是时间变化的
        # 所以我们仍然需要通过 K_z0 转换
        flux_lc = net_rate / K_z0  # erg/cm²/s
        
        # 4. 计算平均 flux（用于归一化）
        # 这里的 mean_flux 应该接近 flux_z0（如果没有很大的时间变化）
        mean_flux = flux_z0
        
        # 5. 归一化 flux 轮廓并插值（使其平滑）
        t_norm_raw = time_t100 / self.t100
        flux_norm_raw = flux_lc / mean_flux
        
        # 插值到更密集的网格（可选，使轮廓更平滑）
        # 这里使用原始采样，如需更平滑可以增加插值点
        t_norm = t_norm_raw
        flux_norm = flux_norm_raw
        
        # 6. 重建计数率轮廓：flux轮廓 × 平均flux × K(z0)
        # 这样得到的是归一化的计数率（相对于平均计数率）
        counts_rate_reconstructed = flux_norm * mean_flux * K_z0
        mean_rate_reconstructed = mean_flux * K_z0
        counts_norm = counts_rate_reconstructed / mean_rate_reconstructed
        
        # 缓存结果和中间参数
        self._normalized_profile = (t_norm, counts_norm)
        self._mean_flux_z0 = mean_flux  # 平均flux [erg/cm²/s]
        self._flux_z0 = flux_z0  # XSPEC计算的flux(z0) [erg/cm²/s]
        self._rate_z0 = rate_z0  # XSPEC计算的rate(z0) [cts/s]
        self._K_z0 = K_z0  # K(z0) [cts/(erg/cm²)]
        self._mean_rate_z0 = mean_rate_reconstructed  # 平均rate [cts/s]
        
        return self._normalized_profile
    
    def _parse_model_structure(self):
        """解析XSPEC模型结构，找到norm参数位置和alpha参数值"""
        # 临时初始化XSPEC以解析模型
        from jinwu.core.heasoft import HeasoftEnvManager
        import xspec as xs
        
        try:
            hem = HeasoftEnvManager()
            hem.init_heasoft()
            
            if not hem.is_heasoft_initialized():
                raise RuntimeError("HEASoft环境未初始化")
            
            # 设置XSPEC环境
            xs.AllData.clear()
            xs.AllModels.clear()
            xs.Xset.abund = 'wilm'
            xs.Xset.xsect = 'vern'
            xs.Xset.cosmo = '67.66 0 0.6888463055445441'
            xs.Xset.allowPrompting = False
            
            # 创建临时模型
            m = xs.Model(self.model)
            
            # 收集所有参数
            param_idx = 0
            self._norm_param_idx = None
            self._alpha_param_idx = None
            self._norm0_base = None
            self._alpha_base = None
            
            for comp_name in m.componentNames:
                comp = getattr(m, comp_name)
                for param_name in comp.parameterNames:
                    # 检查是否是norm参数（通常在最后一个分量）
                    if param_name.lower() == 'norm' and comp_name == m.componentNames[-1]:
                        self._norm_param_idx = param_idx
                        if param_idx < len(self.params):
                            self._norm0_base = float(self.params[param_idx])
                    
                    # 检查是否是谱指数参数
                    if param_name.lower() in ('phoindex', 'index', 'alpha'):
                        self._alpha_param_idx = param_idx
                        if param_idx < len(self.params):
                            self._alpha_base = float(self.params[param_idx])
                    
                    param_idx += 1
            
            # 清理
            xs.AllData.clear()
            xs.AllModels.clear()
            
        except Exception as e:
            # 如果解析失败，使用默认值
            print(f"警告: 无法解析模型结构: {e}")
            self._norm_param_idx = None
            self._alpha_param_idx = None
            self._norm0_base = None
            self._alpha_base = None
    
    def _adjust_params_for_redshift(self, z: float) -> Tuple[float, ...]:
        """根据红移调整模型参数（主要是norm参数）
        
        参数：
        - z: 目标红移
        
        返回：
        - 调整后的参数元组
        """
        if self._norm_param_idx is None or self._norm0_base is None or self._alpha_base is None:
            # 无法调整，返回原参数
            return self.params
        
        # 计算norm缩放因子
        try:
            dc0 = cosmo.comoving_distance(self.z0).value
            dcz = cosmo.comoving_distance(z).value
            factor = (dc0 / dcz) ** 2
            
            # 计算新的norm值
            norm_new = self._norm0_base * ((1 + self.z0) / (1 + z)) ** self._alpha_base * factor
            
            # 复制参数并更新norm
            params_list = list(self.params)
            params_list[self._norm_param_idx] = norm_new
            
            return tuple(params_list)
        except Exception as e:
            print(f"警告: 调整norm参数失败: {e}")
            return self.params
    
    def get_lightcurve_info(self) -> Dict[str, Any]:
        """
        获取光变曲线的基本信息
        
        返回：
        - dict: 包含光变曲线统计信息的字典
        """
        total_src = float(np.sum(self.counts_src))
        total_back = float(np.sum(self.counts_back))
        total_time = float(self.time[-1] - self.time[0]) if len(self.time) > 1 else float(self.dt)
        
        return {
            'z0': self.z0,
            't0': self.t0,
            't100': self.t100,
            'duration': float(self.time.max() - self.time.min()),
            'n_bins': len(self.time),
            'dt': self.dt,
            'area_ratio': self.area_ratio,
            'total_src_counts': total_src,
            'total_back_counts': total_back,
            'total_net_counts': total_src - total_back * self.area_ratio,
            'mean_src_rate': total_src / total_time,
            'mean_back_rate': total_back / total_time,
            'snr_estimate': total_src / np.sqrt(total_back * self.area_ratio) if total_back > 0 else np.inf,
        }
    
    def plot_lightcurve(self, ax=None, show_background=True):
        """
        绘制光变曲线
        
        参数：
        - ax: matplotlib axis对象（如果None，创建新图）
        - show_background: 是否显示背景
        
        返回：
        - ax: matplotlib axis对象
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # 计算计数率（使用实际时间间隔）
        # 对于第 i 个 bin，使用 dt_array[i] 作为时间间隔
        # 但由于 counts 是 bin的计数，而 dt_array 是 bin之间的间隔
        # 我们使用中位数 dt 作为近似，或者将 dt_array 扩展一个元素
        if len(self.dt_array) == len(self.counts_src) - 1:
            dt_for_rate = np.append(self.dt_array, self.dt_array[-1])
        elif len(self.dt_array) == len(self.counts_src):
            dt_for_rate = self.dt_array
        else:
            dt_for_rate = np.full(len(self.counts_src), self.dt)
        
        src_rate = self.counts_src / dt_for_rate
        back_rate = self.counts_back / dt_for_rate
        
        # 绘制源计数率
        ax.step(self.time, src_rate, where='mid', label='Source', color='black', linewidth=1.5)
        
        if show_background:
            # 绘制背景计数率（缩放到源区域面积）
            ax.step(self.time, back_rate * self.area_ratio, where='mid', 
                   label='Background (scaled)', color='red', linewidth=1, alpha=0.7)
        
        # 标记T100
        ax.axvline(self.t100, color='blue', linestyle='--', linewidth=1.5, 
                  label=f'T100 = {self.t100:.1f} s', alpha=0.7)
        
        ax.set_xlabel('Time since T0 (s)', fontsize=12)
        ax.set_ylabel('Count Rate (cts/s)', fontsize=12)
        ax.set_title(f'Light Curve (z = {self.z0})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax

    def _generate_lightcurve_at_redshift(
        self,
        z: float,
        n_curves: int = 10000,
        rng: Optional[np.random.Generator] = None
    ) -> list[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        在给定红移z生成n_curves条模拟光变曲线
        
        物理流程（更新后）：
        1. 获取归一化轮廓 (T/T100, 计数率/平均计数率)
           - 该轮廓已经通过 flux 插值重建，确保平滑
        2. 计算K(z)转化因子
        3. 红移缩放：时间 × (1+z)/(1+z0)
        4. 计算源信号期望计数率：
           - 归一化轮廓 × 平均计数率(z0) × K(z)/K(z0)
        5. 从背景后验采样背景计数率
        6. 对每条曲线分别 Poisson 采样：
           - 源区域 = Poisson(源信号) + Poisson(ON背景)
           - 背景区域 = Poisson(OFF背景)
        
        参数：
        - z: 红移
        - n_curves: 生成曲线数量（默认10000）
        - rng: 随机数生成器
        
        返回：
        - list of (time, counts_on, counts_off) tuples
        """
        from jinwu.lightcurve.lcfake import generate_redshift_lightcurves, XspecConfig
        
        if rng is None:
            rng = np.random.default_rng()

        # 1. 获取归一化轮廓（已经通过 flux 重建）
        t_norm, counts_norm = self._normalize_lightcurve_profile()
        
        # 2. 从缓存中获取平均计数率和K(z0)（如果没有缓存则重新计算）
        if not hasattr(self, '_mean_rate_z0') or not hasattr(self, '_K_z0'):
            # 重新触发归一化以计算缓存值
            self._normalized_profile = None
            t_norm, counts_norm = self._normalize_lightcurve_profile()
        
        mean_rate_z0 = self._mean_rate_z0
        K_z0 = self._K_z0
        
        # 3. 为目标红移z调整参数（主要是norm参数）
        params_z = self._adjust_params_for_redshift(z)
        
        # 4. 创建 XSPEC 配置对象
        cfg = XspecConfig(
            arf=self.arf,
            rmf=self.rmf,
            background=self.background,
            model=self.model,
            params=params_z,
            band=self.band,
            exposure=self.exposure,
            back_exposure=self.back_exposure,
            xspec_abund='wilm',
            xspec_xsect='vern',
            xspec_cosmo='67.66 0 0.6888463055445441',
            allow_prompting=False
        )
        
        # 5. 调用统一的光变生成函数
        # 注意：dt 只是一个参考值，实际的时间间隔由 t_norm 和 t100 决定
        return generate_redshift_lightcurves(
            time_norm=t_norm,
            counts_norm=counts_norm,
            z0=self.z0,
            t100=self.t100,
            dt=self.dt,  # 传递中位数作为参考
            mean_rate_z0=mean_rate_z0,
            K_z0=K_z0,
            z=z,
            background_posterior=self._background_posterior,
            cfg=cfg,
            n_curves=n_curves,
            rng=rng
        )

    def _can_trigger_at_redshift(self, z: float, n_mc: int = 10000) -> bool:
        """
        检查在红移z是否能触发（SNR≥7）- 使用可配置的触发窗口策略
        
        参数：
        - z: 红移
        - n_mc: 蒙特卡洛模拟数量（默认10000）
        
        返回：
        - bool: 是否至少有90%的曲线能触发
        """
        # 生成n_mc条模拟光变曲线
        lightcurves = self._generate_lightcurve_at_redshift(z, n_curves=n_mc)

        # 检查每条曲线是否能触发
        triggered_count = 0
        for time, counts_on, counts_off in lightcurves:
            # 创建BackgroundSimple对象（使用OFF区域的实际计数）
            # 这里t_off_ref是整个光变曲线的时长
            t_duration = time[-1] - time[0]
            n_off_total = float(np.sum(counts_off))
            
            bg_simple = BackgroundSimple(
                area_ratio=self.area_ratio,
                t_off_ref=t_duration,
                n_off_ref=n_off_total
            )

            # 创建TriggerDecider（使用ON区域计数）
            # 计算实际的dt（如果时间数组长度>1）
            if len(time) > 1:
                dt_actual = float(np.median(np.diff(time)))
            else:
                dt_actual = self.dt
            
            decider = TriggerDecider.from_counts(
                time=time,
                counts=counts_on,
                dt=dt_actual,
                bg=bg_simple
            )
            
            # 根据配置的方法进行触发判断
            if self.trigger_method == 'sliding_window':
                triggered, info = decider.sliding_window(window=self.trigger_window)
            elif self.trigger_method == 'head_window':
                triggered, info = decider.head_window(window=self.trigger_window)
            elif self.trigger_method == 'cumulative_from_t0':
                triggered, info = decider.cumulative_from_t0(target=7.0)
            else:
                raise ValueError(f"未知的触发方法: {self.trigger_method}")
            
            if triggered:
                triggered_count += 1

        # 检查是否达到要求的触发比例
        actual_fraction = triggered_count / n_mc
        print(f"  z={z:.3f}: {triggered_count}/{n_mc} ({actual_fraction*100:.1f}%) triggered ({self.trigger_method}, window={self.trigger_window}s, require≥{self.trigger_fraction*100:.0f}%)")
        return actual_fraction >= self.trigger_fraction

    def extrapolate_max_redshift(
        self,
        z_min: float = 0.001,
        z_max: float = 20.0,
        tol: float = 0.01,
        n_mc_per_check: int = 1000
    ) -> Dict[str, Any]:
        """
        二分搜索找到最大可触发红移
        
        参数：
        - z_min: 最小搜索红移（默认0.01）
        - z_max: 最大搜索红移（默认10.0）
        - tol: 红移搜索精度（默认0.01）
        - n_mc_per_check: 每次检查的蒙特卡洛模拟数量（默认1000，可提高到10000以获得更准确结果）

        返回：
        - max_z: 最大红移
        - can_trigger_at_max: 是否在max_z能触发
        - search_history: 搜索历史
        """
        search_history = []

        left, right = z_min, z_max
        max_z = z_min

        print(f"搜索范围: z ∈ [{z_min:.3f}, {z_max:.3f}]")

        while right - left > tol:
            mid = (left + right) / 2
            print(f"\n检查 z = {mid:.3f} (区间: [{left:.3f}, {right:.3f}])")
            can_trigger_mid = self._can_trigger_at_redshift(mid, n_mc=n_mc_per_check)

            search_history.append({
                'z': mid,
                'can_trigger': can_trigger_mid,
                'interval': (left, right)
            })

            if can_trigger_mid:
                # 能触发，尝试更高
                max_z = mid
                left = mid
                print(f"  ✓ 可触发！向上搜索...")
            else:
                # 不能触发，降低上限
                right = mid
                print(f"  ✗ 不能触发，向下搜索...")

        print("\n" + "=" * 60)
        print(f"搜索完成！最大可触发红移: z_max = {max_z:.3f}")
        print(f"总共迭代 {len(search_history)} 次")
        print("=" * 60)

        return {
            'max_z': max_z,
            'can_trigger_at_max': self._can_trigger_at_redshift(max_z, n_mc=n_mc_per_check*2),
            'search_history': search_history
        }