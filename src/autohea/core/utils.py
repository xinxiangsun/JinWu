'''
Date: 2025-05-30 17:43:59
LastEditors: Xinxiang Sun sunxx@nao.cas.cn
LastEditTime: 2025-10-10 11:56:03
LastEditTime: 2025-09-25 20:34:19
FilePath: /research/autohea/src/autohea/core/utils.py
'''
import numpy as np
from autohea.core.heasoft import HeasoftEnvManager as hem
import xspec
try :
    import soxs
    from soxs import ConvolvedSpectrum
except:
    pass

from pathlib import Path
import matplotlib.pyplot as plt
from autohea.core.file import ArfReader, RmfReader, RspReader
from astropy import units as u
from astropy import constants as const
from IPython.display import display, Math, Latex
from astropy.cosmology import Planck18 as cosmo
from functools import lru_cache
import os


def generate_download_url(isot_time):
    """
    根据给定的 isot (YYYY-MM-DDTHH:MM:SS) 时间生成 GBM poshist 文件的下载 URL。

    参数:
    - isot_time (str): ISOT 格式时间字符串，例如 "2024-01-01T12:00:00"

    返回:
    - url (str): 生成的 poshist 文件下载 URL
    """
    # 解析时间

    # 提取年份、月份、日期
    year = isot_time.strftime('%y')
    yr2 = isot_time.datetime.year
    month = f"{isot_time.datetime.month:02d}"  # 两位数格式
    day = f"{isot_time.datetime.day:02d}"

    # 生成文件名
    filename = f"glg_poshist_all_{year}{month}{day}_v00.fit"

    # 生成完整的下载路径
    # https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/daily/2025/01/01/current/
    # url = f"https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/daily/{yr2}/{isot_time.strftime('%m/%d/')}current/{filename}"
    url = f"https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/daily/{yr2}/{isot_time.strftime('%m/%d/')}current"
    return url



def snr_li_ma(n_src, n_bkg, alpha_area_time):
    """
    Calculate the signal-to-noise ratio (SNR) using the Li & Ma formula.

    Parameters:
    n_src (int): 源区域的计数
    n_bkg (int): 背景区域的计数
    alpha_area_time (float): 	•	\alpha：背景区域与源区域之间的归一化因子，反映暴露时间或面积比：
    \alpha_area_time = \frac{t_{\text{on}} A_{\text{on}}}{t_{\text{off}} A_{\text{off}}}

    Returns:
    float: The calculated SNR.
    """
    if n_bkg == 0:
        return np.inf  # Avoid division by zero, return infinity if no background counts
    part1 = n_src*np.log((1 + alpha_area_time) * n_src / alpha_area_time /(n_bkg+n_src))
    part2 = n_bkg*np.log((1+alpha_area_time)*n_bkg/(n_bkg+n_src))
    snr = np.sqrt(2 * (part1 + part2))
    return snr


class RedshiftExtrapolator():
    
    def __init__(self, z0, bkgnum, duration, model, par, arfpath: Path | str, rmfpath: Path | str | None = None, area_ratio: float = 1/12):
        '''红移外推器 - 基于原有代码保守重构'''
        # 基本参数
        self._z0 = float(z0)
        self._model = str(model)
        self._par = list(par)
        self._duration = float(duration)
        self._bkgnum = float(bkgnum)
        self._area_ratio = float(area_ratio)
        
        # 文件路径
        self._arfpath = Path(arfpath).expanduser().resolve()
        self._rmfpath = Path(rmfpath).expanduser().resolve() if rmfpath is not None else None
        # self._bkgpath = Path(bkgpath).expanduser().resolve() if bkgpath is not None else None
        
        # 为 XSPEC fakeit 保存文件路径字符串
        self._arf_file = str(self._arfpath)
        self._rmf_file = str(self._rmfpath) if self._rmfpath is not None else None
        
        # 验证ARF文件存在
        if not self._arfpath.exists():
            raise FileNotFoundError(f"ARF文件不存在: {self._arfpath}")
        if self._rmfpath is not None and not self._rmfpath.exists():
            print(f"警告: RMF文件不存在: {self._rmfpath}")
            self._rmfpath = None
            self._rmf_file = None


    @property
    def srcnum(self):
        """源区域的计数"""
        return getattr(self, '_srcnum', 0)
    
    @srcnum.setter
    def srcnum(self, value):
        if value < 0:
            raise ValueError("源区域的计数必须大于等于0")
        self._srcnum = value
    
    @property
    def bkgnum(self):
        """背景区域的计数"""
        return self._bkgnum
    
    @bkgnum.setter
    def bkgnum(self, value):
        if value < 0:
            raise ValueError("背景区域的计数必须大于等于0")
        self._bkgnum = value
    
    @property
    def area_ratio(self):
        """源区域与背景区域的面积比"""
        return self._area_ratio
    
    @area_ratio.setter
    def area_ratio(self, value):
        if value <= 0:
            raise ValueError("源区域与背景区域的面积比必须大于0")
        self._area_ratio = value
    
        

    @property
    def z0(self):
        """红移z0的属性访问器"""
        return self._z0

    @z0.setter
    def z0(self, value):
        if value < 0:
            raise ValueError("红移z0必须大于等于0")
        self._z0 = value



    @property
    def nh(self):
        """中性氢柱密度的属性访问器"""
        return getattr(self, '_nh', 0.0)
    
    @nh.setter
    def nh(self, value):
        if value < 0:
            raise ValueError("中性氢柱密度必须大于等于0")
        self._nh = value
    
    
    @property
    def model(self):
        """模型的属性访问器"""
        return self._model
    
    @model.setter
    def model(self, value):
        if not isinstance(value, str):
            raise ValueError("模型必须是字符串类型")
        self._model = value

    @property
    def par(self):
        """参数的属性访问器"""
        return self._par
    
    @par.setter
    def par(self, value):
        if not isinstance(value, list):
            raise ValueError("参数必须是一个列表")
        if len(value) == 0:
            raise ValueError("参数列表不能为空")
        if not all(isinstance(v, (int, float)) for v in value):
            raise ValueError("参数列表中的所有元素必须是数字")
        self._par = value


    def _set_model(self):
        """设置XSPEC模型"""
        if isinstance(self._model, str) and isinstance(self._par, list):
            _hem = hem()
            _hem.init_heasoft()
            if _hem.is_heasoft_initialized():
                xspec.AllData.clear()
                xspec.AllModels.clear()
                xspec.Xset.abund = 'wilm'
                xspec.Xset.xsect = 'vern'
                xspec.Xset.cosmo = '67.66 0 0.6888463055445441'
                self._m1 = xspec.Model(self._model)
                
            else:
                raise RuntimeError("HEASoft 环境未初始化")
        else:
            raise ValueError("模型必须是字符串类型，参数必须是列表")

    def is_last_component_z(self):
        """检查模型最后一个分量是否以'z'开头"""
        return self._m1.componentNames[-1].lower().startswith('z')

    def _set_par(self):
        """设置xspec模型的所有参数"""
        self._components = self._m1.componentNames
        param_objs = []
        param_names = []
        self._par_dict = {}

        for comp in self._components:
            comp_obj = getattr(self._m1, comp)
            for pname in comp_obj.parameterNames:
                param_objs.append(getattr(comp_obj, pname))
                param_names.append(f"{comp}.{pname}")

        if len(self._par) != len(param_objs):
            raise ValueError(f"参数数量不匹配: 提供了{len(self._par)}个参数，但模型需要{len(param_objs)}个参数")
        
        # 设置参数数值
        for pobj, val, pname in zip(param_objs, self._par, param_names):
            pobj.values = val
            self._par_dict[pname] = val

        # 识别和处理红移参数
        redshift_components = []
        self._par_z = None
        
        for comp in self._components:
            comp_obj = getattr(self._m1, comp)
            # 检查是否是红移分量（通常以z开头或包含redshift参数）
            if comp.lower().startswith('z') or hasattr(comp_obj, 'Redshift'):
                redshift_components.append(comp)
                if self._par_z is None:  # 使用第一个找到的红移参数
                    try:
                        self._par_z = getattr(comp_obj, 'Redshift')
                        self._z_base = float(self._par_z.values[0])
                    except Exception:
                        pass

        # 如果有多个红移分量，链接它们（通常第二个链接到第一个）
        if len(redshift_components) > 1:
            try:
                first_z_comp = getattr(self._m1, redshift_components[0])
                first_z_param = getattr(first_z_comp, 'Redshift')
                
                for comp_name in redshift_components[1:]:
                    comp_obj = getattr(self._m1, comp_name)
                    z_param = getattr(comp_obj, 'Redshift')
                    z_param.link = first_z_param
                    print(f"链接红移参数: {comp_name}.Redshift -> {redshift_components[0]}.Redshift")
            except Exception as e:
                print(f"警告: 红移参数链接失败: {e}")

        # 如果没有找到红移参数，使用初始红移值
        if self._par_z is None:
            self._z_base = float(self._z0)
            print(f"警告: 模型中未找到红移参数，使用初始红移值 z={self._z_base}")

        # 冻结非关键参数（保持红移和归一化参数可变）
        for pobj, pname in zip(param_objs, param_names):
            # 不冻结红移参数和归一化参数，以便后续调整
            if not (pname.lower().endswith('.redshift') or pname.lower().endswith('.norm')):
                pobj.frozen = True

        # 缓存基线参数
        try:
            _last = getattr(self._m1, self._m1.componentNames[-1])
            self._norm_param = getattr(_last, "norm", None)
            if self._norm_param is not None:
                self._norm0_base = float(self._norm_param.values[0])
            else:
                self._norm0_base = None

            # 捕获谱指数
            self._alpha_base = None
            for pname in getattr(_last, "parameterNames", []):
                if pname.lower() in ("phoindex", "index", "alpha"):
                    self._alpha_base = float(getattr(_last, pname).values[0])
                    break
        except Exception:
            self._norm0_base = None
            self._alpha_base = None

    def validate_model_setup(self):
        """验证模型设置的正确性"""
        if not hasattr(self, '_m1'):
            raise RuntimeError("模型尚未初始化，请先调用 init_model()")
        
        print("🔍 模型验证报告:")
        print(f"  模型表达式: {self._model}")
        print(f"  分量数量: {len(self._components)}")
        print(f"  分量列表: {self._components}")
        
        # 检查参数数量
        total_params = sum(len(getattr(getattr(self._m1, comp), 'parameterNames', [])) 
                          for comp in self._components)
        print(f"  总参数数: {total_params}, 提供参数数: {len(self._par)}")
        
        if total_params != len(self._par):
            print(f"  ⚠️  参数数量不匹配!")
        else:
            print(f"  ✅ 参数数量匹配")
        
        # 检查红移参数
        redshift_count = 0
        redshift_params = []
        for comp in self._components:
            comp_obj = getattr(self._m1, comp)
            if hasattr(comp_obj, 'Redshift'):
                redshift_count += 1
                z_param = getattr(comp_obj, 'Redshift')
                redshift_params.append({
                    'component': comp,
                    'value': z_param.values[0],
                    'frozen': z_param.frozen,
                    'linked': z_param.link != ''
                })
        
        print(f"  红移参数数量: {redshift_count}")
        for i, rp in enumerate(redshift_params):
            status = []
            if rp['frozen']:
                status.append("冻结")
            if rp['linked']:
                status.append("已链接")
            status_str = ", ".join(status) if status else "自由"
            print(f"    {i+1}. {rp['component']}.Redshift = {rp['value']:.3f} ({status_str})")
        
        # 检查归一化参数
        last_comp = getattr(self._m1, self._components[-1])
        if hasattr(last_comp, 'norm'):
            norm_param = getattr(last_comp, 'norm')
            print(f"  归一化参数: {self._components[-1]}.norm = {norm_param.values[0]:.2e}")
            print(f"    冻结状态: {'是' if norm_param.frozen else '否'}")
        else:
            print(f"  ⚠️  最后分量没有 norm 参数")
        
        # 检查谱指数
        if self._alpha_base is not None:
            print(f"  谱指数: {self._alpha_base:.2f}")
        else:
            print(f"  ⚠️  未找到谱指数参数")
        
        print(f"  基线红移: z₀ = {self._z0:.3f}")
        if hasattr(self, '_z_base'):
            print(f"  模型红移: z = {self._z_base:.3f}")
        
        return redshift_count > 0 and hasattr(last_comp, 'norm')

    def init_model(self):
        """初始化模型"""
        self._set_model()
        self._set_par()
        # 可选择性验证
        # self.validate_model_setup()

    def get_param_obj(self, comp_name, param_name):
        """根据分量名和参数名获取参数对象"""
        try:
            comp_obj = getattr(self._m1, comp_name)
            return getattr(comp_obj, param_name)
        except AttributeError as e:
            raise ValueError(f"无法找到参数 {comp_name}.{param_name}: {e}")

    def find_redshift_param(self):
        """查找模型中的红移参数"""
        for comp in self._components:
            comp_obj = getattr(self._m1, comp)
            if hasattr(comp_obj, 'Redshift'):
                return getattr(comp_obj, 'Redshift')
        return None

    # ---------------- Redshift parameter limit helpers -----------------
    def _get_redshift_param_limits(self):
        """返回(redshift_top, redshift_max)，若不存在返回(None, None)"""
        if getattr(self, '_par_z', None) is None:
            return (None, None)
        try:
            if getattr(self, '_par_z', None) is None:
                return (None, None)
            vals = list(self._par_z.values)  # type: ignore[attr-defined]
            if len(vals) >= 6:
                return (float(vals[4]), float(vals[5]))
        except Exception:
            pass
        return (None, None)

    def _extend_redshift_param_limit(self, new_limit: float = 20.0):
        """当需要搜索更高红移时，动态提升XSPEC红移参数的 top/max 上限。

        参数:
            new_limit: 希望扩展到的上限 (同时作用于top与max)
        """
        if getattr(self, '_par_z', None) is None:
            return False
        try:
            if getattr(self, '_par_z', None) is None:
                return False
            vals = list(self._par_z.values)  # type: ignore[attr-defined]
            # values = [val, delta, min, bottom, top, max]
            if len(vals) < 6:
                return False
            cur_top, cur_max = float(vals[4]), float(vals[5])
            if cur_top >= new_limit - 1e-6 and cur_max >= new_limit - 1e-6:
                return False  # 已满足
            # 扩展
            vals[4] = max(new_limit, cur_top)
            vals[5] = max(new_limit, cur_max)
            # 确保当前值不超过新的top
            if vals[0] > vals[4]:
                vals[0] = vals[4]
            self._par_z.values = vals  # type: ignore[attr-defined]
            print(f"🔧 已扩展红移参数上限: top={vals[4]}, max={vals[5]}")
            return True
        except Exception as e:
            print(f"⚠️ 扩展红移参数上限失败: {e}")
            return False

    def _build_soxs_responses(self):
        """构建并缓存soxs的ARF/RMF对象"""
        if not hasattr(self, "_soxs_arf") or self._soxs_arf is None:
            os.chdir(str(self._arfpath.parent))
            self._soxs_arf = soxs.AuxiliaryResponseFile(str(self._arfpath))
        
        if self._rmfpath is not None and (not hasattr(self, "_soxs_rmf") or self._soxs_rmf is None):
            try:
                self._soxs_rmf = soxs.RedistributionMatrixFile(str(self._rmfpath))
                print(f"成功加载RMF文件: {self._rmfpath}")
            except Exception as e:
                print(f"警告: 加载RMF文件失败: {e}")
                self._soxs_rmf = None
        else:
            # 如果没有RMF文件路径，设置为None
            self._soxs_rmf = None

    def _current_alpha_index(self):
        """获取当前谱指数"""
        last_comp = getattr(self._m1, self._m1.componentNames[-1])
        for pname in getattr(last_comp, "parameterNames", []):
            if pname.lower() in ("phoindex", "index", "alpha"):
                return getattr(last_comp, pname).values[0]
        return None

    def _snr_at(self, z, band=(0.5, 4.0)):
        """计算给定红移下的信噪比"""
        try:
            z_safe = max(float(z), 1e-6)
            
            # 确保SOXS对象已构建
            self._build_soxs_responses()
            
            # 更新红移参数
            if self._par_z is not None:
                self._par_z.values = z_safe
            
            # 更新归一化参数（若存在）
            if (
                hasattr(self, '_norm_param') and self._norm_param is not None and
                hasattr(self, '_norm0_base') and self._norm0_base is not None and
                hasattr(self, '_alpha_base') and self._alpha_base is not None
            ):
                try:
                    factor = (cosmo.comoving_distance(self._z0).value / cosmo.comoving_distance(z_safe).value) ** 2  # type: ignore[attr-defined]
                    norm0 = float(self._norm0_base)
                    alpha_val = float(self._alpha_base)
                    self._norm_param.values = norm0 * ((1 + self._z0) / (1 + z_safe)) ** alpha_val * factor
                except Exception:
                    pass
            


            # 使用 XSPEC fakeit 方法计算计数率（新方法）
            try:
                import xspec as xs
                from xspec import FakeitSettings, AllData
                
                # 确保响应文件路径可用
                if not hasattr(self, '_rmf_file') or self._rmf_file is None:
                    # 如果没有直接的文件路径，使用 SOXS 备用方法
                    raise Exception("缺少RMF响应文件路径")
                
                # 创建 FakeitSettings - 只使用必需参数
                fakeit_settings = FakeitSettings(
                    response=self._rmf_file,
                    arf=self._arf_file,
                    exposure=str(self._duration)
                )
                
                # 使用 fakeit 生成模拟谱
                AllData.fakeit(1, fakeit_settings, noWrite=True)
                
                # 获取折叠后的谱（计数谱）
                folded_counts = self._m1.folded(1)  # 获取第1组数据的折叠谱
                
                # 简单计算：假设所有能道都在感兴趣的能段内
                # TODO: 可以进一步优化以精确选择能段对应的能道
                rate_src_only = sum(folded_counts) / self._duration
                
                
                    
            except Exception as fakeit_error:
                # 如果 XSPEC fakeit 失败，回退到 SOXS 方法
                print(f"警告: XSPEC fakeit 方法失败 ({fakeit_error})，回退到 SOXS")
                
                # SOXS 备用方法
                spec = soxs.Spectrum.from_pyxspec_model(self._m1)
                newspec = spec.new_spec_from_band(band[0], band[1])
                
                # 初始化 cspec2 变量
                cspec2 = None
                
                # 进行卷积
                try:
                    # if hasattr(self, '_soxs_rmf') and self._soxs_rmf is not None:
                    #     cspec2 = ConvolvedSpectrum.convolve(newspec, self._soxs_arf, rmf=self._soxs_rmf)
                    # else:
                    cspec2 = ConvolvedSpectrum.convolve(newspec, self._soxs_arf)
                        
                    # 设置曝光时间
                    cspec2.exp_time = (self._duration, "s")
                    
                    # 计算源计数率
                    rate_src_only = cspec2.rate.sum().value
                    
                except Exception as conv_error:
                    print(f"警告: 卷积失败: {conv_error}")
                    return 0.0
            
            # 检查是否成功计算了计数率
            if rate_src_only is None:
                print(f"警告: 无法计算z={z:.3f}处的计数率")
                return 0.0
                
            # 计算总计数（源信号 + 背景贡献）
            n_src = rate_src_only * self._duration + self.bkgnum * self.area_ratio
            n_bkg = self.bkgnum
            
            # 计算SNR
            snr = snr_li_ma(n_src=n_src, n_bkg=n_bkg, alpha_area_time=self.area_ratio)
            return float(snr)
            
        except Exception as e:
            print(f"警告: 计算z={z:.3f}处SNR失败: {e}")
            return 0.0

    def compute_grid(self, z_grid, band=(0.5, 4.0)):
        """
        基于给定红移网格，计算每个z在指定能段内的观测/物理量。

        参数:
        - z_grid: array-like，需要计算的红移数组
        - band: tuple(float, float)，能段范围（单位：keV），例如(0.5, 4.0)

        返回:
        - dict，各键对应numpy.ndarray（长度与z_grid一致）：
            - z: 红移z（float）
            - rate: on区域总计数率[cts/s]
            - net_rate: 源计数率（卷积后、带宽内光子率）[ph/s]
            - flux: 未卷积的能通量（带宽内）[erg/(cm^2 s)]
            - flux_convolved: 卷积后的能通量（带宽内）[erg/s]
            - snr: Li & Ma公式计算的信噪比
        """
        cwd = os.getcwd()
        # ---- 临时关闭进度条 (tqdm) ----
        _tqdm_mod = None
        _old_disable_flag = None
        try:
            import tqdm as _tqdm_mod  # type: ignore
            # 记录原状态并关闭
            if hasattr(_tqdm_mod, 'tqdm'):
                _old_disable_flag = getattr(_tqdm_mod.tqdm, 'disable', None)
                _tqdm_mod.tqdm.disable = True  # 全局静默
        except Exception:
            _tqdm_mod = None

        self._build_soxs_responses()

        # 安全数值提取
        def _as_scalar(x):
            try:
                if hasattr(x, "value"):
                    return float(x.value)
                if isinstance(x, (tuple, list)):
                    if len(x) == 0:
                        return float("nan")
                    x0 = x[0]
                    if hasattr(x0, "value"):
                        return float(x0.value)
                    return float(x0)
                return float(x)
            except Exception:
                return float("nan")

        # 取参：最后一项的norm和可能的谱指数alpha
        last_comp_name = self._m1.componentNames[-1]
        last_comp = getattr(self._m1, last_comp_name)
        norm_param = getattr(last_comp, "norm", None)
        if norm_param is None:
            raise ValueError(f"模型最后一项 {last_comp_name} 没有 norm 参数")

        # 使用缓存的基线归一化与谱指数
        if hasattr(self, "_norm0_base") and (self._norm0_base is not None):
            norm0 = float(self._norm0_base)
        else:
            norm0 = norm_param.values[0] if hasattr(norm_param, "values") else float(norm_param)
        
        if hasattr(self, "_alpha_base") and (self._alpha_base is not None):
            alpha_val = float(self._alpha_base)
        else:
            alpha_val = self._current_alpha_index()

        # z参数 - 使用更健壮的查找方法
        if getattr(self, "_par_z", None) is None:
            self._par_z = self.find_redshift_param()
            

        # 背景计数率
        bkgrate_off = self._bkgnum / self._duration if self._duration and self._duration > 0 and self._bkgnum is not None else 0.0

        dc0 = cosmo.comoving_distance(self._z0).value  # type: ignore[attr-defined]
        dcz = cosmo.comoving_distance(z_grid).value  # type: ignore[attr-defined]
        factor_grid = (dc0 / dcz) ** 2

        rate_list = []
        net_rate_list = []
        flux_list = []
        snr_list = []
        convolved_flux_list = []
        
        for i, z in enumerate(z_grid):
            # 调整z与归一化
            try:
                if self._par_z is not None:
                    self._par_z.values = float(z)  # type: ignore[attr-defined]
            except Exception:
                pass
            
            if alpha_val is not None:
                norm_param.values = float(norm0) * ((1 + self._z0) / (1 + z)) ** float(alpha_val) * factor_grid[i]
            else:
                norm_param.values = float(norm0) * factor_grid[i]

            # 构造谱并限定到能段
            spec = soxs.Spectrum.from_pyxspec_model(self._m1)
            newspec = spec.new_spec_from_band(band[0], band[1])

            # 卷积响应
            if getattr(self, "_soxs_rmf", None) is not None:
                try:
                    cspec2 = ConvolvedSpectrum.convolve(newspec, self._soxs_arf, rmf=self._soxs_rmf)
                except Exception as e:
                    cspec2 = ConvolvedSpectrum.convolve(newspec, self._soxs_arf)
            else:
                cspec2 = ConvolvedSpectrum.convolve(newspec, self._soxs_arf)
            
            cspec2.exp_time = (self._duration, "s")

            # 源净计数率与on区域总计数率
            rate_src_only = cspec2.rate.sum().value
            rate_on_total = rate_src_only + bkgrate_off * self._area_ratio

            # Li-Ma SNR
            n_off = bkgrate_off * (self._duration if self._duration else 0.0)
            n_on = rate_src_only * self._duration + self._area_ratio * n_off
            snr = snr_li_ma(n_src=n_on, n_bkg=n_off, alpha_area_time=self._area_ratio)

            # 带宽内的净计数率和能通量
            net_rate_raw, flux_after_arf_raw = cspec2.get_flux_in_band(band[0], band[1])
            flux_without_arf_raw = newspec.get_flux_in_band(band[0], band[1])

            rate_list.append(float(rate_on_total))
            net_rate_list.append(_as_scalar(net_rate_raw))
            flux_list.append(_as_scalar(flux_without_arf_raw))
            convolved_flux_list.append(_as_scalar(flux_after_arf_raw))
            snr_list.append(float(snr))

        # 恢复XSPEC模型到基线
        try:
            if self._par_z is not None:
                self._par_z.values = float(self._z_base if hasattr(self, "_z_base") else self._z0)
        except Exception:
            pass
        try:
            if norm_param is not None:
                norm_param.values = float(norm0)
        except Exception:
            pass

        os.chdir(cwd)
        result_dict = {
            "z": np.asarray(z_grid, dtype=float),
            "rate": np.asarray(rate_list, dtype=float) * u.photon / u.s,  # type: ignore[attr-defined]
            "net_rate": np.asarray(net_rate_list, dtype=float) * u.photon / u.s,  # type: ignore[attr-defined]
            "flux": np.asarray(flux_list, dtype=float) * u.erg / u.s / u.cm**2,  # type: ignore[attr-defined]
            "flux_convolved": np.asarray(convolved_flux_list, dtype=float) * u.erg / u.s,  # type: ignore[attr-defined]
            "snr": np.asarray(snr_list, dtype=float),
        }

        # ---- 恢复 tqdm 状态 ----
        if _tqdm_mod is not None and _old_disable_flag is not None:
            try:
                _tqdm_mod.tqdm.disable = _old_disable_flag  # type: ignore
            except Exception:
                pass

        return result_dict

    def compute_table(self, z0=None, width=1.0, npts=100, band=(0.5, 4.0)):
        """在[z0, z0+width]上生成z/flux/rate/net_rate/snr表格"""
        if z0 is None:
            z0 = self._z0
        z_grid = np.linspace(z0, z0 + width, npts)
        return self.compute_grid(z_grid, band=band)

    def compute(self, snr_target=7.0):
        """计算满足指定SNR阈值的红移估计值"""
        if not hasattr(self, "_m1"):
            self.init_model()
        return self.find_redshift_for_snr(snr_target=snr_target)

    def find_redshift_for_snr(self, snr_target=7.0, zmin=None, zmax=None, tol=1e-5, max_depth=50, depth=0, max_expand=6):
        """递归网格 + 动态扩展红移参数上限 (直接修改XSPEC参数 top/max) 查找 SNR=目标 对应红移。

        逻辑:
        1. 初始在当前范围 [zmin, zmax] 构造8点网格。
        2. 若最高点 SNR 仍 > 目标 且 z_max ≥ 9 且 redshift 参数 top < 15，则调用 _extend_redshift_param_limit(15)。
        3. 扩展后继续扩大 zmax 直到 SNR 下降穿越目标或达到 top/max=15。
        4. 一旦发现 SNR 发生跨越，在该区间内递归细分，直至 tol 或 max_depth。
        5. 若至上限仍未低于目标，返回参数上限(redshift top)。
        """
        if not hasattr(self, "_m1"):
            self.init_model()

        if zmin is None:
            zmin = self._z0
        if zmax is None:
            # 默认尝试向上 1.0
            top_limit, _ = self._get_redshift_param_limits()
            if top_limit is None:
                top_limit = 10.0
            zmax = min(zmin + 1.0, top_limit)

        # 获取当前可用上限
        top_limit, max_limit = self._get_redshift_param_limits()
        if top_limit is None:
            top_limit = 10.0
        if max_limit is None:
            max_limit = top_limit

        # 保证 zmax 不超过当前的 top_limit
        zmax = min(zmax, top_limit)
        if zmax <= zmin:
            return float(zmin)

        # 生成网格并计算 SNR
        z_grid = np.linspace(zmin, zmax, 8)
        cwd = os.getcwd()
        snr_grid = np.array([self._snr_at(z) for z in z_grid], dtype=float)
        os.chdir(cwd)
        # 查找第一次 SNR 低于目标的位置
        idx = np.where(snr_grid < snr_target)[0]

        # 情况 A: 找到跨越点
        if len(idx) > 0:
            # 若跨越发生在第一个点，直接返回
            if idx[0] == 0:
                return float(z_grid[0])
            # 取跨越区间
            z1 = z_grid[idx[0]-1]
            z2 = z_grid[idx[0]]
            s1 = snr_grid[idx[0]-1]
            s2 = snr_grid[idx[0]]
            if (z2 - z1) < tol or depth >= max_depth or s1 == s2:
                if s1 == s2:
                    return float(0.5*(z1+z2))
                # 线性插值
                z_target = z1 + (snr_target - s1)*(z2 - z1)/(s2 - s1)
                return float(z_target)
            # 递归细化
            return self.find_redshift_for_snr(
                snr_target=snr_target,
                zmin=z1,
                zmax=z2,
                tol=tol,
                max_depth=max_depth,
                depth=depth+1,
                max_expand=max_expand
            )

        # 情况 B: 该网格内 SNR 全部 >= 目标
        boundary_snr = snr_grid[-1]
        # 如果已经达到当前参数上限且已经扩展过或无法再扩展
        if abs(z_grid[-1] - top_limit) < 1e-9:
            # 若 top < 15 且 z≥9 尝试扩展一次
            if top_limit < 20.0 - 1e-6 and z_grid[-1] >= 9.0:
                extended = self._extend_redshift_param_limit(20.0)
                if extended:
                    new_top, _ = self._get_redshift_param_limits()
                    if new_top is None:
                        new_top = 20.0
                    if max_expand > 0:
                        return self.find_redshift_for_snr(
                            snr_target=snr_target,
                            zmin=z_grid[-1],
                            zmax=new_top,
                            tol=tol,
                            max_depth=max_depth,
                            depth=depth,
                            max_expand=max_expand-1
                        )
            # 如果已经到 15 或扩展失败
            if top_limit >= 20.0 - 1e-6:
                if boundary_snr >= snr_target:
                    print(f"⚠️ 在最大允许红移 z={top_limit} 处 SNR={boundary_snr:.2f} 仍 ≥ 目标 {snr_target}，返回上限值。")
                    return float(top_limit)
            # 尝试再做一点小拓展 (若仍有expand次数且 top_limit<15)
            if max_expand > 0 and top_limit < 20.0 - 1e-6:
                return self.find_redshift_for_snr(
                    snr_target=snr_target,
                    zmin=z_grid[-1],
                    zmax=min(top_limit + (z_grid[-1]-z_grid[0]), 20.0),
                    tol=tol,
                    max_depth=max_depth,
                    depth=depth,
                    max_expand=max_expand-1
                )
            return float(z_grid[-1])

        # 尚未到参数上限，可继续向上扩展搜索
        if max_expand > 0:
            # 新的 zmax 尝试向上扩展（不超过 top_limit）
            span = (zmax - zmin)
            proposed = zmax + span
            proposed = min(proposed, top_limit)
            if proposed > zmax + 1e-9:
                return self.find_redshift_for_snr(
                    snr_target=snr_target,
                    zmin=zmin,
                    zmax=proposed,
                    tol=tol,
                    max_depth=max_depth,
                    depth=depth,
                    max_expand=max_expand-1
                )

        # 无法继续扩展
        return float(z_grid[-1])


class GeneralRelativity:
    """基础相对论/多普勒与辐射变换工具类"""

    def __init__(self):
        self._v = None

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, value):
        if hasattr(value, 'unit'):
            if value.value < 0:
                raise ValueError("速度必须大于等于0")
            self._v = value.to(u.meter/u.second)  # type: ignore
        else:
            if value < 0:
                raise ValueError("速度必须大于等于0")
            self._v = value * u.meter/u.second  # type: ignore

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
        return 0.0  # 占位实现

    @property
    def lorentz_factor(self):
        beta = self.beta
        return 1 / np.sqrt(1 - beta ** 2)

    @classmethod
    def show_formula(cls, formula_type="all"):
        formulas = {
            "lorentz": r"\\text{洛伦兹因子:}\\quad \\gamma = \\frac{1}{\\sqrt{1-\\beta^2}}",
            "doppler": (r"\\text{Doppler因子:}\\quad "
                        r"\\mathcal{D} = \\frac{1}{\\gamma (1 - \\beta \\cos\\theta)}"
                        r"= \\gamma (1 + \\beta \\cos\\theta')"),
            "volume": r"\\text{体积变换:}\\quad ds = \\mathcal{D}\\,ds',\\quad dV = D\\,dV'",
            "length": r"\\text{长度变换:}\\quad ds = \\mathcal{D}\\,ds'",
            "time": r"\\text{时间变换:}\\quad dt = \\mathcal{D}^{-1} \\,dt'",
            "energry": r"\\text{能量变换:}\\quad E = \\mathcal{D}E'",
            "dcos_theta": (r"\\text{微分余弦变换:}" r"\\quad d\\cos\\theta = \\frac{d\\cos\\theta'}{\\gamma^2(1 + \\beta\\cos\\theta')^2} = D^{-2} d\\cos\\theta'"),
            "cos_theta": r"\\text{余弦变换:}\\quad \\cos\\theta = \\frac{\\cos\\theta' + \\beta}{1 + \\beta\\cos\\theta'}",
            "sin_theta": r"\\text{正弦变换:}\\quad \\sin\\theta = \\frac{\\sin\\theta'}{\\gamma(1 + \\beta\\cos\\theta')}",
            "tan_theta": r"\\text{正切变换:}\\quad \\tan\\theta = \\frac{\\sin\\theta'}{\\gamma(\\cos\\theta' + \\beta)}",
            "solid_angle": r"\\text{立体角变换:}\\quad d\\Omega = \\mathcal{D}^{-2} d\\Omega'",
            "time_ratio_simple": (r"\\Delta t_{\\text{eng}} : \\Delta t_e : \\Delta t_e' : \\Delta t_{\\text{obs}} \\simeq 1 : 2\\gamma^2 : 2\\gamma : 1."),
            "time_ratio_full": (r"\\Delta t_{\\text{eng}} : \\Delta t_e : \\Delta t_e' : \\Delta t_{\\text{obs}} = "
                                 r"\\frac{1-\\beta}{1-\\beta\\cos\\theta} : \\frac{1}{1-\\beta\\cos\\theta} : \\frac{1}{\\gamma(1-\\beta\\cos\\theta)} : 1."),
            "tobs_teng": (r"\\Delta t_{\\text{obs}} = \\frac{1-\\beta\\cos\\theta}{1-\\beta} \\Delta t_{\\text{eng}}."),
            "intensity": (r"\\text{辐射强度变换:}\\quad I_\\nu(\\nu) = \\mathcal{D}^3 I'_{\\nu'}(\\nu')"),
        }
        header = r"\text{带'}\text{的是共动系，不带的是近邻观测者系}\\"
        note = r"\text{尤其需要特别注意的事情是: 近邻观测者系仍然需要经过宇宙学的变换才能得到观测的结果}"
        note2 = r"\text{另外由于视超光速效应,引擎系下两束光的间隔在辐射过程中会导致间隔观测到的信号间隔变短,这完全不涉及相对论}"
        if formula_type == "all":
            display(Math(header))
            display(Math(note))
            for key in formulas:
                display(Math(formulas[key]))
        else:
            display(Math(header))
            display(Math(note))
            display(Math(formulas.get(formula_type, r"\\text{未知公式类型}")))

    @classmethod
    def show_radiation_transform(cls, formula_type="all"):
        """展示常用辐射变换公式"""
        formulas = {
            "flux1": (r"F_\\nu(\\nu_{\\text{obs}}) = \\frac{(1+z)\\mathcal{D}^3 j'_{\\nu'}(\\nu')V'}{D_L^2}."),
            "flux2": (r"F_\\nu(\\nu_{\\text{obs}}) = \\frac{(1+z)L_{\\nu,\\text{iso}}(\\nu)}{4\\pi D_L^2},"),
            "l_iso": (r"L_{\\text{iso}}(\\nu) = \\nu L_{\\nu,\\text{iso}}(\\nu) = \\mathcal{D}^4 (\\nu' L'_{\\nu'}(\\nu'))."),
            "l_nu_iso": (r"L_{\\nu,\\text{iso}}(\\nu) = \\mathcal{D}^3 L'_{\\nu'}(\\nu')."),
            "l_nu": (r"L_\\nu(\\nu) = \\mathcal{D} L'_{\\nu'}(\\nu')."),
            "l": (r"L(\\nu) = \\mathcal{D}^2 L'_{\\nu'}(\\nu')."),
            "intensity": (r"I_\\nu(\\nu) = \\mathcal{D}^3 I'_{\\nu'}(\\nu'),"),
            "emissivity": (r"j_\\nu(\\nu) = \\mathcal{D}^2 j'_{\\nu'}(\\nu'),"),
            "absorption": (r"\\alpha_\\nu(\\nu) = \\mathcal{D}^{-1} \\alpha'_{\\nu'}(\\nu')."),
        }
        header = r"\\text{带'}\\text{的是共动系，不带的是近邻观测者系}\\"
        if formula_type == "all":
            display(Math(header))
            for key in formulas:
                display(Math(formulas[key]))
        else:
            display(Math(header))
            display(Math(formulas.get(formula_type, r"\\text{未知公式类型}")))

    @classmethod
    def show_grmhd_equations(cls):
        """显示理想磁流体的GRMHD方程组（MHD守恒形式）"""
        eqs = [
            r"\\frac{\\partial (\\gamma \\rho)}{\\partial t} + \\nabla \\cdot (\\gamma \\rho \\mathbf{v}) = 0",
            r"\\frac{\\partial}{\\partial t} \\left( \\frac{\\gamma^2 h}{c^2} \\mathbf{v} + \\frac{\\mathbf{E} \\times \\mathbf{B}}{4\\pi c} \\right)"\
            r"+ \\nabla \\cdot \\left[ \\frac{\\gamma^2 h}{c^2} \\mathbf{v} \\otimes \\mathbf{v} + \\left( p + \\frac{E^2 + B^2}{8\\pi} \\right) \\mathbf{I} - \\frac{\\mathbf{E} \\otimes \\mathbf{E} + \\mathbf{B} \\otimes \\mathbf{B}}{4\\pi} \\right] = 0",
            r"\\frac{\\partial}{\\partial t} \\left( \\gamma^2 h - p - \\gamma \\rho c^2 + \\frac{B^2 + E^2}{8\\pi} \\right)"\
            r"+ \\nabla \\cdot \\left[ (\\gamma^2 h - \\gamma \\rho c^2) \\mathbf{v} + \\frac{c}{4\\pi} \\mathbf{E} \\times \\mathbf{B} \\right] = 0",
            r"\\frac{\\partial \\mathbf{B}}{\\partial t} + c \\nabla \\times \\mathbf{E} = 0"
        ]
        display(Math(r"注意方程组中\\otimes表示张量积,通过假设E=B=0, GRMHD方程可以演化为一般的广义相对论流体力学方程"))
        for eq in eqs:
            display(Math(eq))









class HydroDynamics:
    """经典/相对论流体力学辅助类"""

    @classmethod
    def show_shock_jump_conditions(cls):
        """
        展示流体力学的激波跳变条件（Rankine-Hugoniot conditions）
        """
        from IPython.display import display, Math
        display(Math(r"\text{激波跳变条件（Rankine-Hugoniot conditions）:}"))
        eqs = [
            r"\frac{\rho_2}{\rho_1} = \frac{v_1}{v_2} = \frac{(\hat{\gamma}+1)M_1^2}{(\hat{\gamma}-1)M_1^2+2}",
            r"\frac{p_2}{p_1} = \frac{2\hat{\gamma} M_1^2 - \hat{\gamma} + 1}{\hat{\gamma} + 1}",
            r"\frac{T_2}{T_1} = \frac{p_2 \rho_1}{p_1 \rho_2} = \frac{(2\hat{\gamma} M_1^2 - \hat{\gamma} + 1)[(\hat{\gamma}-1)M_1^2+2]}{(\hat{\gamma}+1)^2 M_1^2}"
        ]
        for eq in eqs:
            display(Math(eq))


class SFH:
    def __init__(self):
        """
        星系形成历史（SFH）类，用于处理和分析星系的形成和演化历史。
        """
        pass





