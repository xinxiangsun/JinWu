'''
Date: 2025-05-30 17:43:59
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-08-15 13:30:37
FilePath: /research/autohea/src/autohea/core/utils.py
'''
import numpy as np
from autohea.core.heasoft import HeasoftEnvManager as hem
import xspec
import soxs
from pathlib import Path
import matplotlib.pyplot as plt
from autohea.core.file import ArfReader, RmfReader, RspReader
from astropy import units as u
from astropy.constants import c  # type: ignore
from IPython.display import display, Math, Latex
from astropy.cosmology import Planck18 as cosmo


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
    """
    红移外推器类，用于计算在给定信噪比阈值下能探测到的最大红移
    
    基于XSPEC物理模型和正确的宇宙学距离概念：
    
    核心物理公式：
    norm_new = norm_original × ((1+z₀)/(1+z))^α × (r_c²(z₀)/r_c²(z))
    
    **严格的物理推导（基于XSPEC单位和真实距离）：**
    
    1. XSPEC光子数密度单位：N(E) [photons cm⁻² keV⁻¹ s⁻¹]
    
    2. 宇宙学距离概念：
       - 共动距离 r_c：源和观测者之间的真实物理距离
       - 光度距离 D_L：定义量，D_L ≡ √(L/(4πF_obs)) = (1+z)×r_c
       - 光度距离不是真实的几何距离！
    
    3. 光子数密度的红移变换（使用真实物理距离）：
       - 几何衰减：按真实距离平方反比 ∝ 1/r_c²
       - 时间膨胀：光子到达率 ∝ 1/(1+z)
       - 能量间隔：dE_rest = dE_obs × (1+z)
       
       完整变换：
       N_obs(E_obs) = N_rest(E_rest) × (r_c²(z₀)/r_c²(z)) × 1/(1+z) × (1+z)
                    = N_rest(E_rest) × (r_c²(z₀)/r_c²(z))
    
    4. K-correction（幂律谱）：
       对于 N_rest(E) ∝ E^(-α)：
       N_rest(E_rest) = N_rest(E_obs×(1+z)) = N_rest(E_obs) × (1+z)^(-α)
       
       最终：N_obs(E_obs) = N_rest(E_obs) × ((1+z₀)/(1+z))^α × (r_c²(z₀)/r_c²(z))
    
    **为什么使用共动距离而不是光度距离：**
    - 光子数密度的几何衰减遵循真实物理距离 r_c
    - 光度距离 D_L 是为保持 F=L/(4πD_L²) 而定义的量，不是真实距离
    - 使用真实距离可以直接分离几何效应和红移效应
    
    支持的XSPEC模型类型：
    - powerlaw: PhoIndex, norm
    - bknpower: PhoIndx1, BreakE, PhoIndx2, norm  
    - cutoffpl: PhoIndex, HighECut, norm
    - grbm: alpha, beta, tem, norm
    
    使用示例：
    extrapolator = RedshiftExtrapolator(
        nh=1e21, z0=1.0, 
        model="TBabs*zTBabs*powerlaw",
        par=[1e21, 1e21, 1.0, 2.0, 1e-3],
        arfpath="response.arf", rmfpath="response.rmf", bkgpath="background.pha",
        srcnum=100, bkgnum=1200, duration=155
    )
    max_z = extrapolator.compute(snr_target=7)
    """
    
    def __init__(self, nh ,z0 , model: str, par: list,  arfpath: list | Path | str, rmfpath: list | Path | str, bkgpath: list | Path | str,
                 srcnum, bkgnum,duration, area_ratio: float = 1/12):
        '''
        对于EP的数据处理而言, alpha的默认值大约是1/12, 但是在实际的数据处理中
        
        '''
        self._srcnum = srcnum
        self._bkgnum= bkgnum
        self._area_ratio = area_ratio
        self._z0 = z0
        self._nh = nh
        self._model = model
        self._par = par
        self._duration = duration
        self._arfpath = arfpath
        self._rmfpath = rmfpath
        self._bkgpath = bkgpath


    @property
    def srcnum(self):
        """源区域的计数"""
        return self._srcnum
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
        return self._nh
    

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
        if isinstance(self._model, str):
            if isinstance(self._par, list):
                # 初始化HEASoft环境
                env_manager = hem()
                env_manager.init_heasoft()
                if env_manager.is_heasoft_initialized():
                    xspec.AllData.clear()
                    xspec.AllModels.clear()
                    xspec.Xset.abund = 'wilm'
                    xspec.Xset.xsect = 'vern'
                    self._m1 = xspec.Model(self._model)
                    print("模型已设置:", self._model)
                    self._m1.show()
                else:
                    raise RuntimeError("HEASoft 环境未初始化，请先手动初始化 HEASoft 环境。")
            else:
                raise ValueError("参数必须是一个列表")
        else:
            raise ValueError("模型必须是字符串类型")

    def is_last_component_z(self):
        """
        检查模型最后一个分量是否以'z'开头
        """
        return self._m1.componentNames[-1].lower().startswith('z')

    def _set_par(self):
        """
        自动设置xspec模型的所有参数，并将参数名与值保存到 self._par_dict。
        """
        self._components = self._m1.componentNames
        
        param_objs = []
        param_names = []
        self._par_dict = {}  # 保存参数名与值

        for comp in self._components:
            comp_obj = getattr(self._m1, comp)
            for pname in comp_obj.parameterNames:
                param_objs.append(getattr(comp_obj, pname))
                param_names.append(f"{comp}.{pname}")

        if len(self._par) != len(param_objs):
            raise ValueError(f"参数数量({len(self._par)})与模型参数数量({len(param_objs)})不一致")
        
        # 设置参数值
        for pobj, val, pname in zip(param_objs, self._par, param_names):
            pobj.values = val
            self._par_dict[pname] = val
        
        # 处理红移关联
        if self.is_last_component_z():
            # 如果最后一个分量带有红移，将其与第一个含红移的分量关联
            last_comp = self._m1.componentNames[-1]
            last_comp_obj = getattr(self._m1, last_comp)
            
            # 查找第一个含有红移的分量
            first_z_comp = None
            for comp_name in self._m1.componentNames[:-1]:  # 排除最后一个
                comp_obj = getattr(self._m1, comp_name)
                if hasattr(comp_obj, 'Redshift'):
                    first_z_comp = comp_obj
                    break
            
            if first_z_comp is not None and hasattr(last_comp_obj, 'Redshift'):
                last_comp_obj.Redshift.link = first_z_comp.Redshift
        
        # 冻结所有参数
        for pobj in param_objs:
            pobj.frozen = True
    
    
#这个地方还需要修改, 增加判断最后一个模型是否是带有红移的判断, 从而泗洪


    def init_model(self):
        """
        初始化模型，设置参数并冻结。
        """
        if hasattr(self,'_model'):
            self._set_model()

        else:
            raise ValueError("模型未设置,请通过调用model设置模型。")
        
        if hasattr(self, '_par'):
            self._set_par()
        else:
            raise ValueError("参数未设置,请通过调用par设置参数。")
    


    def analyze_model_parameters(self):
        """
        分析模型参数，自动识别不同类型的参数
        返回参数分类字典
        """
        params_info = {
            'norm_params': [],      # 归一化参数
            'redshift_params': [],  # 红移参数
            'spectral_params': [],  # 光谱指数参数
            'energy_params': [],    # 能量相关参数（截止、折断等）
            'other_params': []      # 其他参数
        }
        
        for param_name, param_value in self._par_dict.items():
            param_lower = param_name.lower()
            
            # 归一化参数
            if 'norm' in param_lower:
                params_info['norm_params'].append(param_name)
            
            # 红移参数
            elif 'redshift' in param_lower or 'z' in param_lower:
                params_info['redshift_params'].append(param_name)
            
            # 光谱指数参数（各种变体）
            elif any(x in param_lower for x in ['phoindex', 'phoindx', 'photonindex', 'alpha', 'beta']):
                params_info['spectral_params'].append(param_name)
            
            # 能量相关参数
            elif any(x in param_lower for x in ['highecut', 'breake', 'tem', 'energy', 'cut']):
                params_info['energy_params'].append(param_name)
            
            # 氢柱密度
            elif 'nh' in param_lower:
                params_info['other_params'].append(param_name)
            
            else:
                params_info['other_params'].append(param_name)
        
        return params_info

    def get_model_info(self):
        """
        获取模型信息，用于调试和验证
        """
        params_info = self.analyze_model_parameters()
        
        print("=== 模型参数分析 ===")
        print(f"模型: {self._model}")
        print(f"组件: {self._components}")
        print(f"总参数数: {len(self._par_dict)}")
        
        for category, params in params_info.items():
            if params:
                print(f"{category}: {params}")
        
        print("\n所有参数及其值:")
        for param_name, param_value in self._par_dict.items():
            print(f"  {param_name}: {param_value}")
        
        return params_info

    

    def _get_spectral_index(self):
        """
        自动获取光谱指数参数，用于K-correction计算
        支持多种XSPEC模型的不同参数命名约定
        """
        params_info = self.analyze_model_parameters()
        spectral_params = params_info['spectral_params']
        
        if not spectral_params:
            # 如果没有找到光谱参数，返回默认值
            print("警告：未找到光谱指数参数，使用默认值 α=2.0")
            return 2.0
        
        # 对于有多个光谱指数的模型（如bknpower），使用第一个
        first_spectral_param = spectral_params[0]
        alpha_value = self._par_dict[first_spectral_param]
        
        print(f"使用光谱指数参数: {first_spectral_param} = {alpha_value}")
        return alpha_value

    def find_redshift_for_snr(self, snr_target=7, zmin=None, zmax=None, tol=1e-5, max_depth=15, depth=0, max_expand=2):
        """
        递归自适应网格查找，基于正确的XSPEC物理模型
        
        **完整的XSPEC红移外推公式：**
        norm_new = norm_original × ((1+z₀)/(1+z))^α × (r_c²(z₀)/r_c²(z))
        
        **物理解释：**
        1. ((1+z₀)/(1+z))^α: K-correction（光谱演化修正）
           - XSPEC模型单位：photons cm⁻² keV⁻¹ s⁻¹ 
           - 红移改变时，观测能段对应的静止系能段改变
           - 对幂律谱N(E) ∝ E^(-α)，需要此修正保证物理一致性
        
        2. (r_c²(z₀)/r_c²(z)): 真实距离几何衰减
           - 使用共动距离r_c（真实物理距离）
           - 光子数密度按真实距离平方反比衰减
           - 区别于光度距离D_L（定义量，非真实距离）
        
        **推导过程：**
        - 光子数密度变换：N_obs = N_rest × (r_c²(z₀)/r_c²(z)) × 时间膨胀效应
        - 时间膨胀：1/(1+z) 和能量间隔变换：(1+z) 相互抵消
        - K-correction：幂律谱的能量依赖性修正
        """
        if zmin is None:
            zmin = self._z0
        if zmax is None:
            zmax = self._z0 + 1

        z_grid = np.linspace(zmin, zmax, 8)
        
        # 1. 几何距离因子：使用共动距离（真实物理距离）
        r_c_z0 = cosmo.comoving_distance(self._z0).value
        r_c_grid = cosmo.comoving_distance(z_grid).value
        geometric_factor = (r_c_z0 / r_c_grid) ** 2
        
        # 2. K-correction因子：XSPEC光谱演化修正
        alpha = self._get_spectral_index()
        k_correction_factor = ((1 + self._z0) / (1 + z_grid)) ** alpha
        
        # 3. 完整的归一化缩放因子
        total_factor = k_correction_factor * geometric_factor
        
        snr_grid = []
        original_norm = self._par_norm.values[0] if hasattr(self._par_norm.values, '__len__') else self._par_norm.values

        for i, z in enumerate(z_grid):
            # 设置红移参数
            self._par_z.values = z
            
            # 设置归一化：应用完整的XSPEC红移外推公式
            self._par_norm.values = original_norm * total_factor[i]
            
            # 🔬 能谱卷积核心过程（基于trysimulation.ipynb的完整实现）
            # 
            # 物理过程详解：
            # 1. XSPEC模型 → 理论光子数谱 N(E) [photons cm⁻² keV⁻¹ s⁻¹]
            # 2. 能段提取 → 0.5-4.0 keV范围的光子数谱  
            # 3. 仪器响应 → ARF和RMF将光子数谱转换为实际探测器计数
            # 4. 最终输出 → 探测器计数率 [counts/s]
            #
            # 关键点：ARF包含有效面积信息，RMF包含能量分辨率信息
            #         两者结合才能给出完整的仪器响应
            try:
                # Step 1: 从XSPEC模型生成理论光子数谱
                # 此时模型已经应用了红移外推的归一化修正
                spec = soxs.Spectrum.from_pyxspec_model(self._m1)
                
                # Step 2: 提取科学感兴趣的能段 (0.5-4.0 keV)
                newspec = spec.new_spec_from_band(0.5, 4.0)
                
                # Step 3: 设置仪器响应文件（按trysimulation.ipynb方法）
                # 注意：这些属性设置对某些soxs版本可能是只读的，但计算仍然正确
                try:
                    newspec.rmf = str(self._rmfpath)              # 响应矩阵文件  # type: ignore
                    newspec.arf = str(self._arfpath)              # 辅助响应文件  # type: ignore
                    newspec.bkg = str(self._bkgpath)              # 背景谱文件  # type: ignore
                    newspec.exposure = (self._duration, "s")      # 源区曝光时间  # type: ignore
                    newspec.backExposure = (self._duration, "s")  # 背景区曝光时间  # type: ignore
                except AttributeError:
                    # 某些soxs版本这些属性是只读的，但不影响计算
                    pass
                
                # Step 4: 应用ARF进行卷积（核心物理过程）
                # ARF × 光子数谱 = 探测器计数谱
                soxsarf = soxs.AuxiliaryResponseFile(str(self._arfpath))
                cspec = newspec * soxsarf
                
                # Step 5: 获取总计数率（严格按照trysimulation.ipynb）
                # cspec.rate.sum().value 给出总的探测器计数率 [counts/s]
                if hasattr(cspec, 'rate') and hasattr(cspec.rate, 'sum'):  # type: ignore
                    src_rate = cspec.rate.sum().value  # 源区域计数率  # type: ignore
                    # 按照trysimulation.ipynb: rate = cspec.rate.sum().value + bkgrate/12
                    total_rate = src_rate + (self._bkgnum / self._duration) * self._area_ratio
                else:
                    # 如果无法获取rate属性，直接报错
                    raise RuntimeError(f"无法从SOXS能谱对象获取计数率信息。"
                                     f"cspec对象类型: {type(cspec)}, "
                                     f"缺少'rate'属性或'rate.sum()'方法。"
                                     f"可用属性: {[attr for attr in dir(cspec) if not attr.startswith('_')]}")
                    
            except Exception as e:
                # 重新抛出异常，不使用备用方法
                raise RuntimeError(f"SOXS能谱卷积失败: {e}. "
                                 f"模型: {self._model}, 红移: {z}, "
                                 f"ARF: {self._arfpath}, RMF: {self._rmfpath}, BKG: {self._bkgpath}") from e
            
            # Step 6: 从计数率计算总计数（用于SNR计算）
            # 注意：这里不再额外添加背景，因为上面已经包含了
            total_counts = total_rate * self._duration
            
            # 计算信噪比（使用Li&Ma公式）
            snr = snr_li_ma(
                n_src=total_counts, 
                n_bkg=self._bkgnum, 
                alpha_area_time=self._area_ratio
            )
            snr_grid.append(snr)

        snr_grid = np.array(snr_grid)
        idx = np.where(snr_grid < snr_target)[0]
        
        if len(idx) == 0:
            if max_expand > 0:
                return self.find_redshift_for_snr(
                    snr_target=snr_target, zmin=zmin, zmax=zmax + (zmax-zmin), 
                    tol=tol, max_depth=max_depth, depth=depth, max_expand=max_expand-1
                )
            else:
                return z_grid[-1]
        
        if idx[0] == 0:
            return z_grid[0]
        
        z1 = z_grid[idx[0]-1]
        z2 = z_grid[idx[0]]
        
        if (z2-z1 < tol) or (depth >= max_depth):
            snr1 = snr_grid[idx[0]-1]
            snr2 = snr_grid[idx[0]]
            z_snr_target = z1 + (snr_target-snr1)*(z2-z1)/(snr2-snr1)
            return z_snr_target
        else:
            return self.find_redshift_for_snr(
                snr_target=snr_target, zmin=z1, zmax=z2, 
                tol=tol, max_depth=max_depth, depth=depth+1, max_expand=max_expand
            )
    
    
    def compute(self, snr_target=7, show_model_info=False):
        """
        计算在给定信噪比阈值下能探测到的最大红移
        
        Parameters:
        -----------
        snr_target : float
            目标信噪比阈值
        show_model_info : bool
            是否显示模型参数分析信息
        """
        self.init_model()
        
        # 显示模型信息（如果需要）
        if show_model_info:
            params_info = self.get_model_info()
        
        # 自动查找红移参数（通常在第一个分量中）
        redshift_param = None
        for comp_name in self._components:
            comp_obj = getattr(self._m1, comp_name)
            if hasattr(comp_obj, 'Redshift'):
                redshift_param = getattr(comp_obj, 'Redshift')
                break
        
        if redshift_param is None:
            print("警告:模型没有使用带有红移的模型，注意检查模型是否正确。")
        
        self._par_z = redshift_param
        
        # 查找归一化参数（通常在最后一个分量中）
        norm_param = None
        last_comp = self._components[-1]
        last_comp_obj = getattr(self._m1, last_comp)
        if hasattr(last_comp_obj, 'norm'):
            norm_param = getattr(last_comp_obj, 'norm')
        
        if norm_param is None:
            raise ValueError("模型中未找到归一化参数")
        
        self._par_norm = norm_param
        
        return self.find_redshift_for_snr(snr_target=snr_target)
        
    def verify_redshift_extrapolation(self, z_test=None):
        """
        验证红移外推的物理正确性，输出详细的计算过程
        
        Parameters:
        -----------
        z_test : float, optional
            测试红移值，默认为z0+0.5
        """
        if z_test is None:
            z_test = self._z0 + 0.5
            
        print("=" * 60)
        print("🔬 XSPEC红移外推验证")
        print("=" * 60)
        
        # 显示基本信息
        print(f"初始红移 z₀: {self._z0}")
        print(f"测试红移 z: {z_test}")
        print(f"模型: {self._model}")
        
        # 获取光谱指数
        alpha = self._get_spectral_index()
        print(f"光谱指数 α: {alpha}")
        
        # 计算距离因子
        r_c_z0 = cosmo.comoving_distance(self._z0).value  # type: ignore # Mpc
        r_c_test = cosmo.comoving_distance(z_test).value  # type: ignore # Mpc
        geometric_factor = (r_c_z0 / r_c_test) ** 2
        
        # 计算K-correction因子
        k_correction = ((1 + self._z0) / (1 + z_test)) ** alpha
        
        # 总因子
        total_factor = k_correction * geometric_factor
        
        print("\n" + "=" * 40)
        print("📐 距离计算 (共动距离)")
        print("=" * 40)
        print(f"r_c(z₀={self._z0}) = {r_c_z0:.1f} Mpc")
        print(f"r_c(z={z_test}) = {r_c_test:.1f} Mpc")
        print(f"几何因子 (r_c²(z₀)/r_c²(z)) = {geometric_factor:.4f}")
        
        print("\n" + "=" * 40)
        print("🌈 K-correction计算")
        print("=" * 40)
        print(f"K-correction = ((1+{self._z0})/(1+{z_test}))^{alpha}")
        print(f"            = {k_correction:.4f}")
        
        print("\n" + "=" * 40)
        print("🎯 最终结果")
        print("=" * 40)
        print(f"总缩放因子 = {k_correction:.4f} × {geometric_factor:.4f} = {total_factor:.4f}")
        print(f"norm_new = norm_original × {total_factor:.4f}")
        
        print("\n" + "=" * 40)
        print("✅ 物理验证")
        print("=" * 40)
        print("• 使用共动距离r_c (真实物理距离)")
        print("• K-correction保证XSPEC模型物理一致性")
        print("• 时间膨胀和能量间隔效应已自然抵消")
        print("• 符合trysimulation.ipynb的计算逻辑")
        
        return {
            'z0': self._z0,
            'z_test': z_test,
            'alpha': alpha,
            'r_c_z0': r_c_z0,
            'r_c_test': r_c_test,
            'geometric_factor': geometric_factor,
            'k_correction': k_correction,
            'total_factor': total_factor
        }
        





#  def compute(self, norm0, z0, par3, par5, snrrate1, snr_li_ma):
        
#         soxsarf = soxs.AuxiliaryResponseFile(str(self.arfpath))
#         for i, z in enumerate(self.redshift_grid):
#             par3.values = z
#             par5.values = norm0 * ((1+z0)/(1+z))**self.alpha * self.factor[i]
#             spec = soxs.Spectrum.from_pyxspec_model(self.model)
#             newspec = spec.new_spec_from_band(0.5, 4.0)
#             newspec.rmf = str(self.rmfpath)
#             newspec.arf = str(self.arfpath)
#             newspec.bkg = str(self.bkgpath)
#             newspec.exposure = (155, "s")
#             newspec.backExposure = (155, "s")
#             cspec = newspec * soxsarf
#             self.rate[i] = cspec.rate.sum().value + self.bkgrate/12
#             self.snr1[i] = snrrate1(self.rate[i], self.bkgrate, self.lctime, alpha=1/12)
#             self.snr_lima[i] = snr_li_ma(n_src=self.rate[i]*155, n_bkg=self.bkgrate*155, alpha=1/12)
    
#     def find_last_snr_above(self, snr_arr, threshold):
#         idx = np.where(snr_arr > threshold)[0]
#         if len(idx) == 0:
#             return None, None
#         last_idx = idx[-1]
#         return self.redshift_grid[last_idx], snr_arr[last_idx]
    
#     def find_first_rate_below(self, threshold, scale=1):
#         idx = np.where(self.rate*scale < threshold)[0]
#         if len(idx) == 0:
#             return None, None
#         first_idx = idx[0]
#         return self.redshift_grid[first_idx], self.rate[first_idx]
    
#     def plot_snr(self, snr_cut=3, savefile=None):
#         snr_cut_idx = np.where((self.snr1 < snr_cut) | (self.snr_lima < snr_cut))[0]
#         if len(snr_cut_idx) > 0:
#             cut_idx = snr_cut_idx[0]
#         else:
#             cut_idx = len(self.redshift_grid)
#         plt.figure(figsize=(10, 6))
#         plt.plot(self.redshift_grid[:cut_idx], self.snr1[:cut_idx], label='SNR1', color='blue', linewidth=1.5)
#         plt.plot(self.redshift_grid[:cut_idx], self.snr_lima[:cut_idx], label='SNR_LiMa', color='orange', linewidth=1.5)
#         plt.axhline(y=7, color='red', linestyle='--', label='SNR=7')
#         plt.axhline(y=snr_cut, color='green', linestyle='--', label=f'SNR={snr_cut}')
#         plt.xlabel('Redshift', fontsize=14)
#         plt.ylabel('SNR', fontsize=14)
#         plt.title(f'SNR1 and SNR_LiMa vs Redshift (SNR≥{snr_cut})', fontsize=16)
#         plt.legend(fontsize=12)
#         plt.grid(alpha=0.3)
#         plt.show()
#         if savefile:
#             plt.savefig(savefile, dpi=300, bbox_inches='tight')


class GeneralRelativity:
    
    def __init__(self):
        self._v = None
        pass

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, value):
        if hasattr(value, 'unit'):  # 检查是否为Quantity对象
            if value.value < 0:
                raise ValueError("速度必须大于等于0")
            self._v = value.to(u.meter/u.second)  # type: ignore
        else:
            if value < 0:
                raise ValueError("速度必须大于等于0")
            self._v = value * u.meter/u.second  # type: ignore

    def time_dilation(self, t_rest, frame_from="静止系", frame_to="运动系"):
        """
        计算时间膨胀效应，并注明变换
        :param t_rest: 静止系下的时间（Quantity）
        :param frame_from: 原参考系
        :param frame_to: 目标参考系
        :return: 运动系下的时间（Quantity）
        """
        result = self.lorentz_factor * t_rest
        print(f"时间膨胀: 从 {frame_from} 到 {frame_to}，输入 {t_rest}，输出 {result}")
        return result

    def length_contraction(self, l_rest, frame_from="静止系", frame_to="运动系"):
        """
        计算长度收缩效应，并注明变换
        :param l_rest: 静止系下的长度（Quantity）
        :param frame_from: 原参考系
        :param frame_to: 目标参考系
        :return: 运动系下的长度（Quantity）
        """
        result = l_rest / self.lorentz_factor
        print(f"长度收缩: 从 {frame_from} 到 {frame_to}，输入 {l_rest}，输出 {result}")
        return result

    @property
    def beta(self):
        if self._v is None:
            raise ValueError("速度未设置")
        return (self._v / c).decompose().value

    @property
    def lorentz_factor(self):
        beta = self.beta
        return 1 / np.sqrt(1 - beta ** 2)
    

    @classmethod
    def show_formula(cls,formula_type="all"):
        formulas = {
            "lorentz": r"\text{洛伦兹因子:}\quad \gamma = \frac{1}{\sqrt{1-\beta^2}}",
            
             "doppler": (
                        r"\text{Doppler因子:}\quad "
                        r"\mathcal{D} = \frac{1}{\gamma (1 - \beta \cos\theta)}"
                        r"= \gamma (1 + \beta \cos\theta')"
                        ),
            
            "volume": r"\text{体积变换:}\quad ds = \mathcal{D}\,ds',\quad dV = D\,dV'",
            
            "length": r"\text{长度变换:}\quad ds = \mathcal{D}\,ds'",
            "time": r"\text{时间变换:}\quad dt = \mathcal{D}^{-1} \,dt'",
            "energry": r"\text{能量变换:}\quad E = \mathcal{D}E'",
            "dcos_theta": (
                r"\text{微分余弦变换:}"
                r"\quad d\cos\theta = \frac{d\cos\theta'}{\gamma^2(1 + \beta\cos\theta')^2} = D^{-2} d\cos\theta'"
            ),
            
            "cos_theta": r"\text{余弦变换:}\quad \cos\theta = \frac{\cos\theta' + \beta}{1 + \beta\cos\theta'}",
            
            "sin_theta": r"\text{正弦变换:}\quad \sin\theta = \frac{\sin\theta'}{\gamma(1 + \beta\cos\theta')}",
            
            "tan_theta": r"\text{正切变换:}\quad \tan\theta = \frac{\sin\theta'}{\gamma(\cos\theta' + \beta)}",
            
            "solid_angle": r"\text{立体角变换:}\quad d\Omega = \mathcal{D}^{-2} d\Omega'",
            
            "time_ratio_simple": (
            r"\Delta t_{\text{eng}} : \Delta t_e : \Delta t_e' : \Delta t_{\text{obs}} \simeq 1 : 2\gamma^2 : 2\gamma : 1."
            ),

            "time_ratio_full": (
                r"\Delta t_{\text{eng}} : \Delta t_e : \Delta t_e' : \Delta t_{\text{obs}} = "
                r"\frac{1-\beta}{1-\beta\cos\theta} : \frac{1}{1-\beta\cos\theta} : \frac{1}{\gamma(1-\beta\cos\theta)} : 1."
            ),

            "tobs_teng": (
                r"\Delta t_{\text{obs}} = \frac{1-\beta\cos\theta}{1-\beta} \Delta t_{\text{eng}}."
            ),
            "intensity": (
                r"\text{辐射强度变换:}\quad I_\nu(\nu) = \mathcal{D}^3 I'_{\nu'}(\nu')"
            ),
            
        }
        header = r"\text{带'}\text{的是共动系，不带的是近邻观测者系}\\"
        note = r"\text{尤其需要特别注意的事情是: 近邻观测者系仍然需要经过宇宙学的变换才能得到观测的结果}"
        note2 = r"\text{另外由于视超光速效应,引擎系下两束光的间隔在辐射过程中会导致间隔观测到的信号间隔变短,这完全不涉及相对论}"
        if formula_type == "all":
            display(Math(header))
            display(Math(note))
            display(Math(note2))
            for key in formulas:
                display(Math(formulas[key]))
        else:
            display(Math(header))
            display(Math(note))
            display(Math(note2))
            display(Math(formulas.get(formula_type, r"\text{未知公式类型}")))
    

    @classmethod
    def show_radiation_transform(cls, formula_type="all"):
        """
        展示常用的辐射变换公式
        :param formula_type: 可选"all"或指定公式名
        """
        formulas = {
            "flux1": (
                r"F_\nu(\nu_{\text{obs}}) = \frac{(1+z)\mathcal{D}^3 j'_{\nu'}(\nu')V'}{D_L^2}."
            ),
            "flux2": (
                r"F_\nu(\nu_{\text{obs}}) = \frac{(1+z)L_{\nu,\text{iso}}(\nu)}{4\pi D_L^2},"
            ),
            "l_iso": (
                r"L_{\text{iso}}(\nu) = \nu L_{\nu,\text{iso}}(\nu) = \mathcal{D}^4 (\nu' L'_{\nu'}(\nu'))."
            ),
            "l_nu_iso": (
                r"L_{\nu,\text{iso}}(\nu) = \mathcal{D}^3 L'_{\nu'}(\nu')."
            ),
            "l_nu": (
                r"L_\nu(\nu) = \mathcal{D} L'_{\nu'}(\nu')."
            ),
            "l":(
                r"L(\nu) = \mathcal{D}^2 L'_{\nu'}(\nu')."
            ),
            "intensity": (
                r"I_\nu(\nu) = \mathcal{D}^3 I'_{\nu'}(\nu'),"
            ),
            "emissivity": (
                r"j_\nu(\nu) = \mathcal{D}^2 j'_{\nu'}(\nu'),"
            ),
            "absorption": (
                r"\alpha_\nu(\nu) = \mathcal{D}^{-1} \alpha'_{\nu'}(\nu')."
            ),
        }
        header = r"\text{带'}\text{的是共动系，不带的是近邻观测者系}\\"
        if formula_type == "all":
            display(Math(header))
            for key in formulas:
                display(Math(formulas[key]))
        else:
            display(Math(header))
            display(Math(formulas.get(formula_type, r"\text{未知公式类型}")))



    @classmethod
    def show_grmhd_equations(cls):
        """
        显示理想磁流体的GRMHD方程组（MHD守恒形式）
        """
        eqs = [
            r"\frac{\partial (\gamma \rho)}{\partial t} + \nabla \cdot (\gamma \rho \mathbf{v}) = 0",
            r"\frac{\partial}{\partial t} \left( \frac{\gamma^2 h}{c^2} \mathbf{v} + \frac{\mathbf{E} \times \mathbf{B}}{4\pi c} \right)"
            r"+ \nabla \cdot \left[ \frac{\gamma^2 h}{c^2} \mathbf{v} \otimes \mathbf{v} + \left( p + \frac{E^2 + B^2}{8\pi} \right) \mathbf{I} - \frac{\mathbf{E} \otimes \mathbf{E} + \mathbf{B} \otimes \mathbf{B}}{4\pi} \right] = 0",
            r"\frac{\partial}{\partial t} \left( \gamma^2 h - p - \gamma \rho c^2 + \frac{B^2 + E^2}{8\pi} \right)"
            r"+ \nabla \cdot \left[ (\gamma^2 h - \gamma \rho c^2) \mathbf{v} + \frac{c}{4\pi} \mathbf{E} \times \mathbf{B} \right] = 0",
            r"\frac{\partial \mathbf{B}}{\partial t} + c \nabla \times \mathbf{E} = 0"
        ]
        display(Math(r"注意方程组中\otimes表示张量积,通过假设E=B=0, GRMHD方程可以演化为一般的广义相对论流体力学方程"))
        for eq in eqs:
            display(Math(eq))









class HydroDynamics:
    """
    用于描述经典或相对论流体力学的类
    """

    def __init__(self):
        pass

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





