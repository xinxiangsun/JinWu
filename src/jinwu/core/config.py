from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Dict, Type, Optional, Tuple, List, Literal
from astropy.units import Quantity
_INSTRUMENT_REGISTRY: Dict[str, Type['InstrumentConfig']] = {}

def register_instrument(cls):
    """装饰器：自动注册仪器"""
    _INSTRUMENT_REGISTRY[cls.__name__] = cls
    return cls

def instrument(name: str, **kwargs) -> 'InstrumentConfig':
    """获取仪器实例"""
    if name not in _INSTRUMENT_REGISTRY:
        raise ValueError(
            f"Unknown instrument: {name}\n"
            f"Available: {list(_INSTRUMENT_REGISTRY.keys())}"
        )
    return _INSTRUMENT_REGISTRY[name](**kwargs)

@dataclass
class InstrumentConfig(ABC):
    """仪器基类"""
    
    # ============ 基本信息 ============
    name: str
    telescope: str
    
    # ============ 能量范围 ============
    Emin_keV: float
    Emax_keV: float
    
    # ============ 波段和位置 ============
    band: Literal['Gamma', 'X', 'UV/Optical/IR'] 
    place: Optional[Literal['space', 'ground']] 
    
    background_type: Literal[
                            'temporal',           # 时间域（GBM polynomial fit）
                            'spatial',            # 空间域（annulus/region）
                            'detector_shadow',    # 探测器遮挡区（部分编码孔径仪器）
                            'blank_sky',          # 空天区观测（Chandra、XMM）
                            ]



    # ============ 光学仪器滤光片 ============
    
    
    # ============ 谱提取参数 ============
    grouping_min_counts: int
    srcregion_type: Literal['circle', 'box', 'ellipse', 'polygon']
    bkgregion_type: Literal['annulus', 'circle', 'box', 'polygon']
    stat_method: Literal['cstat', 'pgstat', 'wstat', 'chi']
    response_type: Literal['rmf', 'rsp']
    

    effective_area: Optional[Quantity]
    fov           : Optional[Quantity]


    filtername: Optional[str] = None
    
    

# ============ 有子仪器/能段的仪器 ============

@register_instrument
class UVOT(InstrumentConfig):
    """Swift UV/Optical Telescope - 多个滤光片"""
    
    # 定义所有可用的滤光片和对应的能段
    FILTERS = {
        'V': (2.3, 3.5),      # 能量范围 (eV)
        'B': (2.5, 3.5),
        'U': (3.0, 4.2),
        'UVW1': (3.2, 4.5),
        'UVM2': (4.0, 5.0),
        'UVW2': (4.5, 6.0),
        'White': (2.0, 6.0),
    }
    
    def __init__(self, filter='V', **kwargs):
        if filter not in self.FILTERS:
            raise ValueError(f"Unknown filter: {filter}. Available: {list(self.FILTERS.keys())}")
        
        energy_range = self.FILTERS[filter]
        
        defaults = {
            'name': f'UVOT_{filter}',
            'telescope': 'Swift',
            'energy_range': energy_range,
            'effective_area': 150.0,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)
        self.filter = filter
    
    def extract_spectrum(self, **kwargs):
        return f"UVOT spectrum (filter: {self.filter})"
    
    @classmethod
    def list_filters(cls):
        """列出所有可用滤光片"""
        return list(cls.FILTERS.keys())

@register_instrument
class GBM(InstrumentConfig):
    """Fermi Gamma-ray Burst Monitor - 多个探测器"""
    
    # 定义所有探测器及其能段
    DETECTORS = {
        'NAI_1': (8.0, 1000.0),
        'NAI_2': (8.0, 1000.0),
        'NAI_3': (8.0, 1000.0),
        'NAI_4': (8.0, 1000.0),
        'NAI_5': (8.0, 1000.0),
        'NAI_6': (8.0, 1000.0),
        'BGO_1': (200.0, 40000.0),
        'BGO_2': (200.0, 40000.0),
    }
    
    def __init__(self, detector='NAI_1', **kwargs):
        if detector not in self.DETECTORS:
            raise ValueError(
                f"Unknown detector: {detector}\n"
                f"Available: {list(self.DETECTORS.keys())}"
            )
        
        energy_range = self.DETECTORS[detector]
        
        defaults = {
            'name': f'GBM_{detector}',
            'telescope': 'Fermi',
            'energy_range': energy_range,
            'effective_area': 100.0,
            'grouping_min_counts': 25,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)
        self.detector = detector
        self.detector_type = 'NAI' if 'NAI' in detector else 'BGO'
    
    def extract_spectrum(self, **kwargs):
        return f"GBM spectrum ({self.detector})"
    
    @classmethod
    def list_detectors(cls):
        """列出所有可用探测器"""
        return list(cls.DETECTORS.keys())
    
    @classmethod
    def get_nai_detectors(cls):
        """获取所有 NaI 探测器"""
        return [d for d in cls.DETECTORS.keys() if 'NAI' in d]
    
    @classmethod
    def get_bgo_detectors(cls):
        """获取所有 BGO 探测器"""
        return [d for d in cls.DETECTORS.keys() if 'BGO' in d]

# ============ 内置仪器 ============

@register_instrument
@dataclass
class WXT(InstrumentConfig):
    """EP/WXT 仪器"""
    def __init__(self, **kwargs):
        defaults = {
            'name': 'WXT',
            'telescope': 'Einstein Probe',
            'energy_range': (0.5, 4.0),
            'effective_area': 500.0
        }
        defaults.update(kwargs)
        super().__init__(**defaults)
    
    def extract_spectrum(self, **kwargs):
        return "WXT spectrum extraction"

@register_instrument
@dataclass
class BAT(InstrumentConfig):
    """Swift/BAT 仪器"""
    def __init__(self, **kwargs):
        defaults = {
            'name': 'BAT',
            'telescope': 'Swift',
            'energy_range': (15.0, 150.0),
            'effective_area': 5200.0
        }
        defaults.update(kwargs)
        super().__init__(**defaults)
    