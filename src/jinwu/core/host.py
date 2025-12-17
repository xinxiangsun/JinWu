"""
Host Galaxy Candidate Finder for Transient Sources

This module provides tools to identify potential host galaxies around transient sources
using multi-catalog cross-matching (NED, GLADE+, PanSTARRS, Gaia).
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
import warnings

from astroquery.vizier import Vizier
from astroquery.ipac.ned import Ned
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table


from ipyaladin import Aladin
from regions import CircleSkyRegion


import plotly.graph_objects as go
from PIL import Image
PLOTLY_AVAILABLE = True


import requests
REQUESTS_AVAILABLE = True
ALADIN_AVAILABLE = True

warnings.filterwarnings('ignore')


class HostGalaxyFinder:
    """
    查找暂现源周围宿主星系候选的工具类
    
    支持多个目录的交叉匹配：NED、GLADE+、PanSTARRS、Gaia
    可生成Aladin交互式图和Plotly可视化图
    
    Parameters
    ----------
    ra : float, Quantity, or None
        源的赤经 (度或Quantity对象). 如果提供coord则忽略
    dec : float, Quantity, or None
        源的赤纬 (度或Quantity对象). 如果提供coord则忽略
    error_radius : float or Quantity
        位置不确定度半径 (角秒或Quantity对象)
    search_radius : float or Quantity
        搜索半径 (角秒或Quantity对象)
    source_name : str, optional
        源名称，用于输出和标题 (默认: "Transient Source")
    output_dir : str or Path, optional
        输出目录，默认为当前路径。None表示使用运行目录
    coord : SkyCoord, optional
        SkyCoord对象，如果提供则忽略ra/dec参数
        
    Examples
    --------
    使用浮点数和角秒::
    
        finder = HostGalaxyFinder(ra=290.38565, dec=4.68515, 
                                  error_radius=2.3, search_radius=30)
    
    使用Quantity对象::
    
        import astropy.units as u
        finder = HostGalaxyFinder(ra=290.38565*u.deg, dec=4.68515*u.deg,
                                  error_radius=2.3*u.arcsec, search_radius=30*u.arcsec)
    
    使用SkyCoord对象::
    
        from astropy.coordinates import SkyCoord
        coord = SkyCoord(ra=290.38565, dec=4.68515, unit=u.deg)
        finder = HostGalaxyFinder(coord=coord, error_radius=2.3*u.arcsec, 
                                  search_radius=30*u.arcsec)
        
    Attributes
    ----------
    ps1_df : pd.DataFrame
        PanSTARRS源数据表
    gaia_df : pd.DataFrame
        Gaia源数据表
    ned_df : pd.DataFrame
        NED源数据表
    glade_df : pd.DataFrame
        GLADE+星系数据表
    """
    
    def __init__(
        self,
        ra=None,
        dec=None,
        error_radius=None,
        search_radius=None,
        source_name: str = "Transient Source",
        output_dir: Optional[str] = None,
        coord: Optional[SkyCoord] = None
    ):
        """
        初始化查找器
        
        Parameters
        ----------
        ra : float, Quantity, or None
            源的赤经 (度或Quantity对象)
        dec : float, Quantity, or None
            源的赤纬 (度或Quantity对象)
        error_radius : float, Quantity, or None
            位置不确定度半径 (角秒或Quantity对象)
        search_radius : float, Quantity, or None
            搜索半径 (角秒或Quantity对象)
        source_name : str, optional
            源名称，用于输出和标题
        output_dir : str or Path, optional
            输出目录，默认为当前路径。None表示使用运行目录
        coord : SkyCoord, optional
            SkyCoord对象，如果提供则忽略ra/dec参数
        """
        self.source_name = source_name
        
        # 处理坐标输入（支持SkyCoord或ra/dec）
        if coord is not None:
            if not isinstance(coord, SkyCoord):
                raise TypeError("coord必须是SkyCoord对象")
            self.coord = coord
            self.ra = coord.ra.deg
            self.dec = coord.dec.deg
        else:
            # 处理ra输入
            if ra is None:
                raise ValueError("必须提供ra或coord参数")
            if isinstance(ra, u.Quantity):
                self.ra = ra.to(u.deg).value
            else:
                self.ra = float(ra)
            
            # 处理dec输入
            if dec is None:
                raise ValueError("必须提供dec或coord参数")
            if isinstance(dec, u.Quantity):
                self.dec = dec.to(u.deg).value
            else:
                self.dec = float(dec)
            
            # 创建坐标对象
            self.coord = SkyCoord(ra=self.ra*u.deg, dec=self.dec*u.deg, frame='icrs')
        
        # 设置输出目录
        if output_dir is None:
            self.output_dir = Path.cwd()
        else:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 处理error_radius（支持float或Quantity）
        if error_radius is None:
            raise ValueError("必须提供error_radius参数")
        if isinstance(error_radius, u.Quantity):
            self.error_radius = error_radius.to(u.arcsec)
        else:
            self.error_radius = float(error_radius) * u.arcsec
        
        # 处理search_radius（支持float或Quantity）
        if search_radius is None:
            raise ValueError("必须提供search_radius参数")
        if isinstance(search_radius, u.Quantity):
            self.search_radius = search_radius.to(u.arcsec)
        else:
            self.search_radius = float(search_radius) * u.arcsec
        
        # 初始化数据框
        self.ps1_df = pd.DataFrame()
        self.gaia_df = pd.DataFrame()
        self.ned_df = pd.DataFrame()
        self.glade_df = pd.DataFrame()
        
        # Vizier实例
        self.vizier = Vizier(columns=['*'], row_limit=-1)
        
        # Aladin小部件（可选）
        self.aladin = None
        self.aladin_screenshot_path = None
        
        # Plotly图表（可选）
        self.plotly_fig = None
        self.plotly_html_path = None
        
        print(f"✓ 初始化完成")
        print(f"  源: {source_name}")
        print(f"  坐标: RA={self.ra:.5f}°, Dec={self.dec:.5f}°")
        print(f"  位置不确定度: {self.error_radius.to(u.arcsec).value:.2f}\"")
        print(f"  搜索半径: {self.search_radius.to(u.arcsec).value:.1f}\"")
        print(f"  输出目录: {self.output_dir.resolve()}")
        print()
    
    def query_all_catalogs(self) -> None:
        """查询所有目录"""
        print("="*60)
        print("开始查询多个目录...")
        print("="*60)
        self.query_ned()
        self.query_glade_panstarrs_gaia()
        print("="*60)
        print("目录查询完成")
        print("="*60)
        print()
    
    def query_ned(self) -> None:
        """查询NED（名称搜索目录）"""
        print("\n1. 查询NED数据库...")
        try:
            ned_result = Ned.query_region(self.coord, radius=self.search_radius)
            print(f"   找到 {len(ned_result)} 个NED源")
            
            if len(ned_result) > 0:
                ned_candidates = []
                for obj in ned_result:
                    obj_coord = SkyCoord(ra=obj['RA']*u.deg, dec=obj['DEC']*u.deg)
                    separation = self.coord.separation(obj_coord)
                    obj_type = obj['Type'] if 'Type' in ned_result.colnames else 'Unknown'
                    
                    ned_candidates.append({
                        'Name': obj['Object Name'],
                        'RA': obj['RA'],
                        'Dec': obj['DEC'],
                        'Type': obj_type,
                        'Redshift': obj['Redshift'] if 'Redshift' in ned_result.colnames else np.nan,
                        'Separation_arcsec': separation.to(u.arcsec).value,
                        'Magnitude': obj['Magnitude and Filter'] if 'Magnitude and Filter' in ned_result.colnames else 'N/A'
                    })
                
                self.ned_df = pd.DataFrame(ned_candidates).sort_values('Separation_arcsec')
                print("\n   NED候选源:")
                print(self.ned_df.to_string(index=False))
        except Exception as e:
            print(f"   NED查询失败: {e}")
            self.ned_df = pd.DataFrame()
    
    def query_glade_panstarrs_gaia(self) -> None:
        """查询GLADE+、PanSTARRS和Gaia目录"""
        # GLADE+
        print("\n2. 查询GLADE+星系目录...")
        try:
            glade_result = self.vizier.query_region(self.coord, radius=self.search_radius, 
                                                    catalog='VII/281/glade2')
            if len(glade_result) > 0:
                glade_table = glade_result[0]
                print(f"   找到 {len(glade_table)} 个GLADE+星系")
                
                glade_candidates = []
                for galaxy in glade_table:
                    gal_coord = SkyCoord(ra=galaxy['RAJ2000']*u.deg, 
                                        dec=galaxy['DEJ2000']*u.deg)
                    separation = self.coord.separation(gal_coord)
                    
                    glade_candidates.append({
                        'GLADE_Name': galaxy['GLADE'] if 'GLADE' in glade_table.colnames else 'N/A',
                        'RA': galaxy['RAJ2000'],
                        'Dec': galaxy['DEJ2000'],
                        'Redshift': galaxy['z'] if 'z' in glade_table.colnames else np.nan,
                        'Separation_arcsec': separation.to(u.arcsec).value,
                        'Bmag': galaxy['Bmag'] if 'Bmag' in glade_table.colnames else np.nan,
                        'Flag': galaxy['Flag'] if 'Flag' in glade_table.colnames else 'N/A'
                    })
                
                self.glade_df = pd.DataFrame(glade_candidates).sort_values('Separation_arcsec')
                print(f"   GLADE+结果: 找到{len(self.glade_df)}个星系")
            else:
                print("   未找到GLADE+星系")
                self.glade_df = pd.DataFrame()
        except Exception as e:
            print(f"   GLADE+查询失败: {e}")
            self.glade_df = pd.DataFrame()
        
        # PanSTARRS
        print("\n3. 查询PanSTARRS测光目录...")
        try:
            ps1_result = self.vizier.query_region(self.coord, radius=self.search_radius, 
                                                 catalog='II/349/ps1')
            if len(ps1_result) > 0:
                ps1_table = ps1_result[0]
                print(f"   找到 {len(ps1_table)} 个PanSTARRS源")
                
                ps1_candidates = []
                for src in ps1_table:
                    src_coord = SkyCoord(ra=src['RAJ2000']*u.deg, 
                                        dec=src['DEJ2000']*u.deg)
                    separation = self.coord.separation(src_coord)
                    source_type, confidence, features = self._classify_ps1_source(src)
                    
                    ps1_candidates.append({
                        'objID': src['objID'] if 'objID' in ps1_table.colnames else 'N/A',
                        'RA': src['RAJ2000'],
                        'Dec': src['DEJ2000'],
                        'Separation_arcsec': separation.to(u.arcsec).value,
                        'Type': source_type,
                        'Confidence': confidence,
                        'Features': '|'.join(features) if features else '无特征',
                        'f_objID': src['f_objID'] if 'f_objID' in ps1_table.colnames else 0,
                        'Qual': src['Qual'] if 'Qual' in ps1_table.colnames else 0,
                        'gmag': src['gmag'] if 'gmag' in ps1_table.colnames else np.nan,
                        'rmag': src['rmag'] if 'rmag' in ps1_table.colnames else np.nan,
                        'imag': src['imag'] if 'imag' in ps1_table.colnames else np.nan,
                        'zmag': src['zmag'] if 'zmag' in ps1_table.colnames else np.nan,
                        'e_gmag': src['e_gmag'] if 'e_gmag' in ps1_table.colnames else np.nan,
                    })
                
                self.ps1_df = pd.DataFrame(ps1_candidates).sort_values('Separation_arcsec')
                type_counts = self.ps1_df['Type'].value_counts()
                print(f"   PanSTARRS源类型分布:")
                for stype, count in type_counts.items():
                    print(f"     {stype}: {count}个")
            else:
                print("   未找到PanSTARRS源")
                self.ps1_df = pd.DataFrame()
        except Exception as e:
            print(f"   PanSTARRS查询失败: {e}")
            self.ps1_df = pd.DataFrame()
        
        # Gaia
        print("\n4. 查询Gaia源并进行星系分类...")
        try:
            gaia_result = self.vizier.query_region(self.coord, radius=self.search_radius, 
                                                   catalog='I/355/gaiadr3')
            if len(gaia_result) > 0:
                gaia_table = gaia_result[0]
                print(f"   ✓ 找到 {len(gaia_table)} 个Gaia源")
                
                gaia_candidates = []
                for src in gaia_table:
                    src_coord = SkyCoord(ra=src['RA_ICRS']*u.deg, 
                                        dec=src['DE_ICRS']*u.deg)
                    separation = self.coord.separation(src_coord)
                    source_type, confidence, features = self._classify_gaia_source(src, gaia_table)
                    
                    gaia_candidates.append({
                        'Source_ID': src['Source'],
                        'RA': src['RA_ICRS'],
                        'Dec': src['DE_ICRS'],
                        'Separation_arcsec': separation.to(u.arcsec).value,
                        'Gmag': src['Gmag'] if 'Gmag' in gaia_table.colnames else np.nan,
                        'BP_RP': src['BP-RP'] if 'BP-RP' in gaia_table.colnames else np.nan,
                        'Parallax': src['Plx'] if 'Plx' in gaia_table.colnames else np.nan,
                        'Parallax_error': src['e_Plx'] if 'e_Plx' in gaia_table.colnames else np.nan,
                        'Type': source_type,
                        'Confidence': confidence,
                        'Features': '|'.join(features) if features else '无特征',
                        'Redshift': np.nan,
                    })
                
                self.gaia_df = pd.DataFrame(gaia_candidates).sort_values('Separation_arcsec')
                gaia_galaxies = self.gaia_df[self.gaia_df['Type'] == 'Galaxy']
                print(f"   ✓ Gaia中找到 {len(gaia_galaxies)} 个星系候选")
            else:
                print("   未找到Gaia源")
                self.gaia_df = pd.DataFrame()
        except Exception as e:
            print(f"   Gaia查询异常: {str(e)[:80]}")
            self.gaia_df = pd.DataFrame()
    
    @staticmethod
    def _classify_ps1_source(row) -> Tuple[str, int, List[str]]:
        """PanSTARRS源分类（星系 vs 恒星）

        按照 notebook 中的精简判定逻辑：
        - 只用 f_objID bit24（PS 延展）或 Qual bit0（PS 延展）任一为真 → Galaxy
        - QSO 标志（ICRF/likely/possible）也视为 Galaxy
        - 其余默认 Star（若高本动则在特征中注明）

        返回:
        - source_type: 'Galaxy' | 'Star'
        - confidence: 0-100 的经验置信度（用于排序/展示）
        - features: 判定依据文本列表
        """

        features: List[str] = []

        dtype_names = getattr(getattr(row, 'dtype', None), 'names', None) or ()
        qual = int(row['Qual']) if 'Qual' in dtype_names else 0
        f_objid = int(row['f_objID']) if 'f_objID' in dtype_names else 0

        # ========== Qual 字段标志位 ==========
        QUAL_FLAG_EXTENDED_PS = 1  # bit0

        # ========== f_objID 标志位（与 notebook 保持一致） ==========
        PS1_FLAG_ICRF_QSAR = 8
        PS1_FLAG_LIKELY_QSO = 16
        PS1_FLAG_POSSIBLE_QSO = 32
        PS1_FLAG_LARGE_PM = 4096
        PS1_FLAG_EXTENDED_PS = 16777216  # bit24

        qual_extended_ps = (qual & QUAL_FLAG_EXTENDED_PS) != 0
        is_extended_ps = (f_objid & PS1_FLAG_EXTENDED_PS) != 0

        is_qso_icrf = (f_objid & PS1_FLAG_ICRF_QSAR) != 0
        is_likely_qso = (f_objid & PS1_FLAG_LIKELY_QSO) != 0
        is_possible_qso = (f_objid & PS1_FLAG_POSSIBLE_QSO) != 0
        is_large_pm = (f_objid & PS1_FLAG_LARGE_PM) != 0

        # 1) 延展源：f_objID bit24 或 Qual bit0 任一满足
        if is_extended_ps or qual_extended_ps:
            if is_extended_ps and qual_extended_ps:
                features.append('Galaxy: PS延展(f_objID bit24 + Qual bit0)')
                return 'Galaxy', 100, features
            if is_extended_ps:
                features.append('Galaxy: PS延展(f_objID bit24)')
                return 'Galaxy', 90, features
            features.append('Galaxy: PS延展(Qual bit0)')
            return 'Galaxy', 90, features

        # 2) QSO 标志：也视为 Galaxy
        if is_qso_icrf or is_likely_qso or is_possible_qso:
            if is_qso_icrf:
                features.append('Galaxy: QSO标志(ICRF,bit3)')
            elif is_likely_qso:
                features.append('Galaxy: QSO标志(bit4,高置信)')
            else:
                features.append('Galaxy: QSO标志(bit5,低置信)')
            return 'Galaxy', 80, features

        # 3) 其余：Star
        if is_large_pm:
            features.append('Star: 高本动(bit12)')
            return 'Star', 50, features

        features.append('Star: 无延展/QSO标志')
        return 'Star', 30, features
    
    @staticmethod
    def _classify_gaia_source(src, gaia_table) -> Tuple[str, int, List[str]]:
        """Gaia源分类"""
        source_type = 'Star'
        confidence = 30
        features = []
        
        if 'PGal' in gaia_table.colnames:
            pgal = src['PGal']
            if not np.isnan(pgal):
                features.append(f"✓ PGal参数: {pgal:.3f} (Gaia星系概率)")
                if pgal > 0.5:
                    return 'Galaxy', int(pgal * 100), features
        
        if 'Gal?' in gaia_table.colnames:
            try:
                gal = int(src['Gal']) if not np.isnan(src['Gal']) else 0
                if gal > 0:
                    features.append(f"✓ Gal标志: {gal} (Gaia星系标志位)")
                    return 'Galaxy', 80, features
            except:
                pass
        
        return source_type, confidence, features
    
    def print_summary(self) -> None:
        """打印汇总结果"""
        print("\n" + "="*60)
        print("宿主星系候选汇总")
        print("="*60)
        
        # PanSTARRS星系
        if not self.ps1_df.empty:
            galaxies = self.ps1_df[self.ps1_df['Type'] == 'Galaxy'].copy()
            galaxies = galaxies.sort_values('Separation_arcsec')
            
            if len(galaxies) > 0:
                print(f"\n✓ PanSTARRS星系候选 ({len(galaxies)}个)")
                summary = []
                for idx, (_, row) in enumerate(galaxies.iterrows(), 1):
                    summary.append({
                        '序号': idx,
                        'ID': f"{row['objID']:.0f}",
                        '距离("）': f"{row['Separation_arcsec']:.2f}",
                        'r星等': f"{row['rmag']:.2f}" if not np.isnan(row['rmag']) else "---",
                        'RA': f"{row['RA']:.5f}",
                        'Dec': f"{row['Dec']:.5f}",
                        '置信度': f"{row['Confidence']:.0f}%"
                    })
                summary_df = pd.DataFrame(summary)
                print(summary_df.to_string(index=False))
                print(f"\n  距离范围: {galaxies['Separation_arcsec'].min():.2f}\" ~ {galaxies['Separation_arcsec'].max():.2f}\"")
                print(f"  平均距离: {galaxies['Separation_arcsec'].mean():.2f}\"")
        
        # Gaia星系
        if not self.gaia_df.empty:
            gaia_galaxies = self.gaia_df[self.gaia_df['Type'] == 'Galaxy'].copy()
            gaia_galaxies = gaia_galaxies.sort_values('Separation_arcsec')
            
            if len(gaia_galaxies) > 0:
                print(f"\n✓ Gaia星系候选 ({len(gaia_galaxies)}个)")
                summary_gaia = []
                for idx, (_, row) in enumerate(gaia_galaxies.iterrows(), 1):
                    summary_gaia.append({
                        '序号': idx,
                        'ID': f"{int(row['Source_ID'])}",
                        '距离("）': f"{row['Separation_arcsec']:.2f}",
                        'G星等': f"{row['Gmag']:.2f}" if not np.isnan(row['Gmag']) else "---",
                        'RA': f"{row['RA']:.5f}",
                        'Dec': f"{row['Dec']:.5f}",
                        '置信度': f"{row['Confidence']:.0f}%"
                    })
                summary_gaia_df = pd.DataFrame(summary_gaia)
                print(summary_gaia_df.to_string(index=False))
                print(f"\n  距离范围: {gaia_galaxies['Separation_arcsec'].min():.2f}\" ~ {gaia_galaxies['Separation_arcsec'].max():.2f}\"")
                print(f"  平均距离: {gaia_galaxies['Separation_arcsec'].mean():.2f}\"")
        
        print("\n" + "="*60 + "\n")
    
    def create_aladin_view(self) -> None:
        """创建Aladin交互式查看器"""
        if not ALADIN_AVAILABLE:
            print("⚠ ipyaladin未安装，跳过Aladin视图")
            return
        
        print("\n创建Aladin交互式查看器...")
        try:
            self.aladin = Aladin(
                target=f"{self.ra} {self.dec}",
                fov=0.02,
                survey='P/PanSTARRS/DR1/color-z-zg-g',
                show_fullscreen_button=True,
                show_layers_control=True
            )
            
            center = SkyCoord(ra=self.ra*u.deg, dec=self.dec*u.deg, frame='icrs')
            
            # 添加误差圆和搜索圆
            error_region = CircleSkyRegion(center=center, radius=self.error_radius)
            self.aladin.add_graphic_overlay_from_region(error_region, color='red', line_width=2)
            
            search_region = CircleSkyRegion(center=center, radius=self.search_radius)
            self.aladin.add_graphic_overlay_from_region(search_region, color='yellow', line_width=2)
            
            # 添加PanSTARRS星系
            if not self.ps1_df.empty:
                galaxies = self.ps1_df[self.ps1_df['Type'] == 'Galaxy'].copy()
                if len(galaxies) > 0:
                    cat_data = Table({
                        'RA': galaxies['RA'].values,
                        'DEC': galaxies['Dec'].values,
                        'name': [f"PS1-{int(objid)}" for objid in galaxies['objID'].values],
                        'rmag': galaxies['rmag'].values,
                        'sep': galaxies['Separation_arcsec'].values,
                        'reason': galaxies['Features'].values
                    })
                    self.aladin.add_table(cat_data, name='PanSTARRS星系', color='orange', 
                                         source_size=12, shape='circle')
            
            # 添加Gaia星系
            if not self.gaia_df.empty:
                gaia_galaxies = self.gaia_df[self.gaia_df['Type'] == 'Galaxy'].copy()
                if len(gaia_galaxies) > 0:
                    gaia_cat_data = Table({
                        'RA': gaia_galaxies['RA'].values,
                        'DEC': gaia_galaxies['Dec'].values,
                        'name': [f"Gaia-{int(sid)}" for sid in gaia_galaxies['Source_ID'].values],
                        'Gmag': gaia_galaxies['Gmag'].values,
                        'sep': gaia_galaxies['Separation_arcsec'].values,
                        'reason': gaia_galaxies['Features'].values
                    })
                    self.aladin.add_table(gaia_cat_data, name='Gaia星系', color='magenta', 
                                         source_size=12, shape='square')
            
            print("✓ Aladin视图创建完成")
            print(f"  - 红色圆圈: GRB误差范围 ({self.error_radius.to(u.arcsec).value:.1f}\")")
            print(f"  - 黄色圆圈: 搜索半径 ({self.search_radius.to(u.arcsec).value:.1f}\")")
            print(f"  - 橙色圆点: PanSTARRS星系候选")
            print(f"  - 紫色方块: Gaia星系候选")
        except Exception as e:
            print(f"✗ Aladin视图创建失败: {e}")
            self.aladin = None
    
    def save_aladin_screenshot(self) -> None:
        """保存Aladin截图"""
        if self.aladin is None:
            print("⚠ Aladin视图未初始化，无法保存截图")
            return
        
        print("\n保存Aladin截图...")
        out_png = self.output_dir / "aladin_auto.png"
        
        try:
            # 优先使用ipyaladin自带的保存接口
            self.aladin.save_view_as_image(path=str(out_png))
            self.aladin_screenshot_path = out_png
            print(f"✓ Aladin截图已保存: {out_png.resolve()}")
        except Exception as e:
            print(f"Aladin截图失败: {e}")
            # 尝试直接下载PanSTARRS底图
            self._download_panstarrs_image(out_png)
    
    def _download_panstarrs_image(self, output_path: Path) -> None:
        """直接下载PanSTARRS彩色底图"""
        if not REQUESTS_AVAILABLE:
            print("⚠ requests未安装，无法下载底图")
            return
        
        print("尝试直接下载PanSTARRS底图...")
        hips_url = "https://alasky.unistra.fr/hips-image-services/hips2fits"
        params = {
            "hips": "P/PanSTARRS/DR1/color-z-zg-g",
            "ra": self.ra,
            "dec": self.dec,
            "fov": (2.0 / 60),
            "width": 800,
            "height": 800,
            "projection": "TAN",
            "format": "jpg"
        }
        
        try:
            resp = requests.get(hips_url, params=params, timeout=20)
            resp.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(resp.content)
            self.aladin_screenshot_path = output_path
            print(f"✓ 底图已下载: {output_path.resolve()}")
        except Exception as e2:
            print(f"✗ 无法获取截图: {e2}")
    
    def create_plotly_visualization(self) -> None:
        """创建Plotly交互式可视化"""
        if not PLOTLY_AVAILABLE:
            print("⚠ plotly未安装，跳过Plotly可视化")
            return
        
        print("\n创建Plotly交互式可视化...")
        
        # 加载底图
        bg_img = None
        if self.aladin_screenshot_path and self.aladin_screenshot_path.exists():
            try:
                bg_img = Image.open(self.aladin_screenshot_path)
                print(f"✓ 使用底图: {self.aladin_screenshot_path.name}, 尺寸: {bg_img.size}")
            except Exception as e:
                print(f"⚠ 无法加载底图: {e}")
        else:
            print("⚠ 未找到底图文件")
        
        # 计算角秒坐标
        ps1_galaxies = self.ps1_df[self.ps1_df['Type'] == 'Galaxy'].copy()
        ps1_stars = self.ps1_df[self.ps1_df['Type'] == 'Star'].copy()
        
        if len(ps1_galaxies) > 0:
            ps1_galaxies['offset_ra'], ps1_galaxies['offset_dec'] = zip(*[
                self._to_arcsec_offset(r['RA'], r['Dec']) 
                for _, r in ps1_galaxies.iterrows()
            ])
        
        if len(ps1_stars) > 0:
            ps1_stars['offset_ra'], ps1_stars['offset_dec'] = zip(*[
                self._to_arcsec_offset(r['RA'], r['Dec']) 
                for _, r in ps1_stars.iterrows()
            ])
        
        if not self.gaia_df.empty:
            gaia_gal = self.gaia_df[self.gaia_df['Type'] == 'Galaxy'].copy()
        else:
            gaia_gal = pd.DataFrame()
            
        if len(gaia_gal) > 0:
            gaia_gal['offset_ra'], gaia_gal['offset_dec'] = zip(*[
                self._to_arcsec_offset(r['RA'], r['Dec']) 
                for _, r in gaia_gal.iterrows()
            ])
        
        # 创建Plotly图表
        fig = go.Figure()
        fov_arcsec = 60
        
        # 添加底图
        if bg_img:
            fig.add_layout_image(
                dict(
                    source=bg_img,
                    xref="x", yref="y",
                    x=fov_arcsec, y=fov_arcsec,
                    sizex=2*fov_arcsec, sizey=2*fov_arcsec,
                    sizing="stretch",
                    opacity=0.8,
                    layer="below"
                )
            )
        
        # 添加源标记
        if len(ps1_galaxies) > 0:
            fig.add_trace(go.Scatter(
                x=ps1_galaxies['offset_ra'], y=ps1_galaxies['offset_dec'],
                mode='markers', name='PanSTARRS星系',
                marker=dict(size=8, color='rgba(0,0,0,0)', symbol='circle', 
                            line=dict(width=2, color='orange')),
                text=[f"ID:{int(r['objID'])}<br>距离:{r['Separation_arcsec']:.2f}\"<br>rmag:{r['rmag']:.2f}" 
                      for _, r in ps1_galaxies.iterrows()],
                hoverinfo='text'
            ))
        
        if len(ps1_stars) > 0:
            fig.add_trace(go.Scatter(
                x=ps1_stars['offset_ra'], y=ps1_stars['offset_dec'],
                mode='markers', name='PanSTARRS恒星',
                marker=dict(size=6, color='rgba(0,0,0,0)', symbol='star',
                            line=dict(width=1.5, color='cyan')),
                text=[f"ID:{int(r['objID'])}<br>距离:{r['Separation_arcsec']:.2f}\"" 
                      for _, r in ps1_stars.iterrows()],
                hoverinfo='text'
            ))
        
        if len(gaia_gal) > 0:
            fig.add_trace(go.Scatter(
                x=gaia_gal['offset_ra'], y=gaia_gal['offset_dec'],
                mode='markers', name='Gaia星系',
                marker=dict(size=8, color='rgba(0,0,0,0)', symbol='diamond',
                            line=dict(width=2, color='magenta')),
                text=[f"ID:{int(r['Source_ID'])}<br>距离:{r['Separation_arcsec']:.2f}\"<br>Gmag:{r['Gmag']:.2f}" 
                      for _, r in gaia_gal.iterrows()],
                hoverinfo='text'
            ))
        
        # 添加GRB中心
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers', name=self.source_name,
            marker=dict(size=10, color='rgba(0,0,0,0)', symbol='x', 
                        line=dict(width=3, color='red')),
            hoverinfo='text', text=f'{self.source_name}<br>RA={self.ra}°<br>Dec={self.dec}°'
        ))
        
        # 添加圆圈
        theta = np.linspace(0, 2*np.pi, 100)
        error_r = self.error_radius.to(u.arcsec).value
        fig.add_trace(go.Scatter(
            x=error_r * np.cos(theta), y=error_r * np.sin(theta),
            mode='lines', name=f'误差范围 ({error_r:.1f}")',
            line=dict(color='red', width=2, dash='dash'),
            hoverinfo='skip'
        ))
        
        search_r = self.search_radius.to(u.arcsec).value
        fig.add_trace(go.Scatter(
            x=search_r * np.cos(theta), y=search_r * np.sin(theta),
            mode='lines', name=f'搜索半径 ({search_r:.0f}")',
            line=dict(color='yellow', width=2, dash='dot'),
            hoverinfo='skip'
        ))
        
        # 更新布局
        fig.update_layout(
            title=dict(
                text=f'{self.source_name} 宿主星系候选<br>RA={self.ra:.5f}°, Dec={self.dec:.5f}°',
                x=0.5, xanchor='center'
            ),
            xaxis=dict(
                title='ΔRA × cos(Dec) (角秒) [东向左]',
                range=[fov_arcsec, -fov_arcsec],
                scaleanchor='y',
                scaleratio=1,
                gridcolor='rgba(255,255,255,0.2)',
                zeroline=False
            ),
            yaxis=dict(
                title='ΔDec (角秒)',
                range=[-fov_arcsec, fov_arcsec],
                showgrid=True,
                gridcolor='rgba(255,255,255,0.2)',
                zeroline=False
            ),
            width=900, height=900,
            hovermode='closest',
            plot_bgcolor='rgba(20, 20, 30, 0.3)',
            paper_bgcolor='white',
            legend=dict(
                x=1.02, y=1,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='black',
                borderwidth=1
            )
        )
        
        self.plotly_fig = fig
        print("✓ Plotly可视化创建完成")
    
    def _to_arcsec_offset(self, ra_deg: float, dec_deg: float) -> Tuple[float, float]:
        """转换为角秒偏移"""
        delta_ra = (ra_deg - self.ra) * np.cos(np.radians(self.dec))
        delta_dec = dec_deg - self.dec
        return delta_ra * 3600, delta_dec * 3600
    
    def save_plotly_html(self, filename: Optional[str] = None) -> Path:
        """保存Plotly图为HTML"""
        if self.plotly_fig is None:
            print("⚠ Plotly图表未创建")
            return None
        
        if filename is None:
            filename = f"{self.source_name.replace(' ', '_')}_host_galaxies.html"
        
        output_path = self.output_dir / filename
        try:
            self.plotly_fig.write_html(str(output_path))
            self.plotly_html_path = output_path
            print(f"✓ Plotly图已保存: {output_path.resolve()}")
            return output_path
        except Exception as e:
            print(f"✗ 保存Plotly图失败: {e}")
            return None
    
    def save_plotly_png(self, filename: Optional[str] = None) -> Path:
        """保存Plotly图为PNG（需要kaleido）"""
        if self.plotly_fig is None:
            print("⚠ Plotly图表未创建")
            return None
        
        if filename is None:
            filename = f"{self.source_name.replace(' ', '_')}_host_galaxies.png"
        
        output_path = self.output_dir / filename
        try:
            self.plotly_fig.write_image(str(output_path))
            print(f"✓ Plotly图(PNG)已保存: {output_path.resolve()}")
            return output_path
        except Exception as e:
            print(f"⚠ 保存PNG失败(通常需要kaleido包): {e}")
            print("  使用 `pip install kaleido` 来启用PNG导出")
            return None
    
    def run(self, save_aladin: bool = True, save_plotly_html: bool = True,
                         aladin_path: Optional[str] = None, plotly_path: Optional[str] = None) -> None:
        """执行完整分析流程"""
        print("\n" + "="*70)
        print(f"开始分析: {self.source_name}")
        print("="*70)
        
        # 1. 查询所有目录
        self.query_all_catalogs()
        
        # 2. 打印汇总
        self.print_summary()
        
        # 3. 创建和保存Aladin视图
        self.create_aladin_view()
        if save_aladin:
            if aladin_path:
                aladin_dir = Path(aladin_path)
                aladin_dir.mkdir(parents=True, exist_ok=True)
                self.output_dir = aladin_dir
            self.save_aladin_screenshot()
        
        # 4. 创建Plotly可视化并保存
        self.create_plotly_visualization()
        if save_plotly_html:
            if plotly_path:
                plotly_dir = Path(plotly_path)
                plotly_dir.mkdir(parents=True, exist_ok=True)
                self.output_dir = plotly_dir
            self.save_plotly_html()
        
        print("\n" + "="*70)
        print("✓ 分析完成")
        print("="*70 + "\n")

