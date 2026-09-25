"""GBM 任务数据 URL / 下载辅助（自 jinwu.core.utils 移植，0.2.0）。

Fermi/GBM 属于可选仪器包，任务数据相关的 URL 生成不应放在核心包
``jinwu.core.utils``（核心层不应内置单仪器逻辑）。原位置保留弃用垫片。
"""

from __future__ import annotations

__all__ = ['generate_download_url']


def generate_download_url(isot_time):
    """
    根据给定的 isot (YYYY-MM-DDTHH:MM:SS) 时间生成 GBM poshist 文件的下载 URL。

    参数:
    - isot_time (str): ISOT 格式时间字符串，例如 "2024-01-01T12:00:00"

    返回:
    - url (str): 生成的 poshist 文件下载 URL

    说明:
    - 文件名为 HEASARC GBM daily 数据目录的姿规律 ``glg_poshist_all_<yy><mm><dd>_v00.fit``；
    - 返回的是 ``.../daily/<YYYY>/<MM>/<DD>/current`` 目录 URL（HEASARC 的
      current 别名指向当日最新版本目录）。
    参考：HEASARC Fermi GBM daily data, https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/daily/
    """
    # 提取年份、月份、日期
    year = isot_time.strftime('%y')
    yr2 = isot_time.datetime.year
    month = f"{isot_time.datetime.month:02d}"  # 两位数格式
    day = f"{isot_time.datetime.day:02d}"

    # 生成文件名
    filename = f"glg_poshist_all_{year}{month}{day}_v00.fit"

    # 生成完整的下载路径（目录 URL；文件名保留供调用方拼接）
    # https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/daily/2025/01/01/current/
    # url = f"https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/daily/{yr2}/{isot_time.strftime('%m/%d/')}current/{filename}"
    url = f"https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/daily/{yr2}/{isot_time.strftime('%m/%d/')}current"
    return url
