"""简单的 PHA 重分箱工具（纯 Python）。

提供 `rebin_pha(pha: PhaData, nbins: int) -> PhaData`：
把连续通道按固定因子物理合并到 `nbins` 个新通道（LINEAR 模式）。

对齐 HEASoft 6.37 语义（`heasptools/ftrbnpha/ftrbnpha.cxx` LINEAR 分支 +
`heacore/heasp/pha.cxx pha::rebinChannels`）：

- LINEAR 模式要求输出通道数整除输入通道数（ftrbnpha.cxx: "the output number
  of channels must be an exact divisor of the input number of channels"），
  否则报错；组因子 = nchan_in / nbins，各组等宽。不再支持旧版的浮点因子
  不均匀分箱与 nbins > nchan 的补零行为。
- 计数组内求和（GroupBin SumMode）；stat_err 取 sqrt(合并计数)，与
  rebinChannels 的 GAUSS 选项 sqrt(factor*pha)/factor 及泊松误差在统计上一致
  （泊松方差可加）。
- 输出通道号从原首通道起连续重编号（rebinChannels: m_Channel[i] =
  FirstOrigChannel + i），不再从 0 起。
- 质量折叠：组质量 = 组首通道质量，其后任一成员质量非零则覆盖（取最后
  一个非零者；rebinChannels 的 "any input element making up a bin has bad
  quality then the whole bin does" 规则）。输入无 quality 时输出 quality=None。
- 物理合并后 GROUPING 归零（rebinChannels 将 m_Group 全置 0），故 grouping=None。

ASCASIS 专用的 FAINT2BRIGHT / BRIGHT2LINEAR 压缩模式与 binfile/PHA 模板
模式不在本简化实现范围内。
"""
from __future__ import annotations

import numpy as np
from ..core.data import PhaData

# 参考：HEASoft 6.37 heasptools/ftrbnpha/ftrbnpha.cxx（LINEAR: binInfo.load(
#       nChanIn/nChanOut, nChanIn)，整除校验见其 msg 分支）；
#       heacore/heasp/pha.cxx pha::rebinChannels（GroupBin SumMode 求和、
#       m_Channel 连续重编号、quality 折叠、m_Group 清零）；
#       heacore/heasp/grouping.cxx grouping::load / GroupBin。
def rebin_pha(pha: PhaData, nbins: int) -> PhaData:
    ch = np.asarray(pha.channels, dtype=int)
    cnt = np.asarray(pha.counts, dtype=float)
    if ch.size == 0:
        return pha
    if nbins <= 0:
        raise ValueError('nbins must be > 0')

    nch = ch.size
    ch_min = int(ch[0])
    # HEASoft LINEAR 模式：输出通道数必须整除输入通道数（含 nbins>nch 的情形）
    if nbins > nch or (nch % nbins) != 0:
        raise ValueError(
            f'For LINEAR mode the output number of channels ({nbins}) must be '
            f'an exact divisor of the input number of channels ({nch}).'
        )
    factor = nch // nbins

    counts = cnt.reshape(nbins, factor).sum(axis=1)
    stat_err = np.sqrt(counts)
    # 输出通道号从原首通道连续重编号（rebinChannels 语义），不再从 0 起
    channels = ch_min + np.arange(nbins, dtype=int)

    qual = pha.quality
    if qual is None:
        quality = None
    else:
        q_in = np.asarray(qual, dtype=int).ravel()
        if q_in.size != nch:  # 输入 quality 长度异常时按无质量处理
            quality = None
        else:
            q_mat = q_in.reshape(nbins, factor)
            quality = q_mat[:, 0].copy()
            for g in range(nbins):
                for m in q_mat[g, 1:]:
                    if m != 0:  # 最后一个非零成员覆盖（HEASoft 折叠规则）
                        quality[g] = m

    return PhaData(path=pha.path, channels=channels, counts=counts, stat_err=stat_err,
                   exposure=pha.exposure, backscal=pha.backscal, areascal=pha.areascal,
                   quality=quality, grouping=None,
                   ebounds=(pha.ebounds if factor == 1 else None), header=pha.header,
                   meta=pha.meta, headers_dump=pha.headers_dump, columns=pha.columns)
