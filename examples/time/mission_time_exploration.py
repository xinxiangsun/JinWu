"""迁移示例 / Migrated example: time/mission_time_exploration.
状态 / Status: legacy_exploration. 包含非法闰秒日期和可选 swiftbat 依赖 / Includes an invalid leap-second date and optional swiftbat dependency
历史探索不保证可完整运行 / Historical exploration is not guaranteed runnable.
见同名 Notebook 与目录 README / See the paired Notebook and README.
"""


# 保持单元共享全局变量语义；仅直接运行脚本时执行。
# Preserve shared Notebook globals; execute only when run as a script.
if __name__ == '__main__':
    # 原 Notebook 单元 / Notebook cell 1

    # 从仓库定位输入，允许通过环境变量覆盖；不移动原始数据。
    # Locate repository inputs with environment overrides; original data stay in place.
    from pathlib import Path
    from datetime import datetime, timezone
    import os
    _anchor = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
    REPO_ROOT = next((p for p in (_anchor, *_anchor.parents)
                      if (p/'packages/jinwu').is_dir()), None)
    if REPO_ROOT is None:
        raise RuntimeError('请从 JinWu 仓库内运行 / Run from within the JinWu checkout')
    RESEARCH_ROOT = Path(os.environ.get('JINWU_EXAMPLE_RESEARCH_ROOT', REPO_ROOT.parent))
    TEST_DATA_ROOT = Path(os.environ.get('JINWU_EXAMPLE_TEST_ROOT', REPO_ROOT/'test'))
    DOWNLOAD_ROOT = Path(os.environ.get('JINWU_EXAMPLE_DOWNLOAD_ROOT', Path.home()/'下载'))
    # 图件写入独立运行目录，不覆盖研究目录中的历史图。
    # Write figures into a unique run directory, preserving historical research plots.
    OUTPUT_DIR = Path(os.environ.get('JINWU_EXAMPLE_OUTPUT_ROOT', REPO_ROOT/'examples/_outputs')) / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)


    # 原 Notebook 单元 / Notebook cell 2

    from jinwu.core.time import Time
    import erfa

    # 原 Notebook 单元 / Notebook cell 3

    time1 = Time('2015-06-30 23:59:59', format = 'iso', scale = 'utc')
    time2 = Time('2015-06-30 23:59:60', format = 'iso', scale = 'utc')
    time3 = Time('2015-07-01 00:00:00', format = 'iso', scale = 'utc')
    time4 = Time('2015-07-01 00:00:01', format = 'iso', scale = 'utc')
    time1.swift, time2.swift, time3.swift, time4.swift

    # 原 Notebook 单元 / Notebook cell 4

    Time(457401612.791011, format='swift').utc.isot,Time(457401613.791011, format='swift').utc.isot,Time(457401614.791011, format='swift').utc.isot

    # 原 Notebook 单元 / Notebook cell 5

    print(Time('2015-07-01 00:00:02', format='iso', scale='utc').swift)
    print(Time('2015-07-01 00:00:03', format='iso', scale='utc').swift)
    print(Time('2015-07-01 00:00:04', format='iso', scale='utc').swift)
    print(Time('2015-07-01 00:00:05', format='iso', scale='utc').swift)
    print(Time('2015-07-01 00:00:06', format='iso', scale='utc').swift)
    print(Time('2015-07-01 00:00:07', format='iso', scale='utc').swift)
    print(Time('2015-07-01 00:00:08', format='iso', scale='utc').swift)
    print(Time('2015-07-01 00:00:09', format='iso', scale='utc').swift)
    print(Time('2015-07-01 01:00:09', format='iso', scale='utc').swift)
    print(Time('2015-07-02 01:00:09', format='iso', scale='utc').swift)


    # 原 Notebook 单元 / Notebook cell 6

    print(Time('2015-07-02 23:59:60', format='iso', scale='utc').swift)

    # 原 Notebook 单元 / Notebook cell 7

    print(Time('2015-07-03 00:00:00', format='iso', scale='utc').swift)

    # 原 Notebook 单元 / Notebook cell 8

    Time('2015-07-03 00:00:00', format='iso', scale='utc').fermi

    # 原 Notebook 单元 / Notebook cell 9

    457491623.7956923 - 457405223.7911908

    # 原 Notebook 单元 / Notebook cell 10

    trigtime = Time('2025-06-15 22:25:19.59', format='iso', scale='utc')
    trigtime.swift

    # 原 Notebook 单元 / Notebook cell 11

    trigtime.ep

    # 原 Notebook 单元 / Notebook cell 12

    trigtime = Time(771719155.392, format='swift')
    trigtime.utc.isot

    # 原 Notebook 单元 / Notebook cell 13

    trigtime.ep

    # 原 Notebook 单元 / Notebook cell 14

    time1.swift, time2.swift, time3.swift, time4.swift

    # 原 Notebook 单元 / Notebook cell 15


    from jinwu.core.time import Time

    print(Time('2016-12-30 23:59:59', format='iso', scale='utc').swift)
    print(Time('2016-12-31 00:00:00', format='iso', scale='utc').swift)
    print(Time('2016-12-31 00:00:01', format='iso', scale='utc').swift)

    # 原 Notebook 单元 / Notebook cell 16

    Time(504835224.38732594, format='swift').utc.isot

    # 原 Notebook 单元 / Notebook cell 18

    print(Time('2016-12-31 22:59:59', format='iso', scale='utc').swift)
    print(Time('2016-12-31 23:59:59', format='iso', scale='utc').swift)
    print(Time('2016-12-31 23:59:60', format='iso', scale='utc').swift)
    print(Time('2017-01-01 00:00:00', format='iso', scale='utc').swift)
    print(Time('2017-01-01 00:00:01', format='iso', scale='utc').swift)

    # 原 Notebook 单元 / Notebook cell 19

    print(Time('2017-01-02 00:00:01', format='iso', scale='utc').swift)

    # 原 Notebook 单元 / Notebook cell 20

    504918015.4337296 - 504918016.392

    # 原 Notebook 单元 / Notebook cell 21

    from swiftbat.clockinfo import utcf

    # 原 Notebook 单元 / Notebook cell 22

    utcf( 457401613.791)

    # 原 Notebook 单元 / Notebook cell 23

    import astropy.units as u
    flux = 1*u.uJy
    flux.to(u.MJy)

    # 原 Notebook 单元 / Notebook cell 24

    energy = 1e-13*u.erg
    area = 1*u.cm**2
    time = 1000*u.s
    flux = energy / area / time
    bandwidth = 1*u.Hz
    flux_density = flux / bandwidth
    flux_density.to(u.Jy)

    # 原 Notebook 单元 / Notebook cell 25

    SWIFT_EPOCH_UTC = Time('2001-01-01T00:00:00.000', format='isot', scale='utc')
    SWIFT_EPOCH_LEAP = (
        (SWIFT_EPOCH_UTC.tai.jd - SWIFT_EPOCH_UTC.utc.jd) * erfa.DAYSEC  # type: ignore[attr-defined]
    )

    # 原 Notebook 单元 / Notebook cell 26

    SWIFT_EPOCH_LEAP

    # 原 Notebook 单元 / Notebook cell 28

    # Swift MET ↔ UTC quick checks around leap seconds
    from astropy.time import Time
    from jinwu.core.time import Time as _TimeAlias  # ensure format classes are registered
    for isot in ['2015-06-30T23:59:59','2015-06-30T23:59:60','2015-07-01T00:00:00',
                 '2016-12-31T23:59:59','2016-12-31T23:59:60','2017-01-01T00:00:00']:
        u = Time(isot, scale='utc')
        met = u.to_value('swift')
        uu = Time(met, format='swift', scale='utc')
        print(isot, '-> MET', met, '-> back', uu.isot, 'Δs', (uu - u).to_value('s'))

    u1 = Time('2015-06-30T23:59:59', scale='utc')
    u2 = Time('2015-07-01T00:00:00', scale='utc')
    print('ΔMET 2015 leap:', u2.to_value('swift') - u1.to_value('swift'))
    u3 = Time('2016-12-31T23:59:59', scale='utc')
    u4 = Time('2017-01-01T00:00:00', scale='utc')
    print('ΔMET 2016 leap:', u4.to_value('swift') - u3.to_value('swift'))
