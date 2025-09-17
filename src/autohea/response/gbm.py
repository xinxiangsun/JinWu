'''
Date: 2025-02-25 16:17:52
LastEditors: Xinxiang Sun sunxinxiang24@mails.ucas.ac.cn
LastEditTime: 2025-04-27 15:35:29
FilePath: /research/autohea/autohea/response/gbm.py
'''
from gdt.missions.fermi.gbm.detectors import GbmDetectors
import os
import subprocess

class contgbmrsp:
    """
    用来生成连续观测的 GBM 响应文件，针对非触发情形。
    需要通过网页安装的 GBM 响应生成的 Perl 程序，并且所有的数据应该满足特定的文件目录格式。
                    赤经，单位是度，天文坐标系下的坐标。
                     赤纬，单位是度，天文坐标系下的坐标。
                            开始时间, Fermi MET 时间，以秒为单位。
                          结束时间, Fermi MET 时间，以秒为单位。
                              探测器代号列表。
            生成用于执行 GBM 响应生成的 Perl 脚本的命令字符串。
            执行生成的命令字符串，并打印输出或错误信息。
                     获取赤经。
                      获取赤纬。
                             获取开始时间(Fermi MET)。
                           获取结束时间(Fermi MET)。
                              获取从探测器代号派生的探测器编号。
                               获取探测器代号。
                    设置赤经。
                     设置赤纬。
                              设置探测器代号。
                        设置开始时间(Fermi MET)。
                      设置结束时间(Fermi MET)。
    """
    """
    A class to generate GBM response files for continuous observations in non-triggered scenarios.
    This class requires a Perl script (`SA_GBM_RSP_GEN.pl`) generated via the GBM response webpage, 
    and all data should adhere to a specific directory structure.
    Attributes:
        ra (float): Right Ascension in degrees (astronomical coordinate system).
        dec (float): Declination in degrees (astronomical coordinate system).
        start_time (float): Start time in Fermi MET (seconds).
        end_time (float): End time in Fermi MET (seconds).
        detector (list[int]): List of detector indices.
    Methods:
        commandpl() -> str:
            Generates the command string to execute the Perl script for GBM response generation.
        gbmrsppl() -> None:
            Executes the generated command string and prints the output or error.
    Properties:
        _ra (float): Getter for the Right Ascension.
        _dec (float): Getter for the Declination.
        _start_time (float): Getter for the start time in Fermi MET.
        _end_time (float): Getter for the end time in Fermi MET.
        _det_num (list[int]): Getter for the detector numbers derived from the detector indices.
        _detector (list[int]): Getter for the detector indices.
    Setters:
        ra (float): Setter for the Right Ascension.
        dec (float): Setter for the Declination.
        detector (list[int]): Setter for the detector indices.
        tstart (float): Setter for the start time in Fermi MET.
        tend (float): Setter for the end time in Fermi MET.
        """
    
    def __init__(self, ra, dec, start_time, end_time, detector):
        #ra 赤经
        # 赤纬 dec
        # 开始时间，用Fermi met，start_time
        # 结束时间， 用Fermi met，end_time
        # 探头
        self._ra = ra  # 初始化赤经
        self._dec = dec  # 初始化赤纬
        self._start_time = start_time  # 初始化开始时间fermi met
        self._end_time = end_time  # 初始化结束时间fermi met
        self._detector = detector  # 初始化探测器代号列表 
        # self._srcname = srcname #实际上这个地方是源的类型，但是实际上对于非触发类型的数据，我prefer直接使用他们的源名当作类型
        
    def commandpl(self):
        det_str = " ".join(f"-d{d}" for d in self._det_num)  # 生成 "-d0 -d5 -d9" 格式
        commands = (
            f"SA_GBM_RSP_GEN.pl -Ccspec {det_str} "
            f"-R{self._ra} -D{self._dec} "
            f"-S{self._start_time} -E{self._end_time} ."
        )
        return commands
    

    def gbmrsppl(self): 
        command = self.commandpl()
        print(f"Executing: {command}")
        try:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            print("Output:", result.stdout)
            print("Error:", result.stderr)
        except subprocess.CalledProcessError as e:
            print("Command failed:", e)

    @property
    def _ra(self):
        return self._ra
    
    @property
    def _dec(self):
        return self._dec
    
    @property
    def _start_time(self):
        return self._start_time
    
    @property
    def _end_time(self):
        return self._end_time
    
    @property
    def _det_num(self):
        detector_list = [GbmDetectors[i] for i in self._detector]
        self._detnum = [d.number for d in detector_list]
        return self._det_num
    
    # @property
    # def _srcname(self):
    #     return self._srcname
    
    @_ra.setter
    def ra(self, newra):
        self._ra = newra
    
    @_dec.setter
    def dec(self, newdec):
        self._dec = newdec
    
    @property
    def _detector(self):
        return self._detector

    @_detector.setter
    def detector(self,newdet):
        self._detector = newdet


    @_start_time.setter
    def tstart(self, newtstart):
        self._start_time = newtstart

    @_end_time.setter
    def tend(self, newtend):
        self._end_time = newtend

    # @_srcname.setter
    # def srcname(self, newsrcname):
    #     self._srcname = newsrcname

