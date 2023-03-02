import sys

import numpy
import scipy

import cupy
import cupyx

import datetime
from dataclasses import dataclass

@dataclass
class ExecutionSetup:
    gpu: bool = False
    precision: str = 'float64'
    precision_np = numpy.float64
    precision_cp = cupy.float64


class CPU_GPU_Abstractor:
    def __init__(self, exec_setup: ExecutionSetup):
        self.exec_setup = exec_setup
        self.__setup_numpy_and_scipy()
        self.__setup_precision()

        self.__pb_started = False

    def __setup_precision(self):
        if self.exec_setup.precision == 'float32':
            self.exec_setup.precision = self.xp.float32
            self.exec_setup.precision_np = numpy.float32
            self.exec_setup.precision_cp = cupy.float32
        elif self.exec_setup.precision == 'float64':
            self.exec_setup.precision = self.xp.float64
            self.exec_setup.precision_np = numpy.float64
            self.exec_setup.precision_cp = cupy.float64
        else:
            raise Exception("Invalid floating point precision specified")

    def __setup_numpy_and_scipy(self):
        if cupy.cuda.runtime.getDeviceCount() > 0 and self.exec_setup.gpu:
            self.xp = cupy
            self.scipy = cupyx.scipy
            self.gpu = True

            device = cupy.cuda.Device(0)
            print(f"GPU-Device ID: {device.id}")
            print(f"GPU-Compute Capability: {device.compute_capability}")
            print(f"GPU-Memory available: {device.mem_info[0]/1e6:.1f}/{device.mem_info[1]/1e6:.1f} MB")

        else:
            self.xp = numpy
            self.scipy = scipy
            self.gpu = False

    def _print_progress_bar(self, progress, max, progress_bar_length=40, title="Progress"):
        progress_ratio = progress/max
        if not self.__pb_started:
            self.__ts = datetime.datetime.now()
            self.__pb_started = True

        tc = datetime.datetime.now()
        tf = (1/progress_ratio -1)*(tc - self.__ts).total_seconds()

        title = f"\r{title}: {progress:5}/{max:5}: "
        success_rate = f" {progress_ratio*100:3.2f}%"
        remaining_time = f" ERT: {int(tf/60):02}:{int(tf%60):02}"
        number_of_progress_indicators = int(progress * progress_bar_length // (max))

        sys.stdout.write(title + "[" + number_of_progress_indicators*"#" + (progress_bar_length - number_of_progress_indicators)*"-" + "]" + success_rate + remaining_time)
