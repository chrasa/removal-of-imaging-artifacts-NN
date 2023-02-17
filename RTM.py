import sys

import numpy
import scipy

import cupy
import cupyx

from os.path import exists
from os.path import sep

from SimulationSetup import SimulationSetup
from benchmark import timeit
from typing import TypeVar

PAST = 2
PRESENT = 1
FUTURE = 0

Array = TypeVar('Array', numpy.array, cupy.array)

class RTM:

    def __init__(self, setup: SimulationSetup, gpu: bool=False):

        self.setup_numpy_and_scipy(gpu)

        self.setup = setup
        self.background_velocity = self.xp.full(self.setup.N**2, setup.background_velocity_value, dtype=self.xp.float64)
        self.nts = 20
        self.delta_t = setup.tau/self.nts

        self.data_folder = "." + sep + "bs_test" + sep    

    def setup_numpy_and_scipy(self, gpu):
        if cupy.cuda.runtime.getDeviceCount() > 0 and gpu:
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

    def generate_sources(self):
        source_locations = numpy.array([15,  25,  35,  44,  54,  64,  74,  84,  93, 103, 113, 123, 133,
                    142, 152, 162, 172, 182, 191, 201, 211, 221, 231, 240, 250, 260,
                    270, 279, 289, 299, 309, 319, 328, 338, 348, 358, 368, 377, 387,
                    397, 407, 417, 426, 436, 446, 456, 466, 475, 485, 495])
        B_delta = numpy.zeros([512*512, 50])

        for idx in range(len(source_locations)):
            B_delta[:,idx] = scipy.signal.unit_impulse(512*512, source_locations[idx])
        
        if self.gpu:
            return cupy.asarray(B_delta)
        else:
            return B_delta

    @timeit 
    def get_A(self):
        c = self.xp.full(self.setup.N**2, self.setup.background_velocity_value, dtype=self.xp.float64)
        I_k = self.scipy.sparse.identity(self.setup.N)

        D_k = (1/self.setup.delta_x**2)*self.scipy.sparse.diags([1,-2,1],[-1,0,1], shape=(self.setup.N,self.setup.N), dtype=self.xp.float64)
        D_k = self.scipy.sparse.csr_matrix(D_k)
        D_k[0, 0] = -1 * (1/self.setup.delta_x**2)

        L = self.scipy.sparse.kron(D_k, I_k) + self.scipy.sparse.kron(I_k, D_k)
        C = self.scipy.sparse.diags(c, 0, dtype=self.xp.float64)
        A = (- C @ L @ C)

        return A

    @timeit
    def calculate_U(self):
        # Load D
        D_0 = self.xp.load("./bs_test/D_fine.npy")

        # Reverse the time 
        D_0 = self.xp.flip(D_0, 0)

        print(f"Shape of D_0: {D_0.shape}")
        Nt = D_0.shape[0]

        u = self.xp.zeros([3,512*512,self.setup.N_s])
        U = self.xp.zeros([int(Nt/self.nts),self.setup.N*self.setup.N])

        B_delta = self.generate_sources()
        A = self.get_A()

        factor = 2*self.scipy.sparse.identity(self.setup.N**2) - self.delta_t**2 * A

        source_idx = 25
        for time_idx in range(Nt):
            self.__print_benchmark_progress(time_idx+1, Nt)
            u[PAST,:,:] = u[PRESENT,:,:]
            u[PRESENT,:,:] = u[FUTURE,:,:]
            # u[FUTURE,:,source_idx] = factor@u[PRESENT,:,source_idx] - u[PAST,:,source_idx] + self.delta_t**2 *  B_delta @ D_0[time_idx,:,source_idx]
            u[FUTURE,:,:] = factor@u[PRESENT,:,:] - u[PAST,:,:] + self.delta_t**2 *  B_delta @ D_0[time_idx,:,:]

            if (time_idx % self.nts) == 0:
                U[int(time_idx/self.nts),:] = u[FUTURE,:,source_idx]
        
        sys.stdout.write("\n")
        self.xp.save("RTM_U.npy", U)
        


    def __print_benchmark_progress(self, progress, max, progress_bar_length=40):
        title = f"\rRTM solver progress: {progress:5}/{max:5}: "
        success_rate = f" {(progress/max)*100:3.2f}%"
        number_of_progress_indicators = int(progress * progress_bar_length // (max))

        sys.stdout.write(title + "[" + number_of_progress_indicators*"#" + (progress_bar_length - number_of_progress_indicators)*"-" + "]" + success_rate)


def main():
    N_t = 10
    use_gpu = False

    for i, arg in enumerate(sys.argv):
        if arg == '-Nt':
            N_t = int(sys.argv[i+1])
        elif arg == '-gpu':
            use_gpu = True

    sim_setup = SimulationSetup(N_t=N_t)
    solver = RTM(sim_setup, use_gpu)
    solver.calculate_U()


if __name__ == "__main__":
    main()

