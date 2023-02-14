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

from matplotlib import pyplot as plt

PAST = 2
PRESENT = 1
FUTURE = 0

Array = TypeVar('Array', numpy.array, cupy.array)

class WaveSolver:

    def __init__(self, setup: SimulationSetup, gpu: bool=False):

        self.setup_numpy_and_scipy(gpu)

        self.setup = setup

        self.imaging_region_indices = self.get_imaging_region_indices()
        self.background_velocity = self.xp.full(self.setup.N**2, setup.background_velocity_value, dtype=self.xp.float64)
        self.delta_t = setup.tau/20

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

    def import_sources(self):
        
        b = self.xp.loadtxt(self.setup.Bsrc_file, delimiter =',', dtype=self.xp.float64)
        self.xp.reshape(b, (self.setup.N_x * self.setup.N_y, self.setup.N_s))

        return b
    
    def get_imaging_region_indices(self):
        im_y_indices = range(self.setup.O_y, self.setup.O_y+self.setup.N_y_im)
        im_x_indices = range(self.setup.O_x, self.setup.O_x+self.setup.N_x_im)
        indices = [y*self.setup.N_x + x for y in im_y_indices for x in im_x_indices] 

        return indices

    @timeit 
    def init_simulation(self, c: Array):
        # I_k is the identity matrix
        I_k = self.scipy.sparse.identity(self.setup.N)

        # D_k is the N_x × N_y tridiagonal matrix which represents the boundary conditions. The D_k matrix is presented in Equation 6
        D_k = (1/self.setup.delta_x**2)*self.scipy.sparse.diags([1,-2,1],[-1,0,1], shape=(self.setup.N,self.setup.N), dtype=self.xp.float64)
        D_k = self.scipy.sparse.csr_matrix(D_k)
        D_k[0, 0] = -1 * (1/self.setup.delta_x**2)

        # Equation 5
        L = self.scipy.sparse.kron(D_k, I_k) + self.scipy.sparse.kron(I_k, D_k)

        # C is a diagonal matrix of size N_x N_y × N_x N_y, which stores background velocity of the medium
        C = self.scipy.sparse.diags(c, 0, dtype=self.xp.float64)

        # Equation 7
        A = (- C @ L @ C)

        # Equation 8
        u = self.xp.zeros((3, self.setup.N_x * self.setup.N_y, self.setup.N_s), dtype=self.xp.float64) # Stores past, current and future instances

        b = self.import_sources()

        # Equation 9
        u[PRESENT] = b
        u[FUTURE] = (-0.5* self.delta_t**2 * A) @ b + b

        # Equation 10
        D = self.xp.zeros((2*self.setup.N_t, self.setup.N_s, self.setup.N_s), dtype=self.xp.float64)
        D[0] = self.xp.transpose(b) @ u[1]

        return u, A, D, b 

    @timeit
    def calculate_U0_and_D0(self, path_to_c):
        """Calculate the orthogonal background snapshots V_0"""
        c = self.xp.load(path_to_c)
        c = self.xp.squeeze(c)

        # image = self.xp.reshape(c, (self.setup.N_x, self.setup.N_y))
        # image = self.xp.reshape(c[self.imaging_region_indices], (self.setup.N_x_im, self.setup.N_y_im))
        # plt.gray()
        # plt.imshow(image)
        # plt.show()
        # exit()
        u_init, A, D_init, b = self.init_simulation(c)
        D_0, U_0 = self.calculate_U_D(u_init, A, D_init, b)

        self.xp.save(self.data_folder + "D_0.npy", D_0)
        self.xp.save(self.data_folder + "U_0.npy", U_0)

        return D_0, U_0

    @timeit
    def calculate_U_D(self, u, A, D, b):
        """Calculate wave fields in the medium and the data at the receivers

        Solve the wave equation for a reference wave speed c0 without any fractures by using a finite-
        difference time domain method and collect the matrices U_0 (the wave field snapshots in the refer-
        ence medium) and D_0 (the data measured at the receivers).
        """
        nts = 20
        T = (self.setup.N_t * 2 - 1) * self.delta_t * nts
        time = self.xp.linspace(0, T, num=2*self.setup.N_t*nts)

        U_0 = self.xp.zeros((self.setup.N_x_im*self.setup.N_y_im, self.setup.N_s, self.setup.N_t))   # Can Fortran ordering be used already here?
        U_0[:,:,0] = u[1][self.imaging_region_indices]      # Check if using a (sparse) projection matrix is faster?
        
        count_storage_D = 0
        count_storage_U_0 = 0
        for i in range(1,len(time)):
            self.__print_benchmark_progress(i+1, len(time))
            u[PAST] = u[PRESENT] 
            u[PRESENT] = u[FUTURE] 
            #ts = ti.time() 
            u[FUTURE] = (-self.delta_t**2 * A) @ u[PRESENT] - u[PAST] + 2*u[PRESENT]
            #te = ti.time()
            #print(f'Function:if Execution time:{(te -ts):10.2f}s')

            if (i % nts) == 0:
                index = int(i/nts)
                D[index] = self.xp.transpose(b) @ u[1]
                D[index] = 0.5*(D[index].T + D[index])

                count_storage_D += 1
                #print(f"Count D = {count_storage_D}")

                if i <= self.setup.N_t*nts-1:
                    U_0[:,:,index] = u[PRESENT][self.imaging_region_indices]

                    #print(f"U_0 current timestep: {index+1}/{self.N_t}")

                    count_storage_U_0 += 1


        sys.stdout.write("\n")
        # self.xp.save(self.data_folder + "U_0.npy", U_0)
        # print(f"Saved U_0 of shape {U_0.shape}")
        
        U_0 = self.xp.reshape(U_0, (self.setup.N_x_im * self.setup.N_y_im, self.setup.N_s * self.setup.N_t),order='F')

        #print(f"Count D = {count_storage_D}")
        #print(f"Count stage U_0 = {count_storage_U_0}")
        return D, U_0


    def __print_benchmark_progress(self, progress, max, progress_bar_length=40):
        title = f"\rWave solver progress: {progress:5}/{max:5}: "
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
    solver = WaveSolver(sim_setup, use_gpu)
    solver.calculate_U0_and_D0("./fracture/images/circle.npy")
    
    # test = solver.import_sources()
    # print(test.shape)

if __name__ == "__main__":
    main()
