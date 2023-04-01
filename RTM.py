import sys

import numpy
import scipy

import cupy
import cupyx

from os import path

from SimulationSetup import SimulationSetup
from benchmark import timeit

from cpu_gpu_abstraction import CPU_GPU_Abstractor, ExecutionSetup

PAST = 2
PRESENT = 1
FUTURE = 0

class RTM(CPU_GPU_Abstractor):
    def __init__(self, setup: SimulationSetup, exec_setup: ExecutionSetup):
        super(RTM, self).__init__(exec_setup=exec_setup)

        self.setup = setup
        self.background_velocity = self.xp.full(self.setup.N**2, setup.background_velocity_value, dtype=self.exec_setup.precision)
        self.nts = 20
        self.delta_t = setup.tau/self.nts
        self.imaging_region_indices = self.get_imaging_region_indices()
        self.I_0_path = self.exec_setup.data_folder + "I_0_RTM.npy"

        self.I_0 = self.get_I0()

    def get_I0(self):
        if path.exists(self.I_0_path):
            return self.xp.load(self.I_0_path)
        else:
            I_0 = self.__calculate_I0()
            self.xp.save(self.I_0_path, I_0)
            return I_0
        
    def __calculate_I0(self):
        self.__calculate_U_RT(D="D_0_fine.npy", U_RT_file_name="U_0_RT.npy")
        return self.__calculate_imaging_function(U_RT_file_name="U_0_RT.npy", U_0_file_name="U_0.npy")

    def generate_sources(self):
        source_locations = numpy.array([15,  25,  35,  44,  54,  64,  74,  84,  93, 103, 113, 123, 133,
                    142, 152, 162, 172, 182, 191, 201, 211, 221, 231, 240, 250, 260,
                    270, 279, 289, 299, 309, 319, 328, 338, 348, 358, 368, 377, 387,
                    397, 407, 417, 426, 436, 446, 456, 466, 475, 485, 495])
        B_delta = numpy.zeros([512*512, 50], dtype=self.exec_setup.precision)

        for idx in range(len(source_locations)):
            B_delta[:,idx] = scipy.signal.unit_impulse(512*512, source_locations[idx])
        
        B_delta = numpy.reshape(B_delta, (512,512,50))
        B_delta = numpy.transpose(B_delta, (1,0,2))
        B_delta = numpy.reshape(B_delta, (512*512,50))

        if self.exec_setup.gpu:
            return cupy.asarray(B_delta)
        else:
            return B_delta
    
    def get_imaging_region_indices(self):
        im_y_indices = range(self.setup.O_y, self.setup.O_y+self.setup.N_y_im)
        im_x_indices = range(self.setup.O_x, self.setup.O_x+self.setup.N_x_im)
        indices = [y*self.setup.N_x + x for y in im_y_indices for x in im_x_indices] 

        return indices

    @timeit 
    def get_A(self):
        c = self.xp.full(self.setup.N**2, self.setup.background_velocity_value, dtype=self.exec_setup.precision)
        I_k = self.scipy.sparse.identity(self.setup.N)

        D_k = (1/self.setup.delta_x**2)*self.scipy.sparse.diags([1,-2,1],[-1,0,1], shape=(self.setup.N,self.setup.N), dtype=self.exec_setup.precision)
        D_k = self.scipy.sparse.csr_matrix(D_k, dtype=self.exec_setup.precision)
        D_k[0, 0] = -1 * (1/self.setup.delta_x**2)

        L = self.scipy.sparse.kron(D_k, I_k) + self.scipy.sparse.kron(I_k, D_k)
        C = self.scipy.sparse.diags(c, 0, dtype=self.exec_setup.precision)
        A = (- C @ L @ C)

        return A

    @timeit
    def __calculate_U_RT(self, D, U_RT_file_name="U_RT.npy"):
        D = self._get_array_from_disk_or_mem(D)
        # Reverse the time 
        D = self.xp.flip(D, 0)

        u = self.xp.zeros([3,512*512,self.setup.N_s], dtype=self.exec_setup.precision)
        U = numpy.memmap(self.exec_setup.data_folder + U_RT_file_name, self.exec_setup.precision, 'w+', shape=(2*self.setup.N_t,self.setup.N_x_im*self.setup.N_y_im, self.setup.N_s))

        B_delta = self.generate_sources()
        A = self.get_A()

        factor = 2*self.scipy.sparse.identity(self.setup.N**2) - self.delta_t**2 * A

        for time_idx in range(2*self.setup.N_t * self.nts):
            self._print_progress_bar(time_idx+1, 2*self.setup.N_t*self.nts, title="Reverse time calculation")
            u[PAST,:,:] = u[PRESENT,:,:]
            u[PRESENT,:,:] = u[FUTURE,:,:]
            u[FUTURE,:,:] = factor@u[PRESENT,:,:] - u[PAST,:,:] + self.delta_t**2 *  B_delta @ D[time_idx,:,:]

            if (time_idx % self.nts) == 0:
                U[int(time_idx/self.nts),:,:] = cupy.asnumpy(u[FUTURE,:,:][self.imaging_region_indices])

        self._end_progress_bar()
        U.flush()

    def __calculate_imaging_function(self, U_RT_file_name="U_RT.npy", U_0_file_name="U_0.npy"):
        U_RT = numpy.memmap(self.exec_setup.data_folder + U_RT_file_name, self.exec_setup.precision_np, 'r', shape=(2*self.setup.N_t, self.setup.N_x_im*self.setup.N_y_im, self.setup.N_s))
        U_0 = numpy.memmap(self.exec_setup.data_folder + U_0_file_name, self.exec_setup.precision_np, 'r', shape=(2*self.setup.N_t, self.setup.N_x_im*self.setup.N_y_im, self.setup.N_s))

        I = self.xp.zeros((self.setup.N_x_im*self.setup.N_y_im, self.setup.N_s), dtype=self.exec_setup.precision)

        for tidx in range(2*self.setup.N_t):
            self._print_progress_bar(tidx+1, 2*self.setup.N_t, title="RTM Integral progress")
            t_idx_backward = 2*self.setup.N_t -tidx -1
            t_idx_forward = tidx

            U_RT_temp = self.xp.array(U_RT[t_idx_forward,:,:])
            U_0_temp = self.xp.array(U_0[t_idx_backward,:,:])

            for s_idx in range(self.setup.N_s):
                I[:,s_idx] += U_RT_temp[:,s_idx] * U_0_temp[:,s_idx] * self.setup.tau
        self._end_progress_bar()

        return self.xp.sum(I, axis=1)
    
    @timeit
    def calculate_I(self, D, U_0_file_name="U_0.npy", I_file_name=""):
        D = self._get_array_from_disk_or_mem(D)
        self.__calculate_U_RT(D, U_RT_file_name="U_RT.npy")
        I = self.__calculate_imaging_function(U_RT_file_name="U_RT.npy", U_0_file_name=U_0_file_name)
        I = I - self.I_0
        if I_file_name:
            self.xp.save(self.exec_setup.data_folder + I_file_name, I)
        
        return cupy.asnumpy(self.xp.squeeze(I))
        

def main():
    use_gpu = False

    for i, arg in enumerate(sys.argv):
        if arg == '-gpu':
            use_gpu = True

    sim_setup = SimulationSetup(N_t=35)
    exec_setup = ExecutionSetup(gpu=use_gpu, precision='float32')
    solver = RTM(sim_setup, exec_setup)
    solver.calculate_I(D="U_RT.npy", I_file_name="I.npy")


if __name__ == "__main__":
    main()

