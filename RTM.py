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

        self.data_folder = f".{path.sep}rtm_data{path.sep}"    

    def generate_sources(self):
        source_locations = numpy.array([15,  25,  35,  44,  54,  64,  74,  84,  93, 103, 113, 123, 133,
                    142, 152, 162, 172, 182, 191, 201, 211, 221, 231, 240, 250, 260,
                    270, 279, 289, 299, 309, 319, 328, 338, 348, 358, 368, 377, 387,
                    397, 407, 417, 426, 436, 446, 456, 466, 475, 485, 495])
        B_delta = numpy.zeros([512*512, 50], dtype=self.exec_setup.precision)

        for idx in range(len(source_locations)):
            B_delta[:,idx] = scipy.signal.unit_impulse(512*512, source_locations[idx])
        
        return cupy.asarray(B_delta)

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
    def calculate_U_RT(self, D_file_name="D_fine.npy", U_RT_file_name="U_RT.npy"):
        D = self.xp.load(self.data_folder + D_file_name)

        # Reverse the time 
        D = self.xp.flip(D, 0)

        Nt = D.shape[0]

        u = self.xp.zeros([3,512*512,self.setup.N_s], dtype=self.exec_setup.precision)
        U = numpy.memmap(self.data_folder + U_RT_file_name, self.exec_setup.precision, 'w+', shape=(int(Nt/self.nts),self.setup.N*self.setup.N, self.setup.N_s))

        B_delta = self.generate_sources()
        A = self.get_A()

        factor = 2*self.scipy.sparse.identity(self.setup.N**2) - self.delta_t**2 * A

        for time_idx in range(Nt):
            self._print_progress_bar(time_idx+1, Nt)
            u[PAST,:,:] = u[PRESENT,:,:]
            u[PRESENT,:,:] = u[FUTURE,:,:]
            u[FUTURE,:,:] = factor@u[PRESENT,:,:] - u[PAST,:,:] + self.delta_t**2 *  B_delta @ D[time_idx,:,:]

            if (time_idx % self.nts) == 0:
                U[int(time_idx/self.nts),:,:] = cupy.asnumpy(u[FUTURE,:,:])

        sys.stdout.write("\n")
        U.flush()
        

def main():
    N_t = 70
    use_gpu = False

    for i, arg in enumerate(sys.argv):
        if arg == '-Nt':
            N_t = int(sys.argv[i+1])
        elif arg == '-gpu':
            use_gpu = True

    sim_setup = SimulationSetup(N_t=N_t)
    exec_setup = ExecutionSetup(gpu=use_gpu, precision='float64')
    solver = RTM(sim_setup, exec_setup)
    solver.calculate_U_RT()


if __name__ == "__main__":
    main()

