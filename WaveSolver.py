import sys
from os import path

import numpy

from benchmark import timeit
from cpu_gpu_abstraction import CPU_GPU_Abstractor, ExecutionSetup
from SimulationSetup import SimulationSetup

from typing import TypeVar

PAST = 2
PRESENT = 1
FUTURE = 0


class WaveSolver(CPU_GPU_Abstractor):

    def __init__(self, setup: SimulationSetup, exec_setup: ExecutionSetup):
        super(WaveSolver, self).__init__(exec_setup=exec_setup)

        self.setup = setup

        self.imaging_region_indices = self.get_imaging_region_indices()
        self.background_velocity = self.xp.full(self.setup.N**2, setup.background_velocity_value, dtype=self.exec_setup.precision)
        self.delta_t = setup.tau/20

    def import_sources(self):
        
        b = self.xp.loadtxt(self.setup.Bsrc_file, delimiter =',', dtype=self.exec_setup.precision)
        self.xp.reshape(b, (self.setup.N_x * self.setup.N_y, self.setup.N_s))

        return b
    
    def get_imaging_region_indices(self):
        im_y_indices = range(self.setup.O_y, self.setup.O_y+self.setup.N_y_im)
        im_x_indices = range(self.setup.O_x, self.setup.O_x+self.setup.N_x_im)
        indices = [x*self.setup.N_x + y for x in im_x_indices for y in im_y_indices] 

        return indices

    @timeit 
    def init_simulation(self, c):
        # I_k is the identity matrix
        I_k = self.scipy.sparse.identity(self.setup.N)

        # D_k is the N_x × N_y tridiagonal matrix which represents the boundary conditions. The D_k matrix is presented in Equation 6
        D_k = (1/self.setup.delta_x**2)*self.scipy.sparse.diags([1,-2,1],[-1,0,1], shape=(self.setup.N,self.setup.N), dtype=self.exec_setup.precision)
        D_k = self.scipy.sparse.csr_matrix(D_k)
        D_k[0, 0] = -1 * (1/self.setup.delta_x**2)

        # Equation 5
        L = self.scipy.sparse.kron(D_k, I_k) + self.scipy.sparse.kron(I_k, D_k)

        # C is a diagonal matrix of size N_x N_y × N_x N_y, which stores background velocity of the medium
        C = self.scipy.sparse.diags(c, 0, dtype=self.exec_setup.precision)

        # Equation 7
        A = (- C @ L @ C)

        # Equation 8
        u = self.xp.zeros((3, self.setup.N_x * self.setup.N_y, self.setup.N_s), dtype=self.exec_setup.precision) # Stores past, current and future instances

        b = self.import_sources()

        # Equation 9
        u[PRESENT] = b
        u[FUTURE] = (-0.5* self.delta_t**2 * A) @ b + b

        # Equation 10
        D = self.xp.zeros((2*self.setup.N_t, self.setup.N_s, self.setup.N_s), dtype=self.exec_setup.precision)
        D[0] = self.xp.transpose(b) @ u[1]

        return u, A, D, b 

    @timeit
    def calculate_U_D(self, c):
        """Calculate wave fields in the medium and the data at the receivers for a given fracture"""
        c = self._get_array_from_disk_or_mem(c)
        c = self.xp.squeeze(c)

        u_init, A, D_init, b = self.init_simulation(c)
        D, D_fine = self.__calculate_U_D(u_init, A, D_init, b)

        self.xp.save(self.exec_setup.data_folder + "D.npy", D)
        self.xp.save(self.exec_setup.data_folder + "D_fine.npy", D_fine)

        return D, D_fine
    
    @timeit
    def calculate_U0_D0(self, file_name="U_0.npy"):
        """Calculate wave fields in the medium and the data at the receivers without any fractures"""

        u_init, A, D_init, b = self.init_simulation(self.background_velocity)
        D_0, D_0_fine = self.__calculate_U_D(u_init, A, D_init, b, file_name)

        self.xp.save(self.exec_setup.data_folder + "D_0.npy", D_0)
        self.xp.save(self.exec_setup.data_folder + "D_0_fine.npy", D_0_fine)


    def __calculate_U_D(self, u, A, D, b, U_file_name="U.npy"):
        nts = 20
        D_fine = self.xp.zeros((2*self.setup.N_t*nts, self.setup.N_s, self.setup.N_s), dtype=self.exec_setup.precision)
        T = (self.setup.N_t * 2 - 1) * self.delta_t * nts
        time = self.xp.linspace(0, T, num=2*self.setup.N_t*nts)

        U_0 = numpy.memmap(self.exec_setup.data_folder+U_file_name, self.exec_setup.precision_np, 'w+', shape=(2*self.setup.N_t, self.setup.N_y_im*self.setup.N_x_im, self.setup.N_s))
        U_0[0,:,:] = self.asnumpy(u[PRESENT][self.imaging_region_indices])      # Check if using a (sparse) projection matrix is faster?
        
        for i in range(1,len(time)):
            self._print_progress_bar(i+1, len(time), title="Wave solver progress")
            u[PAST] = u[PRESENT] 
            u[PRESENT] = u[FUTURE] 
            u[FUTURE] = (-self.delta_t**2 * A) @ u[PRESENT] - u[PAST] + 2*u[PRESENT]

            D_fine[i] = self.xp.transpose(b) @ u[PRESENT]
            D_fine[i] = 0.5*(D_fine[i].T + D_fine[i])

            if (i % nts) == 0:
                index = int(i/nts)
                D[index] = self.xp.transpose(b) @ u[PRESENT]
                D[index] = 0.5*(D[index].T + D[index])
                
                U_0[index,:,:] = self.asnumpy(u[PRESENT][self.imaging_region_indices])

        self._end_progress_bar()
        U_0.flush()

        return D, D_fine


def main():
    use_gpu = False

    for i, arg in enumerate(sys.argv):
        if arg == '-gpu':
            use_gpu = True

    sim_setup = SimulationSetup(N_t=65,
                                N_x_im=140,
                                N_y_im=155,
                                O_x=25,
                                O_y=180)
    exec_setup = ExecutionSetup(
        gpu=use_gpu,
        precision='float64')
    solver = WaveSolver(sim_setup, exec_setup)
    solver.calculate_U0_D0()
    # solver.calculate_U_D("old_fwd/fractures/circle.npy")


if __name__ == "__main__":
    main()
