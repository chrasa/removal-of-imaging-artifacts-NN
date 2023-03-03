import numpy as np
from scipy import sparse
from cholesky import mblockchol
#import cupyx
import cupy as cp
from os.path import exists
from os.path import sep
from SimulationSetup import SimulationSetup
from benchmark import timeit
from cpu_gpu_abstraction import CPU_GPU_Abstractor, ExecutionSetup

class BackgroundSnapshotsSolver(CPU_GPU_Abstractor):

    def __init__(self, setup: SimulationSetup, exec_setup: ExecutionSetup):
        super(BackgroundSnapshotsSolver, self).__init__(exec_setup=exec_setup)

        self.setup = setup
        self.N = setup.N_x
        self.N_s = setup.N_s
        self.N_x = setup.N_x
        self.N_y = setup.N_y
        self.delta_x = setup.delta_x
        self.N_x_im = setup.N_x_im
        self.N_y_im = setup.N_y_im
        self.O_x = setup.O_x
        self.O_y = setup.O_y
        self.Bsrc_file = setup.Bsrc_file 

        self.imaging_region_indices = self.get_imaging_region_indices()

        self.background_velocity_value = setup.background_velocity_value
        self.background_velocity = np.full(self.N**2, setup.background_velocity_value, dtype=np.float64)

        self.tau = setup.tau
        self.N_t = setup.N_t
        self.delta_t = setup.tau/20

        self.data_folder = "." + sep + "rom_data" + sep
        
        # Calculate or load the orthogonal background snapshots V0
        if exists(self.data_folder + "V_0.np."):
            self.V_0 = self.xp.load(self.data_folder + "V_0.np.")
        else:
            self.V_0 = self.calculate_V0()
            self.xp.save(self.data_folder + "V_0.npy", self.V_0)
        
        if not exists(self.data_folder + "I_0.np."):
            R = np.load(self.data_folder + "R_0.npy")
            I_0 = self.calculate_imaging_func(R)
            np.save(self.data_folder + "I_0.npy", I_0)

    def import_sources(self):
        
        b = self.xp.loadtxt(self.Bsrc_file, delimiter =',', dtype=np.float64)
        self.xp.reshape(b, (self.N_x * self.N_y, self.N_s))

        return b
    
    def get_imaging_region_indices(self):
        im_y_indices = range(self.O_y, self.O_y+self.N_y_im)
        im_x_indices = range(self.O_x, self.O_x+self.N_x_im)
        indices = [y*self.N_x + x for y in im_y_indices for x in im_x_indices] 

        return indices

    @timeit 
    def init_simulation(self, c: np.array):
        # I_k is the identity matrix
        I_k = self.scipy.sparse.identity(self.N)

        # D_k is the N_x × N_y tridiagonal matrix which represents the boundary conditions. The D_k matrix is presented in Equation 6
        D_k = (1/self.delta_x**2)*self.scipy.sparse.diags([1,-2,1],[-1,0,1], shape=(self.N,self.N), dtype=np.float64)
        D_k = self.scipy.sparse.csr_matrix(D_k)
        D_k[0, 0] = -1 * (1/self.delta_x**2)

        # Equation 5
        L = self.scipy.sparse.kron(D_k, I_k) + self.scipy.sparse.kron(I_k, D_k)

        # C is a diagonal matrix of size N_x N_y × N_x N_y, which stores background velocity of the medium
        C = self.scipy.sparse.diags(c, 0, dtype=np.float64)

        # Equation 7
        A = (- C @ L @ C)

        # Equation 8
        u = self.xp.zeros((3, self.N_x * self.N_y, self.N_s), dtype=np.float64) # Stores past, current and future instances

        b = self.import_sources()

        # Equation 9
        u[1] = b
        u[0] = (-0.5* self.delta_t**2 * A) @ b + b

        # Equation 10
        D = self.xp.zeros((2*self.N_t, self.N_s, self.N_s), dtype=np.float64)
        D[0] = self.xp.transpose(b) @ u[1]

        return u, A, D, b 

    @timeit
    def calculate_V0(self):
        """Calculate the orthogonal background snapshots V_0"""
        u_init, A, D_init, b = self.init_simulation(self.background_velocity)
        D_0, U_0 = self.calculate_U_D(u_init, A, D_init, b)
        M_0 = self.calculate_mass_matrix(D_0)
        V_0 = U_0 @ np.linalg.inv(M_0)

        if not exists(self.data_folder + "M_0.np."):
            np.save(self.data_folder + "M_0.npy", M_0)

        return V_0

    def find_indices(self,j):
        ind_t = np.linspace(0, self.N_s, self.N_s) + self.N_s*j 
        ind_list = [int(x) for x in ind_t]
        return ind_list

    @timeit
    def calculate_U_D(self, u, A, D, b):
        """Calculate wave fields in the medium and the data at the receivers

        Solve the wave equation for a reference wave speed c0 without any fractures by using a finite-
        difference time domain method and collect the matrices U_0 (the wave field snapshots in the refer-
        ence medium) and D_0 (the data measured at the receivers).
        """
        nts = 20
        T = (self.N_t * 2 - 1) * self.delta_t * nts
        time = self.xp.linspace(0, T, num=2*self.N_t*nts)

        # U_0 = self.xp.zeros((self.N_x_im*self.N_y_im, self.N_s, self.N_t))   # Can Fortran ordering be used already here?
        U_0 = np.memmap(self.data_folder+"U_0.npy", np.float32, 'w+', shape=(self.N_x_im*self.N_y_im, self.N_s, self.N_t))
        U_0[:,:,0] = cp.asnumpy(u[1][self.imaging_region_indices])      # Check if using a (sparse) projection matrix is faster?
        
        for i in range(1,len(time)):
            self._print_progress_bar(i+1, len(time), title="Calculate U and D")
            u[2] = u[1] 
            u[1] = u[0] 
            u[0] = (-self.delta_t**2 * A) @ u[1] - u[2] + 2*u[1]

            if (i % nts) == 0:
                index = int(i/nts)
                D[index] = np.transpose(b) @ u[1]
                D[index] = 0.5*(D[index].T + D[index])

        #         if i <= self.N_t*nts-1:
        #             U_0[:,:,index] = cp.asnumpy(u[1][self.imaging_region_indices])

        # U_0 = np.array(U_0) # Load array completely into CPU memory
        # U_0 = np.reshape(U_0, (self.N_x_im * self.N_y_im, self.N_s * self.N_t))
        # np.save(self.data_folder + "U_0.npy", U_0)

        self.xp.save(self.data_folder + "D_0.npy", D)

        return D, U_0

    @timeit
    def calculate_mass_matrix(self, D):
        """Calculate the Grammian matrix, also revered to as "Mass Matrix" using Block-Cholesky factorization"""
        M = np.zeros((self.N_s*self.N_t, self.N_s*self.N_t), dtype=np.float64)

        for i in range(self.N_t):
            for j in range(self.N_t):
                ind_i = self.find_indices(i)
                ind_j = self.find_indices(j)

                M[ind_i[0]:ind_i[-1],ind_j[0]:ind_j[-1]] = 0.5 * (D[abs(i-j)] + D[abs(i+j)])

        R = mblockchol(M, self.N_s, self.N_t)

        return R
    
    @timeit
    def calculate_imaging_func(self, R):
        I = self.V_0 @ R
        I = np.square(I)

        I = I.sum(axis=1)
        
        return I



def main():
    sim_setup = SimulationSetup()
    exec_setup = ExecutionSetup(gpu=True)

    solver = BackgroundSnapshotsSolver(sim_setup, exec_setup)

if __name__ == "__main__":
    main()
