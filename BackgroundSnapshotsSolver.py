import time as ti
import numpy as np
from scipy import sparse
from cholesky import mblockchol
#import cupyx
#import cupy as cp
from os.path import exists
from os.path import sep
import sys
from SimulationSetup import SimulationSetup
from benchmark import timeit

class BackgroundSnapshotsSolver:

    def __init__(self, setup: SimulationSetup):

        self.setup = setup

        self.imaging_region_indices = self.get_imaging_region_indices()
        self.background_velocity = np.full(self.setup.N**2, setup.background_velocity_value, dtype=np.float64)
        self.delta_t = setup.tau/20

        self.data_folder = "." + sep + "bs_test" + sep
        
        # Calculate or load the orthogonal background snapshots V0
        if exists(self.data_folder + "V_0.np."):
            self.V_0 = np.load(self.data_folder + "V_0.np.")
        
        else:
            self.V_0 = self.calculate_V0()
            np.save(self.data_folder + "V_0.npy", self.V_0)
        
        if not exists(self.data_folder + "I_0.np."):
            M_0 = np.load(self.data_folder + "M_0.npy")
            I_0 = self.calculate_imaging_func(M_0)
            np.save(self.data_folder + "I_0.npy", I_0)

    def import_sources(self):
        
        b = np.loadtxt(self.setup.Bsrc_file, delimiter =',', dtype=np.float64)
        np.reshape(b, (self.setup.N_x * self.setup.N_y, self.setup.N_s))

        return b
    
    def get_imaging_region_indices(self):
        im_y_indices = range(self.setup.O_y, self.setup.O_y+self.setup.N_y_im)
        im_x_indices = range(self.setup.O_x, self.setup.O_x+self.setup.N_x_im)
        indices = [y*self.setup.N_x + x for y in im_y_indices for x in im_x_indices] 

        return indices

    @timeit 
    def init_simulation(self, c: np.array):
        # I_k is the identity matrix
        I_k = sparse.identity(self.setup.N)

        # D_k is the N_x × N_y tridiagonal matrix which represents the boundary conditions. The D_k matrix is presented in Equation 6
        D_k = (1/self.setup.delta_x**2)*sparse.diags([1,-2,1],[-1,0,1], shape=(self.setup.N,self.setup.N), dtype=np.float64)
        D_k = sparse.csr_matrix(D_k)
        D_k[0, 0] = -1 * (1/self.setup.delta_x**2)

        # Equation 5
        L = sparse.kron(D_k, I_k) + sparse.kron(I_k, D_k)

        # C is a diagonal matrix of size N_x N_y × N_x N_y, which stores background velocity of the medium
        C = sparse.diags(c, 0, dtype=np.float64)

        # Equation 7
        A = (- C @ L @ C)

        # Equation 8
        u = np.zeros((3, self.setup.N_x * self.setup.N_y, self.setup.N_s), dtype=np.float64) # Stores past, current and future instances

        b = self.import_sources()

        # Equation 9
        u[1] = b
        u[0] = (-0.5* self.delta_t**2 * A) @ b + b

        # Equation 10
        D = np.zeros((2*self.setup.N_t, self.setup.N_s, self.setup.N_s), dtype=np.float64)
        D[0] = np.transpose(b) @ u[1]

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
        ind_t = np.linspace(0, self.setup.N_s, self.setup.N_s) + self.setup.N_s*j 
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
        T = (self.setup.N_t * 2 - 1) * self.delta_t * nts
        time = np.linspace(0, T, num=2*self.setup.N_t*nts)

        U_0 = np.zeros((self.setup.N_x_im*self.setup.N_y_im, self.setup.N_s, self.setup.N_t))   # Can Fortran ordering be used already here?
        U_0[:,:,0] = u[1][self.imaging_region_indices]      # Check if using a (sparse) projection matrix is faster?
        
        count_storage_D = 0
        count_storage_U_0 = 0
        for i in range(1,len(time)):
            self.__print_benchmark_progress(i+1, len(time))
            u[2] = u[1] 
            u[1] = u[0] 
            #ts = ti.time() 
            u[0] = (-self.delta_t**2 * A) @ u[1] - u[2] + 2*u[1]
            #te = ti.time()
            #print(f'Function:if Execution time:{(te -ts):10.2f}s')

            if (i % nts) == 0:
                index = int(i/nts)
                D[index] = np.transpose(b) @ u[1]
                D[index] = 0.5*(D[index].T + D[index])

                count_storage_D += 1
                #print(f"Count D = {count_storage_D}")

                if i <= self.setup.N_t*nts-1:
                    U_0[:,:,index] = u[1][self.imaging_region_indices]

                    #print(f"U_0 current timestep: {index+1}/{self.N_t}")

                    count_storage_U_0 += 1


        sys.stdout.write("\n")
        np.save(self.data_folder + "U_0.npy", U_0)
        print(f"Saved U_0 of shape {U_0.shape}")
        
        U_0 = np.reshape(U_0, (self.setup.N_x_im * self.setup.N_y_im, self.setup.N_s * self.setup.N_t),order='F')

        #print(f"Count D = {count_storage_D}")
        #print(f"Count stage U_0 = {count_storage_U_0}")
        return D, U_0

    @timeit
    def calculate_mass_matrix(self, D):
        """Calculate the Grammian matrix, also revered to as "Mass Matrix" using Block-Cholesky factorization"""
        M = np.zeros((self.setup.N_s*self.setup.N_t, self.setup.N_s*self.setup.N_t), dtype=np.float64)

        for i in range(self.setup.N_t):
            for j in range(self.setup.N_t):
                ind_i = self.find_indices(i)
                ind_j = self.find_indices(j)

                M[ind_i[0]:ind_i[-1],ind_j[0]:ind_j[-1]] = 0.5 * (D[abs(i-j)] + D[abs(i+j)])

        R = mblockchol(M, self.setup.N_s, self.setup.N_t)

        return R
    
    @timeit
    def calculate_imaging_func(self, R):
        """
        # for i in range(self.N_x_im*self.N_y_im):
        #     I[i] = np.linalg.norm(V_0[i, :] @ R, 2)**2
        """
        I = self.V_0 @ R
        I = np.square(I)

        I = I.sum(axis=1)
        
        return I

    def __print_benchmark_progress(self, progress, max, progress_bar_length=40):
        title = f"\rWave solver progress: {progress:5}/{max:5}: "
        success_rate = f" {(progress/max)*100:3.2f}%"
        number_of_progress_indicators = int(progress * progress_bar_length // (max))

        sys.stdout.write(title + "[" + number_of_progress_indicators*"#" + (progress_bar_length - number_of_progress_indicators)*"-" + "]" + success_rate)


def main():
    sim_setup = SimulationSetup(N_t=6)

    solver = BackgroundSnapshotsSolver(sim_setup)

if __name__ == "__main__":
    main()
