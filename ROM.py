import numpy as np
from scipy import sparse
from cholesky import mblockchol
from scipy import ndimage
from os.path import exists
from SimulationSetup import SimulationSetup
from benchmark import timeit

class ROM:

    def __init__(self, setup: SimulationSetup = SimulationSetup()):

        self.setup = setup

        self.imaging_region_indices = self.get_imaging_region_indices()

        self.background_velocity_value = setup.background_velocity_value
        self.background_velocity = np.full(self.setup.N**2, setup.background_velocity_value, dtype=np.float64)

        self.delta_t = setup.tau/20
        
        if exists("./rom_data/V_0.npy"):
            self.V_0 = np.load("./rom_data/V_0.npy")
        
        else:
            self.V_0 = self.calculate_V0()
            np.save("./rom_data/V_0.npy", self.V_0)
        
        if not exists("./rom_data/I_0.npy"):
            R = np.load("./rom_data/R_0.npy")
            I_0 = self.calculate_imaging_func(R)
            np.save("./rom_data/I_0.npy", I_0)

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
        # Calculate operators
        I_k = sparse.identity(self.setup.N)
        D_k = (1/self.setup.delta_x**2)*sparse.diags([1,-2,1],[-1,0,1], shape=(self.setup.N,self.setup.N), dtype=np.float64)
        D_k = sparse.csr_matrix(D_k)
        D_k[0, 0] = -1 * (1/self.setup.delta_x**2)

        L = sparse.kron(D_k, I_k) + sparse.kron(I_k, D_k)
        C = sparse.diags(c, 0, dtype=np.float64)

        A = (- C @ L @ C)

        u = np.zeros((3, self.setup.N_x * self.setup.N_y, self.setup.N_s), dtype=np.float64) # Stores past, current and future instances

        b = self.import_sources()

        u[1] = b
        u[0] = (-0.5* self.delta_t**2 * A) @ b + b

        D = np.zeros((2*self.setup.N_t, self.setup.N_s, self.setup.N_s), dtype=np.float64)
        D[0] = np.transpose(b) @ u[1]

        return u, A, D, b 

    @timeit
    def calculate_V0(self):
        u_init, A, D_init, b = self.init_simulation(self.background_velocity)
        D, U_0 = self.calculate_u_d(u_init, A, D_init, b) 
        R = self.calculate_mass_matrix(D)
        V_0 = U_0 @ np.linalg.inv(R)

        if not exists("./rom_data/R_0.npy"):
            np.save("./rom_data/R_0.npy", R)

        return V_0

    def find_indices(self,j):
        ind_t = np.linspace(0, self.setup.N_s, self.setup.N_s) + self.setup.N_s*j 
        ind_list = [int(x) for x in ind_t]
        return ind_list

    def calculate_u_d(self, u, A, D, b):
        D = np.load("./rtm_data/D_0.npy")

        U_0_mem = np.memmap("./rtm_data/U_0.npy", np.float64, 'r', shape=(2*self.setup.N_t, self.setup.N_y_im*self.setup.N_x_im, self.setup.N_s))
        U_0 = np.array(U_0_mem)
        U_0 = U_0[0:self.setup.N_t,:,:]
        U_0 = np.moveaxis(U_0, 0, -1)
        U_0 = np.reshape(U_0, (self.setup.N_x_im * self.setup.N_y_im, self.setup.N_s*self.setup.N_t), order='F')
        
        return D, U_0

    def calculate_d(self, u, A, D, b):
        D = np.load("./rtm_data/D.npy")
        return D

    @timeit
    def calculate_intensity(self, C: np.array):
        u_init, A_init, D_init, b = self.init_simulation(C)
        D = self.calculate_d(u_init, A_init, D_init, b)
        R = self.calculate_mass_matrix(D)
        I = self.calculate_imaging_func(R)
        
        return I

    @timeit
    def calculate_mass_matrix(self, D):
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
        I = self.V_0 @ R
        I = np.square(I)

        I = I.sum(axis=1)
        
        return I

    def get_image_derivative(self, I):
        I = I.reshape((self.setup.N_y_im, self.setup.N_x_im))
        dx = -1 * np.array([[-1, 0, 1]])
        I_x = ndimage.convolve(I, dx)
        I_x = I_x.reshape((-1, 1))

        return I_x
    
    def calculate_I_matrices(self, n_images: int, plot: bool, output_file: str = ""):
        c = np.load(f"./old_fwd/fractures/circle.npy")
        I = self.calculate_intensity(c)
        np.save("./rom_data/I.npy", I)


def main():
    setup = SimulationSetup(
        N_t=65,
        N_x_im=140,
        N_y_im=155,
        O_x=25,
        O_y=180
    )

    solver = ROM(setup)

    solver.calculate_I_matrices(10, True, False)

if __name__ == "__main__":
    main()