import numpy as np
from scipy import sparse
from cholesky import mblockchol
from scipy import ndimage
from os.path import exists

class ROM:

    def __init__(self, 
                N_x: int = 512,
                N_y: int = 512,
                N_s: int = 50,
                delta_x: float = 0.0063,
                tau: float = 3.0303*10**(-5),
                N_t: int = 70,
                background_velocity_value: float = 1000,
                Bsrc_file: str = "Bsrc_T.txt",
                N_x_im: int = 175,
                N_y_im: int = 350,
                O_x: int = 25,
                O_y: int = 81):

        self.N = N_x
        self.N_s = N_s
        self.N_x = N_x
        self.N_y = N_y
        self.delta_x = delta_x
        self.N_x_im = N_x_im
        self.N_y_im = N_y_im
        self.O_x = O_x
        self.O_y = O_y
        self.Bsrc_file = Bsrc_file 

        self.imaging_region_indices = self.get_imaging_region_indices()

        self.background_velocity_value = background_velocity_value
        self.background_velocity = np.full(self.N**2,
                                           background_velocity_value,
                                           dtype=np.float64)

        self.tau = tau
        self.N_t = N_t
        self.delta_t = tau/20
        
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
        
        b = np.loadtxt(self.Bsrc_file, delimiter =',', dtype=np.float64)
        np.reshape(b, (self.N_x * self.N_y, self.N_s))

        return b
    
    def get_imaging_region_indices(self):
        im_y_indices = range(self.O_y, self.O_y+self.N_y_im)
        im_x_indices = range(self.O_x, self.O_x+self.N_x_im)
        indices = [y*self.N_x + x for y in im_y_indices for x in im_x_indices] 

        return indices
        
    def init_simulation(self, c: np.array):
        # Calculate operators
        I_k = sparse.identity(self.N)
        D_k = (1/self.delta_x**2)*sparse.diags([1,-2,1],[-1,0,1], shape=(self.N,self.N), dtype=np.float64)
        D_k = sparse.csr_matrix(D_k)
        D_k[0, 0] = -1 * (1/self.delta_x**2)

        L = sparse.kron(D_k, I_k) + sparse.kron(I_k, D_k)
        C = sparse.diags(c, 0, dtype=np.float64)

        A = (- C @ L @ C)

        u = np.zeros((3, self.N_x * self.N_y, self.N_s), dtype=np.float64) # Stores past, current and future instances

        b = self.import_sources()

        u[1] = b
        u[0] = (-0.5* self.delta_t**2 * A) @ b + b

        D = np.zeros((2*self.N_t, self.N_s, self.N_s), dtype=np.float64)
        D[0] = np.transpose(b) @ u[1]

        return u, A, D, b 

    def calculate_V0(self):
        u_init, A, D_init, b = self.init_simulation(self.background_velocity)
        D, U_0 = self.calculate_u_d(u_init, A, D_init, b) 
        R = self.calculate_mass_matrix(D)
        V_0 = U_0 @ np.linalg.inv(R)

        if not exists("./rom_data/R_0.npy"):
            np.save("./rom_data/R_0.npy", R)

        return V_0

    def find_indices(self,j):
        ind_t = np.linspace(0, self.N_s, self.N_s) + self.N_s*j 
        ind_list = [int(x) for x in ind_t]
        return ind_list

    def calculate_u_d(self, u, A, D, b):
        D = np.load("./rtm_data/D_0.npy")

        U_0_mem = np.memmap("./rtm_data/U_0.npy", np.float64, 'r', shape=(2*self.N_t, self.N_y_im*self.N_x_im, self.N_s))
        U_0 = np.array(U_0_mem)
        U_0 = U_0[0:self.N_t,:,:]
        U_0 = np.moveaxis(U_0, 0, -1)
        U_0 = np.reshape(U_0, (self.N_x_im * self.N_y_im, self.N_s*self.N_t), order='F')
        
        return D, U_0

    def calculate_d(self, u, A, D, b):
        D = np.load("./rtm_data/D.npy")
        return D

    def calculate_intensity(self, C: np.array):
        u_init, A_init, D_init, b = self.init_simulation(C)
        D = self.calculate_d(u_init, A_init, D_init, b)
        R = self.calculate_mass_matrix(D)
        I = self.calculate_imaging_func(R)
        I = self.get_image_derivative(I)

        return I

    def calculate_mass_matrix(self, D):
        M = np.zeros((self.N_s*self.N_t, self.N_s*self.N_t), dtype=np.float64)

        for i in range(self.N_t):
            for j in range(self.N_t):
                ind_i = self.find_indices(i)
                ind_j = self.find_indices(j)

                M[ind_i[0]:ind_i[-1],ind_j[0]:ind_j[-1]] = 0.5 * (D[abs(i-j)] + D[abs(i+j)])

        R = mblockchol(M, self.N_s, self.N_t)

        return R
    
    def calculate_imaging_func(self, R):
        I = self.V_0 @ R
        I = np.square(I)

        I = I.sum(axis=1)
        
        return I

    def get_image_derivative(self, I):
        I = I.reshape((self.N_y_im, self.N_x_im))
        dx = -1 * np.array([[-1, 0, 1]])
        I_x = ndimage.convolve(I, dx)
        I_x = I_x.reshape((-1, 1))

        return I_x
    
    def calculate_I_matrices(self, n_images: int, plot: bool, output_file: str = ""):
        c = np.load(f"./old_fwd/fractures/circle.npy")
        I = self.calculate_intensity(c)
        np.save("./rom_data/I.npy", I)


def main():
    solver = ROM( 
                N_x = 512,
                N_y = 512,
                N_s = 50,
                delta_x = 0.0063,
                tau = 3.0303*10**(-5),
                N_t = 65,
                background_velocity_value = 1000,
                Bsrc_file = "Bsrc_T.txt",
                N_x_im = 140,
                N_y_im = 155,
                O_x = 25,
                O_y = 180
    )

    solver.calculate_I_matrices(10, True, False)

if __name__ == "__main__":
    main()