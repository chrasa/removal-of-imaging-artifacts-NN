from cholesky import mblockchol
from os.path import exists
from SimulationSetup import SimulationSetup
from benchmark import timeit
from cpu_gpu_abstraction import CPU_GPU_Abstractor, ExecutionSetup


class ROM(CPU_GPU_Abstractor):

    def __init__(self, setup: SimulationSetup, exec_setup: ExecutionSetup):
        super(ROM, self).__init__(exec_setup=exec_setup)

        self.setup = setup

        if exists("./rom_data/V_0.npy"):
            self.V_0 = self.xp.load("./rom_data/V_0.npy")
        
        else:
            self.V_0 = self.calculate_V0()
            self.xp.save("./rom_data/V_0.npy", self.V_0)
        
        if not exists("./rom_data/I_0.npy"):
            R = self.xp.load("./rom_data/R_0.npy")
            I_0 = self.calculate_imaging_func(R)
            self.xp.save("./rom_data/I_0.npy", I_0)

    @timeit
    def calculate_V0(self):
        D_0, U_0 = self.load_D0_and_U0() 
        R = self.calculate_mass_matrix(D_0)
        V_0 = U_0 @ self.xp.linalg.inv(R)

        if not exists("./rom_data/R_0.npy"):
            self.xp.save("./rom_data/R_0.npy", R)

        return V_0

    def find_indices(self,j):
        ind_t = self.xp.linspace(0, self.setup.N_s, self.setup.N_s) + self.setup.N_s*j 
        ind_list = [int(x) for x in ind_t]
        return ind_list

    def load_D0_and_U0(self):
        D = self.xp.load("./rtm_data/D_0.npy")

        U_0_mem = self.xp.memmap("./rtm_data/U_0.npy", self.exec_setup.precision, 'r', shape=(2*self.setup.N_t, self.setup.N_y_im*self.setup.N_x_im, self.setup.N_s))
        U_0 = self.xp.array(U_0_mem)
        U_0 = U_0[0:self.setup.N_t,:,:]
        U_0 = self.xp.moveaxis(U_0, 0, -1)
        U_0 = self.xp.reshape(U_0, (self.setup.N_x_im * self.setup.N_y_im, self.setup.N_s*self.setup.N_t), order='F')
        
        return D, U_0

    @timeit
    def calculate_intensity(self):
        D = self.xp.load("./rtm_data/D.npy")
        R = self.calculate_mass_matrix(D)
        I = self.calculate_imaging_func(R)
        
        return I

    @timeit
    def calculate_mass_matrix(self, D):
        M = self.xp.zeros((self.setup.N_s*self.setup.N_t, self.setup.N_s*self.setup.N_t), dtype=self.exec_setup.precision)

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
        I = self.xp.square(I)

        I = I.sum(axis=1)
        
        return I

    def get_image_derivative(self, I):
        I = I.reshape((self.setup.N_y_im, self.setup.N_x_im))
        dx = -1 * self.xp.array([[-1, 0, 1]])
        I_x = self.scipy.ndimage.convolve(I, dx)
        I_x = I_x.reshape((-1, 1))

        return I_x
    
    def calculate_I_matrices(self):
        I = self.calculate_intensity()
        I_0 = self.xp.load("./rom_data/I_0.npy")

        I = I - I_0
        I = self.get_image_derivative(I)
        self.xp.save("./rom_data/I.npy", I)


def main():
    setup = SimulationSetup(
        N_t=65,
        N_x_im=140,
        N_y_im=155,
        O_x=25,
        O_y=180
    )

    exec_setup = ExecutionSetup(
        gpu=False,
        precision='float64',
        data_folder='rom_data'
    )

    solver = ROM(setup, exec_setup)
    solver.calculate_I_matrices()

if __name__ == "__main__":
    main()