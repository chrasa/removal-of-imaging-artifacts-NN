from cholesky import mblockchol
from os.path import exists, sep
from SimulationSetup import SimulationSetup
from benchmark import timeit
from cpu_gpu_abstraction import CPU_GPU_Abstractor, ExecutionSetup


class ROM(CPU_GPU_Abstractor):

    def __init__(self, setup: SimulationSetup, exec_setup: ExecutionSetup):
        super(ROM, self).__init__(exec_setup=exec_setup)

        self.setup = setup
        self.V_0_path = self.exec_setup.data_folder + "V_0.npy"
        self.I_0_path = self.exec_setup.data_folder + "I_0_ROM.npy"
        self.R_0_path = self.exec_setup.data_folder + "R_0.npy"

        self.V_0 = self.get_V0()
        self.I_0 = self.get_I0()

    def get_V0(self):
        if exists(self.V_0_path):
            return self.xp.load(self.V_0_path)
        else:
            V_0 = self.__calculate_V0()
            self.xp.save(self.V_0_path, V_0)
            return V_0
        
    def get_I0(self):
        if not exists(self.I_0_path):
            I_0 = self.__calculate_I0()
            self.xp.save(self.I_0_path, I_0)
            return I_0
        else:
            return self.xp.load(self.I_0_path)

    @timeit
    def __calculate_I0(self):
        if not exists(self.R_0_path):
            D_0, _ = self.load_D0_and_U0() 
            R_0 = self._calculate_mass_matrix(D_0)
        else:
            R_0 = self.xp.load(self.R_0_path)
        I_0 = self._calculate_imaging_function(R_0)
        return I_0
    
    @timeit
    def __calculate_V0(self):
        D_0, U_0 = self._load_D0_and_U0() 
        R = self._calculate_mass_matrix(D_0)
        V_0 = U_0 @ self.xp.linalg.inv(R)

        if not exists(self.R_0_path):
            self.xp.save(self.R_0_path, R)

        return V_0

    def _find_indices(self,j):
        ind_t = self.xp.linspace(0, self.setup.N_s, self.setup.N_s) + self.setup.N_s*j 
        ind_list = [int(x) for x in ind_t]
        return ind_list

    def _load_D0_and_U0(self, D_0_path="D_0.npy", U_0_path="U_0.npy"):
        D = self.xp.load(self.exec_setup.data_folder + D_0_path)

        U_0_mem = self.xp.memmap(self.exec_setup.data_folder + U_0_path, self.exec_setup.precision, 'r', shape=(2*self.setup.N_t, self.setup.N_y_im*self.setup.N_x_im, self.setup.N_s))
        U_0 = self.xp.array(U_0_mem)
        U_0 = U_0[0:self.setup.N_t,:,:]
        U_0 = self.xp.moveaxis(U_0, 0, -1)
        U_0 = self.xp.reshape(U_0, (self.setup.N_x_im * self.setup.N_y_im, self.setup.N_s*self.setup.N_t), order='F')
        
        return D, U_0

    @timeit
    def _calculate_mass_matrix(self, D):
        M = self.xp.zeros((self.setup.N_s*self.setup.N_t, self.setup.N_s*self.setup.N_t), dtype=self.exec_setup.precision)

        for i in range(self.setup.N_t):
            for j in range(self.setup.N_t):
                ind_i = self._find_indices(i)
                ind_j = self._find_indices(j)

                M[ind_i[0]:ind_i[-1],ind_j[0]:ind_j[-1]] = 0.5 * (D[abs(i-j)] + D[abs(i+j)])

        R = mblockchol(M, self.setup.N_s, self.setup.N_t)

        return R
    
    @timeit
    def _calculate_imaging_function(self, R):
        I = self.V_0 @ R
        I = self.xp.square(I)
        I = I.sum(axis=1)
        
        return I

    def _get_image_derivative(self, I):
        I = I.reshape((self.setup.N_y_im, self.setup.N_x_im))
        dx = -1 * self.xp.array([[-1, 0, 1]])
        I_x = self.scipy.ndimage.convolve(I, dx)
        I_x = I_x.reshape((-1, 1))

        return I_x
    
    def calculate_I(self, D, I_file_name=""):
        D = self._get_array_from_disk_or_mem(D)
        R = self._calculate_mass_matrix(D)
        I = self._calculate_imaging_function(R)
        I = I - self.I_0
        I = self._get_image_derivative(I)
        if I_file_name:
            self.xp.save(self.exec_setup.data_folder + "I_ROM.npy", I)
        return self.asnumpy(self.xp.squeeze(I))


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
    solver.calculate_I("./rtm_data/D.npy")

if __name__ == "__main__":
    main()