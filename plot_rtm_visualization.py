import sys
from os import path
import numpy as np
from setup import SimulationSetup
from cpu_gpu_abstraction import ExecutionSetup
from benchmark import ProgressBar
from matplotlib import pyplot as plt
from cmcrameri import cm


class RtmVisualizer(ProgressBar):
    def __init__(self, setup: SimulationSetup, exec_setup: ExecutionSetup, U_RT_file_name="U_RT.npy", U_0_file_name="U_0.npy", U_file_name="U.npy", fracture_name="im0000.npy") -> None:
        super().__init__()
        self.setup = setup
        self.exec_setup = exec_setup

        print("Loading wave fields into memory...")
        self.U_RT = np.array(np.memmap(self.exec_setup.data_folder + U_RT_file_name, self.exec_setup.precision_np, 'r', shape=(2*self.setup.N_t, self.setup.N_x_im*self.setup.N_y_im, self.setup.N_s)))
        self.U_0 = np.array(np.memmap(self.exec_setup.data_folder + U_0_file_name, self.exec_setup.precision_np, 'r', shape=(2*self.setup.N_t, self.setup.N_x_im*self.setup.N_y_im, self.setup.N_s)))
        self.U = np.array(np.memmap(self.exec_setup.data_folder + U_file_name, self.exec_setup.precision_np, 'r', shape=(2*self.setup.N_t, self.setup.N_x_im*self.setup.N_y_im, self.setup.N_s)))

        print("Normalizing wave fields...")
        self.U_RT = self.__normalize_array(self.U_RT)
        self.U_0 = self.__normalize_array(self.U_0)
        self.U = self.__normalize_array(self.U)

        self.fracture = np.load(f"fractures{path.sep}{fracture_name}")
        self.fracture = self.fracture - self.setup.background_velocity_value
        self.fracture = self.fracture[self.get_imaging_region_indices()]
        self.fracture = self.fracture.reshape(self.setup.N_x_im, self.setup.N_y_im)

        self.source_index = 25

    def __normalize_array(self, a):
        return np.multiply(a, 1/np.max(a))

    def get_imaging_region_indices(self):
        im_y_indices = range(self.setup.O_y, self.setup.O_y+self.setup.N_y_im)
        im_x_indices = range(self.setup.O_x, self.setup.O_x+self.setup.N_x_im)
        indices = [x*self.setup.N_y + y for x in im_x_indices for y in im_y_indices] 
        return indices


    def plot_image(self, tidx, U_0_frame, U_RT_frame, U_forward_frame, U_backward_frame, I):
        fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 3), dpi=300)

        U_0_frame = U_0_frame.reshape(self.setup.N_x_im, self.setup.N_y_im)
        U_RT_frame = U_RT_frame.reshape(self.setup.N_x_im, self.setup.N_y_im)
        U_forward_frame = U_forward_frame.reshape(self.setup.N_x_im, self.setup.N_y_im)
        U_backward_frame = U_backward_frame.reshape(self.setup.N_x_im, self.setup.N_y_im)
        I = I.reshape(self.setup.N_x_im, self.setup.N_y_im)

        U_vmax = 1.0
        U_vmin = -U_vmax

        I_vmax = 1.85e-05
        I_vmin = -I_vmax

        f_vmax = 1000
        f_vmin = -f_vmax
        
        ax[0].set_title("U_0")
        ax[0].imshow(U_0_frame.T, cmap=cm.broc, vmax=U_vmax, vmin=U_vmin)

        ax[1].set_title("U_RT")
        ax[1].imshow(U_RT_frame.T, cmap=cm.broc, vmax=U_vmax, vmin=U_vmin)

        ax[2].set_title("I")
        ax[2].imshow(I.T, cmap=cm.vik, vmax=I_vmax, vmin=I_vmin)

        ax[3].set_title("Fracture")
        ax[3].imshow(self.fracture.T, cmap=cm.vik, vmax=f_vmax, vmin=f_vmin)

        # ax[2,0].set_title("U forward")
        # ax[2,0].imshow(U_forward_frame.T, cmap=cm.broc, vmax=U_vmax, vmin=U_vmin)

        # ax[2,1].set_title("U backward")
        # ax[2,1].imshow(U_backward_frame.T, cmap=cm.broc, vmax=U_vmax, vmin=U_vmin)

        for axis in ax.flatten():
            axis.grid(color='black', linestyle='--', linewidth=0.5)

        fig.tight_layout()
        fig.savefig(f'rtm_visualization{path.sep}frame_{tidx:03d}.jpg')
        # if tidx == 22:
        #     plt.show()
        plt.close()


    def plot_rtm(self):
        I = np.zeros((self.setup.N_x_im*self.setup.N_y_im), dtype=self.exec_setup.precision)

        timesteps = 2*self.setup.N_t

        for tidx in range(timesteps):
            self.print_progress(tidx+1, timesteps)
            t_idx_backward = 2*self.setup.N_t -tidx -1
            t_idx_forward = tidx

            U_RT = np.array(self.U_RT[t_idx_forward,:,self.source_index])
            U_0 = np.array(self.U_0[t_idx_backward,:,self.source_index])
            U_forward = np.array(self.U[t_idx_forward,:,self.source_index])
            U_backward = np.array(self.U[t_idx_backward,:,self.source_index])

            I[:] += U_RT[:] * U_0[:] * self.setup.tau

            self.plot_image(tidx, U_0, U_RT, U_forward, U_backward, I)
        
        self.end()
        # return np.sum(I, axis=1)


def main():
    setup = SimulationSetup()
    exec_setup = ExecutionSetup()

    visualizer = RtmVisualizer(setup, exec_setup)
    visualizer.plot_rtm()


if __name__ == "__main__":
     try:
        main()
     except KeyboardInterrupt:
         print("\nAborted")
