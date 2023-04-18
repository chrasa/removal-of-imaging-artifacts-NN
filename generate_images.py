import sys
from rtm import RTM
from rom import ROM
from wave_solver import WaveSolver
from setup import SimulationSetup
from cpu_gpu_abstraction import ExecutionSetup
from os import path
import numpy as np
from benchmark import timeit


@timeit
def main(n_images=3, precision='float64', data_folder=f".{path.sep}data{path.sep}", use_gpu=False):
    sim_setup = SimulationSetup()
    
    ws_exec_setup = ExecutionSetup(gpu=use_gpu, precision=precision, data_folder=data_folder)
    rtm_exec_setup = ExecutionSetup(gpu=use_gpu, precision=precision, data_folder=data_folder)
    rom_exec_setup = ExecutionSetup(gpu=False, precision=precision, data_folder=data_folder)

    training_data = np.zeros((n_images, 3, sim_setup.N_x_im*sim_setup.N_y_im))

    wave_solver = WaveSolver(sim_setup, ws_exec_setup)
    wave_solver.calculate_U0_D0()

    rtm = RTM(sim_setup, rtm_exec_setup)
    rom = ROM(sim_setup, rom_exec_setup)

    for image_idx in range(n_images):
        fracture_image = np.load(f"./fractures/im{image_idx:04}.npy")
        fracture_image_in_imaging_region = fracture_image[rtm.get_imaging_region_indices()]
        D, D_fine = wave_solver.calculate_U_D(fracture_image)

        I_rtm = rtm.calculate_I(D_fine, I_file_name="I_RTM.npy")
        I_rom = rom.calculate_I(D, I_file_name="I_ROM.py")

        training_data[image_idx, 0] = fracture_image_in_imaging_region
        training_data[image_idx, 1] = I_rtm
        training_data[image_idx, 2] = I_rom
    
        np.save(data_folder + "training_data.npy", training_data)


if __name__ == "__main__":
    n_images = 10
    use_gpu = False

    for i, arg in enumerate(sys.argv):
        if arg == '-n':
            n_images = int(sys.argv[i+1])
        elif arg == '-gpu':
            use_gpu = True
    main(n_images, use_gpu=use_gpu)
