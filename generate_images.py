import sys
from RTM import RTM
from ROM import ROM
from WaveSolver import WaveSolver
from SimulationSetup import SimulationSetup
from cpu_gpu_abstraction import ExecutionSetup
from os import path
import numpy as np
from FractureGenerator import FractureGenerator, FractureSetup


def main(n_images=3, precision='float64', data_folder=f".{path.sep}data{path.sep}"):
    sim_setup = SimulationSetup(
        N_t=40,
        N_x_im=140,
        N_y_im=155,
        O_x=25,
        O_y=180)
    
    ws_exec_setup = ExecutionSetup(gpu=True, precision=precision, data_folder=data_folder)
    rtm_exec_setup = ExecutionSetup(gpu=True, precision=precision, data_folder=data_folder)
    rom_exec_setup = ExecutionSetup(gpu=False, precision=precision, data_folder=data_folder)

    fracture_setup = FractureSetup(
        O_x=25,
        O_y=180,
        fractured_region_height=140,
        fractured_region_width=155,
        n_fractures=3
    )

    training_data = np.zeros((n_images, 3, sim_setup.N_x_im*sim_setup.N_y_im))

    fracture_generator = FractureGenerator(fracture_setup)

    wave_solver = WaveSolver(sim_setup, ws_exec_setup)
    wave_solver.calculate_U0_D0()

    rtm = RTM(sim_setup, rtm_exec_setup)
    rom = ROM(sim_setup, rom_exec_setup)

    for image_idx in range(n_images):
        fracture_image, fracture_image_in_imaging_region = fracture_generator.generate_fractures()
        D, D_fine = wave_solver.calculate_U_D(fracture_image)

        D = np.load(rom_exec_setup.data_folder + "D.npy")
        D_fine = np.load(rtm_exec_setup.data_folder + "D_fine.npy")

        I_rtm = rtm.calculate_I(D_fine, I_file_name="I_RTM.npy")
        I_rom = rom.calculate_I(D, I_file_name="I_ROM.py")

        training_data[image_idx, 0] = fracture_image_in_imaging_region
        training_data[image_idx, 1] = I_rtm
        training_data[image_idx, 2] = I_rom

    np.save(data_folder + "training_data.npy", training_data)


if __name__ == "__main__":
    n_images = 10
    for i, arg in enumerate(sys.argv):
        if arg == '-n':
            n_images = sys.argv[i+1]
    main(n_images)
