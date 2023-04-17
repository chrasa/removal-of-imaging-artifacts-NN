import sys
import numpy as np
import matplotlib.pyplot as plt
from os import path
from setup import SimulationSetup

def get_imaging_region_indices(setup: SimulationSetup):
    im_y_indices = range(setup.O_y, setup.O_y+setup.N_y_im)
    im_x_indices = range(setup.O_x, setup.O_x+setup.N_x_im)
    indices = [y*setup.N_x + x for y in im_y_indices for x in im_x_indices] 

    return indices


def plot_rtm_image(I_rom_path, I_rtm_path, fracture_path, setup: SimulationSetup):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)

    rom_image = np.load(I_rom_path)
    rom_image = np.squeeze(rom_image)
    rom_image = rom_image.reshape(setup.N_y_im, setup.N_x_im)
    clipping_max_rom = np.max(np.abs(rom_image))

    ax1.set_title("ROM Image")
    ax1.imshow(rom_image, vmin=-clipping_max_rom, vmax=clipping_max_rom)

    rtm_image = np.load(I_rtm_path)
    rtm_image_0 = np.load("./data/I_0_RTM.npy")
    rtm_image = rtm_image - rtm_image_0
    rtm_image = np.squeeze(rtm_image)
    rtm_image = rtm_image.reshape(setup.N_y_im, setup.N_x_im)
    clipping_max_rtm = np.max(np.abs(rtm_image))

    ax2.set_title("RTM Image")
    ax2.imshow(rtm_image, vmin=-clipping_max_rtm, vmax=clipping_max_rtm)

    fracture_image = np.load(fracture_path)
    fracture_image = np.squeeze(fracture_image)
    fracture_image = fracture_image[get_imaging_region_indices(setup)]
    fracture_image = np.reshape(fracture_image, (setup.N_y_im, setup.N_x_im))

    ax3.set_title("Fracture Image")
    ax3.imshow(fracture_image)

    plt.show()

def plot_rtm_image_from_single_file(data_path, image_idx, setup: SimulationSetup):
    training_data = np.load(data_path)

    f, (ax1, ax2, ax3) = plt.subplots(1, 3)

    rom_image = np.squeeze(training_data[image_idx, 2])
    rom_image = rom_image.reshape(setup.N_y_im, setup.N_x_im)
    clipping_max_rom = np.max(np.abs(rom_image))

    ax1.set_title("ROM Image")
    ax1.imshow(rom_image.T, vmin=-clipping_max_rom, vmax=clipping_max_rom)

    rtm_image = np.squeeze(training_data[image_idx, 1])
    rtm_image = rtm_image.reshape(setup.N_y_im, setup.N_x_im)
    clipping_max_rtm = np.max(np.abs(rtm_image))

    ax2.set_title("RTM Image")
    ax2.imshow(rtm_image.T, vmin=-clipping_max_rtm, vmax=clipping_max_rtm)

    fracture_image = np.squeeze(training_data[image_idx, 0])
    fracture_image = np.reshape(fracture_image, (setup.N_y_im, setup.N_x_im))

    ax3.set_title("Fracture Image")
    ax3.imshow(fracture_image.T)

    plt.show()

if __name__ == "__main__":
    i = 0

    for i, arg in enumerate(sys.argv):
        if arg == '-n':
            i = int(sys.argv[i+1])

    I_rom_path = "./data/I_ROM.npy"
    I_rtm_path = "./data/I_RTM.npy"
    fracture_path = "./old_fwd/fractures/circle.npy"
    setup = SimulationSetup(
        N_y_im=155, 
        N_x_im=140,
        O_x=25,
        O_y=180
    )

    # plot_rtm_image(I_rom_path, I_rtm_path, fracture_path, setup)
    plot_rtm_image_from_single_file("./data/training_data.npy", i, setup)