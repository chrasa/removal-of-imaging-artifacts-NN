import sys
import numpy as np
import matplotlib.pyplot as plt
from os import path
from setup import SimulationSetup
from benchmark import ProgressBar
from cmcrameri import cm

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

def plot_rtm_image_from_single_file(training_data, setup: SimulationSetup, output_file=False):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14,4))

    rom_image = np.squeeze(training_data[2])
    rom_image = rom_image.reshape(setup.N_x_im, setup.N_y_im)
    clipping_max_rom = np.max(np.abs(rom_image))

    ax1.set_title("ROM Image")
    ax1.imshow(rom_image.T, vmin=-clipping_max_rom, vmax=clipping_max_rom, cmap=cm.broc)
    ax1.grid(color='black', linestyle='--', linewidth=0.5)

    rtm_image = np.squeeze(training_data[1])
    rtm_image = rtm_image.reshape(setup.N_x_im, setup.N_y_im)
    clipping_max_rtm = np.max(np.abs(rtm_image))

    ax2.set_title("RTM Image")
    ax2.imshow(rtm_image.T, vmin=-clipping_max_rtm, vmax=clipping_max_rtm, cmap=cm.broc)
    ax2.grid(color='black', linestyle='--', linewidth=0.5)

    fracture_image = np.squeeze(training_data[0])
    fracture_image = np.reshape(fracture_image, (setup.N_x_im, setup.N_y_im))
    fracture_image = fracture_image - setup.background_velocity_value
    fracture_vmax = np.max([np.abs(np.max(fracture_image)), np.abs(np.min(fracture_image))])

    ax3.set_title("Fracture Image")
    ax3.imshow(fracture_image.T, vmax=fracture_vmax, vmin=-fracture_vmax, cmap=cm.broc)
    ax3.grid(color='black', linestyle='--', linewidth=0.5)

    f.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    idx = 0
    training_data_path = "./data/training_data.npy"
    series_plot = False
    output_path = False

    for i, arg in enumerate(sys.argv):
        if arg == '-n':
            idx = int(sys.argv[i+1])
        elif arg == '-p':
            training_data_path = sys.argv[i+1]
        elif arg == '-series':
            series_plot = True
        elif arg == '-out':
            output_path = sys.argv[i+1]

    setup = SimulationSetup()
    training_data = np.load(training_data_path)

    if series_plot:
        if not output_path:
            raise Exception("No output path provided")
        n_training_samples = training_data.shape[0]
        progress_bar = ProgressBar()
        for i in range(n_training_samples):
            progress_bar.print_progress(i+1, n_training_samples)
            plot_rtm_image_from_single_file(training_data[i], setup, f"{output_path}im{i:04}.jpg")
        progress_bar.end()
    else:
        plot_rtm_image_from_single_file(training_data[idx], setup)

    # I_rom_path = "./data/I_ROM.npy"
    # I_rtm_path = "./data/I_RTM.npy"
    # fracture_path = "./old_fwd/fractures/circle.npy"
    # plot_rtm_image(I_rom_path, I_rtm_path, fracture_path, setup)