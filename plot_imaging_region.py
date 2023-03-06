import numpy as np
from SimulationSetup import SimulationSetup
from matplotlib import pyplot as plt


def get_imaging_region_indices(setup: SimulationSetup):
        im_y_indices = range(setup.O_y, setup.O_y+setup.N_y_im)
        im_x_indices = range(setup.O_x, setup.O_x+setup.N_x_im)
        indices = [y*setup.N_x + x for y in im_y_indices for x in im_x_indices]
        return indices



def main():
    setup = SimulationSetup()
    imaging_region = get_imaging_region_indices(setup)

    full_image = np.zeros((setup.N_x * setup.N_y))
    full_image[imaging_region] = 1
    full_image = np.reshape(full_image, (setup.N_x, setup.N_y))

    print(full_image.shape)

    plt.imshow(full_image)
    plt.show()

if __name__ == "__main__":
      main()