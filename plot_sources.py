import numpy as np
import matplotlib.pyplot as plt
from setup import SimulationSetup as setup


def import_sources():
    b = np.loadtxt(setup.Bsrc_file, delimiter =',', dtype=np.float64)
    b = np.reshape(b, (setup.N_x, setup.N_y, setup.N_s))
    b = np.transpose(b, axes=(1,0,2))
    b = np.reshape(b, (setup.N_x*setup.N_y, setup.N_s))

    return b


def main():
    sources = import_sources()
    sources = np.reshape(sources, (setup.N_x, setup.N_y, setup.N_s))
    slice = sources[:,:,0]
    plt.imshow(slice)
    plt.colorbar(orientation="horizontal")
    plt.show()


if __name__ == "__main__":
    main()