import numpy as np
from SimulationSetup import SimulationSetup as setup
import matplotlib.pyplot as plt


def import_sources():
    
    b = np.loadtxt(setup.Bsrc_file, delimiter =',', dtype=np.float64)
    b = np.reshape(b, (setup.N_x, setup.N_y, setup.N_s))
    b = np.transpose(b, axes=(1,0,2))
    b = np.reshape(b, (setup.N_x*setup.N_y, setup.N_s))

    return b


def main():
    test = import_sources()
    test = np.reshape(test, (setup.N_x, setup.N_y, setup.N_s))
    print(test.shape)
    
    slice = test[:,:,0]
    plt.imshow(slice)
    plt.colorbar(orientation="horizontal")
    plt.show()


if __name__ == "__main__":
    main()