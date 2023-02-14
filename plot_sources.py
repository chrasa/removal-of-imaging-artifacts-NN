import numpy as np
from SimulationSetup import SimulationSetup as setup
import matplotlib.pyplot as plt


def import_sources():
    
    b = np.loadtxt(setup.Bsrc_file, delimiter =',', dtype=np.float64)
    b = np.reshape(b, (setup.N_x, setup.N_y, setup.N_s))

    return b


def main():
    test = import_sources()
    # np.save("./sources/Bsrc_T.npy", test)
    print(test.shape)

    # slice = test[15,0:50,0]
    # plt.plot(slice)
    
    slice = test[:,:,49]
    plt.imshow(slice)
    plt.colorbar(orientation="horizontal")
    plt.show()


if __name__ == "__main__":
    main()