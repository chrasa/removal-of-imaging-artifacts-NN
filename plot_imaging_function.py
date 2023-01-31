import sys
import numpy as np
import matplotlib.pyplot as plt
from os import path

def plot_imaging_function(I_0_path):
    I_0 = np.load(I_0_path)
    I_0 = np.reshape(I_0, (175, 350), order='F')
    print(f"Shape of I_0: {I_0.shape}")

    plt.imshow(I_0)
    plt.colorbar(orientation="horizontal")
    plt.show()


if __name__ == "__main__":
    I_0_path = "bs_test" + path.sep + "I_0.npy"

    for i, arg in enumerate(sys.argv):
        if arg == '-p':
            I_0_path = sys.argv[i+1]
    plot_imaging_function(I_0_path)