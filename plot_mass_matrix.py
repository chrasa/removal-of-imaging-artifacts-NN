import sys
import numpy as np
import matplotlib.pyplot as plt
from os import path

def plot_mass_matrix(R_0_path):
    R_0 = np.load(R_0_path)
    print(f"Shape of M_0: {R_0.shape}")

    plt.imshow(R_0)
    plt.colorbar(orientation="horizontal")
    plt.title("Mass Matrix")
    plt.show()

if __name__ == "__main__":
    R_0_path = "bs_test" + path.sep + "R_0.npy"

    for i, arg in enumerate(sys.argv):
        if arg == '-p':
            R_0_path = int(sys.argv[i+1])
    plot_mass_matrix(R_0_path)