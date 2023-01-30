import sys
import numpy as np
import matplotlib.pyplot as plt
from os import path

def plot_mass_matrix(M_0_path):
    M_0 = np.load(M_0_path)
    print(f"Shape of M_0: {M_0.shape}")

    plt.imshow(M_0)
    plt.colorbar(orientation="horizontal")
    plt.title("Mass Matrix")
    plt.show()

if __name__ == "__main__":
    M_0_path = "bs_test" + path.sep + "M_0.npy"

    for i, arg in enumerate(sys.argv):
        if arg == '-p':
            M_0_path = int(sys.argv[i+1])
    plot_mass_matrix(M_0_path)