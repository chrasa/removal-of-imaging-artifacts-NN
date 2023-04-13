import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from os import path
from FractureSetup import FractureSetup

import glob
glob.glob("*.pdf")


def plot_fracture(fracture_path, frac_setup: FractureSetup):
    image = np.load(fracture_path)
    print(image.shape)
    image = image.reshape(frac_setup.image_height, frac_setup.image_width)
    
    fig, ax = plt.subplots()
    ax.set_title(f"{fracture_path}")
    ax.imshow(image.T)
    rect = patches.Rectangle((frac_setup.O_x, frac_setup.O_y), frac_setup.fractured_region_width, frac_setup.fractured_region_height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # plt.show()
    plt.savefig(f'fractures{ path.sep }img{ path.sep }{fracture_path.split(path.sep)[-1]}.jpg', dpi=150)
    plt.clf()
    plt.cla()

if __name__ == "__main__":
    fractures = glob.glob(f"fractures{ path.sep }*.npy")
    fracture_setup = FractureSetup(
        O_x=156,
        O_y=25,
        fractured_region_height=100,
        fractured_region_width=200,
    )

    for fracture in fractures:
        plot_fracture(fracture, fracture_setup)
