import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from os import path
from fracture.FractureSetup import FractureSetup

def plot_fracture(fracture_path, frac_setup: FractureSetup):
    image = np.load(fracture_path)
    image = image.reshape(frac_setup.image_height, frac_setup.image_width, 1)
    
    fig, ax = plt.subplots()
    ax.set_title(f"{fracture_path}")
    ax.imshow(np.squeeze(image))
    rect = patches.Rectangle((frac_setup.O_x, frac_setup.O_y), frac_setup.fractured_region_width, frac_setup.fractured_region_height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()

if __name__ == "__main__":
    fracture_path = "fracture" + path.sep + "images" + path.sep + "im0.npy"
    fracture_setup = FractureSetup()

    for i, arg in enumerate(sys.argv):
        if arg == '-p':
            fracture_path = sys.argv[i+1]
    plot_fracture(fracture_path, fracture_setup)