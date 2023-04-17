import sys
import glob
import numpy as np
from os import path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from setup import FractureSetup


def plot_fracture(fracture_image, frac_setup: FractureSetup, out_path=False, title='Fracture'):
    fracture_image = fracture_image.reshape(frac_setup.N_y, frac_setup.N_x)
    
    fig, ax = plt.subplots()
    ax.set_title(f"{title}")
    ax.imshow(fracture_image.T)
    rect = patches.Rectangle((frac_setup.O_x, frac_setup.O_y), frac_setup.N_x_im, frac_setup.N_y_im, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if out_path:
        plt.savefig(f'{ out_path }img{ path.sep }{title}.jpg', dpi=150)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    fracture_path = f"fractures{ path.sep }im0000.npy"
    plot_series = False
    fracture_setup = FractureSetup()

    for i, arg in enumerate(sys.argv):
        if arg == '-p':
            fracture_path = sys.argv[i+1]
        elif arg == '-series':
            plot_series = True

    if plot_series:
        fracture_images = glob.glob(f"{ fracture_path }*.npy")
        for image_path in fracture_images:
            print(f"Generating {image_path}")
            fracture_image = np.load(image_path)
            title = image_path.split(path.sep)[-1]
            plot_fracture(fracture_image, fracture_setup, fracture_path, title)

    else:
        fracture_image = np.load(fracture_path)
        plot_fracture(fracture_image, fracture_setup)