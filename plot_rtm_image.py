import sys
import numpy as np
import matplotlib.pyplot as plt
from os import path

plt.rcParams["font.family"] = "aakar"

def plot_rtm_image(rtm_path, fracture_path, image_height:int=512, image_width:int=512):
    image = np.load(rtm_path)
    image = image.reshape(image_height, image_width)
    clipping_max = np.max(np.abs(image)) *0.1

    if fracture_path:
        fracture = np.load(fracture_path)
        fracture = fracture.reshape(image_height, image_width)

        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_title("Input Fracture")
        ax1.imshow(fracture)
        ax2.set_title("RTM Output")
        ax2.imshow(image, vmin=-clipping_max, vmax=clipping_max)
    else:
        plt.imshow(image, vmin=-clipping_max, vmax=clipping_max)
        plt.colorbar(orientation="horizontal")
    
    plt.show()

if __name__ == "__main__":
    path = "I.npy"
    fracture_path = False

    for i, arg in enumerate(sys.argv):
        if arg == '-p':
            path = sys.argv[i+1]
        elif arg == '-f':
            fracture_path = sys.argv[i+1]
    plot_rtm_image(path, fracture_path)