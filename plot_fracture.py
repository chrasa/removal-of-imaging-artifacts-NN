import sys
import numpy as np
import matplotlib.pyplot as plt
from os import path

def plot_fracture(fracture_path, image_height:int=512, image_width:int=512):
    image = np.load(fracture_path)

    image = image.reshape(image_height, image_width, 1)
    plt.gray()
    plt.title(f"{fracture_path}")
    plt.imshow(np.squeeze(image))
    plt.show()

if __name__ == "__main__":
    fracture_path = "fracture" + path.sep + "images" + path.sep + "im0.npy"

    for i, arg in enumerate(sys.argv):
        if arg == '-p':
            fracture_path = int(sys.argv[i+1])
    plot_fracture(fracture_path)