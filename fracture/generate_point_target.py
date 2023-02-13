import numpy as np


def circle(width=512, height=512, c0=1000, c_circle=100):
    xx, yy = np.mgrid[:width, :height]
    circle = (xx - width/2) ** 2 + (yy - height/2) ** 2

    background = np.full((width, height), c0)
    circle = (circle < 100) * (c_circle - c0)

    return background + circle

img = circle()

img = img.reshape((512*512,1))
np.save(f"./images/circle.npy", img)