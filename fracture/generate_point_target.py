import numpy as np


def circle(width=512, height=512, x=256, y=256, c0=1000, c_circle=100):
    xx, yy = np.mgrid[:width, :height]
    circle = (xx - x) ** 2 + (yy - y) ** 2

    background = np.full((width, height), c0)
    circle = (circle < 100) * (c_circle - c0)

    return background + circle

img = circle(y=100)

img = img.reshape((512*512,1))
np.save(f"./images/circle.npy", img)