import numpy as np
from FractureSetup import FractureSetup


def circle(setup: FractureSetup, x=256, y=256, c_circle=2500):
    xx, yy = np.mgrid[:setup.image_width, :setup.image_height]
    circle = (xx - x) ** 2 + (yy - y) ** 2

    background = np.full((setup.image_width, setup.image_height), setup.background_velocity)
    circle = (circle < 100) * (c_circle - setup.background_velocity)

    fracture = background + circle
    return fracture.reshape((setup.image_width*setup.image_height))

if __name__ == "__main__":
    frac_setup = FractureSetup()
    
    img = circle(frac_setup, x=160)
    np.save(f"./images/circle.npy", img)