import sys
import numpy as np
import matplotlib.pyplot as plt
from os import path

# Convert frames to images with: ffmpeg -framerate 12 -pattern_type glob -i '*.jpg'   -c:v libx264 -pix_fmt yuv420p out.mp4

def plot_wave(source, number_of_frames):
    U_0 = np.load("./bs_test/U_0.npy")
    # print(f"Shape of U_0: {U_0.shape}")

    U_0 = np.reshape(U_0, (175, 350, 50, number_of_frames),order='F')


    for timestep in range(number_of_frames):
        slice = U_0[:,:,source,timestep]
        plt.imshow(slice)
        plt.colorbar(orientation="horizontal")
        print(f"Write frame: {timestep+1}/{number_of_frames}")
        plt.savefig(f'wave_frames{ path.sep }frame_{timestep:03d}.jpg', dpi=300)
        plt.clf()
        plt.cla()
    #plt.show()

if __name__ == "__main__":
    Ns = 25
    n = 25
    for i, arg in enumerate(sys.argv):
        if arg == '-Ns':
            Ns = int(sys.argv[i+1])
        elif arg == '-n':
            n = int(sys.argv[i+1])
    plot_wave(Ns, n)