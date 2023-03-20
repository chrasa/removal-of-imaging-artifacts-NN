import sys
import numpy as np
import matplotlib.pyplot as plt
from os import path
from SimulationSetup import SimulationSetup

# Convert frames to images with: ffmpeg -framerate 12 -pattern_type glob -i '*.jpg'   -c:v libx264 -pix_fmt yuv420p out.mp4

def plot_wave(setup: SimulationSetup, source, file_path):
    U_0 = np.memmap(file_path, np.float32, 'r', shape=(2*setup.N_t, setup.N_y_im*setup.N_x_im, setup.N_s))
    U_0 = np.reshape(U_0, (2*setup.N_t, setup.N_y_im, setup.N_x_im, setup.N_s))

    for timestep in range(2*setup.N_t):
        slice = U_0[timestep,:,:,source]
        plt.imshow(slice)
        plt.colorbar(orientation="horizontal")
        print(f"Write frame: {timestep+1}/{setup.N_t}")
        plt.savefig(f'wave_frames{ path.sep }frame_{timestep:03d}.jpg', dpi=300)
        plt.clf()
        plt.cla()
    #plt.show()

if __name__ == "__main__":
    Ns = 25
    N_t = 70
    file_path = "bs_test" + path.sep + "U_0.npy"

    for i, arg in enumerate(sys.argv):
        if arg == '-Ns':
            Ns = int(sys.argv[i+1])
        elif arg == '-n':
            N_t = int(sys.argv[i+1])
        elif arg == '-p':
            file_path = sys.argv[i+1]

    setup = SimulationSetup(N_t=N_t)

    plot_wave(setup, Ns, file_path)