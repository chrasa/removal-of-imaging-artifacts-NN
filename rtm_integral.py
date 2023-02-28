import numpy as np
from SimulationSetup import SimulationSetup
from os import path
import sys

from matplotlib import pyplot as plt

setup = SimulationSetup()

U_rtm = np.memmap("U_RT.npy", np.float64, 'r', shape=(2*setup.N_t, setup.N*setup.N, setup.N_s))
U_0 = np.memmap("U_0.npy", np.float64, 'r', shape=(2*setup.N_t,setup.N*setup.N, setup.N_s))
U = np.memmap("U.npy", np.float64, 'r', shape=(2*setup.N_t,setup.N*setup.N, setup.N_s))

I = np.zeros(setup.N*setup.N, dtype=np.float64)

def plot_rtm(U_0_frame, U_RTM_frame, fracture, I, U_frame, U_reverse_frame):
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 12), dpi=150)

    U_0_frame = U_0_frame.reshape(512, 512)
    U_RTM_frame = U_RTM_frame.reshape(512, 512)
    
    ax[0,0].set_title("U_0")
    ax[0,0].imshow(U_0_frame)

    ax[0,1].set_title("U_RTM")
    ax[0,1].imshow(U_RTM_frame)

    ax[1,0].set_title("Fracture")
    ax[1,0].imshow(np.squeeze(fracture))

    I = I.reshape(512,512)
    ax[1,1].set_title("I")
    ax[1,1].imshow(I, vmin=-1e-15, vmax=1e-15)

    ax[2,0].set_title("reverse U")
    ax[2,0].imshow(U_reverse_frame.reshape(512,512))

    ax[2,1].set_title("U")
    ax[2,1].imshow(U_frame.reshape(512,512))

    fig.savefig(f'rtm_debug_frames{path.sep}frame_{tidx:03d}.jpg')
    # plt.show()
    # input()
    plt.close()

def progress_bar(progress, max, progress_bar_length=40):
    title = f"\rIntegral progress: {progress:5}/{max:5}: "
    success_rate = f" {(progress/max)*100:3.2f}%"
    number_of_progress_indicators = int(progress * progress_bar_length // (max))

    sys.stdout.write(title + "[" + number_of_progress_indicators*"#" + (progress_bar_length - number_of_progress_indicators)*"-" + "]" + success_rate)

fracture = np.load("fracture/images/circle.npy")
fracture = fracture.reshape(512, 512, 1)

# Integrate
si = 25
# for sidx in range(setup.N_s):
print("Load arrays to memory..")
# U_rtm_temp = np.array(U_rtm[:,:,si])
# U_0_temp = np.array(U_0[:,:,si])
# U_temp = np.array(U[:,:,si])

U_rtm_temp = np.load("U_rtm_temp.npy")
U_0_temp = np.load("U_0_temp.npy")
U_temp = np.load("U_temp.npy")

# np.save("U_rtm_temp.npy", U_rtm_temp)
# np.save("U_0_temp.npy", U_0_temp)
# np.save("U_temp.npy", U_temp)

print("Calculating")
for tidx in range(setup.N_t):
    progress_bar(tidx+1, 2*setup.N_t)
    t_idx_backward = 2*setup.N_t -tidx -1
    t_idx_forward = tidx

    temp = U_rtm_temp[t_idx_forward,:]
    temp = temp.reshape(512,512)
    temp = temp.T
    temp = temp.reshape(512*512)

    # print(f"forward idx: {t_idx_forward}, backward idx: {t_idx_backward}")
    # I += 0.5 * U_0_temp[t_idx_backward,:] * U_rtm_temp[t_idx_forward,:] * setup.tau

    I += 0.5 * U_0_temp[t_idx_backward,:] * temp * setup.tau

    # plot_rtm(U_0_temp[t_idx_backward,:], U_rtm_temp[t_idx_forward,:], fracture, I, U_temp[t_idx_forward,:], U_temp[t_idx_backward,:])
    #if tidx == 57:
    plot_rtm(U_0_temp[t_idx_backward,:], temp, fracture, I, U_temp[t_idx_forward,:], U_temp[t_idx_backward,:])

print("\n")
np.save("I.npy", I)