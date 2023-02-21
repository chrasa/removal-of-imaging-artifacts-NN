import numpy as np
from SimulationSetup import SimulationSetup

setup = SimulationSetup()

U_rtm = np.memmap("RTM_U_Test.npy", np.float64, 'r', shape=(2*setup.N_t, setup.N*setup.N, setup.N_s))
U_0 = np.memmap("U_0.npy", np.float64, 'r', shape=(2*setup.N_t,setup.N*setup.N, setup.N_s))

I = np.zeros(setup.N*setup.N, dtype=np.float64)

U_rtm_temp = np.zeros([2*setup.N_t, setup.N*setup.N], dtype=np.float64)
U_0_temp = np.zeros([2*setup.N_t, setup.N*setup.N], dtype=np.float64)


# Integrate
for sidx in range(setup.N_s):
    print("Init arrays")
    U_rtm_temp[:,:] = U_rtm[:,:,sidx]
    U_0_temp[:,:] = U_0[:,:,sidx]
    print("Calculating")
    for tidx in range(2*setup.N_t-1):
        t_idx_backward = 2*setup.N_t -1 - tidx
        t_idx_forward = tidx

        print(f"forward idx: {t_idx_forward}, backward idx: {t_idx_backward}")

        I_0 = U_0_temp[t_idx_backward,:] * U_rtm_temp[t_idx_forward,:]
        I_1 = U_0_temp[t_idx_backward-1,:] * U_rtm_temp[t_idx_forward+1,:]

        I += (I_1 - I_0)/setup.tau
        print(sidx)
