import numpy as np
import matplotlib.pyplot as plt
import scipy

FUTURE = 2
PRESENT = 1
PAST = 0

source_locations = np.array([15,  25,  35,  44,  54,  64,  74,  84,  93, 103, 113, 123, 133,
                    142, 152, 162, 172, 182, 191, 201, 211, 221, 231, 240, 250, 260,
                    270, 279, 289, 299, 309, 319, 328, 338, 348, 358, 368, 377, 387,
                    397, 407, 417, 426, 436, 446, 456, 466, 475, 485, 495])
B_delta = np.zeros([512*512, 50])

for idx in range(len(source_locations)):
    B_delta[:,idx] = scipy.signal.unit_impulse(512*512, source_locations[idx])

# Load D
D_0 = np.load("./bs_test/D_fine.npy")

# Reverse the time 
D_0 = np.flip(D_0, 0)

print(f"Shape of D_0: {D_0.shape}")
Nt = D_0.shape[0]
nts = 20
Ns = 50
N = 512

background_velocity = 1000
c = np.full(N**2, background_velocity, dtype=np.float64)
I_k = scipy.sparse.identity(N)

delta_x = 0.0063
D_k = (1/delta_x**2)*scipy.sparse.diags([1,-2,1],[-1,0,1], shape=(N,N), dtype=np.float64)
D_k = scipy.sparse.csr_matrix(D_k)
D_k[0, 0] = -1 * (1/delta_x**2)

L = scipy.sparse.kron(D_k, I_k) + scipy.sparse.kron(I_k, D_k)
C = scipy.sparse.diags(c, 0, dtype=np.float64)
A = (- C @ L @ C)

delta_t = 3.0303*10**(-5)/20.0

u = np.zeros([3,512*512,Ns])

U = np.zeros([int(Nt/nts),N*N])

source_idx = 25
for time_idx in range(400):
    u[PAST,:,source_idx] = u[PRESENT,:,source_idx]
    u[PRESENT,:,source_idx] = u[FUTURE,:,source_idx]
    # Use identity matrix for 2*identity - Delta_t * A
    #u[FUTURE,:,source_idx] = (2*u[PRESENT,:,source_idx]) - (delta_t**2 * A*u[PRESENT,:,source_idx]) - u[PAST,:,source_idx] + delta_t**2 *  B_delta @ D_0[time_idx,:,source_idx]
    u[FUTURE,:,source_idx] = (2*scipy.sparse.identity(N**2) - delta_t**2 * A)*u[PRESENT,:,source_idx] - u[PAST,:,source_idx] + delta_t**2 *  B_delta @ D_0[time_idx,:,source_idx]

    print(time_idx)
    if (time_idx % nts) == 0:
        U[int(time_idx/nts),:] = u[FUTURE,:,source_idx]

np.save("RTM_U.npy", U)
# slice = U[10,:]
# slice = np.reshape(slice, [512,512])

# plt.imshow(slice)
# plt.colorbar(orientation="horizontal")
# plt.show()
# plt.plot(D_0[0,0,:])
# plt.show()