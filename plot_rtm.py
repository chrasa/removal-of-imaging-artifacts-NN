import numpy as np
from matplotlib import pyplot as plt
import sys

n = 10

for i, arg in enumerate(sys.argv):
    if arg == '-n':
        n = int(sys.argv[i+1])

U = np.load("RTM_U.npy")

slice = U[n,:]
slice = np.reshape(slice, [512,512])

print(np.max(slice))

plt.imshow(slice)
plt.colorbar(orientation="horizontal")
plt.show()
