import sys
import numpy as np
import matplotlib.pyplot as plt
from os import path

def plot_reciever_data(D_path, D_old_path, tx_sensor, rx_sensor):
    D = np.load(D_path)
    print(f"Shape of D: {D.shape} (Time, TX-Sensor, RX-Sensor)")

    D_old = np.load(D_old_path)
    print(f"Shape of D: {D_old.shape} (Time, TX-Sensor, RX-Sensor)")

    diff = D-D_old
    print(np.max(diff))
    print(D[0:10,0,0])

    plt.plot(D[:,tx_sensor,rx_sensor], label="D")
    plt.plot(D_old[:,tx_sensor, rx_sensor], label="D_old")
    plt.title("Receiver Data")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    D_path = "rtm_data" + path.sep + "D.npy"
    D_old_path = "old_fwd" + path.sep + "D.npy"
    tx_sensor = 0
    rx_sensor = 0

    for i, arg in enumerate(sys.argv):
        if arg == '-p':
            D_0_path = sys.argv[i+1]
        elif arg == '-tx':
            tx_sensor = int(sys.argv[i+1])
        elif arg == '-rx':
            rx_sensor = int(sys.argv[i+1])
    plot_reciever_data(D_path, D_old_path, tx_sensor, rx_sensor)