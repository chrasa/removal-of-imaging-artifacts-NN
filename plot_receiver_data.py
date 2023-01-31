import sys
import numpy as np
import matplotlib.pyplot as plt
from os import path

def plot_reciever_data(D_0_path, tx_sensor, rx_sensor):
    D_0 = np.load(D_0_path)
    print(f"Shape of M_0: {D_0.shape} (Time, TX-Sensor, RX-Sensor)")

    plt.plot(D_0[:,tx_sensor,rx_sensor])
    plt.title("Receiver Data")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    D_0_path = "bs_test" + path.sep + "D_0.npy"
    tx_sensor = 0
    rx_sensor = 0

    for i, arg in enumerate(sys.argv):
        if arg == '-p':
            D_0_path = sys.argv[i+1]
        elif arg == '-tx':
            tx_sensor = int(sys.argv[i+1])
        elif arg == '-rx':
            rx_sensor = int(sys.argv[i+1])
    plot_reciever_data(D_0_path, tx_sensor, rx_sensor)