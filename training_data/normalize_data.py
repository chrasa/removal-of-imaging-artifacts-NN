import sys
import numpy as np

class DataNormalizer:
    def __init__(self) -> None:
        self.training_data_file_name = "training_data.npy"
        print("Loading training data...")
        self.training_data = np.load(self.training_data_file_name)

    def normalize_data(self):
        self.__remove_background_velocity()

        rom_max = np.max(self.training_data[:,2,:])
        print(f"ROM max value: {rom_max}")
        rom_min = np.min(self.training_data[:,2,:])
        print(f"ROM min value: {rom_min}")
        rom_abs_max = np.max(np.abs([rom_max, rom_min]))
        self.training_data[:,2,:] = self.training_data[:,2,:]/rom_abs_max
        
        rtm_max = np.max(self.training_data[:,1,:])
        print(f"RTM max value: {rtm_max}")
        rtm_min = np.min(self.training_data[:,1,:])
        print(f"RTM min value: {rtm_min}")
        rtm_abs_max = np.max(np.abs([rtm_max, rtm_min]))
        self.training_data[:,1,:] = self.training_data[:,1,:]/rtm_abs_max

        frac_max = np.max(self.training_data[:,0,:])
        print(f"Fracture max value: {frac_max}")
        frac_min = np.min(self.training_data[:,0,:])
        print(f"Fracture min value: {frac_min}")
        frac_abs_max = np.max(np.abs([frac_max, frac_min]))
        self.training_data[:,0,:] = self.training_data[:,0,:]/frac_abs_max


        print("New max and min values")
        rom_max = np.max(self.training_data[:,2,:])
        print(f"ROM max value: {rom_max}")
        rom_min = np.min(self.training_data[:,2,:])
        print(f"ROM min value: {rom_min}")
        
        rtm_max = np.max(self.training_data[:,1,:])
        print(f"RTM max value: {rtm_max}")
        rtm_min = np.min(self.training_data[:,1,:])
        print(f"RTM min value: {rtm_min}")

        frac_max = np.max(self.training_data[:,0,:])
        print(f"Fracture max value: {frac_max}")
        frac_min = np.min(self.training_data[:,0,:])
        print(f"Fracture min value: {frac_min}")

    def remove_invalid_data(self):
        frac_data = self.training_data[:,0,:]
        invalid_data_idx = []
        for i in range(frac_data.shape[0]):
            if np.min(frac_data[i]) < 10:
                print(f"Images with index {i} are invalid")
                invalid_data_idx.append(i)

        self.training_data = np.delete(self.training_data, invalid_data_idx, axis=0)
        print(f"Shape of cleaned training data: {self.training_data.shape}")

    def save_training_data(self):
        np.save("training_data_normalized.npy", self.training_data)

    def __remove_background_velocity(self):
        self.training_data[:,0,:] = self.training_data[:,0,:] - 1000.0



def main():
    data_normalizer = DataNormalizer()
    data_normalizer.remove_invalid_data()
    data_normalizer.normalize_data()
    data_normalizer.save_training_data()


if __name__ == "__main__":
    main()
