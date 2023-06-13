import os
import numpy as np

class DataLoader:
    def __init__(self) -> None:
        self.training_data_folder = f"..{os.path.sep}uppmax_data{os.path.sep}"
        self.training_data_file_name = "training_data.npy"
        self.folders = self.__get_directory_names()
        
    def __get_directory_names(self):
        folders = os.listdir(self.training_data_folder)
        folders.sort()
        return folders

    def __get_training_data_file_path(self, folder):
        return self.training_data_folder + folder + os.path.sep + self.training_data_file_name

    def __get_number_of_training_images(self):
        n_training_images = 0
        for folder in self.folders:
            file_path = self.__get_training_data_file_path(folder)
            data = np.load(file_path)
            print(f"File {file_path} contains {data.shape[0]} training images")
            n_training_images += data.shape[0]
            del data
        return n_training_images
    
    def load_and_merge_training_data(self):
        data = np.load(self.__get_training_data_file_path(self.folders[0]))
        for folder in self.folders[1:-1]:
            file_path = self.__get_training_data_file_path(folder)
            print(f"Loading file: {file_path}")
            new_data = np.load(file_path)
            print(new_data.shape)
            data = np.append(data, new_data, axis=0)
            del new_data
        print(f"Merged all data sets. Size of merged data: {data.shape}. Saving merged data to disc...")
        np.save(self.training_data_file_name, data)



def main():
    data_loader = DataLoader()
    data_loader.load_and_merge_training_data()


if __name__ == "__main__":
    main()
