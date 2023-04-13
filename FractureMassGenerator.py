import os
import sys
from time import sleep
import numpy as np
from multiprocessing import Process, Value
from FractureSetup import FractureSetup
from FractureGenerator import FractureGenerator
from retry import retry
from benchmark import ProgressBar

progress = Value('i', 0)

class FractureMassGenerator(Process):
    def __init__(self, n, start_idx, fracture_setup: FractureSetup) -> None:
        super().__init__()
        self.n = n
        self.start_index = start_idx
        self.fracture_generator = FractureGenerator(fracture_setup)

    def run(self):
        img_idx = self.start_index
        for _ in range(self.n):
            self._generate_fracture_image(img_idx)
            img_idx += 1
            self.__increase_progress_bar()

    @retry(tries=3)
    def _generate_fracture_image(self, idx):
        img, _ = self.fracture_generator.generate_fractures()
        np.save(f"fractures/im{idx}.npy", img)


    def __increase_progress_bar(self):
        with progress.get_lock():
                progress.value += 1



class MpProgressBar(Process):
    def __init__(self, n_images) -> None:
        super().__init__()
        self.n_images = n_images
        self.progress_bar = ProgressBar()

    def run(self):
        old_progress = progress.value

        while progress.value < self.n_images:
            if old_progress == progress.value:
                sleep(0.1)
            else:
                old_progress = progress.value
                self.progress_bar.print_progress(old_progress, self.n_images)
        
        self.progress_bar.end()




def main():
    n_processes = os.cpu_count()
    n_images = n_processes * 10

    for i, arg in enumerate(sys.argv):
        if arg == '-n':
            n_images = int(sys.argv[i+1])
        elif arg == '-p':
            n_processes = int(sys.argv[i+1])

    if (n_images%n_processes) != 0:
        raise Exception("Number of Images needs to be a multiple of the number of processes")
    
    images_per_generator = int(n_images/n_processes)
    print(f"Images per generator: {images_per_generator}")

    fracture_setup = FractureSetup(
        O_x=156,
        O_y=25,
        fractured_region_height=100,
        fractured_region_width=200,
        n_fractures_min=2,
        n_fractures_max=4,
        max_iterations=200
    )

    fracture_generators = [FractureMassGenerator(images_per_generator, start_idx, fracture_setup) for start_idx in range(0, n_images, images_per_generator)]
    progress_bar = MpProgressBar(n_images)
    progress_bar.start()

    for generator in fracture_generators:
        generator.start()

    for generator in fracture_generators:
        generator.join()
    
    if progress.value < n_images:
        print("Failed to generate all the fracture images with the given number of iterations and retries")
        progress_bar.terminate()
    else:
        progress_bar.join()


if __name__ == "__main__":
    main()