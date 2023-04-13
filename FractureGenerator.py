import sys
import numpy as np
from scipy.stats import uniform
from FractureSetup import FractureSetup
from FracturePlacer import FracturePlacer
from benchmark import ProgressBar

class FractureGenerator(FracturePlacer):
    def __init__(self, fracture_setup: FractureSetup = FractureSetup() ):
        super(FractureGenerator, self).__init__(fracture_setup=fracture_setup)
        self.n_fractures_distribution = uniform(loc=self.setup.n_fractures_min, scale=(self.setup.n_fractures_max-self.setup.n_fractures_min + 1))


    def generate_fractures(self):
        self.__reset_fracture_image()
        n_fractures_to_place = self.n_fractures_distribution.rvs().astype(int)
        n_fractures = 0

        while n_fractures < n_fractures_to_place:
            n_iterations = 0
            fracture_is_valid = False

            while n_iterations < self.setup.max_iterations: 
                n_iterations += 1

                if self._binomial_distribution():
                    fracture_is_valid = self.add_random_single_fracture()
                    n_new_fractures = 1
                else:
                    fracture_is_valid = self.add_random_double_fracture()
                    n_new_fractures = 2

                if fracture_is_valid:
                    n_fractures += n_new_fractures
                    break
                else:
                    continue             

            if not fracture_is_valid:
                raise RuntimeError("Unable to fit fracture in image")
            
        # self.draw_point_target(256, 100, 10, 500)

        # Produce the resulting image
        self.fracture_image[self.fracture_image == -1] = self.setup.background_velocity # Remove the buffer
        # fracture_image = self._blur_fracture_edges(fracture_image)
        # fracture_image = self._add_noise(fracture_image, 1, 0.1)
        # fracture_image = tf.convert_to_tensor(fracture_image)
        # resulting_image = tf.math.add(image, fracture_image)
        self.fracture_image = self.fracture_image.reshape(self.setup.image_width*self.setup.image_height)

        return self.fracture_image, self.fracture_image[self.get_imaging_region_indices()]
    
    def __reset_fracture_image(self):
        self.fracture_image = np.full((self.setup.image_height, self.setup.image_width), self.setup.background_velocity)


def main():
    n_images = 10

    for i, arg in enumerate(sys.argv):
        if arg == '-n':
            n_images = int(sys.argv[i+1])

    fracture_setup = FractureSetup(
        O_x=156,
        O_y=25,
        fractured_region_height=100,
        fractured_region_width=200,
        n_fractures_min=2,
        n_fractures_max=4,
        max_iterations=200
    )

    generator = FractureGenerator(fracture_setup)
    progress_bar = ProgressBar()

    for i in range(n_images):
        progress_bar.print_progress(i+1, n_images)
        img, _ = generator.generate_fractures()
        np.save(f"fractures/im{i}.npy", img)
    progress_bar.end()
    
    

if __name__ == "__main__":
    main()
