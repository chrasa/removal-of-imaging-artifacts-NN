import sys
import random
import numpy as np
from scipy.stats import truncnorm, norm, uniform
from FractureSetup import FractureSetup
from FractureDrawer import FractureDrawer

class FractureGenerator(FractureDrawer):
    def __init__(self, fracture_setup: FractureSetup = FractureSetup() ):
        super(FractureGenerator, self).__init__(fracture_setup=fracture_setup)

        self.setup = fracture_setup
        self.init_distributions()

    def init_distributions(self):
        mean_length = (self.setup.max_length + self.setup.max_length) / 2
        a_length = (self.setup.min_length - mean_length) / self.setup.std_dev_length
        b_length = (self.setup.max_length - mean_length) / self.setup.std_dev_length

        self.length_distribution = truncnorm(a=a_length, b=b_length, loc=mean_length, scale=self.setup.std_dev_length)
        #self.length_distribution = uniform(loc=self.setup.min_length, scale=self.setup.max_length)
        self.angle_distribution = norm(loc=-90, scale=self.setup.std_dev_angle)
        self.n_fractures_distribution = uniform(loc=self.setup.n_fractures_min, scale=(self.setup.n_fractures_max-self.setup.n_fractures_min + 1))

        a_low = (0.3 - 0.45) / 0.05
        b_low = (0.6 - 0.45) / 0.05 
        low_velocity_modifier = truncnorm(a_low, b_low, loc=0.45, scale=0.05)

        a_high = (1.5 - 2.25) / 0.25
        b_high = (3.0 - 2.25) / 0.25
        high_velocity_modifier = truncnorm(a_high, b_high, loc=2.25, scale=0.25)
        self.modifier_distributions = [low_velocity_modifier, high_velocity_modifier]

    def generate_fractures(self):
        self.fracture_image = np.full((self.setup.image_height, self.setup.image_width), self.setup.background_velocity)
        n_fractures = self.n_fractures_distribution.rvs().astype(int)
        for _ in range(n_fractures):
            n_iterations = 0
            fracture_is_valid = False
            fracture_velocity = np.random.choice(self.modifier_distributions).rvs() * self.setup.background_velocity

            while n_iterations < self.setup.max_iterations: 
                n_iterations += 1
                fracture_length = self.length_distribution.rvs().astype(int)
                fracture_angle = self.angle_distribution.rvs()
                xs, ys = self._get_fracture_starting_position()
                fracture_is_valid = self.draw_fracture(xs, ys, fracture_length, fracture_angle, fracture_velocity)

                if fracture_is_valid:
                    break
                else:
                    continue             

            if not fracture_is_valid:
                raise RuntimeError("Unable to fit fracture in image")

        # Produce the resulting image
        self.fracture_image[self.fracture_image == -1] = self.setup.background_velocity # Remove the buffer
        # fracture_image = self._blur_fracture_edges(fracture_image)
        # fracture_image = self._add_noise(fracture_image, 1, 0.1)
        # fracture_image = tf.convert_to_tensor(fracture_image)
        # resulting_image = tf.math.add(image, fracture_image)
        self.fracture_image = self.fracture_image.reshape(self.setup.image_width*self.setup.image_height)

        return self.fracture_image, self.fracture_image[self.get_imaging_region_indices()]
    
    def _get_fracture_starting_position(self):
        for _ in range(self.setup.max_iterations):
            xs, ys = self._sample_coordinates()
            if not self._is_invalid_pixel(xs, ys):
                return xs, ys
        raise RuntimeError("Unable to fit fracture in image")
    
    def generate_point_target(self, x=256, y=256, c_circle=2500):
        xx, yy = np.mgrid[:self.setup.image_width, :self.setup.image_height]
        circle = (xx - x) ** 2 + (yy - y) ** 2

        background = np.full((self.setup.image_width, self.setup.image_height), self.setup.background_velocity)
        circle = (circle < 100) * (c_circle - self.setup.background_velocity)

        fracture = background + circle
        return fracture.reshape((self.setup.image_width*self.setup.image_height))
    
    def _binomial_distribution(self):
        return random.choice([0,1])


def normalize_image(image: np.array):
    normalized = image.astype(np.float32)
    # normalized = normalized / tf.reduce_max(tf.abs(normalized))
    normalized = normalized / np.max(normalized)

    return normalized

def print_progress(progress, max, progress_bar_length=40):
        title = f"\rImages generated: {progress:5}/{max:5}: "
        success_rate = f" {(progress/max)*100:3.2f}%"
        number_of_progress_indicators = int(progress * progress_bar_length // (max))

        sys.stdout.write(title + "[" + number_of_progress_indicators*"#" + (progress_bar_length - number_of_progress_indicators)*"-" + "]" + success_rate)


def main():
    n_images = 10

    for i, arg in enumerate(sys.argv):
        if arg == '-n':
            n_images = int(sys.argv[i+1])

    fracture_setup = FractureSetup(
        O_x=25,
        O_y=180,
        fractured_region_height=140,
        fractured_region_width=155,
        n_fractures_min=2,
        n_fractures_max=4,
        max_iterations=60
    )

    generator = FractureGenerator(fracture_setup)

    for i in range(n_images):
        print_progress(i+1, n_images)
        image, _ = generator.generate_fractures()
        np.save(f"./fractures/im{i}.npy", image)
    
    sys.stdout.write("\n")

    # circle = generator.generate_point_target(y=100)
    # np.save(f"./fractures/circle.npy", circle)

if __name__ == "__main__":
    main()
