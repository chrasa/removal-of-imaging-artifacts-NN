import numpy as np
from scipy.stats import truncnorm, norm, uniform
from scipy.signal import convolve2d as conv2
import sys
from dataclasses import dataclass
import random

@dataclass
class FractureSetup:
    image_width: int = 512
    image_height: int = 512
    fractured_region_width: int = 350
    fractured_region_height: int = 175
    O_x: int = 25
    O_y: int = 81
    n_fractures_min: int = 2
    n_fractures_max: int = 6
    fracture_width: int = 4
    buffer_size: int = 40
    max_length: float = 50.0 
    min_length: float = 20.0
    std_dev_length: float = 10.0
    std_dev_angle: float = 30.0
    mean_noise: float = 1.0
    std_dev_noise: float = 0.2
    max_iterations: int = 15
    background_velocity: float = 1000.0


class FractureGenerator:
    def __init__(self, fracture_setup: FractureSetup = FractureSetup() ):

        self.setup = fracture_setup

        mean_length = (self.setup.max_length + self.setup.max_length) / 2
        a_length = (self.setup.min_length - mean_length) / self.setup.std_dev_length
        b_length = (self.setup.max_length - mean_length) / self.setup.std_dev_length

        self.length_distribution = truncnorm(a=a_length, b=b_length, loc=mean_length, scale=self.setup.std_dev_length)
        #self.length_distribution = uniform(loc=self.setup.min_length, scale=self.setup.max_length)
        self.angle_distribution = norm(loc=-90, scale=self.setup.std_dev_angle)
        self.n_fractures_distribution = uniform(loc=self.setup.n_fractures_min, scale=(self.setup.n_fractures_max-self.setup.n_fractures_min + 1))

        self.x_low = self.setup.O_x
        self.x_high = self.x_low + self.setup.fractured_region_width


        self.y_low = self.setup.O_y
        self.y_high = self.y_low + self.setup.fractured_region_height

        a_low = (0.3 - 0.45) / 0.05
        b_low = (0.6 - 0.45) / 0.05 
        self.low_velocity_modifier = truncnorm(a_low, b_low, loc=0.45, scale=0.05)

        a_high = (1.5 - 2.25) / 0.25
        b_high = (3.0 - 2.25) / 0.25
        self.high_velocity_modifier = truncnorm(a_high, b_high, loc=2.25, scale=0.25)
        self.modifier_distributions = [self.low_velocity_modifier, self.high_velocity_modifier]

    def generate_fractures(self):
        self.fracture_image = np.full((self.setup.image_height, self.setup.image_width), self.setup.background_velocity)
        n_fractures = self.n_fractures_distribution.rvs().astype(int)
        for _ in range(n_fractures):
            n_iterations = 0
            fracture_is_valid = False
            selected_modifier = np.random.choice(self.modifier_distributions)
            modifier_value = selected_modifier.rvs()

            while (not fracture_is_valid) and (n_iterations < self.setup.max_iterations): 
                fracture_length = self.length_distribution.rvs().astype(int)
                fracture_angle = self.angle_distribution.rvs()
                pixels_to_fracture = []

                # Sample a valid starting position for the fracture
                xs, ys = self._get_fracture_starting_position()                

                pixels_to_fracture.append((xs, ys))

                fracture_is_valid = self._draw_fracture(xs, ys, fracture_length, fracture_angle, pixels_to_fracture)

                if not fracture_is_valid:
                    continue

                # Create the fracture
                self._create_buffer(self.fracture_image, pixels_to_fracture)
                for x, y in pixels_to_fracture:
                    self._fracture_pixel(self.fracture_image, x, y, modifier_value)

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

    def _draw_fracture(self, xs, ys, fracture_length, fracture_angle, pixels_to_fracture):
        
        fractured_pixels = 1
        while fractured_pixels < fracture_length:
            xs = xs + np.cos(fracture_angle)
            ys = ys + np.sin(fracture_angle)

            x_int = xs.astype(int)
            y_int = ys.astype(int)

            if self._is_invalid_pixel(x_int, y_int):
                return False

            if (x_int, y_int) not in pixels_to_fracture:
                pixels_to_fracture.append((x_int, y_int))
                fractured_pixels += 1
        return True

    def _get_fracture_starting_position(self):
        for _ in range(self.setup.max_iterations):
            xs, ys = self._sample_coordinates()
            if not self._is_invalid_pixel(xs, ys):
                return xs, ys
        raise RuntimeError("Unable to fit fracture in image")
    
    def _create_buffer(self, image, pixels_to_fracture):
        for x, y in pixels_to_fracture:
            for i in range(x - self.setup.buffer_size, x + self.setup.buffer_size):
                for j in range(y - self.setup.buffer_size, y + self.setup.buffer_size):
                    if not self._out_of_bounds(i, j):
                        image[j, i] = -1

    def _fracture_pixel(self, image, x, y, modifier_value):
        for i in range(x-int(self.setup.fracture_width/2), x+int(self.setup.fracture_width/2)):
            for j in range(y-int(self.setup.fracture_width/2), y+int(self.setup.fracture_width/2)):
                image[j, i] = self.setup.background_velocity*modifier_value
    
    def _blur_fracture_edges(self, image):
        convolved = image.copy()

        psf = np.array([[1, 2, 1], [2, 3, 2], [1, 2,1 ]], dtype='float32')
        psf *= 1 / np.sum(psf)

        convolved = conv2(convolved, psf, 'valid')

        return convolved

    def _out_of_bounds(self, x, y):
        return x < self.setup.O_x or \
               x >= self.setup.O_x + self.setup.fractured_region_width or \
               y < self.setup.O_y or \
               y >= self.setup.O_y + self.setup.fractured_region_height

    def _sample_coordinates(self):
        xs = int(np.random.uniform(self.x_low, self.x_high))
        ys = int(np.random.uniform(self.y_low, self.y_high))
        return xs, ys

    def _is_invalid_pixel(self, x, y):
        if self._out_of_bounds(x, y):
            return True
        
        elif self._collides_with_fracture(x, y):
            return True
        
        elif self._pixel_in_buffer(x, y):
            return True

        else:
            return False

    def _collides_with_fracture(self, x, y):
        collides = self.fracture_image[y, x] < self.setup.background_velocity and self.fracture_image[y, x] > -1 \
            or self.fracture_image[y, x] > self.setup.background_velocity

        return collides

    def _pixel_in_buffer(self, x, y):
        return self.fracture_image[y, x] == -1

    def _add_noise(self, image: np.array, mean_noise: int, std_dev_noise: int):
        gauss_noise = np.random.normal(loc=self.setup.mean_noise,
                                       scale=self.setup.std_dev_noise,
                                       size=image.size
                                       ).astype(np.float32)

        gauss_noise = gauss_noise.reshape(*image.shape)
        noisy_image = np.add(image, gauss_noise)

        return noisy_image
        """
        rvs_vec = np.vectorize(self._noise_helper,
                        excluded=['distribution'],
                        otypes=['float32'])
        return rvs_vec(p=image, distribution=self.noise_distribution)
        """

    def _noise_helper(self, p, distribution):
        return p + distribution.rvs()

    def get_imaging_region_indices(self):
        im_y_indices = range(self.setup.O_y, self.setup.O_y+self.setup.fractured_region_width)
        im_x_indices = range(self.setup.O_x, self.setup.O_x+self.setup.fractured_region_height)
        indices = [y*self.setup.image_height + x for y in im_y_indices for x in im_x_indices] 

        return indices
    
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
