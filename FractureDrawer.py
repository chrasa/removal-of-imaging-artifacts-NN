import numpy as np
from FractureSetup import FractureSetup
from scipy.signal import convolve2d as conv2


class FractureDrawer:
    def __init__(self, fracture_setup: FractureSetup) -> None:
        self.setup = fracture_setup

        self.x_low = self.setup.O_x
        self.x_high = self.x_low + self.setup.fractured_region_width
        self.y_low = self.setup.O_y
        self.y_high = self.y_low + self.setup.fractured_region_height
    
    def draw_fracture(self, xs, ys, length, angle, velocity):
        self.pixels_to_fracture = []
        self.pixels_to_fracture.append((xs, ys))

        fracture_is_valid = self._draw_line(xs, ys, length, angle)

        if not fracture_is_valid:
            return False

        self._create_buffer()
        for x, y in self.pixels_to_fracture:
            self._fracture_pixel(x, y, velocity)
        return True

    def _draw_line(self, xs, ys, fracture_length, fracture_angle):

        fractured_pixels = 1
        while fractured_pixels < fracture_length:
            xs = xs + np.cos(fracture_angle)
            ys = ys + np.sin(fracture_angle)

            x_int = xs.astype(int)
            y_int = ys.astype(int)

            if self._is_invalid_pixel(x_int, y_int):
                return False

            if (x_int, y_int) not in self.pixels_to_fracture:
                self.pixels_to_fracture.append((x_int, y_int))
                fractured_pixels += 1
        return True
    
    def _create_buffer(self):
        for x, y in self.pixels_to_fracture:
            for i in range(x - self.setup.buffer_size, x + self.setup.buffer_size):
                for j in range(y - self.setup.buffer_size, y + self.setup.buffer_size):
                    if not self._out_of_bounds(i, j):
                        self.fracture_image[j, i] = -1

    def _fracture_pixel(self, x, y, velocity):
        for i in range(x-int(self.setup.fracture_width/2), x+int(self.setup.fracture_width/2)):
            for j in range(y-int(self.setup.fracture_width/2), y+int(self.setup.fracture_width/2)):
                self.fracture_image[j, i] = velocity
    
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