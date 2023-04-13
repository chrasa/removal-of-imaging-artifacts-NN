import sys
import random
import numpy as np
from scipy.stats import truncnorm, norm, uniform
from FractureSetup import FractureSetup
from FractureDrawer import FractureDrawer

class FracturePlacer(FractureDrawer):
    def __init__(self, fracture_setup: FractureSetup = FractureSetup() ):
        super(FracturePlacer, self).__init__(fracture_setup=fracture_setup)
        self.__init_distributions()


    def __init_distributions(self):
        mean_length = (self.setup.max_length + self.setup.max_length) / 2
        a_length = (self.setup.min_length - mean_length) / self.setup.std_dev_length
        b_length = (self.setup.max_length - mean_length) / self.setup.std_dev_length

        self.length_distribution = truncnorm(a=a_length, b=b_length, loc=mean_length, scale=self.setup.std_dev_length)
        #self.length_distribution = uniform(loc=self.setup.min_length, scale=self.setup.max_length)
        self.angle_distribution = norm(loc=0, scale=self.setup.std_dev_angle)
        self.double_fracture_radius_distribution = uniform(loc=self.setup.double_fracture_radius_min, scale=(self.setup.double_fracture_radius_max-self.setup.double_fracture_radius_min+1))
        self.double_fracture_start_point_angle_distribution = uniform(loc=0, scale=360)
        self.double_fracture_angle_distribution = norm(loc=0, scale=self.setup.double_fracture_std_dev_angle)

        self.y_fracture_length_distribution = norm(loc=self.setup.y_fracture_mean_length, scale=self.setup.y_fracture_std_dev_length)
        self.y_fracture_angle_distribution = norm(loc=self.setup.y_fracture_mean_angle, scale=self.setup.y_fracture_std_dev_angle)
        self.y_fracture_arms_angle_distribution = norm(loc=self.setup.y_fracture_mean_arms_angle, scale=self.setup.y_fracture_std_dev_arms_angle)

        self.polygon_n_lines_distribution = uniform(loc=self.setup.polygon_fracture_n_lines_min, scale=(self.setup.polygon_fracture_n_lines_max-self.setup.polygon_fracture_n_lines_min+1))
        self.polygon_start_angle_distribution = norm(loc=self.setup.polygon_fracture_start_angle_mean, scale=self.setup.polygon_fracture_start_angle_std_dev)
        self.polygon_next_angle_distribution = norm(loc=self.setup.polygon_fracture_next_angle_mean, scale=self.setup.polygon_fracture_next_angle_std_dev)
        self.polygon_length_distribution = norm(loc=self.setup.polygon_fracture_mean_length, scale=self.setup.polygon_fracture_std_dev_length)

        a_low = (0.3 - 0.45) / 0.05
        b_low = (0.6 - 0.45) / 0.05 
        low_velocity_modifier = truncnorm(a_low, b_low, loc=0.45, scale=0.05)

        a_high = (1.5 - 2.25) / 0.25
        b_high = (3.0 - 2.25) / 0.25
        high_velocity_modifier = truncnorm(a_high, b_high, loc=2.25, scale=0.25)
        self.modifier_distributions = [low_velocity_modifier, high_velocity_modifier]

    
    def add_random_double_fracture(self):
        fracture_length1 = self.length_distribution.rvs().astype(int)
        fracture_angle1 = self.angle_distribution.rvs()
        if self._binomial_distribution():
            fracture_angle1 += 180
        fracture_velocity1 = np.random.choice(self.modifier_distributions).rvs() * self.setup.background_velocity

        fracture_length2 = self.length_distribution.rvs().astype(int)
        fracture_angle2 = fracture_angle1 + self.double_fracture_angle_distribution.rvs()
        fracture_velocity2 = np.random.choice(self.modifier_distributions).rvs() * self.setup.background_velocity

        xs1, ys1 = self._get_fracture_starting_position()

        s2_radius = self.double_fracture_radius_distribution.rvs()
        s2_angle = self.double_fracture_start_point_angle_distribution.rvs()
        xs2 = int( xs1 + np.cos(np.deg2rad(s2_angle)) * s2_radius )
        ys2 = int( ys1 + np.sin(np.deg2rad(s2_angle)) * s2_radius )
        return self.draw_two_fractures(xs1, ys1, fracture_length1, fracture_angle1, fracture_velocity1, xs2, ys2, fracture_length2, fracture_angle2, fracture_velocity2)


    def add_random_single_fracture(self):
        fracture_length = self.length_distribution.rvs().astype(int)
        fracture_angle = self.angle_distribution.rvs()
        fracture_velocity = np.random.choice(self.modifier_distributions).rvs() * self.setup.background_velocity
        xs, ys = self._get_fracture_starting_position()
        return self.draw_fracture(xs, ys, fracture_length, fracture_angle, fracture_velocity)
    

    def add_random_Y_fracture(self):
        lengths = [self.y_fracture_length_distribution.rvs().astype(int) for _ in range(3)]
        angles = [self.y_fracture_angle_distribution.rvs(),
                  self.y_fracture_arms_angle_distribution.rvs(),
                  -self.y_fracture_arms_angle_distribution.rvs()]
        velocity = np.random.choice(self.modifier_distributions).rvs() * self.setup.background_velocity
        xs, ys = self._get_fracture_starting_position()
        return self.draw_Y_fracture(xs, ys, lengths, angles, velocity)
    

    def add_random_ploygon_fracture(self):
        n_lines = self.polygon_n_lines_distribution.rvs().astype(int)
        xs, ys = self._get_fracture_starting_position()
        start_angle = self.polygon_start_angle_distribution.rvs()
        fracture_velocity = np.random.choice(self.modifier_distributions).rvs() * self.setup.background_velocity

        polygon_pixels = []
        for _ in range(n_lines):
            pixels = self._draw_next_polygon_line(xs, ys, start_angle)
            if pixels is False:
                return False
            xs, ys = pixels[-1]
            for pixel in pixels:
                polygon_pixels.append(pixel)
        
        self._create_buffer(polygon_pixels)
        for x, y in polygon_pixels:
            self._fracture_pixel(x, y, fracture_velocity)
        return True
    
    def _draw_next_polygon_line(self, xs, ys, previous_angle):
        n_iterations = 0
        while n_iterations < self.setup.max_iterations:
            n_iterations += 1
            pixels = self._draw_line(xs, ys, self.polygon_length_distribution.rvs().astype(int), previous_angle + self.polygon_next_angle_distribution.rvs())
            if pixels is False:
                continue
            else:
                return pixels
        raise Exception("Unable to place next polygon line")
    
    
    def _get_fracture_starting_position(self):
        for _ in range(self.setup.max_iterations):
            xs, ys = self._sample_coordinates()
            if not self._is_invalid_pixel(xs, ys):
                return xs, ys
        raise RuntimeError("Unable to fit fracture in image")
    
    def _sample_coordinates(self):
        xs = int(np.random.uniform(self.setup.O_x, self.setup.O_x+self.setup.fractured_region_width))
        ys = int(np.random.uniform(self.setup.O_y, self.setup.O_y+self.setup.fractured_region_height))
        return xs, ys
    
    def _binomial_distribution(self):
        return random.choice([0,1])

