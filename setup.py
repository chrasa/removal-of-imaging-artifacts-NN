from dataclasses import dataclass

@dataclass
class ImageSetup:
    """Class for defining image properties"""
    N_x: int = 512
    N_y: int = 512
    N_x_im: int = 155
    N_y_im: int = 140
    O_x: int = 180
    O_y: int = 25


@dataclass
class FractureSetup(ImageSetup):
    """Class for defining the properties for generating the fracture images"""
    n_fractures_min: int = 3
    n_fractures_max: int = 7
    fracture_width: int = 4
    buffer_size: int = 40
    max_length: float = 50.0 
    min_length: float = 20.0
    std_dev_length: float = 10.0
    std_dev_angle: float = 30.0
    mean_noise: float = 1.0
    std_dev_noise: float = 0.2
    max_iterations: int = 200
    background_velocity: float = 1000.0
    double_fracture_radius_min: int = 20
    double_fracture_radius_max: int = 30
    double_fracture_std_dev_angle: float = 10
    y_fracture_mean_length: int = 25
    y_fracture_std_dev_length: float = 5.0
    y_fracture_mean_angle: float = 0.0
    y_fracture_std_dev_angle: float = 20.0
    y_fracture_mean_arms_angle: float = 45.0
    y_fracture_std_dev_arms_angle: float = 10.0
    polygon_fracture_n_lines_min: int = 3
    polygon_fracture_n_lines_max: int = 5
    polygon_fracture_start_angle_mean: float = 0.0
    polygon_fracture_start_angle_std_dev: float = 10.0
    polygon_fracture_next_angle_mean: float = 0
    polygon_fracture_next_angle_std_dev: float = 20.0
    polygon_fracture_mean_length: float = 10.0
    polygon_fracture_std_dev_length: float = 2.0
    double_fracture_probability: float = 0.2
    y_fracture_probability: float = 0.15
    polygon_fracture_probability: float = 0.4


@dataclass
class SimulationSetup(ImageSetup):
    """Class for defining the simulation properties"""
    N_s: int = 50
    delta_x: float = 0.0063
    tau: float = 3.0303*10**(-5)
    N_t: int = 70
    background_velocity_value: float = 1000
    Bsrc_file: str = "Bsrc_T.txt"
