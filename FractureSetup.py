from dataclasses import dataclass

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
    fracture_width: int = 2
    buffer_size: int = 40
    max_length: float = 50.0 
    min_length: float = 20.0
    std_dev_length: float = 10.0
    std_dev_angle: float = 30.0
    mean_noise: float = 1.0
    std_dev_noise: float = 0.2
    max_iterations: int = 15
    background_velocity: float = 1000.0
    double_fracture_radius_min: int = 5
    double_fracture_radius_max: int = 20
    double_fracture_std_dev_angle: float = 10
    y_fracture_mean_length: int = 15
    y_fracture_std_dev_length: float = 5.0
    y_fracture_mean_angle: float = 0.0
    y_fracture_std_dev_angle: float = 20.0
    y_fracture_mean_arms_angle: float = 45.0
    y_fracture_std_dev_arms_angle: float = 10.0
    double_fracture_probability: float = 0.25
    y_fracture_probability: float = 0.25
