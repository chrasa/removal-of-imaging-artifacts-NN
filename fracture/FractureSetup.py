from dataclasses import dataclass

@dataclass
class FractureSetup:
    image_width: int = 512
    image_height: int = 512
    fractured_region_width: int = 350
    fractured_region_height: int = 175
    O_x: int = 25
    O_y: int = 81
    n_fractures: int = 4
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
