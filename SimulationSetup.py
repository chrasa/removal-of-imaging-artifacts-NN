from dataclasses import dataclass

@dataclass
class SimulationSetup:
    """Class for defining the simulation properties"""
    N_x: int = 512
    N_y: int = 512
    N_s: int = 50
    delta_x: float = 0.0063
    tau: float = 3.0303*10**(-5)
    N_t: int = 70
    background_velocity_value: float = 1000
    Bsrc_file: str = "Bsrc_T.txt"
    N_x_im: int = 175
    N_y_im: int = 350
    O_x: int = 25
    O_y: int = 81
    