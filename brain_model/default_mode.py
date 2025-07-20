import numpy as np

class DefaultModeNetworkModule:
    def __init__(self, input_dim: int, noise_level: float = 0.1):
        self.input_dim = input_dim
        self.noise_level = noise_level

    def set_noise_level(self, alpha: float):
        self.noise_level = alpha

    def step_spontaneous(self) -> np.ndarray:
        return np.random.randn(self.input_dim) * self.noise_level
