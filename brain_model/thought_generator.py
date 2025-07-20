import numpy as np
from typing import List

class ThoughtGeneratorModule:
    def __init__(self, dim: int, noise_level: float = 0.1):
        self.dim = dim
        self.noise_level = noise_level

    def set_sequence_noise(self, level: float):
        self.noise_level = level

    def run_cycle(self, num_steps: int) -> List[np.ndarray]:
        activations = []
        state = np.zeros(self.dim)
        for _ in range(num_steps):
            state = state + np.random.randn(self.dim) * self.noise_level
            activations.append(np.copy(state))
        return activations
