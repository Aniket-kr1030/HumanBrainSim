import numpy as np
from typing import List
from .memory import MemorySystem

class GenerativeReplayModule:
    def __init__(self, memory: MemorySystem):
        self.memory = memory

    def replay(self, num_patterns: int, noise_sigma: float = 0.1):
        outputs = []
        for _ in range(num_patterns):
            pattern = self.memory.recall(np.random.randn(self.memory.dim))
            noisy = pattern + np.random.randn(*pattern.shape) * noise_sigma
            outputs.append(noisy)
        return outputs
