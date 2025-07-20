import numpy as np
from typing import List

class MemorySystem:
    def __init__(self, dim: int):
        self.dim = dim
        self.store_matrix = []

    def store(self, pattern: np.ndarray):
        self.store_matrix.append(pattern)

    def recall(self, cue: np.ndarray, steps: int = 5) -> np.ndarray:
        if not self.store_matrix:
            return cue
        sim = [np.dot(p, cue) for p in self.store_matrix]
        idx = int(np.argmax(sim))
        return self.store_matrix[idx]

    def initialize(self, patterns: List[np.ndarray]):
        for p in patterns:
            self.store(p)
