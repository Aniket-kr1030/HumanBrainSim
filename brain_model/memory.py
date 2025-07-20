import numpy as np
from typing import List

class MemorySystem:
    def __init__(self, dim: int):
        self.dim = dim
        self.store_matrix = []

    def store(self, pattern: np.ndarray):
        self.store_matrix.append(pattern)

    def recall(self, cue: np.ndarray, k: int = 1) -> np.ndarray:
        """Return the average of the top ``k`` stored patterns by cosine similarity."""
        if not self.store_matrix:
            return cue
        cue_norm = cue.astype(float)
        cue_norm /= np.linalg.norm(cue_norm) + 1e-8
        mat = np.array([p for p in self.store_matrix], dtype=float)
        mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
        sims = mat_norm @ cue_norm
        idx = np.argsort(sims)[-k:]
        return mat[idx].mean(axis=0)

    def initialize(self, patterns: List[np.ndarray]):
        for p in patterns:
            self.store(p)
