import numpy as np
from typing import List, Tuple

class BrainResponder:
    """Return stored sentences based on hippocampal recall vectors."""

    def __init__(self):
        self.memory: List[Tuple[np.ndarray, str]] = []

    def add(self, activation: np.ndarray, sentence: str) -> None:
        self.memory.append((activation.astype(float), sentence))

    def respond(self, cue: np.ndarray) -> str:
        if not self.memory:
            return "..."
        cue_norm = cue.astype(float)
        cue_norm /= np.linalg.norm(cue_norm) + 1e-8
        sims = []
        for vec, _ in self.memory:
            v = vec.astype(float)
            v /= np.linalg.norm(v) + 1e-8
            sims.append(float(np.dot(v, cue_norm)))
        idx = int(np.argmax(sims))
        return self.memory[idx][1]
