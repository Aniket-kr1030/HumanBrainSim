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
        sims = [float(np.dot(vec, cue)) for vec, _ in self.memory]
        idx = int(np.argmax(sims))
        return self.memory[idx][1]
