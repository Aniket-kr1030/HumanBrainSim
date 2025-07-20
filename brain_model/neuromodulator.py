import numpy as np

class Neuromodulator:
    """Simple neuromodulator with reward-dependent dynamics."""

    def __init__(self, name: str, baseline: float = 0.0, decay: float = 0.01):
        self.name = name
        self.level = baseline
        self.decay = decay

    def step(self, reward: float) -> float:
        """Update neuromodulator level based on reward."""
        self.level += reward
        self.level *= (1.0 - self.decay)
        return self.level
