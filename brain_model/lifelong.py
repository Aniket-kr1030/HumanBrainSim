import numpy as np

class LifelongLearningModule:
    """Elastic Weight Consolidation-style consolidation."""
    def __init__(self, lam: float = 0.1):
        self.lam = lam
        self.prev_params = None
        self.fisher = None

    def record_fisher(self, params: np.ndarray, grads: np.ndarray):
        if self.fisher is None:
            self.fisher = grads ** 2
        else:
            self.fisher = 0.9 * self.fisher + 0.1 * (grads ** 2)
        self.prev_params = params.copy()

    def compute_ewc_penalty(self, params: np.ndarray) -> float:
        if self.prev_params is None or self.fisher is None:
            return 0.0
        diff = params - self.prev_params
        return float(self.lam * np.sum(self.fisher * diff ** 2))

    def consolidation_replay(self, memory, num_patterns: int):
        return memory.recall(np.random.randn(memory.dim), steps=num_patterns)
