import numpy as np

class MetaLearningModule:
    """Minimal MAML-style meta-learning stub."""
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def inner_adapt(self, grad: np.ndarray) -> np.ndarray:
        return -self.lr * grad

    def outer_update(self, params: np.ndarray, meta_grad: np.ndarray) -> np.ndarray:
        return params - self.lr * meta_grad
