import numpy as np

class RLValuator:
    def __init__(self, state_dim: int, gamma: float = 0.9, lr: float = 0.1):
        self.state_dim = state_dim
        self.gamma = gamma
        self.lr = lr
        self.weights = np.zeros(state_dim)

    def resize(self, state_dim: int):
        new_w = np.zeros(state_dim)
        n = min(state_dim, self.state_dim)
        new_w[:n] = self.weights[:n]
        self.state_dim = state_dim
        self.weights = new_w

    def value(self, state: np.ndarray) -> float:
        return np.dot(self.weights, state)

    def td_update(self, state: np.ndarray, reward: float, next_state: np.ndarray) -> float:
        v = self.value(state)
        v_next = self.value(next_state)
        td_error = reward + self.gamma * v_next - v
        self.weights += self.lr * td_error * state
        return td_error
