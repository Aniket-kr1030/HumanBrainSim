import numpy as np

class SocialModule:
    def __init__(self, state_dim: int):
        self.state_dim = state_dim
        self.social_weights = np.random.randn(state_dim)

    def resize(self, state_dim: int):
        new_w = np.random.randn(state_dim)
        n = min(state_dim, self.state_dim)
        new_w[:n] = self.social_weights[:n]
        self.state_dim = state_dim
        self.social_weights = new_w

    def predict_others(self, state: np.ndarray) -> np.ndarray:
        return np.tanh(self.social_weights * state)

    def compute_social_reward(self, predictions: np.ndarray, outcomes: np.ndarray) -> float:
        error = outcomes - predictions
        return -np.sum(error ** 2)
