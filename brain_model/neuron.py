import numpy as np

class Neuron:
    def __init__(self, n_inputs: int):
        self.n_inputs = n_inputs
        self.weights = np.random.randn(n_inputs) * 0.1

    def respond(self, x: np.ndarray) -> float:
        return np.dot(self.weights, x)
