import numpy as np

class InterneuronMicrocircuitModule:
    """Simple excitatory-inhibitory oscillation generator."""
    def __init__(self, size: int):
        self.size = size
        self.excit = np.zeros(size)
        self.inhib = np.zeros(size)

    def connect(self):
        self.W_ei = np.random.randn(self.size, self.size) * 0.1
        self.W_ie = np.random.randn(self.size, self.size) * 0.1

    def run_circuit(self, x: np.ndarray) -> np.ndarray:
        self.excit = np.tanh(self.W_ei @ self.inhib + x)
        self.inhib = np.tanh(self.W_ie @ self.excit)
        return self.excit - self.inhib

    def resize(self, size: int):
        self.size = size
        self.excit = np.zeros(size)
        self.inhib = np.zeros(size)
        self.connect()
