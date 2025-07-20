import numpy as np

class SynapticPruningModule:
    def __init__(self, connections: np.ndarray):
        self.connections = connections
        self.prune_rate = 0.01

    def prune(self, threshold: float):
        mask = np.abs(self.connections) >= threshold
        self.connections = self.connections * mask

    def set_prune_rate(self, rate: float):
        self.prune_rate = rate

class SynapticRewiringModule:
    def __init__(self, connections: np.ndarray):
        self.connections = connections
        self.params = {'window': 10, 'k': 5}

    def rewire(self, num_new: int):
        for _ in range(num_new):
            i = np.random.randint(self.connections.shape[0])
            j = np.random.randint(self.connections.shape[1])
            self.connections[i, j] = np.random.randn() * 0.01

    def set_rewire_params(self, coactivity_window: int, selection_k: int):
        self.params['window'] = coactivity_window
        self.params['k'] = selection_k
