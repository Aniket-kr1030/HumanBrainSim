import numpy as np
from typing import List
from .predictive_coding import PredictiveCodingLayer

class CortexModule:
    def __init__(self, layer_sizes: List[int], device: str = "cpu"):
        self.layers = []
        self.device = device
        input_dim = layer_sizes[0]
        for size in layer_sizes:
            self.layers.append(PredictiveCodingLayer(input_dim, size, device=device))
            input_dim = size

    def resize(self, factor: float):
        """Resize all layers by the given factor."""
        input_dim = self.layers[0].input_dim
        for layer in self.layers:
            new_size = max(1, int(layer.size * factor))
            layer.resize(new_size, input_dim)
            input_dim = new_size

    def forward(self, inputs: np.ndarray):
        activations = inputs
        errors = []
        top_down = np.zeros_like(inputs)
        for layer in self.layers:
            err = layer.forward(activations, top_down)
            errors.append(err)
            activations = err
            top_down = err
        return activations, errors
