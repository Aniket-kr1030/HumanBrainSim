import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch may not be installed
    torch = None

try:
    import coremltools as ct
except ImportError:  # pragma: no cover - optional
    ct = None

class PredictiveCodingLayer:
    """Predictive-coding layer with optional hardware acceleration."""

    def __init__(self, input_dim: int, size: int, device: str = "cpu"):
        self.input_dim = input_dim
        self.size = size
        self.device = device
        self.weights = np.random.randn(size, input_dim).astype(np.float32) * 0.1
        self._prepare_backend()

    def _prepare_backend(self):
        if self.device == "mps" and torch is not None:
            self.tweights = torch.tensor(self.weights, device="mps")
        elif self.device == "ne" and ct is not None:
            inp = ct.TensorType(shape=(self.input_dim,))
            out = ct.TensorType(shape=(self.size,))
            builder = ct.models.neural_network.NeuralNetworkBuilder(
                [("input", inp)], [("output", out)]
            )
            builder.add_inner_product(
                name="fc",
                W=self.weights,
                b=None,
                input_channels=self.input_dim,
                output_channels=self.size,
                has_bias=False,
                input_name="input",
                output_name="output",
            )
            spec = builder.spec
            self.mlmodel = ct.models.MLModel(spec)
        else:
            self.tweights = None
            self.mlmodel = None

    def resize(self, new_size: int, new_input_dim: int = None):
        """Resize the layer while preserving existing weights where possible."""
        if new_input_dim is None:
            new_input_dim = self.input_dim
        new_w = np.random.randn(new_size, new_input_dim) * 0.1
        rows = min(self.size, new_size)
        cols = min(self.input_dim, new_input_dim)
        new_w[:rows, :cols] = self.weights[:rows, :cols]
        self.size = new_size
        self.input_dim = new_input_dim
        self.weights = new_w.astype(np.float32)
        self._prepare_backend()

    def forward(self, bottom_up: np.ndarray, top_down: np.ndarray) -> np.ndarray:
        if self.device == "mps" and torch is not None:
            x = torch.tensor(bottom_up, dtype=torch.float32, device="mps")
            output_t = torch.matmul(self.tweights, x)
            output = output_t.cpu().numpy()
        elif self.device == "ne" and self.mlmodel is not None:
            output = self.mlmodel.predict({"input": bottom_up.astype(np.float32)})["output"]
        else:
            output = self.weights @ bottom_up
        if top_down.shape == output.shape:
            error = top_down - output
        else:
            error = np.zeros_like(output)
        if self.device == "mps" and torch is not None:
            err_t = torch.tensor(error, dtype=torch.float32, device="mps")
            bu_t = x
            self.tweights += 0.01 * err_t.unsqueeze(1) @ bu_t.unsqueeze(0)
            self.weights = self.tweights.cpu().numpy()
        else:
            self.weights += 0.01 * np.outer(error, bottom_up)
        return output + error
