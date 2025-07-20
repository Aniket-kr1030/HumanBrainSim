import numpy as np

class AttentionModule:
    """Simple multi-head self-attention stub."""
    def __init__(self, dim: int, num_heads: int = 1):
        self.dim = dim
        self.num_heads = num_heads
        self.W_q = np.random.randn(num_heads, dim, dim) * 0.1
        self.W_k = np.random.randn(num_heads, dim, dim) * 0.1
        self.W_v = np.random.randn(num_heads, dim, dim) * 0.1

    def resize(self, dim: int):
        new_q = np.random.randn(self.num_heads, dim, dim) * 0.1
        new_k = np.random.randn(self.num_heads, dim, dim) * 0.1
        new_v = np.random.randn(self.num_heads, dim, dim) * 0.1
        m = min(dim, self.dim)
        new_q[:, :m, :m] = self.W_q[:, :m, :m]
        new_k[:, :m, :m] = self.W_k[:, :m, :m]
        new_v[:, :m, :m] = self.W_v[:, :m, :m]
        self.dim = dim
        self.W_q, self.W_k, self.W_v = new_q, new_k, new_v

    def forward_qkv(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        outputs = []
        for i in range(self.num_heads):
            q_proj = self.W_q[i] @ q
            k_proj = self.W_k[i] @ k
            v_proj = self.W_v[i] @ v
            attn_weights = np.exp(q_proj @ k_proj / np.sqrt(self.dim))
            attn_weights = attn_weights / (attn_weights.sum() + 1e-8)
            outputs.append(v_proj * attn_weights)
        return np.mean(outputs, axis=0)
