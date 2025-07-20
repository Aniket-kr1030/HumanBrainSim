from .memory import MemorySystem

class HippocampusModule(MemorySystem):
    def resize(self, dim: int):
        """Resize memory dimension and truncate stored patterns."""
        self.dim = dim
        self.store_matrix = [p[:dim] for p in self.store_matrix]
