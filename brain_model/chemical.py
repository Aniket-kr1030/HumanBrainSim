class ChemicalCompositionModule:
    """Track multiple neuromodulator concentrations."""
    def __init__(self):
        self.state = {
            'DA': 0.0,
            '5HT': 0.0,
            'NE': 0.0,
            'CORT': 0.0,
            'OXYT': 0.0,
            'ENDO': 0.0,
        }

    def compute_target_state(self):
        return {k: 0.0 for k in self.state}

    def update_state(self, deltas):
        for k, v in deltas.items():
            if k in self.state:
                self.state[k] += v

    def get_current_state(self):
        return dict(self.state)

    def register_custom_state(self, name, value=0.0):
        self.state[name] = value
