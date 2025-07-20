class HomeostasisModule:
    def __init__(self):
        self.state = {
            'energy': 1.0,
            'hydration': 1.0,
            'temperature': 1.0,
            'threat_level': 0.0,
        }

    def update_states(self, delta: dict):
        for k, v in delta.items():
            if k in self.state:
                self.state[k] += v

    def compute_drive_signal(self) -> float:
        drive = 0.0
        for k, v in self.state.items():
            if k == 'threat_level':
                drive += v
            else:
                drive += abs(1.0 - v)
        return drive
