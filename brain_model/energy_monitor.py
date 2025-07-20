class EnergyMonitorModule:
    """Check energy usage and enable scaling of activity."""
    def __init__(self, max_usage: float = 1.0):
        self.max_usage = max_usage
        self.current_usage = 0.0

    def update_usage(self, usage: float):
        self.current_usage = usage

    def check_and_scale(self) -> float:
        if self.current_usage > self.max_usage:
            return self.max_usage / (self.current_usage + 1e-8)
        if self.current_usage < 0.5 * self.max_usage and self.current_usage > 0:
            return min(2.0, self.max_usage / (self.current_usage + 1e-8))
        return 1.0
