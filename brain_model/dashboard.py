class DashboardServer:
    """Placeholder for a monitoring dashboard."""
    def __init__(self):
        self.running = False

    def launch(self):
        self.running = True
        print("Dashboard launched")

    def update_plots(self, data):
        if self.running:
            print("Dashboard update", data.keys())
