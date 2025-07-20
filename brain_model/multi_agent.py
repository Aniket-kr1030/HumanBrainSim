class MultiAgentModule:
    """Minimal multi-agent communication manager."""
    def __init__(self):
        self.agents = {}
        self.messages = {}

    def register_agent(self, agent_id, config=None):
        self.agents[agent_id] = config or {}
        self.messages[agent_id] = []

    def broadcast_state(self, agent_id, state):
        for aid in self.agents:
            if aid != agent_id:
                self.messages[aid].append((agent_id, state))

    def receive_messages(self, agent_id):
        msgs = self.messages.get(agent_id, [])
        self.messages[agent_id] = []
        return msgs
