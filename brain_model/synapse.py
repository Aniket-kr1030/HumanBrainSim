class Synapse:
    def __init__(self, pre, post, init_w=0.0):
        self.pre = pre
        self.post = post
        self.weight = init_w

    def stdp_update(self, pre_spike: bool, post_spike: bool, modulator):
        if pre_spike and post_spike:
            self.weight += 0.01 * modulator.level
        elif pre_spike:
            self.weight -= 0.005
