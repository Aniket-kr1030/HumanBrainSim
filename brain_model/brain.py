import numpy as np
from typing import List, Dict, Optional
from scipy.spatial.distance import pdist, squareform

from .neuromodulator import Neuromodulator
from .cortex import CortexModule
from .hippocampus import HippocampusModule
from .rl import RLValuator
from .default_mode import DefaultModeNetworkModule
from .homeostasis import HomeostasisModule
from .social import SocialModule
from .generative_replay import GenerativeReplayModule
from .thought_generator import ThoughtGeneratorModule
from .attention import AttentionModule
from .meta_learning import MetaLearningModule
from .energy_monitor import EnergyMonitorModule
from .lifelong import LifelongLearningModule
from .chemical import ChemicalCompositionModule
from .interneuron_microcircuit import InterneuronMicrocircuitModule
from .dashboard import DashboardServer

class HierarchicalBrainModel:
    def __init__(self, input_dim: int, cortex_layers: List[int], hippocampus_dim: int, ach_thresh: float = 0.5, device: str = "cpu"):
        self.input_dim = input_dim
        self.cortex = CortexModule(cortex_layers, device=device)
        self.hippocampus = HippocampusModule(hippocampus_dim)
        self.dopamine = Neuromodulator("DA")
        self.ach = Neuromodulator("ACh")
        self.critic = RLValuator(cortex_layers[-1])
        self.default_mode = DefaultModeNetworkModule(input_dim)
        self.homeostasis = HomeostasisModule()
        self.social = SocialModule(cortex_layers[-1])
        self.replay = GenerativeReplayModule(self.hippocampus)
        self.thought_gen = ThoughtGeneratorModule(input_dim)
        self.attention = AttentionModule(cortex_layers[-1])
        self.meta = MetaLearningModule()
        self.energy_monitor = EnergyMonitorModule()
        self.lifelong = LifelongLearningModule()
        self.chemicals = ChemicalCompositionModule()
        self.microcircuit = InterneuronMicrocircuitModule(cortex_layers[-1])
        self.dashboard = DashboardServer()
        self.microcircuit.connect()
        self.ach_thresh = ach_thresh

    def step(self, x: np.ndarray, reward: float, next_x: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Process one timestep of external or internal input."""
        if next_x is None:
            next_x = self.default_mode.step_spontaneous()

        activations, errors = self.cortex.forward(x)
        next_act, _ = self.cortex.forward(next_x)
        activations = self.attention.forward_qkv(activations, activations, activations)
        osc = self.microcircuit.run_circuit(activations)
        activations = activations + osc
        usage = float(np.linalg.norm(activations))
        self.energy_monitor.update_usage(usage)
        scale = self.energy_monitor.check_and_scale()
        activations = activations * scale

        drive = self.homeostasis.compute_drive_signal()
        predictions = self.social.predict_others(activations)
        social_reward = self.social.compute_social_reward(predictions, next_act)

        total_reward = reward + drive + social_reward

        td_error = self.critic.td_update(activations, total_reward, next_act)
        self.dopamine.step(td_error)
        ach_level = self.ach.step(0.0)

        stored = False
        if ach_level > self.ach_thresh:
            self.hippocampus.store(activations)
            stored = True

        recall = self.hippocampus.recall(activations)
        penalty = self.lifelong.compute_ewc_penalty(self.cortex.layers[-1].weights)
        self.dashboard.update_plots({"reward": total_reward, "penalty": penalty})
        return {
            "activations": activations,
            "errors": errors,
            "td_error": td_error,
            "recall": recall,
            "penalty": penalty,
            "stored": stored,
        }

    def run_dataset(self, loader, reward_fn):
        for x, next_x in loader:
            r = reward_fn(x)
            yield self.step(x, r, next_x)

    def step_spontaneous(self) -> Dict[str, np.ndarray]:
        """Run a spontaneous cycle with internally generated input."""
        x = self.default_mode.step_spontaneous()
        return self.step(x, reward=0.0, next_x=None)

    def compare_with_neuro_data(self, neural_data: np.ndarray, model_signals: np.ndarray, metric: str = "correlation") -> float:
        """Return RSA correlation between neural and model data."""
        data_rdm = squareform(pdist(neural_data, metric))
        model_rdm = squareform(pdist(model_signals, metric))
        corr = np.corrcoef(data_rdm.ravel(), model_rdm.ravel())[0, 1]
        return float(corr)

    def adjust_capacity(self, factor: float):
        """Increase or decrease neural capacity uniformly across modules."""
        self.cortex.resize(factor)
        new_dim = self.cortex.layers[-1].size
        self.hippocampus.resize(new_dim)
        self.critic.resize(new_dim)
        self.social.resize(new_dim)
        self.attention.resize(new_dim)
        self.microcircuit.resize(new_dim)
