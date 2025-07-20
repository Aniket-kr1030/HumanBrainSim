"""Brain model package."""

from .brain import HierarchicalBrainModel
from .attention import AttentionModule
from .meta_learning import MetaLearningModule
from .energy_monitor import EnergyMonitorModule
from .multi_agent import MultiAgentModule
from .chemical import ChemicalCompositionModule
from .lifelong import LifelongLearningModule
from .interneuron_microcircuit import InterneuronMicrocircuitModule
from .dashboard import DashboardServer
from .speech import SpeechRecognitionModule, TextToSpeechModule
from .text_gen import MarkovResponder
from .brain_responder import BrainResponder
from .hardware_interface import (
    open_camera,
    read_frame,
    show_frame,
    close_camera,
    record_audio,
)

__all__ = [
    "HierarchicalBrainModel",
    "AttentionModule",
    "MetaLearningModule",
    "EnergyMonitorModule",
    "MultiAgentModule",
    "ChemicalCompositionModule",
    "LifelongLearningModule",
    "InterneuronMicrocircuitModule",
    "DashboardServer",
    "SpeechRecognitionModule",
    "TextToSpeechModule",
    "MarkovResponder",
    "BrainResponder",
    "open_camera",
    "read_frame",
    "show_frame",
    "close_camera",
    "record_audio",
]
