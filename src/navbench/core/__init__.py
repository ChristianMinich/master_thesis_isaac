"""Core benchmark logic: ports (interfaces), orchestration, metrics, config, logging.

This package MUST NOT import Isaac Sim, Isaac Lab, torch, or any other heavy
external dependency. Concrete implementations live behind adapters
(``navbench.sim``) and plugins (``navbench.plugins``).
"""

from navbench.core.types import Action, Observation, Pose, StepResult
from navbench.core.scenario import Perturbation, Scenario
from navbench.core.dataset import Episode, EpisodeDataset, InMemoryEpisodeDataset
from navbench.core.agent import Agent
from navbench.core.simulator import SimulatorAdapter
from navbench.core.trainer import Trainer, TrainingResult
from navbench.core.inference import InferenceRunner
from navbench.core.metrics import Metric, MetricResult
from navbench.core.evaluator import EpisodeRecord, Evaluator
from navbench.core.benchmark import BenchmarkRun, BenchmarkRunner
from navbench.core.plugin import ApproachPlugin, PluginRegistry

__all__ = [
    "Action",
    "Agent",
    "ApproachPlugin",
    "BenchmarkRun",
    "BenchmarkRunner",
    "Episode",
    "EpisodeDataset",
    "EpisodeRecord",
    "Evaluator",
    "InMemoryEpisodeDataset",
    "InferenceRunner",
    "Metric",
    "MetricResult",
    "Observation",
    "Perturbation",
    "PluginRegistry",
    "Pose",
    "Scenario",
    "SimulatorAdapter",
    "StepResult",
    "Trainer",
    "TrainingResult",
]