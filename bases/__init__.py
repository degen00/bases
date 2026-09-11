"""Bases: Dots and Boxes with a tabular Q-learning opponent."""
__version__ = "0.2.0"

__all__ = ["Bases", "Board", "Geometry", "Transition", "HumanPlayer",
           "QLearningAgent", "RandomAgent", "GreedyAgent", "MinimaxAgent",
           "train_agents", "evaluate", "evaluate_hyperparameters",
           "hyperparameter_tuning", "load_config", "cfg"]

from .board import Board, Geometry, Transition
from .config import load_config
from .game import Bases
from .player import HumanPlayer
from .agent import QLearningAgent, RandomAgent, GreedyAgent, MinimaxAgent
from .train import train_agents, evaluate, evaluate_hyperparameters, hyperparameter_tuning

#: Configuration loaded at import time (see :mod:`bases.config` for lookup order).
cfg = load_config()
