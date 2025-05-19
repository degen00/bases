__all__ = ["Bases", "HumanPlayer", "QLearningAgent", "RandomAgent",
           "train_agents", "evaluate_hyperparameters", "hyperparameter_tuning"]

from .game import Bases
from .player import HumanPlayer
from .agent import QLearningAgent
from .agent import RandomAgent
from .train import train_agents
from .train import evaluate_hyperparameters
from .train import hyperparameter_tuning
import yaml
from pathlib import Path

yaml_file = 'bases.yml'
with open(yaml_file, 'r') as f:
    cfg = yaml.safe_load(f)
POLICY_PATH = cfg['policy_path']
LINES_LOG_PATH = cfg['lines_log']

Path(cfg['policy_path']).parent.mkdir(parents=True, exist_ok=True)
Path(cfg['lines_log']).parent.mkdir(parents=True, exist_ok=True)
Path(cfg['game_log']).parent.mkdir(parents=True, exist_ok=True)
Path(cfg['hyperparameter_tuning_results']).parent.mkdir(
    parents=True, exist_ok=True)
