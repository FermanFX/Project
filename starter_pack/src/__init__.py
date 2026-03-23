"""
Math4AI Capstone Project - Starter Pack

Models and training utilities for the capstone project:
- Softmax Regression (linear baseline)
- One-Hidden-Layer Neural Network
"""

from .models import SoftmaxRegression, OneHiddenLayerNN, softmax_stable
from .optimizers import SGD, Momentum, Adam, create_optimizer
from .trainer import Trainer, SoftmaxTrainer, NNTrainer, TrainingHistory
from .evaluation import Evaluator, RepeatedSeedResult
from .data_loader import DataLoader

__all__ = [
    'SoftmaxRegression',
    'OneHiddenLayerNN',
    'softmax_stable',
    'SGD',
    'Momentum',
    'Adam',
    'create_optimizer',
    'Trainer',
    'SoftmaxTrainer',
    'NNTrainer',
    'TrainingHistory',
    'Evaluator',
    'RepeatedSeedResult',
    'DataLoader'
]
