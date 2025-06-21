"""
Training utilities and scripts for CATVis models
"""

from .train_classifier import ClassifierTrainer
from .train_contrastive import ContrastiveTrainer

__all__ = ['ClassifierTrainer', 'ContrastiveTrainer'] 