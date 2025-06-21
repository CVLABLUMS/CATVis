"""
Training utilities and scripts for CATVis models
"""

from .train_classifier import ClassifierTrainer, train_eeg_classifier
from .train_contrastive import ContrastiveTrainer, train_contrastive_model

__all__ = ['ClassifierTrainer', 'ContrastiveTrainer', 'train_eeg_classifier', 'train_contrastive_model'] 