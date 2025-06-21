"""
Neural network models for CATVis
"""

from .eeg_classifier import EEGClassifier
from .contrastive_encoder import ContrastiveEncoder, clip_style_contrastive_loss

__all__ = ['EEGClassifier', 'ContrastiveEncoder', 'clip_style_contrastive_loss'] 