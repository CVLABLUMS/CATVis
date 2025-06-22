"""
Data loading and preprocessing modules for CATVis.
"""

from .data_loader import CATVisDataLoader, load_config
from .preprocessor import DataPreprocessor, EEGDataset, EEGTextDataset

import torch
import numpy as np
import os


def setup_deterministic_environment(seed: int = 45, use_deterministic_algorithms: bool = True):
    """
    Setup comprehensive deterministic environment for reproducible results.
    
    Args:
        seed: Random seed to use
        use_deterministic_algorithms: Whether to enable deterministic algorithms
    """
    # Set all random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Configure cuDNN for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enable deterministic algorithms (may impact performance)
    if use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
        
        # Set environment variable for cuBLAS deterministic behavior
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


__all__ = [
    'CATVisDataLoader',
    'DataPreprocessor',
    'EEGDataset',
    'EEGTextDataset',
    'load_config',
    'setup_deterministic_environment'
] 